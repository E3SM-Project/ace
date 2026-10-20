"""A fast xarray backend for netCDF classic files, aimed at CDF-5.

netCDF-C (4.9.3, as used by netCDF4-python and so by xarray's ``netcdf4``
engine) reads classic-format files through a page cache that issues 8 KiB
``read()`` calls. On Lustre that caps a reader at 30-75 MB/s on the E3SM EAM
history files (CDF-5, ~600 MB per record), against ~1 GB/s for plain large
reads. In a classic file, the slab of a record variable at one timestep is
contiguous at a known offset, so this backend reads it with a single
``pread``.

The returned dataset is built from the same raw variables and attributes
that the ``netcdf4`` engine exposes and goes through xarray's own CF decoding,
so ``mask_and_scale``, time decoding and character-array concatenation behave
identically.

CDF-1 (classic) and CDF-2 (64-bit offset) files share the format and are also
supported. Only local paths are supported.

Format reference:
https://docs.unidata.ucar.edu/netcdf-c/current/file_format_specifications.html
"""

import contextlib
import dataclasses
import os
import struct
import threading
from collections.abc import Iterable
from typing import Any

import numpy as np
import xarray as xr
from xarray.backends import BackendArray, BackendEntrypoint
from xarray.backends.common import AbstractDataStore, T_PathFileOrDataStore
from xarray.backends.store import StoreBackendEntrypoint
from xarray.core import indexing
from xarray.core.utils import FrozenDict

_NC_TYPES: dict[int, str] = {
    1: ">i1",  # NC_BYTE
    2: "S1",  # NC_CHAR
    3: ">i2",  # NC_SHORT
    4: ">i4",  # NC_INT
    5: ">f4",  # NC_FLOAT
    6: ">f8",  # NC_DOUBLE
    7: ">u1",  # NC_UBYTE (CDF-5 only, as are the rest)
    8: ">u2",  # NC_USHORT
    9: ">u4",  # NC_UINT
    10: ">i8",  # NC_INT64
    11: ">u8",  # NC_UINT64
}
_NC_CHAR = 2
_NC_DIMENSION = 10
_NC_VARIABLE = 11
_NC_ATTRIBUTE = 12
_STREAMING = 0xFFFFFFFF
_INITIAL_HEADER_BYTES = 256 * 1024


def _pad4(n: int) -> int:
    return (n + 3) // 4 * 4


class _HeaderTruncated(Exception):
    """The header extends beyond the bytes read so far."""


class _HeaderParser:
    def __init__(self, buf: bytes, version: int):
        self._buf = buf
        self.pos = 4
        # CDF-5 widens element counts (and so dimension lengths and vsize) to
        # 64 bits; CDF-2 and CDF-5 widen variable offsets to 64 bits.
        self._count = ">q" if version == 5 else ">i"
        self._offset = ">i" if version == 1 else ">q"

    def _unpack(self, fmt: str) -> int:
        size = struct.calcsize(fmt)
        if self.pos + size > len(self._buf):
            raise _HeaderTruncated()
        (value,) = struct.unpack_from(fmt, self._buf, self.pos)
        self.pos += size
        return value

    def _bytes(self, n: int) -> bytes:
        if self.pos + _pad4(n) > len(self._buf):
            raise _HeaderTruncated()
        out = self._buf[self.pos : self.pos + n]
        self.pos += _pad4(n)
        return out

    def tag(self) -> int:
        return self._unpack(">i")

    def count(self) -> int:
        return self._unpack(self._count)

    def offset(self) -> int:
        return self._unpack(self._offset)

    def name(self) -> str:
        return self._bytes(self.count()).decode("utf-8")

    def list_header(self, expected_tag: int) -> int:
        """Read a list's (tag, nelems) and return nelems."""
        tag, n = self.tag(), self.count()
        if tag == 0 and n == 0:
            return 0
        if tag != expected_tag:
            raise ValueError(
                f"Malformed netCDF header: expected tag {expected_tag}, got {tag}."
            )
        return n

    def attributes(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for _ in range(self.list_header(_NC_ATTRIBUTE)):
            name = self.name()
            nc_type = self.tag()
            n = self.count()
            dtype = np.dtype(_NC_TYPES[nc_type])
            raw = self._bytes(n * dtype.itemsize)
            out[name] = _attribute_value(name, nc_type, raw, dtype)
        return out


def _attribute_value(name: str, nc_type: int, raw: bytes, dtype: np.dtype) -> Any:
    """Convert an attribute to the type netCDF4-python's getncattr returns."""
    if nc_type == _NC_CHAR:
        if name == "_FillValue":
            return raw
        return raw.decode("utf-8", errors="replace").replace("\x00", "")
    values = np.frombuffer(raw, dtype).astype(dtype.newbyteorder("="))
    if values.size == 1:
        return values[0]
    return values


@dataclasses.dataclass(frozen=True)
class _VariableLayout:
    dims: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: np.dtype  # on-disk (big-endian) dtype
    attrs: dict[str, Any]
    begin: int
    is_record: bool

    @property
    def slab_shape(self) -> tuple[int, ...]:
        """Shape of the contiguous block on disk: one record, or the whole."""
        return self.shape[1:] if self.is_record else self.shape

    @property
    def native_dtype(self) -> np.dtype:
        return self.dtype.newbyteorder("=")


@dataclasses.dataclass(frozen=True)
class _Header:
    numrecs: int
    dims: dict[str, int]  # the record dimension has length numrecs
    unlimited_dims: frozenset[str]
    attrs: dict[str, Any]
    variables: dict[str, _VariableLayout]
    recsize: int


def _parse_header(buf: bytes) -> _Header:
    if buf[:3] != b"CDF" or buf[3] not in (1, 2, 5):
        raise ValueError("Not a netCDF classic (CDF-1, CDF-2 or CDF-5) file.")
    p = _HeaderParser(buf, version=buf[3])
    numrecs = p.count()
    if numrecs == _STREAMING:
        raise ValueError("Streaming netCDF files (numrecs unknown) are unsupported.")
    dim_names: list[str] = []
    dims: dict[str, int] = {}
    unlimited: set[str] = set()
    for _ in range(p.list_header(_NC_DIMENSION)):
        name = p.name()
        length = p.count()
        if length == 0:
            unlimited.add(name)
            length = numrecs
        dim_names.append(name)
        dims[name] = length
    attrs = p.attributes()
    variables: dict[str, _VariableLayout] = {}
    for _ in range(p.list_header(_NC_VARIABLE)):
        name = p.name()
        ndims = p.count()
        var_dims = tuple(dim_names[p.count()] for _ in range(ndims))
        var_attrs = p.attributes()
        dtype = np.dtype(_NC_TYPES[p.tag()])
        p.count()  # vsize; recomputed below because CDF-1/2 clip it at 2**32-1
        begin = p.offset()
        variables[name] = _VariableLayout(
            dims=var_dims,
            shape=tuple(dims[d] for d in var_dims),
            dtype=dtype,
            attrs=var_attrs,
            begin=begin,
            is_record=len(var_dims) > 0 and var_dims[0] in unlimited,
        )
    record_vars = [v for v in variables.values() if v.is_record]
    slab_bytes = [int(np.prod(v.slab_shape)) * v.dtype.itemsize for v in record_vars]
    if len(record_vars) == 1:
        # The one case in which record slabs are not padded to 4 bytes.
        recsize = slab_bytes[0]
    else:
        recsize = sum(_pad4(n) for n in slab_bytes)
    return _Header(
        numrecs=numrecs,
        dims=dims,
        unlimited_dims=frozenset(unlimited),
        attrs=attrs,
        variables=variables,
        recsize=recsize,
    )


def _read_header(fd: int) -> _Header:
    size = os.fstat(fd).st_size
    nbytes = _INITIAL_HEADER_BYTES
    while True:
        buf = os.pread(fd, nbytes, 0)
        try:
            return _parse_header(buf)
        except _HeaderTruncated:
            if len(buf) >= size:
                raise ValueError("Truncated netCDF header.")
            nbytes *= 4


def _pread_into(fd: int, buf: memoryview, offset: int):
    """Fill ``buf`` from the file at ``offset``, retrying short reads."""
    done = 0
    while done < len(buf):
        n = os.preadv(fd, [buf[done:]], offset + done)
        if n == 0:
            raise OSError(
                f"Unexpected end of file reading {len(buf)} bytes at offset {offset}."
            )
        done += n


class _CDF5File:
    """A parsed header plus a file descriptor opened lazily in each process.

    Reads use ``os.preadv``, which does not move a shared file offset, so one
    descriptor is safe to use from several threads. After a fork the child
    opens its own descriptor rather than sharing the parent's.
    """

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self._fd: int | None = None
        self._pid: int | None = None
        try:
            self.header = _read_header(self.fd)
        except Exception:
            self.close()
            raise

    @property
    def fd(self) -> int:
        pid = os.getpid()
        if self._fd is None or self._pid != pid:
            with self._lock:
                if self._fd is None or self._pid != pid:
                    self._fd = os.open(self.path, os.O_RDONLY)
                    self._pid = pid
        return self._fd

    def close(self):
        with self._lock:
            if self._fd is not None and self._pid == os.getpid():
                os.close(self._fd)
            self._fd = None
            self._pid = None

    def __del__(self):
        # Like xarray's file managers, release the descriptor of a dataset
        # that was never closed explicitly. Errors are suppressed because this
        # may run during interpreter shutdown.
        with contextlib.suppress(Exception):
            self.close()

    def __getstate__(self) -> dict[str, Any]:
        return {"path": self.path, "header": self.header}

    def __setstate__(self, state: dict[str, Any]):
        self.path = state["path"]
        self.header = state["header"]
        self._lock = threading.Lock()
        self._fd = None
        self._pid = None

    def read_slab(
        self, name: str, record: int | None, lo: int, hi: int, out: np.ndarray
    ):
        """Read rows [lo, hi) of the leading slab axis into ``out``.

        ``out`` must be C-contiguous with the on-disk dtype. For a record
        variable the slab is one record; for a scalar or other non-record
        variable it is the whole variable (a scalar has a single "row").
        """
        var = self.header.variables[name]
        row_bytes = int(np.prod(var.slab_shape[1:])) * var.dtype.itemsize
        offset = var.begin + lo * row_bytes
        if record is not None:
            offset += record * self.header.recsize
        buf = memoryview(out.reshape(-1)).cast("B")
        if len(buf) != (hi - lo) * row_bytes:
            raise ValueError("Output buffer does not match the requested rows.")
        if len(buf) > 0:
            _pread_into(self.fd, buf, offset)


def _normalize_axis_key(key: Any, n: int) -> tuple[int, int, Any]:
    """Bound one axis indexer to the rows [lo, hi) it touches.

    Returns ``(lo, hi, local_key)`` where ``local_key`` indexes the [lo, hi)
    block the same way ``key`` indexes the full axis.
    """
    if isinstance(key, slice):
        start, stop, step = key.indices(n)
        rows = range(start, stop, step)
        if len(rows) == 0:
            return 0, 0, slice(0, 0)
        lo, hi = min(rows), max(rows) + 1
        if step == 1:
            return lo, hi, slice(0, hi - lo)
        return lo, hi, np.asarray(rows) - lo
    if isinstance(key, np.ndarray):
        if key.size == 0:
            return 0, 0, key
        idx = np.where(key < 0, key + n, key)
        if idx.min() < 0 or idx.max() >= n:
            raise IndexError(f"Index out of bounds for axis of length {n}.")
        lo = int(idx.min())
        return lo, int(idx.max()) + 1, idx - lo
    i = int(key)
    if i < 0:
        i += n
    if not 0 <= i < n:
        raise IndexError(f"Index {key} out of bounds for axis of length {n}.")
    return i, i + 1, 0


def _outer_index(array: np.ndarray, key: tuple) -> np.ndarray:
    """Apply an OUTER indexer (ints, slices, 1-d int arrays) to ``array``."""
    if all(isinstance(k, slice) and k == slice(None) for k in key):
        return array
    # Last axis first, so the position of the axes still to index is unchanged
    # by an int dropping an axis.
    for axis in reversed(range(len(key))):
        k = key[axis]
        if isinstance(k, np.ndarray):
            array = np.take(array, k, axis=axis)
        else:
            # the Ellipsis keeps a 0-d result an array rather than a scalar
            array = array[(slice(None),) * axis + (k, Ellipsis)]
    return array


class _CDF5Array(BackendArray):
    def __init__(self, file: _CDF5File, name: str):
        self._file = file
        self._name = name
        var = file.header.variables[name]
        self.shape = var.shape
        self.dtype = var.native_dtype

    def __getitem__(self, key: indexing.ExplicitIndexer) -> np.ndarray:
        return indexing.explicit_indexing_adapter(
            key, self.shape, indexing.IndexingSupport.OUTER, self._getitem
        )

    def _getitem(self, key: tuple) -> np.ndarray:
        # Only the rows [lo, hi) of the leading slab axis that the key touches
        # are read, with one pread per record: a latitude band (the tile of a
        # spatial H split) costs a proportionally smaller read. The other axes
        # are cut from those rows in memory. Reading each row's segment of a
        # longitude band separately instead was ~4x slower on Lustre for the
        # 720x1440 EAM files (notes/tile-loading.md), so a W split reads full
        # rows.
        var = self._file.header.variables[self._name]
        if var.is_record:
            records = np.arange(var.shape[0])[key[0]]
            slab_key = key[1:]
        else:
            records = None
            slab_key = key
        slab_shape = var.slab_shape
        if len(slab_shape) == 0:  # scalar variable or 1-d record variable
            lo, hi, local_key = 0, 1, slab_key
            block_shape: tuple[int, ...] = ()
        else:
            lo, hi, first = _normalize_axis_key(slab_key[0], slab_shape[0])
            local_key = (first, *slab_key[1:])
            block_shape = (hi - lo, *slab_shape[1:])
        if records is None:
            block = np.empty(block_shape, dtype=var.dtype)
            self._file.read_slab(self._name, None, lo, hi, block)
            return _to_native(_outer_index(block, local_key))
        record_list = np.atleast_1d(records)
        selected_shape = _outer_index(_dummy(block_shape, var.dtype), local_key).shape
        out = np.empty((len(record_list), *selected_shape), dtype=var.dtype)
        if _selects_whole_block(local_key, block_shape):
            # read each record straight into its place in the output
            for i, record in enumerate(record_list):
                self._file.read_slab(self._name, int(record), lo, hi, out[i, ...])
        else:
            block = np.empty(block_shape, dtype=var.dtype)
            for i, record in enumerate(record_list):
                self._file.read_slab(self._name, int(record), lo, hi, block)
                out[i] = _outer_index(block, local_key)
        if np.ndim(records) == 0:
            out = out[0, ...]
        return _to_native(out)


def _dummy(shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    """A zero-memory array of the given shape, for working out result shapes."""
    return np.lib.stride_tricks.as_strided(
        np.empty(1, dtype=dtype), shape=shape, strides=(0,) * len(shape)
    )


def _selects_whole_block(key: tuple, shape: tuple[int, ...]) -> bool:
    return all(
        isinstance(k, slice) and k.indices(n) == (0, n, 1)
        for k, n in zip(key, shape, strict=True)
    )


def _to_native(array: np.ndarray) -> np.ndarray:
    """Convert a big-endian array to native byte order, in place if possible."""
    # array.copy rather than np.ascontiguousarray, which turns 0-d into 1-d
    if not array.flags.c_contiguous or not array.flags.writeable:
        array = array.copy(order="C")
    native = array.dtype.newbyteorder("=")
    if array.dtype == native:
        return array
    return array.byteswap(inplace=True).view(native)


class _CDF5DataStore(AbstractDataStore):
    def __init__(self, path: str):
        self._path = path
        self._file = _CDF5File(path)

    def get_variables(self) -> FrozenDict:
        out = {}
        for name, var in self._file.header.variables.items():
            data = indexing.LazilyIndexedArray(_CDF5Array(self._file, name))
            attrs = dict(var.attrs)
            if var.dtype.kind == "S" and "_FillValue" in attrs:
                # matches xarray's netcdf4 backend
                attrs["_FillValue"] = np.bytes_(attrs["_FillValue"])
            encoding = {
                "dtype": var.native_dtype,
                "source": self._path,
                "original_shape": var.shape,
            }
            out[name] = xr.Variable(var.dims, data, attrs, encoding)
        return FrozenDict(out)

    def get_attrs(self) -> FrozenDict:
        return FrozenDict(self._file.header.attrs)

    def get_dimensions(self) -> FrozenDict:
        return FrozenDict(self._file.header.dims)

    def get_encoding(self) -> dict[str, Any]:
        return {"unlimited_dims": set(self._file.header.unlimited_dims)}

    def close(self, **kwargs):
        self._file.close()


class CDF5BackendEntrypoint(BackendEntrypoint):
    """Read local netCDF classic files (CDF-5, CDF-2, CDF-1) with large preads.

    Use by passing the class as the engine:
    ``xr.open_dataset(path, engine=CDF5BackendEntrypoint)``.
    """

    description = "Open local netCDF classic (CDF-1/2/5) files with large preads"
    open_dataset_parameters = (
        "filename_or_obj",
        "mask_and_scale",
        "decode_times",
        "concat_characters",
        "decode_coords",
        "drop_variables",
        "use_cftime",
        "decode_timedelta",
    )

    def open_dataset(
        self,
        filename_or_obj: T_PathFileOrDataStore,
        *,
        mask_and_scale: bool = True,
        decode_times: Any = True,
        concat_characters: bool = True,
        decode_coords: bool = True,
        drop_variables: str | Iterable[str] | None = None,
        use_cftime: bool | None = None,
        decode_timedelta: Any = None,
    ) -> xr.Dataset:
        # isinstance narrows xarray's union of accepted inputs (buffers,
        # bytes, data stores) to the local paths this backend can pread from
        if not isinstance(filename_or_obj, str | os.PathLike):
            raise TypeError(
                "The cdf5 engine reads local files only, got "
                f"{type(filename_or_obj).__name__}."
            )
        path = os.path.abspath(os.path.expanduser(os.fspath(filename_or_obj)))
        store = _CDF5DataStore(path)
        try:
            return StoreBackendEntrypoint().open_dataset(
                store,
                mask_and_scale=mask_and_scale,
                decode_times=decode_times,
                concat_characters=concat_characters,
                decode_coords=decode_coords,
                drop_variables=drop_variables,
                use_cftime=use_cftime,
                decode_timedelta=decode_timedelta,
            )
        except Exception:
            store.close()
            raise

    def guess_can_open(self, filename_or_obj: T_PathFileOrDataStore) -> bool:
        if not isinstance(filename_or_obj, str | os.PathLike):
            return False
        try:
            with open(os.fspath(filename_or_obj), "rb") as f:
                magic = f.read(4)
        except OSError:
            return False
        return magic[:3] == b"CDF" and magic[3:] in (b"\x01", b"\x02", b"\x05")


def get_xarray_engine(engine: str) -> str | type[BackendEntrypoint]:
    """Map an fme engine name to what ``xr.open_dataset(engine=...)`` takes."""
    if engine == "cdf5":
        return CDF5BackendEntrypoint
    return engine
