"""Minimal reader for netCDF classic CDF-5 (64-bit data) files.

netCDF-C reads these files with 8 KiB read() calls (measured with strace on
the HR EAM files: 1018 x 8192-byte reads for two 4 MB fields), which runs at
30-75 MB/s per reader on Lustre against ~1.2 GB/s for plain sequential reads.
Here a record variable's slice for one timestep is a single pread at a known
offset.

    f = CDF5File(path)
    f.read("PS", 40)          # (lat, lon) float32 array for record 40
    f.read_records("PS", 0, 8)
"""

import os
import struct

import numpy as np

_TYPES = {
    1: ">i1", 2: "S1", 3: ">i2", 4: ">i4", 5: ">f4", 6: ">f8",
    7: ">u1", 8: ">u2", 9: ">u4", 10: ">i8", 11: ">u8",
}  # fmt: skip
_NC_DIMENSION, _NC_VARIABLE, _NC_ATTRIBUTE = 10, 11, 12


class _Header:
    def __init__(self, buf: bytes):
        self.buf, self.pos = buf, 0

    def take(self, fmt: str):
        v = struct.unpack_from(fmt, self.buf, self.pos)
        self.pos += struct.calcsize(fmt)
        return v[0] if len(v) == 1 else v

    def name(self) -> str:
        n = self.take(">q")
        s = self.buf[self.pos : self.pos + n].decode()
        self.pos += (n + 3) // 4 * 4
        return s

    def values(self, nc_type: int, n: int):
        dt = np.dtype(_TYPES[nc_type])
        nbytes = n * dt.itemsize
        raw = self.buf[self.pos : self.pos + nbytes]
        self.pos += (nbytes + 3) // 4 * 4
        if nc_type == 2:
            return raw.decode(errors="replace")
        return np.frombuffer(raw, dt).astype(dt.newbyteorder("="))

    def attrs(self) -> dict:
        tag, n = self.take(">i"), self.take(">q")
        if tag == 0:
            return {}
        assert tag == _NC_ATTRIBUTE, tag
        out = {}
        for _ in range(n):
            k = self.name()
            t, m = self.take(">i"), self.take(">q")
            out[k] = self.values(t, m)
        return out


class CDF5File:
    def __init__(self, path: str, header_bytes: int = 4 * 2**20):
        self.path = path
        self._fd = os.open(path, os.O_RDONLY)
        h = _Header(os.pread(self._fd, header_bytes, 0))
        assert h.buf[:4] == b"CDF\x05", f"{path} is not CDF-5"
        h.pos = 4
        self.numrecs = h.take(">q")
        self.dims: dict[str, int] = {}
        tag, n = h.take(">i"), h.take(">q")
        dim_names = []
        if tag:
            assert tag == _NC_DIMENSION
            for _ in range(n):
                k = h.name()
                dim_names.append(k)
                self.dims[k] = h.take(">q")
        self.attrs = h.attrs()
        self.vars: dict[str, dict] = {}
        tag, n = h.take(">i"), h.take(">q")
        assert tag in (0, _NC_VARIABLE)
        for _ in range(n):
            k = h.name()
            ndim = h.take(">q")
            dimids = [h.take(">q") for _ in range(ndim)]
            vattrs = h.attrs()
            t = h.take(">i")
            vsize, begin = h.take(">q"), h.take(">q")
            dims = tuple(dim_names[i] for i in dimids)
            record = len(dims) > 0 and self.dims[dims[0]] == 0
            shape = tuple(
                self.numrecs if j == 0 and record else self.dims[d]
                for j, d in enumerate(dims)
            )
            self.vars[k] = dict(
                dims=dims, shape=shape, dtype=np.dtype(_TYPES[t]), attrs=vattrs,
                vsize=vsize, begin=begin, record=record,
            )  # fmt: skip
        rec = [v for v in self.vars.values() if v["record"]]
        if len(rec) == 1:  # the one case where record entries are unpadded
            v = rec[0]
            self.recsize = int(np.prod(v["shape"][1:])) * v["dtype"].itemsize
        else:
            self.recsize = sum(v["vsize"] for v in rec)

    def read_records(self, name: str, start: int, stop: int) -> np.ndarray:
        """Records [start, stop) of a record variable, native byte order."""
        v = self.vars[name]
        assert v["record"], name
        inner = v["shape"][1:]
        nbytes = int(np.prod(inner)) * v["dtype"].itemsize
        out = np.empty((stop - start, *inner), dtype=v["dtype"].newbyteorder("="))
        for i, t in enumerate(range(start, stop)):
            buf = os.pread(self._fd, nbytes, v["begin"] + t * self.recsize)
            out[i] = np.frombuffer(buf, v["dtype"]).reshape(inner)
        return out

    def read(self, name: str, t: int | None = None) -> np.ndarray:
        v = self.vars[name]
        if v["record"]:
            assert t is not None
            return self.read_records(name, t, t + 1)[0]
        nbytes = int(np.prod(v["shape"])) * v["dtype"].itemsize
        buf = os.pread(self._fd, nbytes, v["begin"])
        return (
            np.frombuffer(buf, v["dtype"])
            .reshape(v["shape"])
            .astype(v["dtype"].newbyteorder("="))
        )

    def close(self):
        os.close(self._fd)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
