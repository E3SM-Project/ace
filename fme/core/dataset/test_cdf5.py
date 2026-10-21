import multiprocessing
import pathlib
import pickle

import netCDF4
import numpy as np
import pytest
import xarray as xr
from xarray.coding.times import CFDatetimeCoder

from fme.core.dataset import cdf5
from fme.core.dataset.cdf5 import CDF5BackendEntrypoint, get_xarray_engine

N_LAT, N_LON, N_LEVEL, N_CHARS = 3, 5, 2, 4
CDF5 = "NETCDF3_64BIT_DATA"
CDF5_ONLY_TYPES = ("u1", "u2", "i8", "u8")


def _write_classic_file(
    path: pathlib.Path, n_records: int = 4, nc_format: str = CDF5
) -> pathlib.Path:
    """Write a netCDF classic file covering the format's corner cases.

    Odd dimension lengths make the 1- and 2-byte record slabs need padding.
    """
    rng = np.random.default_rng(0)
    with netCDF4.Dataset(path, "w", format=nc_format) as ds:
        ds.createDimension("time", None)
        ds.createDimension("lat", N_LAT)
        ds.createDimension("lon", N_LON)
        ds.createDimension("level", N_LEVEL)
        ds.createDimension("chars", N_CHARS)
        ds.title = "classic test file"
        ds.version = np.int32(3)
        ds.weights = np.array([0.5, 1.5], dtype="f8")

        lat = ds.createVariable("lat", "f8", ("lat",))
        lat.units = "degrees_north"
        lat[:] = np.linspace(-45, 45, N_LAT)
        lon = ds.createVariable("lon", "f4", ("lon",))
        lon[:] = np.arange(N_LON)
        scalar = ds.createVariable("P0", "f8", ())
        scalar.units = "Pa"
        scalar.assignValue(1e5)
        level_index = ds.createVariable("level_index", "i1", ("level",))
        level_index[:] = [1, -2]
        static = ds.createVariable("static", "f4", ("lat", "lon"), fill_value=-1.0)
        static_values = rng.standard_normal((N_LAT, N_LON)).astype("f4")
        static_values[0, 0] = -1.0
        static[:] = static_values

        time = ds.createVariable("time", "f8", ("time",))
        time.units = "days since 0001-01-01 00:00:00"
        time.calendar = "noleap"
        time.bounds = "time_bnds"
        time[:] = 365.0 + 0.25 * np.arange(n_records)
        time_bnds = ds.createVariable("time_bnds", "f8", ("time", "level"))
        time_bnds[:] = 365.0 + 0.25 * np.arange(n_records)[:, None] + [-0.25, 0.0]
        date_written = ds.createVariable("date_written", "S1", ("time", "chars"))
        date_written[:] = np.full((n_records, N_CHARS), b"x", dtype="S1")
        nsteph = ds.createVariable("nsteph", "i4", ("time",))
        nsteph[:] = np.arange(n_records)

        shape = (n_records, N_LAT, N_LON)
        ps = ds.createVariable("PS", "f4", ("time", "lat", "lon"), fill_value=1e20)
        ps.units = "Pa"
        ps.long_name = "Surface pressure"
        ps_values = rng.standard_normal(shape).astype("f4")
        ps_values[:, 1, 2] = 1e20
        ps[:] = ps_values
        packed = ds.createVariable("packed", "i2", ("time", "lat", "lon"))
        packed.scale_factor = np.float32(0.5)
        packed.add_offset = np.float32(10.0)
        packed.missing_value = np.int16(-999)
        packed_values = rng.integers(-100, 100, shape).astype("i2")
        packed_values[:, 0, 0] = -999
        packed[:] = packed_values
        profile = ds.createVariable("profile", "f8", ("time", "level", "lat"))
        profile[:] = rng.standard_normal((n_records, N_LEVEL, N_LAT))
        byte_field = ds.createVariable("byte_field", "i1", ("time", "lon"))
        byte_field[:] = rng.integers(-128, 127, (n_records, N_LON)).astype("i1")
        if nc_format == CDF5:
            for dtype in CDF5_ONLY_TYPES:
                var = ds.createVariable(f"field_{dtype}", dtype, ("time", "lon"))
                var.valid_max = np.array(7, dtype=dtype)
                var[:] = rng.integers(0, 100, (n_records, N_LON)).astype(dtype)
    return path


def _open(path, engine, mask_and_scale=False, decode_times=True) -> xr.Dataset:
    return xr.open_dataset(
        path,
        engine=engine,
        mask_and_scale=mask_and_scale,
        decode_times=CFDatetimeCoder(use_cftime=True) if decode_times else False,
        decode_timedelta=False,
        cache=False,
        chunks=None,
    )


def _assert_same_as_netcdf4(path, **open_kwargs):
    with (
        _open(path, CDF5BackendEntrypoint, **open_kwargs) as actual,
        _open(path, "netcdf4", **open_kwargs) as expected,
    ):
        assert actual.encoding["unlimited_dims"] == expected.encoding["unlimited_dims"]
        for name in expected.variables:
            assert actual[name].dtype == expected[name].dtype, name
            assert actual[name].encoding["dtype"] == expected[name].encoding["dtype"]
            for key, value in expected[name].attrs.items():
                assert type(actual[name].attrs[key]) is type(value), (name, key)
        for key, value in expected.attrs.items():
            assert type(actual.attrs[key]) is type(value), key
        xr.testing.assert_identical(actual.load(), expected.load())


@pytest.mark.parametrize("nc_format", [CDF5, "NETCDF3_64BIT_OFFSET", "NETCDF3_CLASSIC"])
@pytest.mark.parametrize("mask_and_scale", [False, True])
@pytest.mark.parametrize("decode_times", [False, True])
def test_open_matches_netcdf4(tmp_path, nc_format, mask_and_scale, decode_times):
    path = _write_classic_file(tmp_path / "data.nc", nc_format=nc_format)
    _assert_same_as_netcdf4(
        path, mask_and_scale=mask_and_scale, decode_times=decode_times
    )


def test_open_matches_netcdf4_with_no_records(tmp_path):
    path = _write_classic_file(tmp_path / "data.nc", n_records=0)
    _assert_same_as_netcdf4(path, decode_times=False)


@pytest.mark.parametrize(
    "isel",
    [
        {"time": 1},
        {"time": -1},
        {"time": slice(1, 3)},
        {"time": slice(None, None, 2)},
        {"time": slice(3, 3)},
        {"time": [0, 2]},
        {"time": [3, 0, 2]},
        {"time": slice(1, 3), "lat": 1},
        {"time": 2, "lat": slice(1, 3), "lon": [4, 0, 2]},
        {"time": [0, 3], "lon": slice(None, None, -2)},
        {"lat": slice(None, None, -1)},
        {"lat": [2, 0]},
        {"lat": -1, "lon": 4},
        {"level": 1},
        {"time": [1, 2], "level": [1], "lat": slice(0, 2)},
    ],
)
def test_isel_matches_netcdf4(tmp_path, isel):
    path = _write_classic_file(tmp_path / "data.nc")
    with (
        _open(path, CDF5BackendEntrypoint, mask_and_scale=True) as actual,
        _open(path, "netcdf4", mask_and_scale=True) as expected,
    ):
        # The reference is selected in memory, since netCDF4 itself rejects
        # some of these indexers (e.g. unsorted arrays).
        expected_selection = expected.load().isel(isel, missing_dims="ignore")
        for name in expected.variables:
            selected = actual[name].variable.isel(isel, missing_dims="ignore")
            np.testing.assert_array_equal(
                selected.values, expected_selection[name].values, err_msg=name
            )
        xr.testing.assert_identical(
            actual.isel(isel, missing_dims="ignore").load(), expected_selection
        )


@pytest.mark.parametrize("dtype", ["i1", "i2", "f4"])
def test_single_record_variable_has_unpadded_records(tmp_path, dtype):
    """With exactly one record variable, record slabs are not padded."""
    path = tmp_path / "single.nc"
    n_odd = 3  # slab of 3 or 6 bytes, which padding would round up
    values = np.arange(5 * n_odd).reshape(5, n_odd).astype(dtype)
    with netCDF4.Dataset(path, "w", format=CDF5) as ds:
        ds.createDimension("time", None)
        ds.createDimension("x", n_odd)
        static = ds.createVariable("static", "i2", ("x",))
        static[:] = [7, 8, 9]
        ds.createVariable("field", dtype, ("time", "x"))[:] = values
    with _open(path, CDF5BackendEntrypoint) as actual:
        np.testing.assert_array_equal(actual["field"].values, values)
        np.testing.assert_array_equal(actual["field"].isel(time=4).values, values[4])
    _assert_same_as_netcdf4(path)


def test_header_larger_than_first_read(tmp_path, monkeypatch):
    monkeypatch.setattr(cdf5, "_INITIAL_HEADER_BYTES", 16)
    path = _write_classic_file(tmp_path / "data.nc")
    _assert_same_as_netcdf4(path)


def test_rejects_non_classic_file(tmp_path):
    path = tmp_path / "data.nc"
    with netCDF4.Dataset(path, "w", format="NETCDF4") as ds:
        ds.createDimension("x", 2)
    assert not CDF5BackendEntrypoint().guess_can_open(path)
    with pytest.raises(ValueError, match="Not a netCDF classic"):
        _open(path, CDF5BackendEntrypoint)


@pytest.mark.parametrize(
    "nc_format, numrecs_bytes",
    [("NETCDF3_CLASSIC", 4), ("NETCDF3_64BIT_OFFSET", 4), (CDF5, 8)],
)
def test_rejects_streaming_file(tmp_path, nc_format, numrecs_bytes):
    path = _write_classic_file(tmp_path / "data.nc", nc_format=nc_format)
    with open(path, "r+b") as f:
        f.seek(4)  # numrecs follows the 4-byte magic
        f.write(b"\xff" * numrecs_bytes)
    with pytest.raises(ValueError, match="Streaming"):
        _open(path, CDF5BackendEntrypoint)


def test_guess_can_open(tmp_path):
    path = _write_classic_file(tmp_path / "data.nc")
    assert CDF5BackendEntrypoint().guess_can_open(path)
    assert not CDF5BackendEntrypoint().guess_can_open(tmp_path / "missing.nc")


def test_get_xarray_engine():
    assert get_xarray_engine("cdf5") is CDF5BackendEntrypoint
    assert get_xarray_engine("netcdf4") == "netcdf4"
    assert get_xarray_engine("zarr") == "zarr"


def test_lazy_dataset_survives_pickle(tmp_path):
    path = _write_classic_file(tmp_path / "data.nc")
    with _open(path, CDF5BackendEntrypoint) as ds:
        ds["PS"].isel(time=0).load()  # open the descriptor before pickling
        restored = pickle.loads(pickle.dumps(ds))
        xr.testing.assert_identical(restored.isel(time=2).load(), ds.isel(time=2))


def _read_in_child(ds: xr.Dataset, file: cdf5._CDF5File, parent_fd: int, queue):
    queue.put((ds["PS"].isel(time=[1, 3]).values, file.fd != parent_fd))


@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded")
def test_forked_child_reads_with_its_own_descriptor(tmp_path):
    """A child forked after the parent has read opens its own descriptor."""
    path = _write_classic_file(tmp_path / "data.nc")
    file = cdf5._CDF5File(str(path))
    with _open(path, CDF5BackendEntrypoint) as ds:
        expected = ds["PS"].isel(time=[1, 3]).values  # parent opens its fd
        context = multiprocessing.get_context("fork")
        queue = context.Queue()
        child = context.Process(target=_read_in_child, args=(ds, file, file.fd, queue))
        child.start()
        actual, child_has_own_fd = queue.get(timeout=60)
        child.join(timeout=60)
        assert child.exitcode == 0
        assert child_has_own_fd
        np.testing.assert_array_equal(actual, expected)
        # the parent's descriptor still works after the child exits
        np.testing.assert_array_equal(ds["PS"].isel(time=[1, 3]).values, expected)
    file.close()
