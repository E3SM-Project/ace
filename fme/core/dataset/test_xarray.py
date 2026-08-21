"""This file contains unit tests of XarrayDataset."""

import dataclasses
import datetime
import os
import pathlib
from collections import namedtuple
from collections.abc import Iterable, Sequence

import cftime
import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr
from xarray.coding.times import CFDatetimeCoder

from fme.core.coordinates import (
    DepthCoordinate,
    HybridSigmaPressureCoordinate,
    LatLonCoordinates,
    NullVerticalCoordinate,
)
from fme.core.dataset.concat import XarrayConcat, get_dataset
from fme.core.dataset.merged import MergedXarrayDataset
from fme.core.dataset.schedule import IntSchedule
from fme.core.dataset.time import RepeatedInterval, TimeSlice
from fme.core.dataset.utils import FillNaNsConfig
from fme.core.dataset.xarray import (
    OverwriteConfig,
    XarrayDataConfig,
    XarrayDataset,
    XarraySubset,
    _combine_expression,
    _get_cumulative_timesteps,
    _get_file_local_index,
    _get_raw_times,
    _get_timestep,
    _get_vertical_coordinate,
    _repeat_and_increment_time,
    get_xarray_dataset,
)
from fme.core.spatial_mask_provider import SpatialMaskProvider
from fme.core.typing_ import Slice

from .utils import as_broadcasted_tensor

SLICE_NONE = slice(None)
MOCK_DATA_FREQ = "3h"
MOCK_DATA_START_DATE = "2003-03"
MOCK_DATA_LAT_DIM, MOCK_DATA_LON_DIM = ("lat", "lon")


@dataclasses.dataclass
class VariableNames:
    time_dependent_names: Iterable[str]
    time_invariant_names: Iterable[str]
    initial_condition_names: Iterable[str]

    def _concat(self, *lists):
        return_value = []
        for list in lists:
            return_value.extend(list)
        return return_value

    @property
    def all_names(self) -> list[str]:
        return self._concat(
            self.time_dependent_names,
            self.time_invariant_names,
            self.initial_condition_names,
        )

    @property
    def spatial_resolved_names(self) -> list[str]:
        return self._concat(self.time_dependent_names, self.time_invariant_names)


MockData = namedtuple(
    "MockData", ("tmpdir", "obs_times", "start_times", "start_indices", "var_names")
)


def _get_data(
    tmp_path_factory,
    dirname,
    start,
    end,
    file_freq,
    step_freq,
    calendar,
    with_nans=False,
    var_names=["foo", "bar"],
    write_extra_vars=True,
    add_ensemble_dim=False,
) -> MockData:
    """Constructs an xarray dataset and saves to disk in netcdf format."""
    obs_times = xr.date_range(
        start,
        end,
        freq=step_freq,
        calendar=calendar,
        inclusive="left",
        use_cftime=True,
    )
    start_times = xr.date_range(
        start,
        end,
        freq=file_freq,
        calendar=calendar,
        inclusive="left",
        use_cftime=True,
    )
    obs_delta = obs_times[1] - obs_times[0]
    n_levels = 2
    n_lat, n_lon = 4, 8
    n_sample = 3

    non_time_dims = ("sample", "lat", "lon") if add_ensemble_dim else ("lat", "lon")
    non_time_shape = (n_sample, n_lat, n_lon) if add_ensemble_dim else (n_lat, n_lon)

    constant_var = xr.DataArray(
        np.random.randn(*non_time_shape).astype(np.float32),
        dims=non_time_dims,
    )
    constant_scalar_var = xr.DataArray(1.0).astype(np.float32)
    ak = {f"ak_{i}": float(i) for i in range(n_levels)}
    bk = {f"bk_{i}": float(i + 1) for i in range(n_levels)}
    tmpdir = tmp_path_factory.mktemp(dirname)
    filenames = []
    for i, first in enumerate(start_times):
        if first != start_times[-1]:
            last = start_times[i + 1]
        else:
            last = obs_times[-1] + obs_delta
        time = xr.date_range(
            first,
            last,
            freq=step_freq,
            calendar=calendar,
            inclusive="left",
            use_cftime=True,
        )
        data_vars: dict[str, float | xr.DataArray] = {**ak, **bk}
        for var_name in var_names:
            data = np.random.randn(len(time), *non_time_shape).astype(np.float32)
            if with_nans:
                data[0, :, 0] = np.nan
            data_vars[var_name] = xr.DataArray(data, dims=("time", *non_time_dims))

        data_varying_scalar = np.random.randn(len(time)).astype(np.float32)
        if with_nans:
            constant_var[0, 0] = np.nan

        if write_extra_vars:
            data_vars["varying_scalar_var"] = xr.DataArray(
                data_varying_scalar, dims=("time",)
            )
            data_vars["constant_var"] = constant_var
            data_vars["constant_scalar_var"] = constant_scalar_var

        coords = {
            "time": xr.DataArray(time, dims=("time",)),
            "lat": xr.DataArray(np.arange(n_lat, dtype=np.float32), dims=("lat",)),
            "lon": xr.DataArray(np.arange(n_lon, dtype=np.float32), dims=("lon",)),
        }
        if add_ensemble_dim:
            coords["sample"] = xr.DataArray(
                np.arange(n_sample, dtype=np.float32), dims=("sample",)
            )
            # variable without the ensemble dimension is useful for checking
            # broadcast behavior
            data_vars["var_no_ensemble_dim"] = xr.DataArray(
                np.random.randn(len(time), n_lat, n_lon).astype(np.float32),
                dims=("time", "lat", "lon"),
            )
            # set values to sample index for testing convenience
            sample_index_values = np.broadcast_to(
                np.arange(n_sample).reshape(1, n_sample, 1, 1),  # shape [1, ns, 1, 1],
                (len(time), n_sample, n_lat, n_lon),
            )
            data_vars["var_matches_sample_index"] = (
                xr.zeros_like(data_vars["foo"]) + sample_index_values
            )

        ds = xr.Dataset(data_vars=data_vars, coords=coords)
        filename = tmpdir / f"{first.strftime('%Y%m%d%H')}.nc"
        ds.to_netcdf(
            filename,
            unlimited_dims=["time"],
            format="NETCDF4",
        )
        filenames.append(filename)

    initial_condition_names = ()
    start_indices = _get_cumulative_timesteps(_get_raw_times(filenames, "netcdf4"))
    if write_extra_vars:
        variable_names = VariableNames(
            time_dependent_names=(*var_names, "varying_scalar_var"),
            time_invariant_names=("constant_var", "constant_scalar_var"),
            initial_condition_names=initial_condition_names,
        )
    else:
        variable_names = VariableNames(
            time_dependent_names=var_names,
            time_invariant_names=(),
            initial_condition_names=initial_condition_names,
        )
    return MockData(tmpdir, obs_times, start_times, start_indices, variable_names)


def get_mock_monthly_netcdfs(
    tmp_path_factory,
    dirname,
    with_nans=False,
    end_date="2003-06",
    var_names=["foo", "bar"],
    write_extra_vars=True,
    add_ensemble_dim=False,
) -> MockData:
    return _get_data(
        tmp_path_factory,
        dirname,
        start=MOCK_DATA_START_DATE,
        end=end_date,
        file_freq="MS",
        step_freq=MOCK_DATA_FREQ,
        calendar="standard",
        with_nans=with_nans,
        var_names=var_names,
        write_extra_vars=write_extra_vars,
        add_ensemble_dim=add_ensemble_dim,
    )


@pytest.fixture(scope="session")
def mock_monthly_netcdfs(tmp_path_factory) -> MockData:
    return get_mock_monthly_netcdfs(tmp_path_factory, "month")


@pytest.fixture(scope="session")
def mock_monthly_netcdfs_another_source(tmp_path_factory) -> MockData:
    return get_mock_monthly_netcdfs(
        tmp_path_factory, "month_another_source", var_names=["baz", "qux"]
    )


@pytest.fixture(scope="session")
def mock_monthly_netcdfs_another_source_diff_time(tmp_path_factory) -> MockData:
    return get_mock_monthly_netcdfs(
        tmp_path_factory,
        "month_another_source",
        end_date="2003-08",
        var_names=["baz", "qux"],
        write_extra_vars=False,
    )


@pytest.fixture(scope="session")
def mock_monthly_netcdfs_with_nans(tmp_path_factory) -> MockData:
    return get_mock_monthly_netcdfs(tmp_path_factory, "month_with_nans", with_nans=True)


@pytest.fixture(scope="session")
def mock_monthly_netcdfs_ensemble_dim(tmp_path_factory) -> MockData:
    return get_mock_monthly_netcdfs(
        tmp_path_factory,
        "month_with_ensemble_dim",
        add_ensemble_dim=True,
        var_names=["foo", "bar", "var_no_ensemble_dim", "var_matches_sample_index"],
    )


@pytest.fixture(scope="session")
def mock_monthly_zarr_ensemble_dim(
    tmp_path_factory, mock_monthly_netcdfs_ensemble_dim
) -> MockData:
    zarr_parent = tmp_path_factory.mktemp("zarr")
    filename = "data.zarr"
    data = load_files_without_dask(
        mock_monthly_netcdfs_ensemble_dim.tmpdir.glob("*.nc")
    )
    data.to_zarr(zarr_parent / filename)
    return MockData(
        zarr_parent,
        mock_monthly_netcdfs_ensemble_dim.obs_times,
        mock_monthly_netcdfs_ensemble_dim.start_times,
        mock_monthly_netcdfs_ensemble_dim.start_indices,
        mock_monthly_netcdfs_ensemble_dim.var_names,
    )


def load_files_without_dask(files, engine="netcdf4") -> xr.Dataset:
    """Load a sequence of files without dask, concatenating along the time dimension.

    We load the data from the files into memory to ensure Datasets are properly closed,
    since xarray cannot concatenate Datasets lazily without dask anyway. This should be
    acceptable for the small datasets we construct for test purposes.
    """
    datasets = []
    for file in sorted(files):
        with xr.open_dataset(
            file,
            decode_times=CFDatetimeCoder(use_cftime=True),
            decode_timedelta=False,
            engine=engine,
        ) as ds:
            datasets.append(ds.load())
    return xr.concat(datasets, dim="time", data_vars="minimal", coords="minimal")


@pytest.fixture(scope="session")
def mock_monthly_zarr(tmp_path_factory, mock_monthly_netcdfs) -> MockData:
    zarr_parent = tmp_path_factory.mktemp("zarr")
    filename = "data.zarr"
    data = load_files_without_dask(mock_monthly_netcdfs.tmpdir.glob("*.nc"))
    data.to_zarr(zarr_parent / filename)
    return MockData(
        zarr_parent,
        mock_monthly_netcdfs.obs_times,
        mock_monthly_netcdfs.start_times,
        mock_monthly_netcdfs.start_indices,
        mock_monthly_netcdfs.var_names,
    )


@pytest.fixture(scope="session")
def mock_monthly_zarr_with_nans(
    tmp_path_factory, mock_monthly_netcdfs_with_nans
) -> MockData:
    zarr_parent = tmp_path_factory.mktemp("zarr")
    filename = "data.zarr"
    data = load_files_without_dask(mock_monthly_netcdfs_with_nans.tmpdir.glob("*.nc"))
    data.to_zarr(zarr_parent / filename)
    return MockData(
        zarr_parent,
        mock_monthly_netcdfs_with_nans.obs_times,
        mock_monthly_netcdfs_with_nans.start_times,
        mock_monthly_netcdfs_with_nans.start_indices,
        mock_monthly_netcdfs_with_nans.var_names,
    )


@pytest.fixture(scope="session")
def mock_yearly_netcdfs(tmp_path_factory):
    return _get_data(
        tmp_path_factory,
        "yearly",
        start="1999",
        end="2001",
        file_freq="YS",
        step_freq="1D",
        calendar="noleap",
    )


@pytest.mark.parametrize(
    "global_idx,expected_file_local_idx",
    [
        pytest.param(0, (0, 0), id="monthly_file_local_idx_2003_03_01_00"),
        pytest.param(1, (0, 1), id="monthly_file_local_idx_2003_03_01_03"),
        pytest.param(30 * 8, (0, 30 * 8), id="monthly_file_local_idx_2003_03_31_00"),
        pytest.param(31 * 8, (1, 0), id="monthly_file_local_idx_2003_04_01_00"),
        pytest.param(
            (31 + 30 + 20) * 8 - 1,
            (2, 20 * 8 - 1),
            id="monthly_file_local_idx_2003_05_20_21",
        ),
    ],
)
def test_monthly_file_local_index(
    mock_monthly_netcdfs, global_idx, expected_file_local_idx
):
    mock_data: MockData = mock_monthly_netcdfs
    file_local_idx = _get_file_local_index(global_idx, mock_data.start_indices)
    assert file_local_idx == expected_file_local_idx
    delta = mock_data.obs_times[1] - mock_data.obs_times[0]
    target_timestamp = np.datetime64(
        cftime.DatetimeGregorian(2003, 3, 1, 0, 0, 0, 0, has_year_zero=False)
        + global_idx * delta
    )
    file_idx, local_idx = file_local_idx
    full_paths = sorted(list(mock_data.tmpdir.glob("*.nc")))
    with xr.open_dataset(
        full_paths[file_idx],
        decode_times=CFDatetimeCoder(use_cftime=True),
        decode_timedelta=False,
    ) as ds:
        assert ds["time"][local_idx].item() == target_timestamp


def xarray_dataset_constructor(
    config: XarrayDataConfig, names: Sequence[str], n_timesteps: IntSchedule | int
) -> XarrayDataset:
    if isinstance(n_timesteps, int):
        n_timesteps = IntSchedule.from_constant(n_timesteps)
    return XarrayDataset(config, names, n_timesteps)


@pytest.mark.parametrize(
    "global_idx",
    [
        pytest.param(31 * 8 - 1, id="monthly_XarrayDataset_2003_03_31_21"),
        pytest.param((31 + 30 + 20) * 8 - 1, id="monthly_XarrayDataset_2003_05_20_21"),
        pytest.param((31 + 30) * 8 - 1, id="2003_04_30_21 (test for GH #1942)"),
    ],
)
@pytest.mark.parametrize(
    "mock_data_fixture, engine, file_pattern, labels",
    [
        ("mock_monthly_netcdfs", "netcdf4", "*.nc", set()),
        ("mock_monthly_zarr", "zarr", "*.zarr", {"foo_label"}),
    ],
)
def test_XarrayDataset_monthly(
    global_idx, mock_data_fixture, engine, file_pattern, request, labels
):
    mock_data: MockData = request.getfixturevalue(mock_data_fixture)
    var_names: VariableNames = mock_data.var_names
    config = XarrayDataConfig(
        data_path=mock_data.tmpdir,
        file_pattern=file_pattern,
        engine=engine,
        labels=labels,
    )
    dataset = xarray_dataset_constructor(config, var_names.all_names, 2)
    expected_n_samples = len(mock_data.obs_times) - 1

    assert len(dataset) == expected_n_samples
    arrays, time, dataset_labels, epoch, _ = dataset[global_idx]
    assert epoch is None
    assert dataset_labels == labels
    ds = load_files_without_dask(mock_data.tmpdir.glob(file_pattern), engine=engine)
    target_times = ds["time"][global_idx : global_idx + 2].drop_vars("time")
    xr.testing.assert_equal(time, target_times)
    lat_dim, lon_dim = MOCK_DATA_LAT_DIM, MOCK_DATA_LON_DIM
    dims = ("time", str(lat_dim), str(lon_dim))
    shape = (2, ds.sizes[lat_dim], ds.sizes[lon_dim])
    time_slice = slice(global_idx, global_idx + 2)
    for var_name in var_names.spatial_resolved_names:
        data = arrays[var_name]
        assert data.shape[0] == 2
        da = ds[var_name]
        if var_name in var_names.time_dependent_names:
            da = da.isel(time=time_slice)
        target_data = as_broadcasted_tensor(da.variable, dims, shape)
        assert torch.equal(data, target_data)

    for var_name in mock_data.var_names.initial_condition_names:
        data = arrays[var_name].detach().numpy()
        assert data.shape[0] == 1
        target_data = ds[var_name][global_idx : global_idx + 1, :, :].values
        assert np.all(data == target_data)


@pytest.mark.parametrize("n_samples", [None, 1])
@pytest.mark.parametrize("labels", [set(), {"foo"}])
def test_XarrayDataset_monthly_n_timesteps(mock_monthly_netcdfs, n_samples, labels):
    """Test that increasing n_timesteps decreases the number of samples."""
    mock_data: MockData = mock_monthly_netcdfs
    if len(mock_data.var_names.initial_condition_names) != 0:
        return
    config = XarrayDataConfig(
        data_path=mock_data.tmpdir, subset=Slice(stop=n_samples), labels=labels
    )
    n_forward_steps = 4
    dataset, properties = get_xarray_dataset(
        config,
        mock_data.var_names.all_names + ["x"],
        IntSchedule(start_value=n_forward_steps + 1, milestones=[]),
    )
    assert properties.all_labels == labels
    if n_samples is None:
        assert len(dataset) == len(mock_data.obs_times) - n_forward_steps
    else:
        assert len(dataset) == n_samples
    assert "x" in dataset[0][0]


@pytest.mark.parametrize(
    "global_idx,expected_file_local_idx",
    [
        pytest.param(365 + 59, (1, 59), id="yearly_file_local_idx_2000_03_01"),
        pytest.param(365, (1, 0), id="yearly_file_local_idx_2000_01_01"),
        pytest.param(364, (0, 364), id="yearly_file_local_idx_1999_12_31"),
    ],
)
def test_yearly_file_local_index(
    mock_yearly_netcdfs, global_idx, expected_file_local_idx
):
    mock_data: MockData = mock_yearly_netcdfs
    file_local_idx = _get_file_local_index(global_idx, mock_data.start_indices)
    assert file_local_idx == expected_file_local_idx
    delta = mock_data.obs_times[1] - mock_data.obs_times[0]
    target_timestamp = (
        cftime.DatetimeNoLeap(1999, 1, 1, 0, 0, 0, 0, has_year_zero=True)
        + global_idx * delta
    )
    file_idx, local_idx = file_local_idx
    full_paths = sorted(list(mock_data.tmpdir.glob("*.nc")))
    with xr.open_dataset(
        full_paths[file_idx],
        decode_times=CFDatetimeCoder(use_cftime=True),
        decode_timedelta=False,
    ) as ds:
        assert ds["time"][local_idx].item() == target_timestamp


@pytest.mark.parametrize(
    "global_idx",
    [
        pytest.param(364, id="yearly_XarrayDataset_1999_12_31"),
        pytest.param(365 + 31 + 28, id="yearly_XarrayDataset_2000_02_28"),
    ],
)
@pytest.mark.parametrize("labels", [set(), {"foo"}])
def test_XarrayDataset_yearly(mock_yearly_netcdfs, global_idx, labels):
    mock_data: MockData = mock_yearly_netcdfs
    config = XarrayDataConfig(data_path=mock_data.tmpdir, labels=labels)
    ds = load_files_without_dask(mock_data.tmpdir.glob("*.nc"))
    for n_steps in [3, 50]:
        dataset = xarray_dataset_constructor(
            config, mock_data.var_names.all_names, n_steps
        )
        assert len(dataset) == len(mock_data.obs_times) - n_steps + 1
        lon_dim, lat_dim = MOCK_DATA_LON_DIM, MOCK_DATA_LAT_DIM
        dims = ("time", lat_dim, lon_dim)
        shape = (n_steps, ds.sizes[lat_dim], ds.sizes[lon_dim])
        time_slice = slice(global_idx, global_idx + n_steps)
        for var_name in mock_data.var_names.spatial_resolved_names:
            da = ds[var_name]
            if var_name in mock_data.var_names.time_dependent_names:
                da = da.isel(time=time_slice)
            target_data = as_broadcasted_tensor(da.variable, dims, shape)
            target_times = ds["time"][global_idx : global_idx + n_steps].drop_vars(
                "time"
            )
            data, time, labels, epoch, _ = dataset[global_idx]
            assert epoch is None
            assert labels == labels
            data_tensor = data[var_name]
            assert data_tensor.shape[0] == n_steps
            assert torch.equal(data_tensor, target_data)
            xr.testing.assert_equal(time, target_times)


def test_dataset_dtype_casting(mock_monthly_netcdfs):
    mock_data: MockData = mock_monthly_netcdfs
    config = XarrayDataConfig(data_path=mock_data.tmpdir, dtype="bfloat16")
    dataset = xarray_dataset_constructor(config, mock_data.var_names.all_names, 2)
    data_properties = dataset.properties
    assert isinstance(data_properties.horizontal_coordinates, LatLonCoordinates)
    assert data_properties.horizontal_coordinates.lat.dtype == torch.bfloat16
    assert data_properties.horizontal_coordinates.lon.dtype == torch.bfloat16
    assert isinstance(
        data_properties.vertical_coordinate, HybridSigmaPressureCoordinate
    )
    assert data_properties.vertical_coordinate.ak.dtype == torch.bfloat16
    assert data_properties.vertical_coordinate.bk.dtype == torch.bfloat16
    data, _, _, _, _ = dataset[0]
    for tensor in data.values():
        assert tensor.dtype == torch.bfloat16


def test_time_invariant_variable_is_repeated(mock_monthly_netcdfs):
    mock_data: MockData = mock_monthly_netcdfs
    config = XarrayDataConfig(data_path=mock_data.tmpdir)
    dataset = xarray_dataset_constructor(config, mock_data.var_names.all_names, 15)
    data = dataset[0][0]
    assert data["constant_var"].shape[0] == 15
    assert data["constant_scalar_var"].shape == (15, 4, 8)


def _count_file_opens_while_reading(
    monkeypatch, dataset: XarrayDataset, indices: Sequence[int]
) -> int:
    """Number of times the dataset opens a file while reading the samples."""
    n_opens = 0
    original = XarrayDataset._open_file

    def counting_open_file(self, idx):
        nonlocal n_opens
        n_opens += 1
        return original(self, idx)

    monkeypatch.setattr(XarrayDataset, "_open_file", counting_open_file)
    for idx in indices:
        dataset[idx]
    return n_opens


def test_time_invariant_variables_do_not_open_files_per_sample(
    mock_monthly_netcdfs, monkeypatch
):
    """Requesting time-invariant variables should not add per-sample file opens.

    They are loaded once at construction, so reading samples costs the same
    number of file opens whether or not they were requested.
    """
    mock_data: MockData = mock_monthly_netcdfs
    config = XarrayDataConfig(data_path=mock_data.tmpdir)
    names = mock_data.var_names
    # samples spread across the underlying monthly files
    indices = [0, 100, 400, 700, 0]

    without = xarray_dataset_constructor(config, list(names.time_dependent_names), 2)
    with_invariant = xarray_dataset_constructor(config, names.all_names, 2)

    n_without = _count_file_opens_while_reading(monkeypatch, without, indices)
    n_with = _count_file_opens_while_reading(monkeypatch, with_invariant, indices)

    assert n_with == n_without


def test_time_invariant_variable_values_match_source(mock_monthly_netcdfs):
    """Caching must not change the values that are returned."""
    mock_data: MockData = mock_monthly_netcdfs
    config = XarrayDataConfig(data_path=mock_data.tmpdir)
    dataset = xarray_dataset_constructor(config, mock_data.var_names.all_names, 3)
    source = xr.open_dataset(
        mock_data.tmpdir / f"{mock_data.start_times[0].strftime('%Y%m%d%H')}.nc",
        decode_times=False,
        decode_timedelta=False,
    )
    # read several samples spanning different files to confirm the cached
    # tensor is not mutated or aliased between reads
    for idx in [0, 250, 500, 0]:
        data = dataset[idx][0]
        expected = torch.as_tensor(source["constant_var"].values)
        np.testing.assert_array_equal(data["constant_var"][0].numpy(), expected.numpy())
        assert data["constant_var"].shape == (3, 4, 8)
    source.close()


def _get_repeat_dataset(
    mock_data: MockData, n_timesteps: int, n_repeats: int
) -> XarrayDataset:
    config = XarrayDataConfig(data_path=mock_data.tmpdir, n_repeats=n_repeats)
    return xarray_dataset_constructor(
        config, mock_data.var_names.all_names, n_timesteps
    )


@pytest.mark.parametrize("n_timesteps", [1, 4])
@pytest.mark.parametrize("n_repeats", [1, 2])
def test_repeat_dataset_num_timesteps(
    mock_monthly_netcdfs: MockData, n_timesteps, n_repeats
):
    unrepeated_dataset = _get_repeat_dataset(mock_monthly_netcdfs, n_timesteps, 1)
    data = _get_repeat_dataset(mock_monthly_netcdfs, n_timesteps, n_repeats)
    offset = n_timesteps - 1
    expected_length = n_repeats * (len(unrepeated_dataset) + offset) - offset
    assert len(data) == expected_length


@pytest.mark.parametrize(
    "glob_pattern, expected_num_files, expected_year_month_tuples",
    [
        ("*.nc", None, None),
        ("2003030100.nc", 1, [(2003, 3)]),
        ("2003??0100.nc", 3, [(2003, i) for i in range(3, 6)]),
    ],
    ids=["all_files", "single_file", "all_2003_files"],
)
def test_glob_file_pattern(
    mock_monthly_netcdfs: MockData,
    glob_pattern,
    expected_num_files,
    expected_year_month_tuples,
):
    config = XarrayDataConfig(
        data_path=mock_monthly_netcdfs.tmpdir, file_pattern=glob_pattern
    )
    dataset = xarray_dataset_constructor(
        config, mock_monthly_netcdfs.var_names.all_names, 2
    )
    if expected_num_files is None:
        expected_num_files = len(mock_monthly_netcdfs.start_times)
    assert expected_num_files == len(dataset.full_paths)

    if expected_year_month_tuples is not None:
        for i, (year, month) in enumerate(expected_year_month_tuples):
            assert f"{year}{month:02d}" in dataset.full_paths[i]


def test_time_slice():
    time_slice = TimeSlice("2001-01-01", "2001-01-05", 2)
    time_index = xr.date_range(
        "2000", "2002", freq="D", calendar="noleap", use_cftime=True
    )
    slice_ = time_slice.slice(time_index)
    assert slice_ == slice(365, 370, 2)


def test_time_index(mock_monthly_netcdfs):
    config = XarrayDataConfig(data_path=mock_monthly_netcdfs.tmpdir)
    n_timesteps = 2
    names = mock_monthly_netcdfs.var_names.all_names
    dataset = xarray_dataset_constructor(config, names, n_timesteps)
    last_sample_init_time = len(mock_monthly_netcdfs.obs_times) - n_timesteps + 1
    obs_times = mock_monthly_netcdfs.obs_times[:last_sample_init_time]
    assert dataset.sample_start_times.equals(xr.CFTimeIndex(obs_times))


@pytest.mark.parametrize("infer_timestep", [True, False])
def test_XarrayDataset_timestep(mock_monthly_netcdfs, infer_timestep):
    config = XarrayDataConfig(
        data_path=mock_monthly_netcdfs.tmpdir, infer_timestep=infer_timestep
    )
    names = mock_monthly_netcdfs.var_names.all_names
    n_timesteps = 2
    dataset = xarray_dataset_constructor(config, names, n_timesteps)
    if infer_timestep:
        expected_timestep = pd.Timedelta(MOCK_DATA_FREQ).to_pytimedelta()
        assert dataset.timestep == expected_timestep
    else:
        assert dataset.timestep is None


@pytest.mark.parametrize(
    ("periods", "freq", "reverse", "expected", "exception"),
    [
        pytest.param(
            2,
            "3h",
            False,
            datetime.timedelta(hours=3),
            None,
            id="2 timesteps, regular freq",
        ),
        pytest.param(
            3,
            "9h",
            False,
            datetime.timedelta(hours=9),
            None,
            id="3 timesteps, regular freq",
        ),
        pytest.param(3, "3h", True, None, ValueError, id="3 timesteps, negative freq"),
        pytest.param(
            3, "MS", False, None, ValueError, id="3 timesteps, irregular freq"
        ),
        pytest.param(1, "D", False, None, ValueError, id="1 timestep"),
    ],
)
def test_get_timestep(periods, freq, reverse, expected, exception):
    index = xr.date_range("2000", periods=periods, freq=freq, use_cftime=True)

    if reverse:
        index = index[::-1]

    if exception is None:
        result = _get_timestep(index.values)
        assert result == expected
    else:
        with pytest.raises(exception):
            _get_timestep(index.values)


@pytest.mark.parametrize("n_repeats", [1, 3])
def test_repeat_and_increment_times(n_repeats):
    freq = "5h"
    delta = pd.Timedelta(freq).to_pytimedelta()

    start_a = cftime.DatetimeGregorian(2000, 1, 1)
    periods_a = 2
    segment_a = xr.date_range(
        start_a, periods=periods_a, freq=freq, use_cftime=True
    ).values

    start_b = segment_a[-1] + delta
    periods_b = 3
    segment_b = xr.date_range(
        start_b, periods=periods_b, freq=freq, use_cftime=True
    ).values

    raw_times = [segment_a, segment_b]
    raw_periods = [periods_a, periods_b]
    raw_total_periods = sum(raw_periods)

    result = _repeat_and_increment_time(raw_times, n_repeats, delta)
    full_periods = [len(times) for times in result]
    full_total_periods = sum(full_periods)

    result_concatenated = np.concatenate(result)
    expected_concatenated = xr.date_range(
        start_a, periods=full_total_periods, freq=freq, use_cftime=True
    ).values

    assert full_periods == n_repeats * raw_periods
    assert full_total_periods == n_repeats * raw_total_periods
    np.testing.assert_equal(result_concatenated, expected_concatenated)


@pytest.mark.parametrize("n_repeats", [1, 3])
def test_all_times(mock_monthly_netcdfs, n_repeats):
    n_timesteps = 2  # Arbitrary for this test
    dataset = _get_repeat_dataset(mock_monthly_netcdfs, n_timesteps, n_repeats)
    expected_periods = n_repeats * len(mock_monthly_netcdfs.obs_times)
    expected = xr.date_range(
        MOCK_DATA_START_DATE,
        periods=expected_periods,
        freq=MOCK_DATA_FREQ,
        use_cftime=True,
    )
    result = dataset.all_times
    assert result.equals(expected)


def test_get_sample_by_time_slice_times_n_repeats(mock_monthly_netcdfs: MockData):
    n_timesteps = 2  # Arbitrary for this test
    n_repeats = 3
    repeated_dataset = _get_repeat_dataset(mock_monthly_netcdfs, n_timesteps, n_repeats)

    # Pick a slice that is outside the range of the unrepeated data
    unrepeated_length = len(repeated_dataset.all_times) // n_repeats
    time_slice = slice(unrepeated_length, unrepeated_length + 3)

    _, result, _, _, _ = repeated_dataset.get_sample_by_time_slice(time_slice)
    expected = xr.DataArray(
        repeated_dataset.all_times[time_slice].values, dims=["time"]
    )
    xr.testing.assert_equal(result, expected)


@pytest.mark.parametrize(
    "dtype,expected_torch_dtype", [("int16", torch.int16), (None, None)]
)
def test_dataset_config_dtype(dtype, expected_torch_dtype):
    config = XarrayDataConfig(data_path="path/to/data", dtype=dtype)
    assert config.torch_dtype == expected_torch_dtype


def test_dataset_config_dtype_raises():
    with pytest.raises(ValueError):
        XarrayDataConfig(data_path="path/to/data", dtype="invalid_dtype")


@pytest.mark.parametrize(
    "mock_data_fixture, engine, file_pattern",
    [
        ("mock_monthly_netcdfs_with_nans", "netcdf4", "*.nc"),
        ("mock_monthly_zarr_with_nans", "zarr", "*.zarr"),
    ],
)
def test_fill_nans(mock_data_fixture, engine, file_pattern, request):
    mock_data: MockData = request.getfixturevalue(mock_data_fixture)
    nan_config = FillNaNsConfig()
    config = XarrayDataConfig(
        data_path=mock_data.tmpdir,
        fill_nans=nan_config,
        engine=engine,
        file_pattern=file_pattern,
    )
    names = mock_data.var_names.all_names
    dataset = xarray_dataset_constructor(config, names, 2)
    data, _, _, _, _ = dataset[0]
    assert torch.all(data["foo"][0, :, 0] == 0)
    assert torch.all(data["constant_var"][:, 0, 0] == 0)


def test_keep_nans(mock_monthly_netcdfs_with_nans):
    config_keep_nan = XarrayDataConfig(data_path=mock_monthly_netcdfs_with_nans.tmpdir)
    names = mock_monthly_netcdfs_with_nans.var_names.all_names
    dataset = xarray_dataset_constructor(config_keep_nan, names, 2)
    data_with_nan, _, _, _, _ = dataset[0]
    assert torch.all(torch.isnan(data_with_nan["foo"][0, :, 0]))
    assert torch.all(torch.isnan(data_with_nan["constant_var"][:, 0, 0]))


def _write_netcdf_with_fill_value(tmp_path, fill_value: float = 1e20):
    """Write a netCDF whose land points are flagged with CF _FillValue.

    This mirrors raw model output (e.g. MPAS remapped files), as opposed to the
    preprocessed zarr stores which carry NaN directly.
    """
    n_time, n_lat, n_lon = 4, 4, 8
    foo = np.random.randn(n_time, n_lat, n_lon).astype(np.float32)
    foo[:, 0, :] = fill_value  # first latitude row is "land"
    bar = np.random.randn(n_lat, n_lon).astype(np.float32)
    bar[0, :] = fill_value  # same "land" row, but time-invariant
    ds = xr.Dataset(
        data_vars={
            "foo": xr.DataArray(foo, dims=("time", "lat", "lon")),
            "bar": xr.DataArray(bar, dims=("lat", "lon")),
        },
        coords={
            "time": xr.DataArray(
                xr.date_range("2000-01-01", periods=n_time, freq="6h", use_cftime=True),
                dims=("time",),
            ),
            "lat": xr.DataArray(np.arange(n_lat, dtype=np.float32), dims=("lat",)),
            "lon": xr.DataArray(np.arange(n_lon, dtype=np.float32), dims=("lon",)),
        },
    )
    ds["foo"].attrs["_FillValue"] = np.float32(fill_value)
    ds["bar"].attrs["_FillValue"] = np.float32(fill_value)
    path = tmp_path / "20000101.nc"
    ds.to_netcdf(path, unlimited_dims=["time"], format="NETCDF4")
    return path


def test_mask_and_scale_decodes_fill_value(tmp_path):
    """_FillValue must become NaN when mask_and_scale is enabled.

    Without this the sentinel (e.g. 1e20) is loaded verbatim, which silently
    poisons losses: spatial output masking writes NaN over the same points
    while the target keeps the sentinel, so the loss's NaN guard (which keys
    off the target) never fires.
    """
    _write_netcdf_with_fill_value(tmp_path)

    default_config = XarrayDataConfig(data_path=str(tmp_path))
    default_data, _, _, _, _ = xarray_dataset_constructor(default_config, ["foo"], 2)[0]
    assert not torch.isnan(default_data["foo"]).any()
    assert torch.all(default_data["foo"][:, 0, :] == 1e20)

    decoded_config = XarrayDataConfig(data_path=str(tmp_path), mask_and_scale=True)
    decoded_data, _, _, _, _ = xarray_dataset_constructor(decoded_config, ["foo"], 2)[0]
    assert torch.all(torch.isnan(decoded_data["foo"][:, 0, :]))
    assert not torch.isnan(decoded_data["foo"][:, 1:, :]).any()


def test_mask_and_scale_applies_to_time_invariant_variables(tmp_path):
    """Time-invariant variables are read on a separate path that must decode too.

    They are loaded once in ``_load_time_invariant_tensors`` and broadcast over
    the sample, rather than being read per sample like time-dependent ones. That
    path opens the file itself, so it has to honour mask_and_scale as well --
    otherwise a static field such as a land mask keeps its raw sentinel while
    the time-dependent fields beside it decode to NaN.
    """
    _write_netcdf_with_fill_value(tmp_path)

    config = XarrayDataConfig(data_path=str(tmp_path), mask_and_scale=True)
    data, _, _, _, _ = xarray_dataset_constructor(config, ["foo", "bar"], 2)[0]
    assert torch.all(torch.isnan(data["bar"][:, 0, :]))
    assert not torch.isnan(data["bar"][:, 1:, :]).any()


def test_mask_and_scale_composes_with_fill_nans(tmp_path):
    """fill_nans only works once _FillValue has been decoded to NaN."""
    _write_netcdf_with_fill_value(tmp_path)
    config = XarrayDataConfig(
        data_path=str(tmp_path), mask_and_scale=True, fill_nans=FillNaNsConfig()
    )
    data, _, _, _, _ = xarray_dataset_constructor(config, ["foo"], 2)[0]
    assert not torch.isnan(data["foo"]).any()
    assert torch.all(data["foo"][:, 0, :] == 0)


def test_overwrite(mock_monthly_netcdfs):
    const = -10
    multiple = 3.5

    overwrite_config = OverwriteConfig(
        constant={"foo": const},
        multiply_scalar={"bar": multiple},
    )

    config = XarrayDataConfig(data_path=mock_monthly_netcdfs.tmpdir)
    n_timesteps = 2
    names = mock_monthly_netcdfs.var_names.all_names
    dataset = xarray_dataset_constructor(config, names, n_timesteps)[0][0]

    config_overwrite = XarrayDataConfig(
        data_path=mock_monthly_netcdfs.tmpdir, overwrite=overwrite_config
    )
    n_timesteps = 2
    dataset_overwrite = xarray_dataset_constructor(
        config_overwrite, names, n_timesteps
    )[0][0]

    for v in ["foo", "bar"]:
        assert dataset_overwrite[v].dtype == dataset[v].dtype
        assert dataset_overwrite[v].device == dataset[v].device
    assert torch.equal(
        dataset_overwrite["foo"], torch.ones_like(dataset["foo"]) * const
    )
    assert torch.equal(dataset_overwrite["bar"], dataset["bar"] * multiple)


def test_overwrite_add_scalar(mock_monthly_netcdfs):
    """add_scalar applies an offset, and composes with multiply_scalar as a*x+b."""
    multiple = 3.5
    addend = 273.15

    config = XarrayDataConfig(data_path=mock_monthly_netcdfs.tmpdir)
    names = mock_monthly_netcdfs.var_names.all_names
    reference = xarray_dataset_constructor(config, names, 2)[0][0]

    config_overwrite = XarrayDataConfig(
        data_path=mock_monthly_netcdfs.tmpdir,
        overwrite=OverwriteConfig(
            add_scalar={"foo": addend},
            multiply_scalar={"bar": multiple},
        ),
    )
    overwritten = xarray_dataset_constructor(config_overwrite, names, 2)[0][0]
    assert overwritten["foo"].dtype == reference["foo"].dtype
    torch.testing.assert_close(overwritten["foo"], reference["foo"] + addend)

    config_affine = XarrayDataConfig(
        data_path=mock_monthly_netcdfs.tmpdir,
        overwrite=OverwriteConfig(
            multiply_scalar={"foo": multiple}, add_scalar={"foo": addend}
        ),
    )
    affine = xarray_dataset_constructor(config_affine, names, 2)[0][0]
    torch.testing.assert_close(affine["foo"], reference["foo"] * multiple + addend)


def test_get_raw_times_is_memoized_and_serial(tmp_path, monkeypatch):
    """Repeated calls for the same file list must not re-read the files.

    Dataset construction asks for the same stream once per dataset (train
    windows, validation, each inference block). Re-reading every time made
    setup slow enough to trip the NCCL watchdog, and the parallel workarounds
    that hid the cost were unsafe (fork deadlock, then HDF5 heap corruption).
    """
    from fme.core.dataset import xarray as xarray_module

    n_files = 13
    times = xr.date_range("2000", freq="6h", periods=2 * n_files, use_cftime=True)
    paths = []
    for i in range(n_files):
        path = os.path.join(tmp_path, f"file_{i}.nc")
        sel = times[2 * i : 2 * i + 2]
        xr.DataArray(
            range(len(sel)), dims=["time"], coords=[sel], name="foo"
        ).to_dataset().to_netcdf(path)
        paths.append(path)

    xarray_module._get_raw_times_cached.cache_clear()
    reads = []
    original = xarray_module._get_raw_times_single_file

    def counting(path, engine=None):
        reads.append(path)
        return original(path, engine=engine)

    monkeypatch.setattr(xarray_module, "_get_raw_times_single_file", counting)

    first = xarray_module._get_raw_times(paths, "netcdf4")
    assert len(reads) == n_files
    second = xarray_module._get_raw_times(paths, "netcdf4")
    assert len(reads) == n_files, "second call re-read the files"
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    # the caller gets its own list, so mutating it cannot poison the cache
    first.append("sentinel")
    assert len(xarray_module._get_raw_times(paths, "netcdf4")) == n_files
    xarray_module._get_raw_times_cached.cache_clear()


def test_get_raw_paths_local_matches_fsspec(mock_monthly_netcdfs):
    """The local fast path must return exactly what the fsspec path returns."""
    import fsspec

    from fme.core.dataset.xarray import get_raw_paths

    tmpdir = str(mock_monthly_netcdfs.tmpdir)
    fast = get_raw_paths(tmpdir, "*.nc")
    reference = sorted(fsspec.filesystem("file").glob(os.path.join(tmpdir, "*.nc")))
    assert fast == reference
    assert len(fast) > 0
    # patterns that match nothing still agree
    assert get_raw_paths(tmpdir, "*.zarr") == sorted(
        fsspec.filesystem("file").glob(os.path.join(tmpdir, "*.zarr"))
    )


def test_combine_rejects_chained_and_overwritten_targets():
    """Chained combines and overwrite-on-a-target are silent no-ops, so reject."""
    with pytest.raises(ValueError, match="combine targets"):
        XarrayDataConfig(
            data_path="path",
            combine={"a": {"x": 1.0}, "b": {"a": 1.0, "y": 1.0}},
        )
    with pytest.raises(ValueError, match="silently do nothing"):
        XarrayDataConfig(
            data_path="path",
            combine={"total": {"x": 1.0, "y": 1.0}},
            overwrite=OverwriteConfig(multiply_scalar={"total": 2.0}),
        )


def test_mask_and_scale_rejected_for_zarr():
    """The zarr read path skips CF decoding, so the flag would half-apply."""
    with pytest.raises(ValueError, match="not supported with"):
        XarrayDataConfig(
            data_path="path",
            file_pattern="x.zarr",
            engine="zarr",
            mask_and_scale=True,
        )


def test_combine_source_missing_raises_even_when_missing_allowed(
    mock_monthly_netcdfs,
):
    """A combine target cannot be built from a source that is not on disk.

    Without this the dataset builds and then raises KeyError from inside a
    dataloader worker on the first batch.
    """
    config = XarrayDataConfig(
        data_path=mock_monthly_netcdfs.tmpdir,
        combine={"total": {"foo": 1.0, "not_on_disk": 1.0}},
    )
    with pytest.raises(ValueError, match="Cannot build combine target"):
        XarrayDataset(
            config,
            ["total"],
            IntSchedule.from_constant(2),
            allow_missing_variables=True,
        )


def test_combine_sums_fields(mock_monthly_netcdfs):
    """A target field can be defined as a linear combination of loaded fields.

    Needed because raw model output may split a field the model wants whole,
    e.g. MPAS carries rainFlux and snowFlux but no total precipitation.
    """
    names = mock_monthly_netcdfs.var_names.all_names
    reference = xarray_dataset_constructor(
        XarrayDataConfig(data_path=mock_monthly_netcdfs.tmpdir), names, 2
    )[0][0]

    config = XarrayDataConfig(
        data_path=mock_monthly_netcdfs.tmpdir,
        combine={"total": {"foo": 1.0, "bar": 1.0}},
    )
    data = xarray_dataset_constructor(config, ["total"], 2)[0][0]
    # sources were loaded to build the target but not requested, so not returned
    assert set(data) == {"total"}
    torch.testing.assert_close(data["total"], reference["foo"] + reference["bar"])


def test_combine_supports_coefficients_and_keeps_requested_sources(
    mock_monthly_netcdfs,
):
    """Coefficients allow differences; explicitly requested sources are kept."""
    names = mock_monthly_netcdfs.var_names.all_names
    reference = xarray_dataset_constructor(
        XarrayDataConfig(data_path=mock_monthly_netcdfs.tmpdir), names, 2
    )[0][0]

    config = XarrayDataConfig(
        data_path=mock_monthly_netcdfs.tmpdir,
        combine={"diff": {"foo": 1.0, "bar": -1.0}},
    )
    data = xarray_dataset_constructor(config, ["diff", "foo"], 2)[0][0]
    assert set(data) == {"diff", "foo"}
    torch.testing.assert_close(data["diff"], reference["foo"] - reference["bar"])


def test_combine_applies_after_overwrite(mock_monthly_netcdfs):
    """overwrite runs first, so unit and sign fixes compose with combine."""
    names = mock_monthly_netcdfs.var_names.all_names
    reference = xarray_dataset_constructor(
        XarrayDataConfig(data_path=mock_monthly_netcdfs.tmpdir), names, 2
    )[0][0]

    config = XarrayDataConfig(
        data_path=mock_monthly_netcdfs.tmpdir,
        overwrite=OverwriteConfig(multiply_scalar={"foo": 1000.0}),
        combine={"total": {"foo": 1.0, "bar": 1.0}},
    )
    data = xarray_dataset_constructor(config, ["total"], 2)[0][0]
    torch.testing.assert_close(
        data["total"], reference["foo"] * 1000.0 + reference["bar"]
    )


def test_combine_target_metadata_describes_the_combination(mock_monthly_netcdfs):
    """The target's long_name states how it was built, not the first source's.

    Inheriting a source's long_name verbatim mislabels the result, which then
    propagates into inference output and plots.
    """
    config = XarrayDataConfig(
        data_path=mock_monthly_netcdfs.tmpdir,
        combine={"total": {"foo": 1.0, "bar": 1.0}},
    )
    dataset = xarray_dataset_constructor(config, ["total"], 2)
    metadata = dataset.properties.variable_metadata
    assert metadata["total"].long_name == "foo + bar"
    # both sources have no units recorded, so neither does the target
    assert metadata["total"].units is None


def _rewrite_mock_netcdfs(source_dir, destination_dir, edit):
    """Copy a mock dataset, applying ``edit`` to each file's xr.Dataset."""
    for path in sorted(pathlib.Path(source_dir).glob("*.nc")):
        with xr.open_dataset(path) as opened:
            ds = opened.load()
        edit(ds)
        ds.to_netcdf(pathlib.Path(destination_dir) / path.name)
    return str(destination_dir)


def test_combine_target_drops_units_when_sources_disagree(
    mock_monthly_netcdfs, tmp_path
):
    """A combination of differently-united fields has no unit to inherit."""

    def relabel(ds):
        ds["foo"].attrs["units"] = "m"
        ds["bar"].attrs["units"] = "K"

    path = _rewrite_mock_netcdfs(mock_monthly_netcdfs.tmpdir, tmp_path, relabel)
    config = XarrayDataConfig(
        data_path=path, combine={"diff": {"foo": 1.0, "bar": -1.0}}
    )
    metadata = xarray_dataset_constructor(
        config, ["diff"], 2
    ).properties.variable_metadata
    assert metadata["diff"].units is None
    assert metadata["diff"].long_name == "foo - bar"


@pytest.mark.parametrize(
    "sources, expected",
    [
        ({"a": 1.0, "b": 1.0}, "a + b"),
        ({"a": 1.0, "b": -1.0}, "a - b"),
        ({"a": -1.0, "b": 2.0}, "-a + 2*b"),
        ({"a": 0.5}, "0.5*a"),
    ],
)
def test_combine_expression(sources, expected):
    assert _combine_expression(sources) == expected


def test_combine_target_shadowing_on_disk_variable_raises(mock_monthly_netcdfs):
    """A computed target that also exists on disk would silently win.

    Worse, it would only win for the datasets that request it, so two loads of
    the same config could disagree about what the name means.
    """
    config = XarrayDataConfig(
        data_path=mock_monthly_netcdfs.tmpdir,
        combine={"foo": {"bar": 1.0, "constant_var": 1.0}},
    )
    with pytest.raises(ValueError, match="also variables in"):
        xarray_dataset_constructor(config, ["foo"], 2)


def test_combine_config_validation():
    with pytest.raises(ValueError):  # empty source mapping
        XarrayDataConfig(data_path="path", combine={"total": {}})
    with pytest.raises(ValueError):  # target is also a source
        XarrayDataConfig(data_path="path", combine={"foo": {"foo": 1.0, "bar": 1.0}})


def test_combine_target_routes_to_correct_merge_member(mock_monthly_netcdfs):
    """A merged dataset must route a combine target to the member producing it."""
    from fme.core.dataset.merged import MergeDatasetConfig, get_per_dataset_names

    producer = XarrayDataConfig(
        data_path=mock_monthly_netcdfs.tmpdir,
        combine={"total": {"foo": 1.0, "bar": 1.0}},
    )
    plain = XarrayDataConfig(data_path=mock_monthly_netcdfs.tmpdir)
    # the producer is listed second, so routing cannot succeed by position alone
    merged = MergeDatasetConfig(merge=[plain, producer])
    per_dataset = get_per_dataset_names(merged, ["constant_var", "total"])
    assert per_dataset == [["constant_var"], ["total"]]


def test_overwrite_skips_absent_variables(mock_monthly_netcdfs):
    """A config may name variables a given load did not request.

    The coupled loader builds several datasets from one XarrayDataConfig, each
    holding only a subset of the names, so overwrite must not raise on the
    names that subset lacks.
    """
    config = XarrayDataConfig(
        data_path=mock_monthly_netcdfs.tmpdir,
        overwrite=OverwriteConfig(multiply_scalar={"foo": 2.0, "bar": 3.0}),
    )
    # request only "foo"; "bar" is named by the overwrite config but not loaded
    subset_data = xarray_dataset_constructor(config, ["foo"], 2)[0][0]
    assert set(subset_data) == {"foo"}

    reference = xarray_dataset_constructor(
        XarrayDataConfig(data_path=mock_monthly_netcdfs.tmpdir), ["foo"], 2
    )[0][0]
    torch.testing.assert_close(subset_data["foo"], reference["foo"] * 2.0)


def test_mask_decoded_to_nan_raises(mock_monthly_netcdfs, tmp_path):
    """mask_and_scale must not be allowed to punch NaN holes in a mask.

    A mask carrying a _FillValue decodes to NaN at those points, which inverts
    the masking there. The data this was written for has no such mask, but the
    failure would otherwise be silent.
    """

    def add_mask(ds):
        mask = xr.zeros_like(ds["constant_var"]) + 1.0
        mask[0, 0] = 1.0e20
        mask.attrs["_FillValue"] = 1.0e20
        ds["mask_2d"] = mask

    path = _rewrite_mock_netcdfs(mock_monthly_netcdfs.tmpdir, tmp_path, add_mask)

    # without decoding, the sentinel is just an odd value and loading succeeds
    xarray_dataset_constructor(XarrayDataConfig(data_path=path), ["foo"], 2)

    with pytest.raises(ValueError, match="contains NaN"):
        xarray_dataset_constructor(
            XarrayDataConfig(data_path=path, mask_and_scale=True), ["foo"], 2
        )


def test_overwrite_unknown_variable_raises(mock_monthly_netcdfs):
    """A name in no file at all can never take effect, so it is a typo."""
    config = XarrayDataConfig(
        data_path=mock_monthly_netcdfs.tmpdir,
        overwrite=OverwriteConfig(multiply_scalar={"foo": 2.0, "fooo": 3.0}),
    )
    with pytest.raises(ValueError, match="which do not exist in"):
        xarray_dataset_constructor(config, ["foo"], 2)


def test_overwrite_constant_and_add_scalar_conflict():
    with pytest.raises(ValueError):
        OverwriteConfig(constant={"foo": 1.0}, add_scalar={"foo": 2.0})


@pytest.mark.parametrize(
    "mock_data_fixture, engine, file_pattern",
    [
        ("mock_monthly_netcdfs", "netcdf4", "*.nc"),
        ("mock_monthly_zarr", "zarr", "*.zarr"),
    ],
)
def test_rename(mock_data_fixture, engine, file_pattern, request):
    """Renamed variables are requested and returned under their new names."""
    mock_data: MockData = request.getfixturevalue(mock_data_fixture)
    rename = {"foo": "renamed_foo", "constant_var": "renamed_constant_var"}
    names = [rename.get(name, name) for name in mock_data.var_names.all_names]

    config = XarrayDataConfig(
        data_path=mock_data.tmpdir,
        engine=engine,
        file_pattern=file_pattern,
        rename=rename,
    )
    dataset = xarray_dataset_constructor(config, names, 2)
    data, _, _, _, _ = dataset[0]

    reference_config = XarrayDataConfig(
        data_path=mock_data.tmpdir, engine=engine, file_pattern=file_pattern
    )
    reference = xarray_dataset_constructor(
        reference_config, mock_data.var_names.all_names, 2
    )
    reference_data, _, _, _, _ = reference[0]

    assert set(data) == set(names)
    for name in mock_data.var_names.all_names:
        assert torch.equal(data[rename.get(name, name)], reference_data[name])


def test_rename_missing_variable_raises(mock_monthly_netcdfs):
    config = XarrayDataConfig(
        data_path=mock_monthly_netcdfs.tmpdir, rename={"not_a_variable": "foo"}
    )
    with pytest.raises(ValueError, match="not_a_variable"):
        xarray_dataset_constructor(config, ["foo"], 2)


def test_repeated_interval_boolean_mask_subset(mock_monthly_netcdfs):
    config = XarrayDataConfig(data_path=mock_monthly_netcdfs.tmpdir)
    names = mock_monthly_netcdfs.var_names.all_names
    dataset = xarray_dataset_constructor(config, names, 1)
    interval = RepeatedInterval(interval_length="1D", block_length="7D", start="3D")
    boolean_mask = interval.get_boolean_mask(len(dataset), dataset.timestep)
    subset = XarraySubset(dataset, boolean_mask)

    # Check that the subset length matches the expected number of intervals
    expected_length = boolean_mask.sum().item()
    assert len(subset) == expected_length


def test_multi_source_xarray_dataset_has_no_duplicates(
    mock_monthly_netcdfs, mock_monthly_netcdfs_another_source
):
    monthly_netcdfs = [mock_monthly_netcdfs, mock_monthly_netcdfs_another_source]
    datasets = []

    for mock_data in monthly_netcdfs:
        config_source = XarrayDataConfig(data_path=mock_data.tmpdir)
        names = mock_data.var_names.all_names
        dataset = xarray_dataset_constructor(config_source, names, 1)
        datasets.append(dataset)

    with pytest.raises(ValueError):
        # duplicate variable names
        MergedXarrayDataset(datasets=datasets)


def test_multi_source_xarray_dataset_has_same_time(
    mock_monthly_netcdfs, mock_monthly_netcdfs_another_source_diff_time
):
    monthly_netcdfs = [
        mock_monthly_netcdfs,
        mock_monthly_netcdfs_another_source_diff_time,
    ]

    datasets = []
    for mock_data in monthly_netcdfs:
        config_source = XarrayDataConfig(data_path=mock_data.tmpdir)
        names = mock_data.var_names.all_names
        dataset = xarray_dataset_constructor(config_source, names, 1)
        datasets.append(dataset)
    # different time index
    with pytest.raises(ValueError):
        MergedXarrayDataset(datasets=datasets)


def test_multi_source_xarray_returns_merged_data(
    mock_monthly_netcdfs, mock_monthly_netcdfs_another_source
):
    config_source1 = XarrayDataConfig(data_path=mock_monthly_netcdfs.tmpdir)
    names1 = mock_monthly_netcdfs.var_names.all_names
    dataset1 = xarray_dataset_constructor(config_source1, names1, 1)

    config_source2 = XarrayDataConfig(
        data_path=mock_monthly_netcdfs_another_source.tmpdir
    )
    names2 = mock_monthly_netcdfs_another_source.var_names.all_names
    # remove duplicates in source 2 requirements
    for name in names1:
        if name in names2:
            names2.remove(name)
    dataset2 = xarray_dataset_constructor(config_source2, names2, 1)
    merged_dataset = MergedXarrayDataset(datasets=[dataset1, dataset2])
    assert len(merged_dataset) == len(dataset1)
    assert type(merged_dataset[0]) is type(dataset1[0])
    assert type(merged_dataset[0]) is type(dataset2[0])
    for key in merged_dataset[0][0].keys():
        if key in dataset1[0][0].keys():
            assert torch.equal(merged_dataset[0][0][key], dataset1[0][0][key])
            assert merged_dataset[0][1].equals(dataset1[0][1])
        if key in dataset2[0][0].keys():
            assert torch.equal(merged_dataset[0][0][key], dataset2[0][0][key])
            assert merged_dataset[0][1].equals(dataset2[0][1])
        else:
            assert KeyError(f"Key {key} is missing in merged dataset")


def test_xarray_subset_has_correct_sample(mock_monthly_netcdfs):
    mock_data: MockData = mock_monthly_netcdfs
    config = XarrayDataConfig(data_path=mock_data.tmpdir)
    config2 = XarrayDataConfig(data_path=mock_data.tmpdir, subset=Slice(stop=1))
    n_timesteps = 5
    names = mock_data.var_names.all_names + ["x"]
    dataset, _ = get_xarray_dataset(
        config, names, IntSchedule(start_value=n_timesteps, milestones=[])
    )
    dataset2, _ = get_xarray_dataset(
        config2, names, IntSchedule(start_value=n_timesteps, milestones=[])
    )
    assert dataset.sample_start_times[0:1].equals(dataset2.sample_start_times)
    assert dataset[0][0]["foo"].equal(dataset2[0][0]["foo"])
    assert dataset[0][1].equals(dataset2[0][1])


def test_xarray_concat_has_correct_sample(mock_monthly_netcdfs):
    mock_data: MockData = mock_monthly_netcdfs
    n_timesteps = 5
    names = mock_data.var_names.all_names + ["x"]
    config1 = XarrayDataConfig(
        data_path=mock_data.tmpdir, subset=TimeSlice("2003-03-01", "2003-03-31")
    )

    config2 = XarrayDataConfig(
        data_path=mock_data.tmpdir, subset=TimeSlice("2003-05-01", "2003-05-31")
    )
    concat, properties = get_dataset(
        [config1, config2], names, IntSchedule(start_value=n_timesteps, milestones=[])
    )
    expected1, _ = get_xarray_dataset(
        config1, names, IntSchedule(start_value=n_timesteps, milestones=[])
    )
    expected2, _ = get_xarray_dataset(
        config2, names, IntSchedule(start_value=n_timesteps, milestones=[])
    )
    expected_times = np.concatenate(
        [expected1.sample_start_times, expected2.sample_start_times]
    )
    expected = xr.CFTimeIndex(expected_times)
    assert concat.sample_start_times.equals(expected)


def test__get_vertical_coordinate_raises():
    data = xr.Dataset({"ak_0": 1.0, "bk_0": 0.5, "idepth_0": 1.0})
    with pytest.raises(ValueError, match="Dataset contains both hybrid"):
        _get_vertical_coordinate(data, dtype=None)


def test__get_vertical_coordinate_null():
    data = xr.Dataset()
    vertical_coordinate = _get_vertical_coordinate(data, dtype=None)
    assert vertical_coordinate == NullVerticalCoordinate()


def test__get_vertical_coordinate_hybrid_sigma_pressure():
    data = xr.Dataset({"ak_0": 1.0, "bk_0": 0.5, "ak_1": 2.0, "bk_1": 1.5})
    vertical_coordinate = _get_vertical_coordinate(data, dtype=None)
    assert isinstance(vertical_coordinate, HybridSigmaPressureCoordinate)
    assert vertical_coordinate.ak[0] == 1.0
    assert vertical_coordinate.bk[0] == 0.5


def test__get_vertical_coordinate_reference_pressure():
    """Dimensionless ak coefficients are scaled to Pa by the reference pressure."""
    data = xr.Dataset(
        {"ak_0": 0.25, "bk_0": 0.0, "ak_1": 0.5, "bk_1": 1.0, "P0": 100000.0}
    )
    vertical_coordinate = _get_vertical_coordinate(
        data, dtype=torch.float32, reference_pressure_name="P0"
    )
    assert isinstance(vertical_coordinate, HybridSigmaPressureCoordinate)
    assert vertical_coordinate.ak[0] == 25000.0
    assert vertical_coordinate.ak[1] == 50000.0
    assert vertical_coordinate.bk[1] == 1.0

    surface_pressure = torch.tensor([100000.0])
    interface_pressure = vertical_coordinate.interface_pressure(surface_pressure)
    torch.testing.assert_close(interface_pressure, torch.tensor([[25000.0, 150000.0]]))


@pytest.mark.parametrize(
    "data_vars, match",
    [
        pytest.param(
            {"ak_0": 0.01, "bk_0": 0.0},
            "not found in the dataset",
            id="missing_reference_pressure",
        ),
        pytest.param(
            {"ak_0": 0.01, "bk_0": 0.0, "P0": ("lat", [1.0, 2.0])},
            "must be a scalar",
            id="non_scalar_reference_pressure",
        ),
        pytest.param(
            {"idepth_0": 1.0, "idepth_1": 2.0, "P0": 100000.0},
            "does not have a hybrid sigma-pressure",
            id="depth_coordinate",
        ),
    ],
)
def test__get_vertical_coordinate_reference_pressure_raises(data_vars, match):
    data = xr.Dataset(data_vars)
    with pytest.raises(ValueError, match=match):
        _get_vertical_coordinate(data, dtype=None, reference_pressure_name="P0")


@pytest.mark.parametrize("has_deptho", [False, True], ids=["no_deptho", "with_deptho"])
def test__get_vertical_coordinate_depth_no_mask(has_deptho):
    data_vars: dict = {"idepth_0": 1.0, "idepth_1": 2.0}
    if has_deptho:
        data_vars["deptho"] = 1.5
    data = xr.Dataset(data_vars)
    vertical_coordinate = _get_vertical_coordinate(data, dtype=None)
    assert isinstance(vertical_coordinate, DepthCoordinate)
    assert vertical_coordinate.idepth[0] == 1.0
    assert vertical_coordinate.mask[0] == 1.0
    if has_deptho:
        assert vertical_coordinate.deptho is not None
        assert float(vertical_coordinate.deptho) == 1.5
    else:
        assert vertical_coordinate.deptho is None


@pytest.mark.parametrize("has_deptho", [False, True], ids=["no_deptho", "with_deptho"])
def test__get_vertical_coordinate_depth_with_lat_dependent_mask(has_deptho):
    data_vars: dict = {
        "idepth_0": 1.0,
        "idepth_1": 2.0,
        "idepth_2": 3.0,
        "mask_0": ("lat", np.array([1.0, 1.0])),
        "mask_1": ("lat", np.array([0.0, 1.0])),
    }
    if has_deptho:
        data_vars["deptho"] = ("lat", np.array([2.5, 3.0]))
    data = xr.Dataset(data_vars, coords={"lat": np.array([1.0, 2.0])})
    vertical_coordinate = _get_vertical_coordinate(data, dtype=None)
    assert isinstance(vertical_coordinate, DepthCoordinate)
    assert vertical_coordinate.idepth[0] == 1.0
    assert vertical_coordinate.idepth.shape == (3,)
    assert vertical_coordinate.mask.shape == (2, 2)
    if has_deptho:
        assert vertical_coordinate.deptho is not None
        assert vertical_coordinate.deptho.shape == (2,)
    else:
        assert vertical_coordinate.deptho is None


def test__get_vertical_coordinate_depth_with_time_dependent_deptho():
    data = xr.Dataset(
        data_vars={
            "idepth_0": 1.0,
            "idepth_1": 2.0,
            "deptho": ("time", np.array([1.5, 1.5])),
        },
        coords={"time": np.array([1.0, 2.0])},
    )
    with pytest.raises(ValueError, match="'deptho' must be time-independent"):
        _get_vertical_coordinate(data, dtype=None)


def test__get_vertical_coordinate_depth_with_time_dependent_mask():
    data = xr.Dataset(
        data_vars={
            "idepth_0": 1.0,
            "idepth_1": 2.0,
            "idepth_2": 3.0,
            "mask_0": ("time", np.array([1.0, 1.0])),
            "mask_1": ("time", np.array([0.0, 1.0])),
        },
        coords={
            "time": np.array([1.0, 2.0]),
        },
    )
    with pytest.raises(ValueError, match="The ocean mask must by time-independent."):
        _get_vertical_coordinate(data, dtype=None)


@pytest.mark.parametrize(
    "kwargs,",
    [
        pytest.param({"spatial_dimensions": "xyz"}, id="invalid_spatial_dimensions"),
        pytest.param(
            {"engine": "zarr", "file_pattern": "*.nc"},
            id="engine_file_pattern_mismatch",
        ),
        pytest.param(
            {"n_repeats": 2, "infer_timestep": False}, id="n_repeats_infer_timestep"
        ),
        pytest.param({"dtype": "foo"}, id="invalid_dtype"),
        pytest.param({"rename": {"time": "valid_time"}}, id="rename_time"),
        pytest.param({"rename": {"foo": "time"}}, id="rename_to_time"),
        pytest.param({"rename": {"foo": "baz", "bar": "baz"}}, id="rename_duplicate"),
    ],
)
def test_invalid_config_field_raises_error(kwargs):
    """Runs shape and length checks on the dataset."""
    with pytest.raises(ValueError):
        XarrayDataConfig(data_path="path", **kwargs)


@pytest.mark.parametrize(
    "mock_data_fixture, engine, file_pattern",
    [
        ("mock_monthly_netcdfs_ensemble_dim", "netcdf4", "*.nc"),
        ("mock_monthly_zarr_ensemble_dim", "zarr", "*.zarr"),
    ],
)
def test_dataset_with_nonspacetime_dim(
    mock_data_fixture, engine, file_pattern, request
):
    mock_data: MockData = request.getfixturevalue(mock_data_fixture)
    config = XarrayDataConfig(
        data_path=mock_data.tmpdir,
        dtype="bfloat16",
        engine=engine,
        file_pattern=file_pattern,
    )
    # Omit the test variable that has mismatch dimensions
    vars = list(set(mock_data.var_names.all_names) - {"var_no_ensemble_dim"})
    dataset = xarray_dataset_constructor(config, vars, 2)
    data, _, _, _, _ = dataset[0]
    assert len(data["foo"].shape) == 4
    assert dataset.dims == ["time", "sample", "lat", "lon"]


@pytest.mark.parametrize(
    "mock_data_fixture, engine, file_pattern",
    [
        ("mock_monthly_netcdfs_ensemble_dim", "netcdf4", "*.nc"),
        ("mock_monthly_zarr_ensemble_dim", "zarr", "*.zarr"),
    ],
)
def test_dataset_raise_error_on_dim_mismatch(
    mock_data_fixture, engine, file_pattern, request
):
    # Should raise error when trying to broadcast variable that is missing
    # ensemble 'sample' dim
    mock_data: MockData = request.getfixturevalue(mock_data_fixture)
    config = XarrayDataConfig(
        data_path=mock_data.tmpdir,
        dtype="bfloat16",
        engine=engine,
        file_pattern=file_pattern,
    )
    dataset = xarray_dataset_constructor(config, mock_data.var_names.all_names, 2)
    with pytest.raises(ValueError):
        dataset[0]


@pytest.mark.parametrize(
    "mock_data_fixture, engine, file_pattern",
    [
        ("mock_monthly_netcdfs_ensemble_dim", "netcdf4", "*.nc"),
        ("mock_monthly_zarr_ensemble_dim", "zarr", "*.zarr"),
    ],
)
def test_xarray_dataset_isel(mock_data_fixture, engine, file_pattern, request):
    mock_data: MockData = request.getfixturevalue(mock_data_fixture)
    config = XarrayDataConfig(
        data_path=mock_data.tmpdir,
        engine=engine,
        file_pattern=file_pattern,
        subset=Slice(start=None, stop=2),
        isel={"sample": 1},
    )
    vars = list(set(mock_data.var_names.all_names) - {"var_no_ensemble_dim"})
    dataset = xarray_dataset_constructor(config, vars, 2)
    data, _, _, _, _ = dataset[0]
    # Original lat/lon sizes are 4, 8
    assert data["var_matches_sample_index"].shape == (2, 4, 8)
    assert data["constant_var"].shape == (2, 4, 8)
    assert "sample" not in dataset.dims
    assert torch.all(data["var_matches_sample_index"] == 1.0)


@pytest.mark.parametrize(
    "isel",
    [
        {"lat": 0},
        {"time": 0},
        {"grid_x": 0},
    ],
)
def test_xarray_dataset_invalid_isel_raises_error(
    mock_monthly_netcdfs_ensemble_dim, isel
):
    mock_data: MockData = mock_monthly_netcdfs_ensemble_dim
    names = mock_data.var_names.all_names

    with pytest.raises(ValueError):
        config = XarrayDataConfig(
            data_path=mock_data.tmpdir,
            subset=TimeSlice("2003-03-01", "2003-03-31"),
            isel=isel,
        )
        get_dataset([config], names, IntSchedule(start_value=5, milestones=[]))


@pytest.mark.parametrize(
    "isel_value",
    [3, Slice(3, 13)],
)
def test_XarrayDataset_error_on_isel_outside_data(
    mock_monthly_netcdfs_ensemble_dim, isel_value
):
    # mock data has sample dimension size 3
    mock_data: MockData = mock_monthly_netcdfs_ensemble_dim
    config = XarrayDataConfig(
        data_path=mock_data.tmpdir,
        subset=Slice(start=None, stop=2),
        isel={"sample": isel_value},
    )
    vars = list(set(mock_data.var_names.all_names) - {"var_no_ensemble_dim"})
    with pytest.raises(ValueError):
        xarray_dataset_constructor(config, vars, 2)


def test_concat_of_XarrayConcat(mock_monthly_netcdfs):
    mock_data: MockData = mock_monthly_netcdfs
    n_timesteps = 5
    names = mock_data.var_names.all_names + ["x"]
    config = XarrayDataConfig(data_path=mock_data.tmpdir, subset=Slice(None, 4))
    concat, _ = get_dataset(
        [config, config], names, IntSchedule(start_value=n_timesteps, milestones=[])
    )
    concat2 = XarrayConcat(datasets=[concat, concat])
    assert len(concat2) == 16


def test__get_raw_times_across_many_files(tmpdir):
    times_per_file = 2
    n_files = 13
    n_times = n_files * times_per_file

    times = xr.date_range("2000", freq="6h", periods=n_times, use_cftime=True)
    da = xr.DataArray(range(len(times)), dims=["time"], coords=[times], name="foo")
    ds = da.to_dataset()

    paths = []
    for i in range(n_files):
        path = os.path.join(tmpdir, f"file_{i}.nc")
        time_slice = slice(times_per_file * i, times_per_file * (i + 1))
        ds.isel(time=time_slice).to_netcdf(path)
        paths.append(path)

    result = np.concatenate(_get_raw_times(paths, engine="netcdf4"))
    np.testing.assert_equal(result, times)


def test_dataset_properties_update_masks(mock_monthly_netcdfs):
    mock_data: MockData = mock_monthly_netcdfs
    config = XarrayDataConfig(data_path=mock_data.tmpdir)
    dataset = xarray_dataset_constructor(config, mock_data.var_names.all_names, 2)
    data_properties = dataset.properties
    assert not data_properties.spatial_mask_provider.masks
    existing_mask = SpatialMaskProvider(masks={"mask_0": torch.ones(4, 8)})
    data_properties.update_spatial_mask_provider(existing_mask)
    assert "mask_0" in dataset.properties.spatial_mask_provider.masks


def test_variable_metadata_includes_all_names(mock_monthly_netcdfs):
    mock_data: MockData = mock_monthly_netcdfs
    config = XarrayDataConfig(data_path=mock_data.tmpdir)
    names = mock_data.var_names.all_names
    dataset = xarray_dataset_constructor(config, names, 2)
    metadata_keys = set(dataset.properties.variable_metadata.keys())
    assert metadata_keys == set(names)


def test_allow_missing_variables_fills_nan_for_missing(mock_monthly_netcdfs):
    mock_data: MockData = mock_monthly_netcdfs
    config = XarrayDataConfig(data_path=mock_data.tmpdir)
    existing_names = list(mock_data.var_names.time_dependent_names)
    names_with_missing = existing_names + ["nonexistent_var"]
    dataset = XarrayDataset(
        config,
        names_with_missing,
        IntSchedule.from_constant(2),
        allow_missing_variables=True,
    )
    sample_data, _, _, _, missing_names = dataset[0]
    assert "nonexistent_var" in sample_data
    assert sample_data["nonexistent_var"].isnan().all()
    assert missing_names == frozenset({"nonexistent_var"})
    for name in existing_names:
        assert name in sample_data
        assert not sample_data[name].isnan().any()


def test_allow_missing_variables_false_raises_on_missing(mock_monthly_netcdfs):
    mock_data: MockData = mock_monthly_netcdfs
    config = XarrayDataConfig(data_path=mock_data.tmpdir)
    existing_names = list(mock_data.var_names.time_dependent_names)
    names_with_missing = existing_names + ["nonexistent_var"]
    with pytest.raises(ValueError, match="Required variable not found"):
        XarrayDataset(
            config,
            names_with_missing,
            IntSchedule.from_constant(2),
            allow_missing_variables=False,
        )
