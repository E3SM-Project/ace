"""Loading a spatial tile must give exactly the global fields sliced to it."""

import os
from typing import Literal

import numpy as np
import pytest
import torch
import xarray as xr

from fme.core.dataset import cdf5
from fme.core.dataset.cdf5 import CDF5BackendEntrypoint
from fme.core.dataset.concat import ConcatDatasetConfig
from fme.core.dataset.merged import MergeNoConcatDatasetConfig
from fme.core.dataset.schedule import IntSchedule
from fme.core.dataset.testing import (
    TILE_TEST_N_LAT,
    TILE_TEST_N_LON,
    write_tile_test_files,
)
from fme.core.dataset.utils import FillNaNsConfig
from fme.core.dataset.xarray import XarrayDataConfig, XarrayDataset
from fme.core.distributed import Distributed

Engine = Literal["netcdf4", "cdf5"]
ENGINE_FORMATS: dict[Engine, str] = {"netcdf4": "NETCDF4", "cdf5": "NETCDF3_64BIT_DATA"}
NAMES = ["foo", "bar", "static", "x", "missing"]
TILES = [
    (slice(0, 5), slice(0, TILE_TEST_N_LON)),  # latitude band
    (slice(5, 9), slice(0, TILE_TEST_N_LON)),
    (slice(0, TILE_TEST_N_LAT), slice(8, 15)),  # longitude band
    (slice(4, 9), slice(0, 8)),  # 2D tile
]


def _build_dataset(path, engine: Engine, n_timesteps: int = 3) -> XarrayDataset:
    config = XarrayDataConfig(
        data_path=str(path),
        engine=engine,
        fill_nans=FillNaNsConfig(value=-1.0),
    )
    return XarrayDataset(
        config,
        NAMES,
        IntSchedule.from_constant(n_timesteps),
        allow_missing_variables=True,
    )


def _assert_items_equal_sliced(tiled, global_, tile):
    tiled_data, tiled_time, _, _, tiled_missing = tiled
    global_data, global_time, _, _, global_missing = global_
    assert tiled_missing == global_missing
    xr.testing.assert_equal(tiled_time, global_time)
    assert set(tiled_data) == set(global_data)
    for name, tensor in global_data.items():
        expected = tensor[(..., *tile)]
        assert tiled_data[name].shape == expected.shape, name
        torch.testing.assert_close(
            tiled_data[name], expected, rtol=0, atol=0, equal_nan=True
        )


@pytest.mark.parametrize("engine", list(ENGINE_FORMATS))
@pytest.mark.parametrize("tile", TILES)
def test_xarray_dataset_tile_matches_global_slice(tmp_path, engine, tile):
    write_tile_test_files(tmp_path, ENGINE_FORMATS[engine])
    reference = _build_dataset(tmp_path, engine)
    tiled = _build_dataset(tmp_path, engine)
    assert tiled.supports_spatial_tile
    tiled.set_spatial_tile(tile)
    assert tiled.properties.horizontal_coordinates.shape == (
        TILE_TEST_N_LAT,
        TILE_TEST_N_LON,
    )
    # index 4 spans the boundary between the two files
    for idx in (0, 4, len(reference) - 1):
        _assert_items_equal_sliced(tiled[idx], reference[idx], tile)
    _assert_items_equal_sliced(
        tiled.get_sample_by_time_slice(slice(2, 9)),
        reference.get_sample_by_time_slice(slice(2, 9)),
        tile,
    )


@pytest.mark.parametrize(
    "tile",
    [
        (slice(0, 9, 2), slice(None)),
        (slice(3, 3), slice(None)),
        (slice(None), slice(10, 4)),
    ],
)
def test_xarray_dataset_rejects_non_contiguous_tile(tmp_path, tile):
    write_tile_test_files(tmp_path, "NETCDF4", n_files=1)
    dataset = _build_dataset(tmp_path, "netcdf4")
    with pytest.raises(ValueError, match="non-empty contiguous slice"):
        dataset.set_spatial_tile(tile)


def test_wrapped_datasets_propagate_tile(tmp_path):
    write_tile_test_files(tmp_path / "a", "NETCDF4")
    write_tile_test_files(tmp_path / "b", "NETCDF4", seed=1)
    tile = (slice(4, 9), slice(0, 8))
    names = ["foo", "bar"]
    concat_config = ConcatDatasetConfig(
        concat=[XarrayDataConfig(data_path=str(tmp_path / d)) for d in ("a", "b")]
    )
    merge_config = MergeNoConcatDatasetConfig(
        merge=[
            XarrayDataConfig(data_path=str(tmp_path / "a")),
            XarrayDataConfig(
                data_path=str(tmp_path / "b"), rename={"foo": "foo_b", "bar": "bar_b"}
            ),
        ]
    )
    for config, config_names in (
        (concat_config, names),
        (merge_config, names + ["foo_b", "bar_b"]),
    ):
        reference, _ = config.build(config_names, IntSchedule.from_constant(2))
        tiled, _ = config.build(config_names, IntSchedule.from_constant(2))
        assert tiled.supports_spatial_tile
        tiled.set_spatial_tile(tile)
        for idx in (0, len(reference) - 1):
            _assert_items_equal_sliced(tiled[idx], reference[idx], tile)


def test_cdf5_reads_only_the_latitude_band(tmp_path, monkeypatch):
    """A latitude slice is one pread of just the band; a longitude slice
    reads the band's full rows once."""
    write_tile_test_files(tmp_path, "NETCDF3_64BIT_DATA", n_files=1)
    calls: list[int] = []
    real_pread_into = cdf5._pread_into

    def recording_pread_into(fd, buf, offset):
        calls.append(len(buf))
        real_pread_into(fd, buf, offset)

    monkeypatch.setattr(cdf5, "_pread_into", recording_pread_into)
    path = os.path.join(tmp_path, "data_0.nc")
    with xr.open_dataset(path, engine=CDF5BackendEntrypoint) as ds:
        global_values = ds["bar"].isel(time=slice(1, 3)).values
        calls.clear()
        band = ds["bar"].isel(time=slice(1, 3), lat=slice(2, 6)).values
        np.testing.assert_array_equal(band, global_values[:, 2:6])
        assert calls == [4 * TILE_TEST_N_LON * 4] * 2
        calls.clear()
        tile = ds["bar"].isel(time=slice(1, 3), lat=slice(2, 6), lon=slice(3, 9)).values
        np.testing.assert_array_equal(tile, global_values[:, 2:6, 3:9])
        assert calls == [4 * TILE_TEST_N_LON * 4] * 2


def test_scatter_spatial_rejects_unexpected_shape():
    dist = Distributed.get_instance()
    data = {"a": torch.zeros(2, 3, 4, 5)}
    local = data["a"][(..., *dist.get_local_slices((4, 5)))]
    assert dist.scatter_spatial(data, (4, 5))["a"].shape == local.shape
    with pytest.raises(ValueError, match="match neither"):
        dist.scatter_spatial(data, (4, 6))
