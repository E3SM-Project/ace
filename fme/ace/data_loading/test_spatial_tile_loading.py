"""Under spatial parallelism each rank loads only its tile of the data.

Every test compares the loaders as built (tile loading) against the same
loaders with tile loading switched off, i.e. reading the global fields and
slicing them with ``get_local_slices`` as before. The two must agree exactly.
Run under torchrun with the model backend, e.g. (H, W) in (2, 1), (1, 2),
(2, 2); serially the tile is the whole grid and the comparison is trivial.
"""

import contextlib
import pathlib

import pytest
import torch
import xarray as xr

from fme.ace.data_loading.augmentation import AugmentationConfig
from fme.ace.data_loading.batch_data import BatchData
from fme.ace.data_loading.config import DataLoaderConfig
from fme.ace.data_loading.getters import get_gridded_data, get_inference_data
from fme.ace.data_loading.inference import ExplicitIndices, InferenceDataLoaderConfig
from fme.ace.data_loading.perturbation import PerturbationSelector, SSTPerturbation
from fme.ace.requirements import DataRequirements, PrognosticStateDataRequirements
from fme.core.dataset import cdf5
from fme.core.dataset.testing import (
    TILE_TEST_N_LAT,
    TILE_TEST_N_LON,
    write_tile_test_files,
)
from fme.core.dataset.xarray import XarrayDataConfig
from fme.core.distributed import Distributed

ENGINE_FORMATS = {"netcdf4": "NETCDF4", "cdf5": "NETCDF3_64BIT_DATA"}
GLOBAL_SHAPE = (TILE_TEST_N_LAT, TILE_TEST_N_LON)
NAMES = ["foo", "bar", "static", "x", "sst", "ocnfrac"]


def _write_on_root(tmp_path: pathlib.Path, engine: str) -> pathlib.Path:
    dist = Distributed.get_instance()
    root: pathlib.Path = dist.scatter_object(tmp_path)  # the root rank's path
    path = root / engine
    if dist.is_root():
        write_tile_test_files(path, ENGINE_FORMATS[engine])
    dist.barrier()
    return path


@contextlib.contextmanager
def _global_loading(monkeypatch):
    """Switch tile loading off: every rank reads the global fields."""
    with monkeypatch.context() as m:
        m.setattr(Distributed, "get_spatial_tile", lambda self, img_shape: None)
        yield


def _local(tensor: torch.Tensor) -> torch.Tensor:
    slices = Distributed.get_instance().get_local_slices(GLOBAL_SHAPE)
    return tensor[(..., *slices)]


def _is_split() -> bool:
    return Distributed.get_instance().get_spatial_tile(GLOBAL_SHAPE) is not None


def _assert_batch_is_global_sliced(local: BatchData, global_: BatchData):
    """``local`` (on device, from the loader as built) equals the global
    batch read with tile loading off, sliced to this rank's tile."""
    xr.testing.assert_equal(local.time, global_.time)
    assert set(local.data) == set(global_.data)
    for name, tensor in global_.data.items():
        assert tuple(tensor.shape[-2:]) == GLOBAL_SHAPE, name
        torch.testing.assert_close(
            local.data[name].cpu(), _local(tensor.cpu()), rtol=0, atol=0, equal_nan=True
        )


def _train_config(path, engine, num_workers, time_buffer, rotate=0.0):
    dist = Distributed.get_instance()
    return DataLoaderConfig(
        dataset=XarrayDataConfig(data_path=str(path), engine=engine),
        batch_size=2 * dist.total_data_parallel_ranks,
        num_data_workers=num_workers,
        time_buffer=time_buffer,
        augmentation=AugmentationConfig(rotate_probability=rotate),
    )


@pytest.mark.parallel
@pytest.mark.parametrize("engine", list(ENGINE_FORMATS))
@pytest.mark.parametrize(
    "train, num_workers, time_buffer", [(False, 0, 0), (True, 2, 1)]
)
def test_train_loader_tiles_match_global_slices(
    tmp_path, monkeypatch, engine, train, num_workers, time_buffer
):
    path = _write_on_root(tmp_path, engine)
    config = _train_config(path, engine, num_workers, time_buffer)
    requirements = DataRequirements(NAMES, n_timesteps=2)
    rows_read: list[int] = []
    real_read_slab = cdf5._CDF5File.read_slab

    def recording_read_slab(self, name, record, lo, hi, out):
        if name in ("foo", "bar", "sst", "ocnfrac"):
            rows_read.append(hi - lo)
        real_read_slab(self, name, record, lo, hi, out)

    monkeypatch.setattr(cdf5._CDF5File, "read_slab", recording_read_slab)
    tiled = get_gridded_data(config, train=train, requirements=requirements)
    with _global_loading(monkeypatch):
        reference = get_gridded_data(config, train=train, requirements=requirements)
    rows_read.clear()
    # one at a time: a loader with a time buffer is its own iterator
    local_batches = list(tiled.loader)
    raw_tiled_batches = list(tiled._loader)
    tiled_rows_read = set(rows_read)
    raw_global_batches = list(reference._loader)

    assert len(local_batches) == len(reference._loader) > 0
    for local_batch, raw_tiled, raw_global in zip(
        local_batches, raw_tiled_batches, raw_global_batches, strict=True
    ):
        _assert_batch_is_global_sliced(local_batch, raw_global)
        # the loader really did deliver tiles, not global fields
        local_hw = tuple(_local(raw_global.data["foo"]).shape[-2:])
        assert tuple(raw_tiled.data["foo"].shape[-2:]) == local_hw
    if engine == "cdf5" and num_workers == 0:
        # each pread covered only this rank's latitude band
        h_slice = Distributed.get_instance().get_local_slices(GLOBAL_SHAPE)[0]
        assert tiled_rows_read == {len(range(*h_slice.indices(TILE_TEST_N_LAT)))}


@pytest.mark.parallel
def test_global_modifier_keeps_global_loading(tmp_path, monkeypatch):
    """Rotating the sphere needs global fields, so tile loading is skipped
    and the result is the same as before."""
    path = _write_on_root(tmp_path, "cdf5")
    config = _train_config(path, "cdf5", num_workers=0, time_buffer=0, rotate=1.0)
    requirements = DataRequirements(NAMES, n_timesteps=2)
    tiled = get_gridded_data(config, train=False, requirements=requirements)
    with _global_loading(monkeypatch):
        reference = get_gridded_data(config, train=False, requirements=requirements)
    for local_batch, ref_batch, raw_tiled in zip(
        list(tiled.loader), list(reference.loader), list(tiled._loader), strict=True
    ):
        assert tuple(raw_tiled.data["foo"].shape[-2:]) == GLOBAL_SHAPE
        for name, tensor in ref_batch.data.items():
            torch.testing.assert_close(
                local_batch.data[name], tensor, rtol=0, atol=0, equal_nan=True
            )


@pytest.mark.parallel
@pytest.mark.parametrize("engine", list(ENGINE_FORMATS))
@pytest.mark.parametrize("num_workers", [0, 2])
def test_inference_loader_tiles_match_global_slices(
    tmp_path, monkeypatch, engine, num_workers
):
    path = _write_on_root(tmp_path, engine)
    dist = Distributed.get_instance()
    config = InferenceDataLoaderConfig(
        dataset=XarrayDataConfig(data_path=str(path), engine=engine),
        start_indices=ExplicitIndices(
            [3 * i for i in range(dist.total_data_parallel_ranks)]
        ),
        num_data_workers=num_workers,
        perturbations=SSTPerturbation(
            sst=[
                PerturbationSelector(
                    type="greens_function",
                    config={
                        "amplitude": 2.0,
                        "lat_center": 10.0,
                        "lon_center": 150.0,
                        "lat_width": 90.0,
                        "lon_width": 200.0,
                    },
                )
            ]
        ),
        persistence_names=["bar"],
    )

    def build():
        return get_inference_data(
            config,
            total_forward_steps=4,
            window_requirements=DataRequirements(NAMES, n_timesteps=3),
            initial_condition=PrognosticStateDataRequirements(
                names=["foo", "sst"], n_timesteps=1
            ),
            surface_temperature_name="sst",
            ocean_fraction_name="ocnfrac",
        )

    tiled = build()
    with _global_loading(monkeypatch):
        reference = build()

    n_batches = 0
    for local_batch, raw_tiled, raw_global in zip(
        list(tiled.loader), list(tiled._loader), list(reference._loader), strict=True
    ):
        _assert_batch_is_global_sliced(local_batch, raw_global)
        if _is_split():
            assert tuple(raw_tiled.data["foo"].shape[-2:]) != GLOBAL_SHAPE
        n_batches += 1
    assert n_batches == len(reference._loader) > 0
    local_ic = tiled.initial_condition.as_batch_data()
    reference_ic = reference.initial_condition.as_batch_data()
    xr.testing.assert_equal(local_ic.time, reference_ic.time)
    for name, tensor in reference_ic.data.items():
        torch.testing.assert_close(
            local_ic.data[name], tensor, rtol=0, atol=0, equal_nan=True
        )
