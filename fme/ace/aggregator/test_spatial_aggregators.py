"""Aggregators produce the same maps, images and metrics under spatial
parallelism as serially.

Run under torchrun with any backend and layout, e.g. the model backend with
(H, W) in (2, 1), (1, 2), (2, 2), alone or with data parallelism. Every rank
records its data-parallel block of samples and its spatial tile of the same
global data that a serial reference, run in-process on every rank with
``Distributed.force_non_distributed``, records in full. The root compares
the logged metrics, the images (their captions and the plotted arrays) and
the per-epoch diagnostics netCDF files. The other ranks draw no images.
Under data parallelism the ensemble mean RMSE and spread-skill ratio are
skipped, as by convention they are means of per-rank values.
"""

import dataclasses
import datetime
import pathlib
from collections.abc import Callable
from typing import Any

import cftime
import numpy as np
import pytest
import torch
import xarray as xr

import fme.ace.aggregator.inference.enso.enso_coefficient as enso_coefficient
import fme.ace.aggregator.plotting as plotting
from fme.ace.aggregator.inference.main import (
    InferenceAggregatorConfig,
    InferenceEvaluatorAggregatorConfig,
)
from fme.ace.aggregator.inference.near_zero_fraction import NearZeroFractionMetricConfig
from fme.ace.aggregator.inference.seasonal import SeasonalMetricConfig
from fme.ace.aggregator.inference.step_diagnostics import StepDiagnosticsMetricConfig
from fme.ace.aggregator.inference.trend import TrendMetricConfig
from fme.ace.aggregator.inference.video import VideoMetricConfig
from fme.ace.aggregator.one_step import OneStepAggregatorConfig
from fme.ace.aggregator.one_step.ensemble import OneStepEnsembleMetricConfig
from fme.ace.aggregator.one_step.reduced import StepMeanMetricConfig
from fme.ace.data_loading.batch_data import BatchData, PairedData, PrognosticState
from fme.ace.stepper import TrainOutput
from fme.ace.testing import DimSize, DimSizes, MonthlyReferenceData
from fme.ace.testing.spatial import (
    assert_logs_close,
    assert_netcdf_files_close,
    shared_dir,
)
from fme.core.coordinates import LatLonCoordinates
from fme.core.dataset.data_typing import VariableMetadata
from fme.core.dataset_info import DatasetInfo
from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.step.step_diagnostics import StepDiagnostics
from fme.core.typing_ import EnsembleTensorDict, TensorDict, TensorMapping

# 9 x 18 splits unevenly over 2 tiles; the grid covers the Nino 3.4 box
N_LAT, N_LON = 9, 18
LAT = torch.linspace(-80.0, 80.0, N_LAT)
LON = torch.arange(N_LON, dtype=torch.float32) * 20.0
NAMES = ["sst", "b"]
SAMPLES_PER_RANK = 2
MONTH = datetime.timedelta(days=30)
# > 5 years of monthly steps: enables the annual, ENSO index and ENSO
# coefficient metrics, and covers every season
N_FORWARD_STEPS = 73
WINDOW = 8


def _dataset_info(timestep: datetime.timedelta) -> DatasetInfo:
    return DatasetInfo(
        horizontal_coordinates=LatLonCoordinates(lat=LAT, lon=LON),
        timestep=timestep,
        variable_metadata={
            name: VariableMetadata(units="K", long_name=f"long {name}")
            for name in NAMES
        },
    )


def _times(n_samples: int, start: int, n_time: int) -> xr.DataArray:
    t0 = cftime.DatetimeJulian(2000, 1, 1)
    times = [[t0 + (start + i) * MONTH for i in range(n_time)]] * n_samples
    return xr.DataArray(np.array(times), dims=["sample", "time"])


class _View:
    """Maps global fields to what this rank holds: everything serially, or
    its data-parallel block of samples and its spatial tile.

    The global data have SAMPLES_PER_RANK samples per data-parallel rank of
    the parallel run, so ``n_data_parallel`` is taken outside
    ``force_non_distributed``.
    """

    def __init__(self, serial: bool, n_data_parallel: int):
        self._serial = serial
        self._n_samples = SAMPLES_PER_RANK * n_data_parallel

    def global_fields(self, seed: int, n_time: int, offset: float = 0.0) -> TensorDict:
        generator = torch.Generator().manual_seed(seed)
        shape = (self._n_samples, n_time, N_LAT, N_LON)
        return {
            name: torch.randn(*shape, generator=generator) + offset for name in NAMES
        }

    def __call__(self, tensor: torch.Tensor, sample_dim: int = 0) -> torch.Tensor:
        if self._serial:
            return tensor.to(get_device())
        dist = Distributed.get_instance()
        start = dist.data_parallel_rank * SAMPLES_PER_RANK
        samples = tensor.narrow(sample_dim, start, SAMPLES_PER_RANK)
        tile = dist.get_local_slices(tuple(tensor.shape[-2:]))
        return samples[(..., *tile)].contiguous().to(get_device())

    def fields(
        self,
        data: TensorMapping,
        select: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
    ) -> TensorDict:
        """This rank's part of ``select`` of each field, in an order that
        differs between ranks, as the names of real data come from sets, which
        each process orders differently.
        """
        names = list(data)
        if not self._serial and Distributed.get_instance().rank % 2 == 1:
            names = names[::-1]
        return {name: self(select(data[name])) for name in names}

    def times(self, start: int, n_time: int) -> xr.DataArray:
        n = self._n_samples if self._serial else SAMPLES_PER_RANK
        return _times(n, start, n_time)


@dataclasses.dataclass
class _Result:
    logs: list[dict[str, Any]]
    images: list[np.ndarray]


def _run(
    serial: bool, monkeypatch, run: Callable[[_View], list[dict[str, Any]]]
) -> _Result:
    """Run ``run`` serially or on every rank, recording every plotted array."""
    images: list[np.ndarray] = []
    real_plot_imshow = plotting.plot_imshow

    def recording_plot_imshow(data, *args, **kwargs):
        images.append(np.array(data))
        return real_plot_imshow(data, *args, **kwargs)

    n_data_parallel = Distributed.get_instance().total_data_parallel_ranks
    with monkeypatch.context() as m:
        m.setattr(plotting, "plot_imshow", recording_plot_imshow)
        m.setattr(enso_coefficient, "plot_imshow", recording_plot_imshow)
        if serial:
            with Distributed.force_non_distributed():
                logs = run(_View(serial=True, n_data_parallel=n_data_parallel))
        else:
            logs = run(_View(serial=False, n_data_parallel=n_data_parallel))
    return _Result(logs=logs, images=images)


def _compare_on_root(serial: _Result, parallel: _Result, dirs=()):
    dist = Distributed.get_instance()
    dist.barrier()
    if not dist.is_root():
        # only the root's images are logged
        assert parallel.images == []
        return
    ignore: tuple[str, ...] = ()
    nc_rtol = 1e-5
    if dist.total_data_parallel_ranks > 1:
        # means over ranks of each rank's RMSE and spread-skill ratio, not
        # the global values
        ignore = ("ensemble_mean_rmse", "ssr_bias")
        # float32 sample means summed in a different order; gen_var, a ratio
        # of variances of a few samples, amplifies the rounding
        nc_rtol = 1e-4
    assert_logs_close(serial.logs, parallel.logs, ignore=ignore)
    assert len(serial.images) == len(parallel.images) > 0
    for s, p in zip(serial.images, parallel.images):
        np.testing.assert_allclose(p, s, rtol=1e-5, atol=1e-6, equal_nan=True)
    for serial_dir, parallel_dir in dirs:
        assert_netcdf_files_close(serial_dir, parallel_dir, rtol=nc_rtol)


def _write_reference_data(path: pathlib.Path) -> tuple[str, str]:
    """Global time-mean and monthly reference files, written on the root."""
    dist = Distributed.get_instance()
    time_mean_path = path / "time_mean_reference.nc"
    monthly = path / "monthly_data" / "monthly.nc"
    if dist.is_root():
        generator = torch.Generator().manual_seed(7)
        xr.Dataset(
            {
                name: (("lat", "lon"), torch.randn(N_LAT, N_LON, generator=generator))
                for name in NAMES
            }
        ).to_netcdf(time_mean_path)
        monthly = MonthlyReferenceData(
            path=path,
            names=NAMES,
            dim_sizes=DimSizes(
                n_time=72,
                horizontal=[DimSize("lat", N_LAT), DimSize("lon", N_LON)],
                nz_interface=1,
            ),
            n_ensemble=3,
        ).data_filename
    dist.barrier()
    return str(time_mean_path), str(dist.scatter_object(monthly))


def _normalize(data: TensorMapping, apply_mean: bool = True) -> TensorDict:
    mean = 0.5 if apply_mean else 0.0
    return {name: (value - mean) * 2.0 for name, value in data.items()}


def _record_inference(agg, view: _View, with_reference: bool) -> list[dict]:
    """Record an initial condition and the forward steps in windows."""
    target = view.global_fields(seed=1, n_time=N_FORWARD_STEPS + 1)
    gen = view.global_fields(seed=2, n_time=N_FORWARD_STEPS + 1, offset=0.1)
    delta = view.global_fields(seed=3, n_time=N_FORWARD_STEPS + 1)
    logs = []
    ic_target = view.fields(target, lambda x: x[:, :1])
    ic_gen = view.fields(gen, lambda x: x[:, :1])
    if with_reference:
        ic: PairedData | PrognosticState = PairedData(
            prediction=ic_gen, reference=ic_target, time=view.times(0, 1)
        )
    else:
        ic = PrognosticState(BatchData(data=ic_gen, time=view.times(0, 1)))
    logs.extend(agg.record_initial_condition(ic))
    for start in range(1, N_FORWARD_STEPS + 1, WINDOW):
        stop = min(start + WINDOW, N_FORWARD_STEPS + 1)
        window = slice(start, stop)
        logs.extend(
            agg.record_batch(
                PairedData(
                    prediction=view.fields(gen, lambda x: x[:, window]),
                    reference=view.fields(target, lambda x: x[:, window]),
                    time=view.times(start, stop - start),
                    step_diagnostics=StepDiagnostics(
                        delta={"b": view(delta["b"][:, window])}
                    ),
                )
            )
        )
    return logs


@pytest.mark.parallel
def test_inference_evaluator_aggregator_matches_serial(tmp_path, monkeypatch):
    path = shared_dir(tmp_path)
    time_mean_reference, monthly_reference = _write_reference_data(path)
    config = InferenceEvaluatorAggregatorConfig(
        step_means=[
            StepMeanMetricConfig(step=2),
            StepMeanMetricConfig(step=2, target="norm"),
        ],
        ensembles=[],
        video=VideoMetricConfig(enabled=True, enable_extended_videos=True),
        seasonal=SeasonalMetricConfig(enabled=True),
        trend=TrendMetricConfig(enabled=True),
        near_zero_fraction=NearZeroFractionMetricConfig(
            enabled=True, variables=["b"], include_maps=True
        ),
        step_diagnostics=StepDiagnosticsMetricConfig(correction_maps=True),
        time_mean_reference_data=time_mean_reference,
        monthly_reference_data=monthly_reference,
    )

    def run(view: _View, output_dir: pathlib.Path) -> list[dict]:
        agg = config.build(
            dataset_info=_dataset_info(MONTH),
            n_ic_steps=1,
            n_forward_steps=N_FORWARD_STEPS,
            initial_time=view.times(0, 1).isel(time=0),
            normalize=_normalize,
            output_dir=str(output_dir),
        )
        logs = _record_inference(agg, view, with_reference=True)
        logs.append(agg.get_summary_logs())
        agg.flush_diagnostics(subdir="epoch_0001")
        return logs

    serial = _run(True, monkeypatch, lambda view: run(view, path / "serial"))
    parallel = _run(False, monkeypatch, lambda view: run(view, path / "parallel"))
    # every map-type metric took part
    for key in (
        "time_mean/gen_map/sst",
        "zonal_mean/gen/sst",
        "video/sst",
        "power_spectrum/mean_abs_norm_bias/sst",
        "trend/maps/sst",
        "seasonal/bias/sst",
        "enso_coefficient/rmse/sst",
        "near_zero_fraction/error_map/b",
        "time_mean_norm/correction_map/b",
    ):
        assert key in serial.logs[-1], key
    _compare_on_root(
        serial,
        parallel,
        dirs=[(path / "serial" / "epoch_0001", path / "parallel" / "epoch_0001")],
    )


@pytest.mark.parallel
def test_inference_aggregator_matches_serial(tmp_path, monkeypatch):
    path = shared_dir(tmp_path)
    time_mean_reference, _ = _write_reference_data(path)
    config = InferenceAggregatorConfig(time_mean_reference_data=time_mean_reference)

    def run(view: _View, output_dir: pathlib.Path) -> list[dict]:
        agg = config.build(
            dataset_info=_dataset_info(MONTH),
            n_timesteps=N_FORWARD_STEPS + 1,
            output_dir=str(output_dir),
        )
        logs = _record_inference(agg, view, with_reference=False)
        logs.append(agg.get_summary_logs())
        agg.flush_diagnostics()
        return logs

    serial = _run(True, monkeypatch, lambda view: run(view, path / "serial"))
    parallel = _run(False, monkeypatch, lambda view: run(view, path / "parallel"))
    assert "enso_index/sst" in str(serial.logs[-1].keys()) or any(
        k.startswith("enso_index") for k in serial.logs[-1]
    )
    _compare_on_root(serial, parallel, dirs=[(path / "serial", path / "parallel")])


@pytest.mark.parallel
def test_one_step_aggregator_matches_serial(tmp_path, monkeypatch):
    path = shared_dir(tmp_path)
    n_ensemble, n_time = 2, 2
    # the RMSE and spread-skill maps are means of per-rank maps (see
    # _compare_on_root), and the recorded images are not keyed
    data_parallel = Distributed.get_instance().total_data_parallel_ranks > 1
    config = OneStepAggregatorConfig(
        ensemble_denorm=OneStepEnsembleMetricConfig(
            target="denorm", log_mean_maps=not data_parallel
        ),
    )

    def run(view: _View, output_dir: pathlib.Path) -> list[dict]:
        agg = config.build(
            _dataset_info(datetime.timedelta(hours=6)), output_dir=str(output_dir)
        )
        for seed in (10, 20):
            target = view.global_fields(seed, n_time)
            gen = {
                name: torch.stack(
                    [value + 0.1 * (i + 1) for i in range(n_ensemble)], dim=1
                )
                for name, value in view.global_fields(seed + 1, n_time).items()
            }
            agg.record_batch(
                TrainOutput(
                    metrics={"loss": torch.tensor(1.0)},
                    target_data=EnsembleTensorDict(
                        {k: v.unsqueeze(1) for k, v in view.fields(target).items()}
                    ),
                    gen_data=EnsembleTensorDict(view.fields(gen)),
                    time=view.times(0, n_time),
                    normalize=_normalize,
                )
            )
        logs = [agg.get_logs(label="val")]
        agg.flush_diagnostics(subdir="epoch_0001")
        return logs

    serial = _run(True, monkeypatch, lambda view: run(view, path / "serial"))
    parallel = _run(False, monkeypatch, lambda view: run(view, path / "parallel"))
    for key in (
        "val/snapshot/image-full-field/sst",
        "val/mean_map/image-error/b",
        "val/power_spectrum/smallest_scale_norm_bias/sst",
        "val/ensemble/crps/b",
        *(() if data_parallel else ("val/ensemble/crps/mean_map/sst",)),
    ):
        assert any(k.startswith(key) for k in serial.logs[0]), key
    _compare_on_root(
        serial,
        parallel,
        dirs=[(path / "serial" / "epoch_0001", path / "parallel" / "epoch_0001")],
    )
