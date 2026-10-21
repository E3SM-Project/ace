"""Spatially parallel inference produces the same outputs as a serial one.

Run under torchrun with the model backend, e.g. (H, W) in (2, 1), (1, 2),
(2, 2). Each test runs a serial reference in-process on the root rank (under
``Distributed.force_non_distributed``) and the same run on every rank, then
compares on the root every netCDF file written (the data writers' files,
initial_condition.nc and restart.nc, the step diagnostics and the reduced
diagnostics) and every logged metric and image.
"""

import dataclasses
import datetime
import pathlib
from collections.abc import Callable

import numpy as np
import pytest
import torch
import xarray as xr
import yaml

from fme.ace.aggregator.inference.main import (
    InferenceAggregatorConfig,
    InferenceEvaluatorAggregatorConfig,
)
from fme.ace.aggregator.inference.step_diagnostics import StepDiagnosticsMetricConfig
from fme.ace.aggregator.inference.video import VideoMetricConfig
from fme.ace.data_loading.inference import InferenceInitialConditionIndices
from fme.ace.inference.data_writer import DataWriterConfig
from fme.ace.inference.data_writer.file_writer import FileWriterConfig
from fme.ace.inference.data_writer.time_coarsen import TimeCoarsenConfig
from fme.ace.inference.evaluator import (
    InferenceEvaluatorConfig,
    run_evaluator_from_config,
)
from fme.ace.inference.inference import main as inference_main
from fme.ace.inference.test_evaluator import save_plus_one_stepper
from fme.ace.inference.test_inference import save_stepper
from fme.ace.testing import DimSize, DimSizes, FV3GFSData
from fme.ace.testing.spatial import (
    assert_logs_close,
    assert_netcdf_files_close,
    shared_dir,
)
from fme.core.corrector.atmosphere import AtmosphereCorrectorConfig
from fme.core.distributed import Distributed
from fme.core.logging_utils import LoggingConfig
from fme.core.testing import mock_wandb
from fme.core.timing import GlobalTimer

TIMESTEP = datetime.timedelta(hours=6)
# 9 latitudes split unevenly over 2 tiles
N_LAT = 9
N_LON = 16
N_FORWARD_STEPS = 4
# differs between runs
IGNORED_LOGS = ("total_steps_per_second",)


def _run_serial_and_parallel(
    path: pathlib.Path, run: Callable[[pathlib.Path], None], monkeypatch
) -> tuple[list[dict], list[dict]]:
    """``run`` serially on the root into path/serial, then on every rank into
    path/parallel, recording the logs of each (but not the timings, which
    differ between runs).
    """
    monkeypatch.setattr(GlobalTimer, "get_durations", lambda self: {})
    dist = Distributed.get_instance()
    serial_logs: list[dict] = []
    if dist.is_root():
        with Distributed.force_non_distributed(), mock_wandb() as wandb:
            wandb.configure(log_to_wandb=True)
            with torch.no_grad():
                run(path / "serial")
            serial_logs = wandb.get_logs()
    dist.barrier()
    with mock_wandb() as wandb:
        wandb.configure(log_to_wandb=True)
        with torch.no_grad():
            run(path / "parallel")
        parallel_logs = wandb.get_logs()
    dist.barrier()
    return serial_logs, parallel_logs


def _data_writer_config() -> DataWriterConfig:
    return DataWriterConfig(
        save_prediction_files=True,
        save_monthly_files=True,
        save_step_diagnostics=True,
        files=[
            FileWriterConfig("region", names=["var"], lat_extent=[2.0, 6.0]),
            FileWriterConfig(
                "coarse", time_coarsen=TimeCoarsenConfig(coarsen_factor=2)
            ),
        ],
    )


def _skip_if_data_parallel():
    if Distributed.get_instance().total_data_parallel_ranks > 1:
        pytest.skip("per-timestep writers do not support data parallelism")


@pytest.mark.parallel
def test_evaluator_matches_serial(tmp_path, monkeypatch):
    _skip_if_data_parallel()
    dist = Distributed.get_instance()
    path = shared_dir(tmp_path)
    names = ["var", "other"]
    n_ic = 2
    dim_sizes = DimSizes(
        n_time=(n_ic - 1) + N_FORWARD_STEPS + 1,
        horizontal=[DimSize("lat", N_LAT), DimSize("lon", N_LON)],
        nz_interface=2,
    )
    stepper_path = path / "stepper"
    data: FV3GFSData | None = None
    if dist.is_root():
        data = FV3GFSData(
            path=path,
            names=names,
            dim_sizes=dim_sizes,
            timestep_days=TIMESTEP.total_seconds() / 86400,
            save_vertical_coordinate=False,
        )
        save_plus_one_stepper(
            stepper_path,
            names,
            names,
            mean=0.0,
            std=1.0,
            data_shape=dim_sizes.shape_nd,
            # modifies the (random) data, so there are step diagnostics
            corrector=AtmosphereCorrectorConfig(force_positive_names=["other"]),
        )
    loader = dist.scatter_object(data).inference_data_loader_config

    def run(experiment_dir: pathlib.Path):
        config = InferenceEvaluatorConfig(
            experiment_dir=str(experiment_dir),
            n_forward_steps=N_FORWARD_STEPS,
            checkpoint_path=str(stepper_path),
            logging=LoggingConfig(
                log_to_screen=False, log_to_file=False, log_to_wandb=True
            ),
            loader=dataclasses.replace(
                loader,
                start_indices=InferenceInitialConditionIndices(
                    n_initial_conditions=n_ic, first=0, interval=1
                ),
            ),
            aggregator=InferenceEvaluatorAggregatorConfig(
                video=VideoMetricConfig(enabled=True),
                step_diagnostics=StepDiagnosticsMetricConfig(correction_maps=True),
            ),
            data_writer=_data_writer_config(),
            forward_steps_in_memory=2,
            allow_incompatible_dataset=True,
        )
        run_evaluator_from_config(config)

    serial_logs, parallel_logs = _run_serial_and_parallel(path, run, monkeypatch)
    if not dist.is_root():
        return
    summary = serial_logs[-1]
    for key in ("time_mean/gen_map/var", "zonal_mean/gen/var", "video/var"):
        assert f"inference/{key}" in summary, key
    assert any(key.startswith("inference/power_spectrum/") for key in summary)
    assert (path / "serial" / "step_diagnostics" / "correction_deltas.nc").exists()
    assert_logs_close(serial_logs, parallel_logs, ignore=IGNORED_LOGS)
    assert_netcdf_files_close(path / "serial", path / "parallel")


@pytest.mark.parallel
def test_standalone_inference_matches_serial(tmp_path, monkeypatch):
    _skip_if_data_parallel()
    dist = Distributed.get_instance()
    path = shared_dir(tmp_path)
    n_ic = 2
    stepper_path = path / "stepper"
    ic_path = path / "ic.nc"
    data: FV3GFSData | None = None
    if dist.is_root():
        data = FV3GFSData(
            path=path,
            names=["forcing_var", "DSWRFtoa", "sst", "ocean_fraction"],
            dim_sizes=DimSizes(
                n_time=(n_ic - 1) + N_FORWARD_STEPS + 1,
                horizontal=[DimSize("lat", N_LAT), DimSize("lon", N_LON)],
                nz_interface=2,
            ),
            timestep_days=TIMESTEP.total_seconds() / 86400,
            save_vertical_coordinate=False,
        )
        save_stepper(
            stepper_path,
            in_names=["prog", "sst", "forcing_var", "DSWRFtoa"],
            out_names=["prog", "sst", "ULWRFtoa", "USWRFtoa"],
            mean=0.0,
            std=1.0,
            horizontal_coords=data.horizontal_coords,
            nz_interface=2,
        )
        generator = np.random.default_rng(0)
        ic = {
            name: xr.DataArray(
                generator.random((n_ic, N_LAT, N_LON), dtype=np.float32),
                dims=["sample", "lat", "lon"],
            )
            for name in ["prog", "sst", "DSWRFtoa"]
        }
        ic["time"] = xr.DataArray(
            xr.open_dataset(data.data_filename).time.values[:n_ic], dims=["time"]
        )
        xr.Dataset(ic).to_netcdf(ic_path)
    data_path = dist.scatter_object(data).data_path

    def run(experiment_dir: pathlib.Path):
        data_writer = dataclasses.asdict(_data_writer_config())
        data_writer["save_step_diagnostics"] = False
        data_writer["files"][0]["names"] = ["prog"]
        config = {
            "experiment_dir": str(experiment_dir),
            "n_forward_steps": N_FORWARD_STEPS,
            "checkpoint_path": str(stepper_path),
            "logging": {
                "log_to_screen": False,
                "log_to_file": False,
                "log_to_wandb": True,
            },
            "initial_condition": {"path": str(ic_path)},
            "forcing_loader": {"dataset": {"data_path": str(data_path)}},
            "forward_steps_in_memory": 2,
            "aggregator": dataclasses.asdict(InferenceAggregatorConfig()),
            "data_writer": data_writer,
            "allow_incompatible_dataset": True,
        }
        config_path = experiment_dir.parent / f"{experiment_dir.name}.yaml"
        if Distributed.get_instance().is_root():
            with open(config_path, "w") as f:
                yaml.dump(config, f)
        Distributed.get_instance().barrier()
        inference_main(str(config_path))

    serial_logs, parallel_logs = _run_serial_and_parallel(path, run, monkeypatch)
    if not dist.is_root():
        return
    assert (path / "serial" / "restart.nc").exists()
    assert any(key.startswith("inference/time_mean/") for key in serial_logs[-1])
    assert_logs_close(serial_logs, parallel_logs, ignore=IGNORED_LOGS)
    assert_netcdf_files_close(path / "serial", path / "parallel")
