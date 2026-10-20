import dataclasses
import datetime
import os
import warnings
from collections.abc import Mapping, Sequence
from typing import TypeAlias

import cftime
import numpy as np
import numpy.typing as npt
import torch

from fme.ace.data_loading.batch_data import (
    _RESERVED_PREFIX,
    BatchData,
    GatheredBatchData,
    PairedData,
    PrognosticState,
)
from fme.core.cloud import to_netcdf_via_inter_filesystem_copy
from fme.core.dataset.data_typing import VariableMetadata
from fme.core.distributed import Distributed
from fme.core.generics.writer import WriterABC
from fme.core.step.step_diagnostics import StepDiagnostics
from fme.core.typing_ import TensorMapping

from .dataset_metadata import DatasetMetadata
from .file_writer import FileWriter, FileWriterConfig, PairedFileWriter
from .monthly import MonthlyDataWriter, PairedMonthlyDataWriter
from .raw import PairedRawDataWriter, RawDataWriter
from .step_diagnostics import STEP_DIAGNOSTICS_DIR, StepDiagnosticsWriter
from .time_coarsen import PairedTimeCoarsen, TimeCoarsen, TimeCoarsenConfig

PairedSubwriter: TypeAlias = (
    PairedRawDataWriter | PairedTimeCoarsen | PairedMonthlyDataWriter | PairedFileWriter
)

Subwriter: TypeAlias = MonthlyDataWriter | RawDataWriter | TimeCoarsen | FileWriter


@dataclasses.dataclass
class DataWriterConfig:
    """
    Configuration for inference data writers.

    Parameters:
        save_prediction_files: Whether to enable writing of netCDF files
            containing the predictions and target values.
        save_monthly_files: Whether to enable writing of netCDF files
            containing the monthly predictions and target values.
        save_step_diagnostics: Whether to enable writing of netCDF files
            containing per-step diagnostics of the prediction, one file per
            named diagnostics dataset under a ``step_diagnostics/`` directory
            (currently the corrector's per-step correction delta, written to
            ``step_diagnostics/correction_deltas.nc``). Nothing is written
            when no corrector modified anything.
        names: Names of variables to save in the prediction and monthly
            netCDF files.
        time_coarsen: Configuration for time coarsening of written outputs to the
            raw data writer.
        files: Configuration for a sequence of individual data writers. Each data
            writer must have a unique label to avoid filename collisions.
    """

    save_prediction_files: bool = True
    save_monthly_files: bool = True
    save_step_diagnostics: bool = False
    names: Sequence[str] | None = None
    time_coarsen: TimeCoarsenConfig | None = None
    files: list[FileWriterConfig] | None = None

    def __post_init__(self):
        if (
            not any(
                [
                    self.save_prediction_files,
                    self.save_monthly_files,
                    self.save_step_diagnostics,
                ]
            )
            and self.names is not None
        ):
            warnings.warn(
                "names provided but all options to "
                "save subsettable output files are False."
            )
        all_filenames = self._get_all_filenames()
        if len(set(all_filenames)) != len(all_filenames):
            raise ValueError(
                "Duplicate filenames found in file writer configurations. "
                f"Filenames: {all_filenames}"
            )

    @property
    def has_subwriters_enabled(self) -> bool:
        """True when any per-timestep output writer is enabled."""
        return (
            self.save_prediction_files
            or self.save_monthly_files
            or self.save_step_diagnostics
            or bool(self.files)
        )

    def _raise_if_unsupported_under_multi_gpu(self, n_ranks: int) -> None:
        """Raise if per-timestep writers are enabled under multi-GPU."""
        if n_ranks > 1 and self.has_subwriters_enabled:
            raise ValueError(
                "Multi-GPU inference does not yet support per-timestep data "
                "writers (prediction files, monthly files, step diagnostics, "
                "or custom file writers). Set save_prediction_files, "
                "save_monthly_files, and save_step_diagnostics to false and "
                "files to null in the data_writer config, or run with a "
                "single GPU."
            )

    @property
    def _has_file_subwriters(self) -> bool:
        """Whether any sub-writer of prediction (and target) data is enabled."""
        return self.save_prediction_files or self.save_monthly_files or bool(self.files)

    def _names_to_gather(self) -> frozenset[str] | None:
        """The variables any sub-writer saves, or None for all of them."""
        name_lists: list[Sequence[str] | None] = []
        if self.save_prediction_files or self.save_monthly_files:
            name_lists.append(self.names)
        name_lists.extend(file.names for file in self.files or [])
        if any(names is None for names in name_lists):
            return None
        return frozenset(name for names in name_lists for name in names or [])

    def _get_all_filenames(self) -> list[str]:
        filenames = []
        for file in self.files or []:
            filenames.extend(file.filenames)
        return filenames

    def validate_time_coarsen(self, forward_steps_in_memory: int, n_forward_steps: int):
        """Validate time coarsening (top-level and per-file) against the schedule."""
        if self.time_coarsen is not None:
            self.time_coarsen.validate(forward_steps_in_memory, n_forward_steps)
        for file_config in self.files or []:
            file_config.validate_time_coarsen(forward_steps_in_memory, n_forward_steps)

    def build_paired(
        self,
        experiment_dir: str,
        initial_condition_times: npt.NDArray[cftime.datetime],
        n_timesteps: int,
        timestep: datetime.timedelta,
        variable_metadata: Mapping[str, VariableMetadata],
        coords: Mapping[str, np.ndarray],
        dataset_metadata: DatasetMetadata,
    ) -> "PairedDataWriter":
        dist = Distributed.get_instance()
        self._raise_if_unsupported_under_multi_gpu(dist.total_data_parallel_ranks)
        writers: list[PairedSubwriter] = []
        # Spatial co-ranks of the root (the only data-parallel rank when
        # sub-writers are enabled) send it their tiles and open no files.
        writes_files = dist.is_root()
        if self.save_prediction_files and writes_files:
            raw_writer: PairedSubwriter = PairedRawDataWriter(
                path=experiment_dir,
                initial_condition_times=initial_condition_times,
                save_names=self.names,
                variable_metadata=variable_metadata,
                coords=coords,
                dataset_metadata=dataset_metadata,
            )
            if self.time_coarsen is not None:
                raw_writer = self.time_coarsen.build_paired(raw_writer)
            writers.append(raw_writer)
        if self.save_monthly_files and writes_files:
            writers.append(
                PairedMonthlyDataWriter(
                    path=experiment_dir,
                    initial_condition_times=initial_condition_times,
                    n_timesteps=n_timesteps,
                    timestep=timestep,
                    save_names=self.names,
                    variable_metadata=variable_metadata,
                    coords=coords,
                    dataset_metadata=dataset_metadata,
                )
            )
        for writer_config in (self.files or []) if writes_files else []:
            writers.append(
                writer_config.build_paired(
                    experiment_dir=experiment_dir,
                    initial_condition_times=initial_condition_times,
                    n_timesteps=n_timesteps,
                    timestep=timestep,
                    variable_metadata=variable_metadata,
                    coords=coords,
                    dataset_metadata=dataset_metadata,
                )
            )
        return PairedDataWriter(
            writers=writers,
            path=experiment_dir,
            variable_metadata=variable_metadata,
            coords=coords,
            dataset_metadata=dataset_metadata,
            gather_names=self._names_to_gather(),
            has_subwriters=self._has_file_subwriters,
            step_diagnostics_writer=self._build_step_diagnostics_writer(
                experiment_dir=experiment_dir,
                initial_condition_times=initial_condition_times,
                variable_metadata=variable_metadata,
                coords=coords,
                dataset_metadata=dataset_metadata,
            ),
        )

    def _build_step_diagnostics_writer(
        self,
        experiment_dir: str,
        initial_condition_times: npt.NDArray[cftime.datetime],
        variable_metadata: Mapping[str, VariableMetadata],
        coords: Mapping[str, np.ndarray],
        dataset_metadata: DatasetMetadata,
    ) -> StepDiagnosticsWriter | None:
        if not self.save_step_diagnostics:
            return None

        def writer_factory(name: str) -> RawDataWriter | TimeCoarsen:
            path = os.path.join(experiment_dir, STEP_DIAGNOSTICS_DIR)
            os.makedirs(path, exist_ok=True)
            writer: RawDataWriter | TimeCoarsen = RawDataWriter(
                path=path,
                label=name,
                initial_condition_times=initial_condition_times,
                save_names=self.names,
                variable_metadata=variable_metadata,
                coords=coords,
                dataset_metadata=dataset_metadata,
            )
            if self.time_coarsen is not None:
                writer = self.time_coarsen.build(writer)
            return writer

        return StepDiagnosticsWriter(writer_factory)

    def build(
        self,
        experiment_dir: str,
        initial_condition_times: npt.NDArray[cftime.datetime],
        n_timesteps: int,
        timestep: datetime.timedelta,
        variable_metadata: Mapping[str, VariableMetadata],
        coords: Mapping[str, np.ndarray],
        dataset_metadata: DatasetMetadata,
    ) -> "DataWriter":
        dist = Distributed.get_instance()
        self._raise_if_unsupported_under_multi_gpu(dist.total_data_parallel_ranks)
        writers: list[Subwriter] = []
        # TODO: handle writing HEALPix data
        # https://github.com/ai2cm/full-model/issues/1089
        # Spatial co-ranks of the root (the only data-parallel rank when
        # sub-writers are enabled) send it their tiles and write nothing.
        if "face" not in coords and dist.is_root():
            if self.save_prediction_files:
                raw_writer: Subwriter = RawDataWriter(
                    path=experiment_dir,
                    label="autoregressive_predictions",
                    initial_condition_times=initial_condition_times,
                    save_names=self.names,
                    variable_metadata=variable_metadata,
                    coords=coords,
                    dataset_metadata=dataset_metadata,
                )
                if self.time_coarsen is not None:
                    raw_writer = self.time_coarsen.build(raw_writer)
                writers.append(raw_writer)
            if self.save_monthly_files:
                writers.append(
                    MonthlyDataWriter(
                        path=experiment_dir,
                        label="monthly_mean_predictions",
                        initial_condition_times=initial_condition_times,
                        timestep=timestep,
                        save_names=self.names,
                        variable_metadata=variable_metadata,
                        coords=coords,
                        dataset_metadata=dataset_metadata,
                    )
                )
            for writer_config in self.files or []:
                writers.append(
                    writer_config.build(
                        experiment_dir=experiment_dir,
                        initial_condition_times=initial_condition_times,
                        n_timesteps=n_timesteps,
                        timestep=timestep,
                        variable_metadata=variable_metadata,
                        coords=coords,
                        dataset_metadata=dataset_metadata,
                    )
                )
        return DataWriter(
            writers=writers,
            path=experiment_dir,
            variable_metadata=variable_metadata,
            coords=coords,
            dataset_metadata=dataset_metadata,
            gather_names=self._names_to_gather(),
            has_subwriters=self._has_file_subwriters and "face" not in coords,
            step_diagnostics_writer=self._build_step_diagnostics_writer(
                experiment_dir=experiment_dir,
                initial_condition_times=initial_condition_times,
                variable_metadata=variable_metadata,
                coords=coords,
                dataset_metadata=dataset_metadata,
            ),
        )


class PairedDataWriter(WriterABC[PrognosticState, PairedData]):
    def __init__(
        self,
        writers: list[PairedSubwriter],
        path: str,
        variable_metadata: Mapping[str, VariableMetadata],
        coords: Mapping[str, np.ndarray],
        dataset_metadata: DatasetMetadata,
        step_diagnostics_writer: StepDiagnosticsWriter | None = None,
        gather_names: frozenset[str] | None = None,
        has_subwriters: bool | None = None,
    ):
        """
        Args:
            writers: The sub-writers to dispatch each batch to, already built by
                `DataWriterConfig.build_paired`. Empty on the spatial co-ranks
                of the root, which only send their tiles.
            path: Path to write single-snapshot netCDF file(s) via `write`.
            variable_metadata: Metadata for each variable to be written to the file.
            coords: Coordinate data to be written to the file.
            dataset_metadata: Metadata for the dataset.
            step_diagnostics_writer: Writer for per-step diagnostics of the
                prediction, or None to not write them. Held separately from
                the paired sub-writers: it consumes only the prediction's
                diagnostics, not the target/prediction pair.
            gather_names: Under spatial parallelism, the variables to
                assemble on the root for the sub-writers, or None for all.
            has_subwriters: Whether the root has sub-writers, i.e. whether
                every rank takes part in assembling each batch. Defaults to
                whether ``writers`` is non-empty.
        """
        self._writers = writers
        self._gather_names = gather_names
        self._has_subwriters = (
            bool(writers) if has_subwriters is None else has_subwriters
        )
        self._step_diagnostics_writer = step_diagnostics_writer
        self.path = path
        self.coords = coords
        self.variable_metadata = variable_metadata
        self.dataset_metadata = dataset_metadata

    def write(self, data: PrognosticState, filename: str):
        """Eagerly write data to a single netCDF file.

        Args:
            data: the data to be written.
            filename: the filename to use for the netCDF file.
        """
        _gather_and_write(
            batch=data.as_batch_data(),
            path=self.path,
            filename=filename,
            variable_metadata=self.variable_metadata,
            coords=self.coords,
            dataset_metadata=self.dataset_metadata,
        )

    def append_batch(
        self,
        batch: PairedData,
    ):
        """
        Append a batch of data to the file.

        Args:
            batch: Prediction and target data.
        """
        if self._has_subwriters:
            target = _gather_spatial_to_root(batch.reference, self._gather_names)
            prediction = _gather_spatial_to_root(batch.prediction, self._gather_names)
            if target is not None and prediction is not None:
                for writer in self._writers:
                    writer.append_batch(
                        target=target,
                        prediction=prediction,
                        batch_time=batch.time,
                    )
        _append_step_diagnostics(self._step_diagnostics_writer, batch)

    def flush(self):
        """
        Flush the data to disk.
        """
        for writer in self._writers:
            writer.flush()
        if self._step_diagnostics_writer is not None:
            self._step_diagnostics_writer.flush()

    def finalize(self):
        for writer in self._writers:
            writer.finalize()
        if self._step_diagnostics_writer is not None:
            self._step_diagnostics_writer.finalize()


def _write(
    data: BatchData | GatheredBatchData,
    path: str,
    filename: str,
    variable_metadata: Mapping[str, VariableMetadata],
    coords: Mapping[str, np.ndarray],
    dataset_metadata: DatasetMetadata,
):
    """Write provided data to a single netCDF at specified path/filename.

    ``BatchData.to_xarray_dataset`` produces the complete dataset: prognostic
    variables under plain names + time (with the single-timestep time dimension
    squeezed), plus the round-trippable extras (stepper_state, labels, data_mask)
    embedded under reserved ``_fme_state__`` variables when present. This adds the
    writer's presentation on top - per-variable metadata attrs, coordinates, and
    dataset metadata. When ``data`` carries no extras the file is byte-identical
    to the data+time-only file written before this feature.

    Args:
        data: Batch data to written.
        path: Directory to write the netCDF file in.
        filename: filename to use for netCDF.
        variable_metadata: Metadata for each variable to be written to the file.
        coords: Coordinate data to be written to the file.
        dataset_metadata: Metadata for the dataset.
    """
    ds = data.to_xarray_dataset()
    for name in ds.data_vars:
        # Metadata applies only to physical prognostic variables, never to
        # ``time`` or the reserved embedded-state variables.
        if str(name) == "time" or str(name).startswith(_RESERVED_PREFIX):
            continue
        if str(name) in variable_metadata:
            ds[name].attrs.update(variable_metadata[str(name)].as_attrs())
    ds = ds.assign_coords(coords)
    ds.attrs.update(dataset_metadata.as_flat_str_dict())
    to_netcdf_via_inter_filesystem_copy(ds, os.path.join(path, filename))


def _gather_spatial_to_root(
    data: TensorMapping, names: frozenset[str] | None = None
) -> dict[str, torch.Tensor] | None:
    """Assemble the global fields of tile-local ``data`` on the spatial root,
    on CPU and one variable at a time; None on the other spatial ranks.

    Collective over the spatial group (every rank must call it with the same
    variables); returns the data as it is without spatial parallelism.

    Args:
        data: Tensors whose last two dimensions are this rank's tile.
        names: The variables to assemble, or None for all of them.
    """
    dist = Distributed.get_instance()
    if names is not None:
        data = {name: value for name, value in data.items() if name in names}
    if not dist.has_spatial_parallelism:
        return dict(data)
    gathered: dict[str, torch.Tensor] = {}
    for name in sorted(data):
        tensor = dist.gather_spatial_tile_to_root(data[name])
        if tensor is not None:
            gathered[name] = tensor.cpu()
    return gathered if dist.is_spatial_root() else None


def _append_step_diagnostics(
    writer: StepDiagnosticsWriter | None, batch: PairedData
) -> None:
    """Append the batch's step diagnostics, assembled on the spatial root,
    to ``writer``. Collective over the spatial group when there is a writer
    and diagnostics; the writer is built on every rank but only appended to
    (and so only creates files) on the root.
    """
    if writer is None or batch.step_diagnostics is None:
        return
    delta = _gather_spatial_to_root(batch.step_diagnostics.delta)
    if delta is not None:
        writer.append_batch(StepDiagnostics(delta=delta).to_datasets(batch.time))


def _gather_and_write(
    batch: BatchData,
    path: str,
    filename: str,
    variable_metadata: Mapping[str, VariableMetadata],
    coords: Mapping[str, np.ndarray],
    dataset_metadata: DatasetMetadata,
    dist: Distributed | None = None,
) -> None:
    """Gather data-parallel BatchData shards to root and write once.

    Under spatial parallelism each rank holds a tile of every field: the
    tiles are first assembled on the root of each spatial group. Each
    data-parallel rank holds a contiguous block of samples.  The gather
    concatenates them along the sample dimension in rank order so the written
    file has the same layout as a serial run.  Non-root ranks skip the write.
    A barrier at the end ensures all ranks see the file before continuing
    (required for segmented inference, where the next segment reads the
    restart).  With a single data-parallel rank the gather is a no-op.
    """
    if dist is None:
        dist = Distributed.get_instance()

    spatial_batch = batch.gather_spatial_to_root()
    if spatial_batch is not None:
        # every rank of a data-parallel group holds the same tile, so a
        # group's ranks are all spatial roots or all skip together
        gathered_batch = spatial_batch.data_parallel_gather(dist)
    else:
        gathered_batch = None

    try:
        if dist.is_root():
            if gathered_batch is None:
                raise RuntimeError("batch.gather returned None on root")
            _write(
                data=gathered_batch,
                path=path,
                filename=filename,
                variable_metadata=variable_metadata,
                coords=coords,
                dataset_metadata=dataset_metadata,
            )
    finally:
        dist.barrier()


class DataWriter(WriterABC[PrognosticState, PairedData]):
    def __init__(
        self,
        writers: list[Subwriter],
        path: str,
        variable_metadata: Mapping[str, VariableMetadata],
        coords: Mapping[str, np.ndarray],
        dataset_metadata: DatasetMetadata,
        step_diagnostics_writer: StepDiagnosticsWriter | None = None,
        gather_names: frozenset[str] | None = None,
        has_subwriters: bool | None = None,
    ):
        """
        Args:
            writers: The sub-writers to dispatch each batch to, already built by
                `DataWriterConfig.build`. Empty for HEALPix (`face`)
                coordinates, which are not yet supported, and on the spatial
                co-ranks of the root, which only send their tiles.
            path: Directory within which to write single-snapshot netCDF file(s)
                via `write`.
            variable_metadata: Metadata for each variable to be written to the file.
            coords: Coordinate data to be written to the file.
            dataset_metadata: Metadata for the dataset.
            step_diagnostics_writer: Writer for per-step diagnostics of the
                prediction, or None to not write them. Held separately from
                the sub-writers: it consumes only the prediction's
                diagnostics.
            gather_names: Under spatial parallelism, the variables to
                assemble on the root for the sub-writers, or None for all.
            has_subwriters: Whether the root has sub-writers, i.e. whether
                every rank takes part in assembling each batch. Defaults to
                whether ``writers`` is non-empty.
        """
        self._writers = writers
        self._gather_names = gather_names
        self._has_subwriters = (
            bool(writers) if has_subwriters is None else has_subwriters
        )
        self._step_diagnostics_writer = step_diagnostics_writer
        self.path = path
        self.variable_metadata = variable_metadata
        self.dataset_metadata = dataset_metadata
        self.coords = coords

    def append_batch(self, batch: PairedData):
        """
        Append prediction data to the file. The prognostic data
        and forcing data are merged before writing.

        Args:
            batch: Paired data to be written.
        """
        merged = {**batch.prediction, **batch.forcing}
        unpaired_batch = BatchData.new_on_device(
            data=merged,
            time=batch.time,
            labels=batch.labels,
        )
        self._append_batch(unpaired_batch)
        _append_step_diagnostics(self._step_diagnostics_writer, batch)

    def _append_batch(self, batch: BatchData):
        if not self._has_subwriters:
            return
        data = _gather_spatial_to_root(batch.data, self._gather_names)
        if data is None:
            return
        for writer in self._writers:
            writer.append_batch(
                data=data,
                batch_time=batch.time,
            )

    def flush(self):
        """
        Flush the data to disk.
        """
        for writer in self._writers:
            writer.flush()
        if self._step_diagnostics_writer is not None:
            self._step_diagnostics_writer.flush()

    def finalize(self):
        for writer in self._writers:
            writer.finalize()
        if self._step_diagnostics_writer is not None:
            self._step_diagnostics_writer.finalize()

    def write(self, data: PrognosticState, filename: str):
        """Eagerly write data to a single netCDF file.

        Args:
            data: the data to be written.
            filename: the filename to use for the netCDF file.
        """
        _gather_and_write(
            batch=data.as_batch_data(),
            path=self.path,
            filename=filename,
            variable_metadata=self.variable_metadata,
            coords=self.coords,
            dataset_metadata=self.dataset_metadata,
        )
