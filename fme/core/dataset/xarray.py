import dataclasses
import datetime
import glob as globlib
import json
import logging
import os
import re
import warnings
from collections import namedtuple
from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Literal
from urllib.parse import urlparse

import fsspec
import numpy as np
import torch
import xarray as xr
from xarray.coding.times import CFDatetimeCoder

from fme.core.coordinates import (
    DepthCoordinate,
    HorizontalCoordinates,
    HybridSigmaPressureCoordinate,
    NullVerticalCoordinate,
    VerticalCoordinate,
)
from fme.core.dataset.config import DatasetConfigABC
from fme.core.dataset.properties import DatasetProperties
from fme.core.dataset.schedule import IntSchedule
from fme.core.dataset.time import RepeatedInterval, TimeSlice
from fme.core.dataset.utils import FillNaNsConfig
from fme.core.spatial_mask_provider import SpatialMaskProvider
from fme.core.stacker import Stacker
from fme.core.typing_ import Slice, TensorDict

from .data_typing import VariableMetadata
from .dataset import DatasetABC, DatasetItem
from .utils import (
    as_alignable_tensor,
    get_horizontal_coordinates,
    get_nonspacetime_dimensions,
    load_series_data,
    load_series_data_zarr_async,
)

SLICE_NONE = slice(None)
logger = logging.getLogger(__name__)

VariableNames = namedtuple(
    "VariableNames",
    (
        "time_dependent_names",
        "time_invariant_names",
        "static_derived_names",
    ),
)


def _get_vertical_coordinate(
    ds: xr.Dataset,
    dtype: torch.dtype | None,
    reference_pressure_name: str | None = None,
) -> VerticalCoordinate:
    """
    Get vertical coordinate from a dataset.

    If the dataset contains variables named `ak_N` and `bk_N` where
    `N` is the level number, then a hybrid sigma-pressure coordinate
    will be returned. If the dataset contains variables named
    `idepth_N` then a depth coordinate will be returned. If neither thing
    is true, a hybrid sigma-pressure coordinate of lenght 0 is returned.

    Args:
        ds: Dataset to get vertical coordinates from.
        dtype: Data type of the returned tensors. If None, the dtype is not
            changed from the original in ds.
        reference_pressure_name: If provided, the name of a scalar variable in
            ``ds`` holding a reference pressure in Pa. The `ak_N` coefficients
            are then taken to be dimensionless and are multiplied by this
            reference pressure, i.e. interface pressures are computed as
            ``p_N = ak_N * P0 + bk_N * PS``. If None, the `ak_N` coefficients
            are taken to already be in Pa.
    """
    ak_mapping = {
        int(v[3:]): torch.as_tensor(ds[v].values)
        for v in ds.variables
        if v.startswith("ak_")
    }
    bk_mapping = {
        int(v[3:]): torch.as_tensor(ds[v].values)
        for v in ds.variables
        if v.startswith("bk_")
    }
    ak_list = [ak_mapping[k] for k in sorted(ak_mapping.keys())]
    bk_list = [bk_mapping[k] for k in sorted(bk_mapping.keys())]

    idepth_mapping = {
        int(v[7:]): torch.as_tensor(ds[v].values)
        for v in ds.variables
        if v.startswith("idepth_")
    }
    idepth_list = [idepth_mapping[k] for k in sorted(idepth_mapping.keys())]

    if len(ak_list) > 0 and len(bk_list) > 0 and len(idepth_list) > 0:
        raise ValueError(
            "Dataset contains both hybrid sigma-pressure and depth coordinates. "
            "Can only provide one, or else the vertical coordinate is ambiguous."
        )

    reference_pressure: float | None = None
    if reference_pressure_name is not None:
        if len(ak_list) == 0 or len(bk_list) == 0:
            raise ValueError(
                f"A reference pressure variable '{reference_pressure_name}' was "
                "configured, but the dataset does not have a hybrid sigma-pressure "
                "vertical coordinate. It is only used to scale ak coefficients."
            )
        if reference_pressure_name not in ds.variables:
            raise ValueError(
                f"Reference pressure variable '{reference_pressure_name}' was not "
                "found in the dataset."
            )
        if ds[reference_pressure_name].size != 1:
            raise ValueError(
                f"Reference pressure variable '{reference_pressure_name}' must be a "
                f"scalar, but it has shape {ds[reference_pressure_name].shape}."
            )
        reference_pressure = float(ds[reference_pressure_name].values.item())

    coordinate: VerticalCoordinate
    deptho = None
    if len(idepth_list) > 0:
        if "mask_0" in ds.data_vars:
            mask_layers = {
                name: torch.as_tensor(ds[name].values, dtype=dtype)
                for name in ds.data_vars
                if re.match(r"mask_(\d+)$", name)
            }
            for name in mask_layers:
                if "time" in ds[name].dims:
                    raise ValueError("The ocean mask must by time-independent.")
            stacker = Stacker({"mask": ["mask_"]})
            mask = stacker("mask", mask_layers)
        else:
            logger.warning(
                "Dataset does not contain a mask. Providing a DepthCoordinate with "
                "mask set to 1 at all layers."
            )
            mask = torch.ones(len(idepth_list) - 1, dtype=dtype)
        if "deptho" in ds.data_vars:
            if "time" in ds["deptho"].dims:
                raise ValueError("'deptho' must be time-independent.")
            deptho = torch.as_tensor(ds["deptho"].values, dtype=dtype)
        else:
            logger.warning(
                "Dataset does not have a variable named 'deptho' (sea floor depth). "
                "The ocean depth integral will not account for partial bottom cells."
            )
        coordinate = DepthCoordinate(
            torch.as_tensor(idepth_list, dtype=dtype), mask, deptho
        )
    elif len(ak_list) > 0 and len(bk_list) > 0:
        ak = torch.as_tensor(ak_list, dtype=dtype)
        if reference_pressure is not None:
            ak = ak * reference_pressure
        coordinate = HybridSigmaPressureCoordinate(
            ak=ak,
            bk=torch.as_tensor(bk_list, dtype=dtype),
        )
    else:
        logger.warning("Dataset does not contain a vertical coordinate.")
        coordinate = NullVerticalCoordinate()

    return coordinate


def _get_raw_times_single_file(path: str, engine: str | None = None) -> np.array:
    with _open_xr_dataset(path, engine=engine) as ds:
        return ds.time.values


@lru_cache(maxsize=32)
def _get_raw_times_cached(
    paths: tuple[str, ...], engine: str
) -> tuple[np.ndarray, ...]:
    """Read each file's time coordinate, serially, memoized on the file list.

    Deliberately serial. Two faster approaches were tried and both break in
    production:

    * ``multiprocessing.Pool`` (the original) is created on a rank that has
      already initialized CUDA and NCCL. Forking such a process deadlocks on
      pool teardown: the rank wedges in ``Pool.__exit__`` -> ``_terminate_pool``
      -> ``join`` while its peers sit in DDP's parameter allgather until the
      30-minute NCCL watchdog fires, reporting a misleading "rank N has
      inconsistent 0 params".
    * ``ThreadPoolExecutor`` deadlocks nothing but corrupts the heap, because
      netCDF4/HDF5 is not thread-safe. It survives small runs and then dies at
      production width with ``corrupted size vs. prev_size`` and SIGSEGV
      partway through dataset construction.

    Memoizing is what makes serial affordable: a config opens the same file
    list once per dataset (train windows, validation, each inference block),
    so the ~20 calls per rank collapse to one per distinct stream. Reading
    1501 files takes ~84 s, so this is the difference between ~4 minutes of
    setup and ~28 minutes.

    The time coordinate of a file cannot change during a run, so the cache
    cannot go stale.
    """
    return tuple(_get_raw_times_single_file(path, engine=engine) for path in paths)


def _get_raw_times(paths: list[str], engine: str) -> list[np.ndarray]:
    return list(_get_raw_times_cached(tuple(paths), engine))


def _repeat_and_increment_time(
    raw_times: list[np.ndarray], n_repeats: int, timestep: datetime.timedelta
) -> list[np.ndarray]:
    """Repeats and increments a collection of arrays of evenly spaced times."""
    n_timesteps = sum(len(times) for times in raw_times)
    timespan = timestep * n_timesteps

    repeated_and_incremented_time = []
    for repeats in range(n_repeats):
        increment = repeats * timespan
        for time in raw_times:
            incremented_time = time + increment
            repeated_and_incremented_time.append(incremented_time)
    return repeated_and_incremented_time


def _get_cumulative_timesteps(time: list[np.ndarray]) -> np.ndarray:
    """Returns a list of cumulative timesteps for each item in a time coordinate."""
    num_timesteps_per_file = [0]
    for time_coord in time:
        num_timesteps_per_file.append(len(time_coord))
    return np.array(num_timesteps_per_file).cumsum()


def _get_file_local_index(index: int, start_indices: np.ndarray) -> tuple[int, int]:
    """
    Return a tuple of the index of the file containing the time point at `index`
    and the index of the time point within that file.
    """
    file_index = np.searchsorted(start_indices, index, side="right") - 1
    time_index = index - start_indices[file_index]
    return int(file_index), time_index


class StaticDerivedData:
    names = ("x", "y", "z")
    metadata = {
        "x": VariableMetadata(units="", long_name="Euclidean x-coordinate"),
        "y": VariableMetadata(units="", long_name="Euclidean y-coordinate"),
        "z": VariableMetadata(units="", long_name="Euclidean z-coordinate"),
    }

    def __init__(self, coordinates: HorizontalCoordinates):
        self._coords = coordinates
        self._x: torch.Tensor | None = None
        self._y: torch.Tensor | None = None
        self._z: torch.Tensor | None = None

    def _get_xyz(self) -> TensorDict:
        if self._x is None or self._y is None or self._z is None:
            coords = self._coords
            x, y, z = coords.xyz

            self._x = torch.as_tensor(x)
            self._y = torch.as_tensor(y)
            self._z = torch.as_tensor(z)

        return {"x": self._x, "y": self._y, "z": self._z}

    def __getitem__(self, name: str) -> torch.Tensor:
        return self._get_xyz()[name]


def _get_protocol(path):
    return urlparse(str(path)).scheme


def _get_fs(path):
    protocol = _get_protocol(path)
    if not protocol:
        protocol = "file"
    proto_kw = _get_fs_protocol_kwargs(path)
    fs = fsspec.filesystem(protocol, **proto_kw)

    return fs


def _preserve_protocol(original_path, glob_paths):
    protocol = _get_protocol(str(original_path))
    if protocol:
        glob_paths = [f"{protocol}://{path}" for path in glob_paths]
    return glob_paths


def _get_fs_protocol_kwargs(path):
    protocol = _get_protocol(path)
    kwargs = {}
    if protocol == "gs":
        # https://gcsfs.readthedocs.io/en/latest/api.html#gcsfs.core.GCSFileSystem
        key_json = os.environ.get("FSSPEC_GS_KEY_JSON", None)
        key_file = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", None)

        if key_json is not None:
            token = json.loads(key_json)
        elif key_file is not None:
            token = key_file
        else:
            logger.warning(
                "GCS currently expects user credentials authenticated using"
                " `gcloud auth application-default login`. This is not recommended for "
                "production use."
            )
            token = "google_default"
        kwargs["token"] = token
    elif protocol == "s3":
        # https://s3fs.readthedocs.io/en/latest/#s3-compatible-storage
        env_vars = [
            "FSSPEC_S3_KEY",
            "FSSPEC_S3_SECRET",
            "FSSPEC_S3_ENDPOINT_URL",
        ]
        for v in env_vars:
            if v not in os.environ:
                warnings.warn(
                    f"An S3 path was specified but environment variable {v} "
                    "was not found. This may cause authentication issues if not "
                    "set and no other defaults are present. See "
                    "https://s3fs.readthedocs.io/en/latest/#s3-compatible-storage"
                    " for details."
                )

    return kwargs


def _expand_combine_sources(
    names: Sequence[str], combine: Mapping[str, Mapping[str, float]]
) -> list[str]:
    """Replace each combine target with the source variables that build it.

    Targets do not exist on disk, so the dataset must load their sources
    instead. Sources already requested in their own right keep their place.
    """
    expanded: list[str] = []
    for name in names:
        if name in combine:
            expanded.extend(s for s in combine[name] if s not in expanded)
        elif name not in expanded:
            expanded.append(name)
    return expanded


def _combine_expression(sources: Mapping[str, float]) -> str:
    """Render a combine target's definition, e.g. ``rainFlux + snowFlux``."""
    terms: list[str] = []
    for i, (name, coefficient) in enumerate(sources.items()):
        magnitude = abs(coefficient)
        term = name if magnitude == 1 else f"{magnitude:g}*{name}"
        if i == 0:
            terms.append(f"-{term}" if coefficient < 0 else term)
        else:
            terms.append(f"{'-' if coefficient < 0 else '+'} {term}")
    return " ".join(terms)


def _open_xr_dataset(path: str, *args, mask_and_scale: bool = False, **kwargs):
    # need the path to get protocol specific arguments for the backend
    protocol_kw = _get_fs_protocol_kwargs(path)
    if protocol_kw:
        kwargs.update({"storage_options": protocol_kw})

    return xr.open_dataset(
        path,
        *args,
        decode_times=CFDatetimeCoder(use_cftime=True),
        decode_timedelta=False,
        mask_and_scale=mask_and_scale,
        cache=False,
        chunks=None,
        **kwargs,
    )


_open_xr_dataset_lru = lru_cache()(_open_xr_dataset)


def _open_file_fh_cached(path, **kwargs):
    protocol = _get_protocol(path)
    if protocol:
        # add an LRU cache for remote zarrs
        return _open_xr_dataset_lru(
            path,
            **kwargs,
        )
    # netcdf4 and h5engine have a filehandle LRU cache in xarray
    # https://github.com/pydata/xarray/blob/cd3ab8d5580eeb3639d38e1e884d2d9838ef6aa1/xarray/backends/file_manager.py#L54 # noqa: E501
    return _open_xr_dataset(
        path,
        **kwargs,
    )


def get_raw_paths(path, file_pattern):
    if not _get_protocol(path):
        path = os.path.expanduser(str(path))
        # fsspec's glob stats every entry in the directory, which on a model run
        # directory of ~10k files is ~250x slower than the stdlib (7s vs 0.03s)
        # for an identical result. That cost is paid once per dataset per rank,
        # and a rank that falls far enough behind its peers trips the NCCL
        # watchdog during DDP setup.
        return sorted(globlib.glob(os.path.join(path, file_pattern), recursive=True))
    fs = _get_fs(path)
    glob_paths = sorted(fs.glob(os.path.join(path, file_pattern)))
    raw_paths = _preserve_protocol(path, glob_paths)
    return raw_paths


def _get_spatial_mask_provider(
    ds: xr.Dataset, dtype: torch.dtype | None
) -> SpatialMaskProvider:
    """Get mask provider from a dataset.

    If the dataset contains time-invariant variables that start with the string
    "mask_" then these variables will be used to instantiate a SpatialMaskProvider
    object. Otherwise, an empty SpatialMaskProvider is returned.

    Args:
        ds: Dataset to get vertical coordinates from.
        dtype: Data type of the returned tensors. If None, the dtype is not
            changed from the original in ds.

    """
    masks: dict[str, torch.Tensor] = {
        name: torch.as_tensor(ds[name].values, dtype=dtype)
        for name in ds.data_vars
        if name.startswith("mask_")
    }
    for name in masks:
        if "time" in ds[name].dims:
            raise ValueError("Masks must be time-independent.")
        # A mask is 0/1 everywhere, so a NaN is never meaningful. It arises when
        # the mask carries a CF _FillValue and XarrayDataConfig.mask_and_scale
        # decodes it, which inverts the masking at those points.
        if bool(torch.isnan(masks[name]).any()):
            raise ValueError(
                f"Mask variable '{name}' contains NaN values. If this dataset "
                "is loaded with mask_and_scale=True, the mask most likely has a "
                "_FillValue attribute that is being decoded to NaN; drop that "
                "attribute from the mask variable."
            )
    spatial_mask_provider = SpatialMaskProvider(masks)
    logging.info(f"Initialized {spatial_mask_provider}.")
    return spatial_mask_provider


@dataclasses.dataclass
class OverwriteConfig:
    """Configuration to overwrite field values in XarrayDataset.

    Applied as ``value * multiply_scalar + add_scalar``, so the two can be
    combined to express an affine unit conversion.

    Parameters:
        constant: Fill field with constant value.
        multiply_scalar: Multiply field by scalar value.
        add_scalar: Add scalar value to field, e.g. 273.15 to convert a
            temperature from degrees Celsius to Kelvin.
    """

    constant: Mapping[str, float] = dataclasses.field(default_factory=dict)
    multiply_scalar: Mapping[str, float] = dataclasses.field(default_factory=dict)
    add_scalar: Mapping[str, float] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        key_overlap = set(self.constant.keys()) & (
            set(self.multiply_scalar.keys()) | set(self.add_scalar.keys())
        )
        if key_overlap:
            raise ValueError(
                "OverwriteConfig cannot have the same variable in both constant "
                f"and multiply_scalar or add_scalar: {key_overlap}"
            )

    def apply(self, tensors: TensorDict) -> TensorDict:
        # Variables not present are skipped rather than raising: a single
        # XarrayDataConfig may be loaded several times for different subsets of
        # its names (the coupled loader splits the atmosphere into separate
        # forcing and target datasets this way), and each such load only holds
        # the names it asked for.
        for var, fill_value in self.constant.items():
            if var not in tensors:
                continue
            data = tensors[var]
            tensors[var] = torch.ones_like(data) * torch.tensor(
                fill_value, dtype=data.dtype, device=data.device
            )
        for var, multiplier in self.multiply_scalar.items():
            if var not in tensors:
                continue
            data = tensors[var]
            tensors[var] = data * torch.tensor(
                multiplier, dtype=data.dtype, device=data.device
            )
        for var, addend in self.add_scalar.items():
            if var not in tensors:
                continue
            data = tensors[var]
            tensors[var] = data + torch.tensor(
                addend, dtype=data.dtype, device=data.device
            )
        return tensors

    @property
    def variables(self):
        return (
            set(self.constant.keys())
            | set(self.multiply_scalar.keys())
            | set(self.add_scalar.keys())
        )


@dataclasses.dataclass
class XarrayDataConfig(DatasetConfigABC):
    """
    Parameters:
        data_path: Path to the data.
        file_pattern: Glob pattern to match files in the data_path.
        n_repeats: Number of times to repeat the dataset (in time). It is up
            to the user to ensure that the input dataset to repeat results in
            data that is reasonably continuous across repetitions.
        engine: Backend used in xarray.open_dataset call.
        spatial_dimensions: Specifies the spatial dimensions for the grid, default
            is lat/lon. If 'latlon', it is assumed that the last two dimensions are
            latitude and longitude, respectively. If 'healpix', it is assumed that the
            last three dimensions are face, height, and width, respectively.
        subset: Slice defining a subset of the XarrayDataset to load. This can
            either be a `Slice` of integer indices or a `TimeSlice` of timestamps.
            This feature is applied directly to the dataset samples. For example,
            if the file(s) have the time coordinate (t0, t1, t2, t3) and
            requirements.n_timesteps=2, then subset=Slice(stop=2) will
            provide two samples: (t0, t1), (t1, t2).
        infer_timestep: Whether to infer the timestep from the provided data.
            This should be set to True (the default) for ACE training. It may
            be useful to toggle this to False for applications like downscaling,
            which do not depend on the timestep of the data and therefore lack
            the additional requirement that the data be ordered and evenly
            spaced in time. It must be set to True if n_repeats > 1 in order
            to be able to infer the full time coordinate.
        dtype: Data type to cast the data to. If None, no casting is done. It is
            required that 'torch.{dtype}' is a valid dtype.
        rename: Mapping from variable names as they appear on disk to the names
            used by FME, e.g. ``{"PRECT": "surface_precipitation_rate"}``. This
            follows the same convention as ``xarray.Dataset.rename``. It is
            applied when the data is opened, so every other name in the
            configuration (including ``overwrite`` and
            ``reference_pressure_name``) refers to the renamed variables.
        reference_pressure_name: Name of a scalar variable holding a reference
            pressure in Pa, e.g. ``"P0"``. If provided, the hybrid
            sigma-pressure ``ak_N`` coefficients in the dataset are taken to be
            dimensionless and interface pressures are computed as
            ``p_N = ak_N * P0 + bk_N * PS``. If None (the default), the ``ak_N``
            coefficients are taken to already be in Pa.
        overwrite: Optional OverwriteConfig to overwrite loaded field values.
        fill_nans: Optional FillNaNsConfig to fill NaNs with a constant value.
        combine: Mapping from a new variable name to a mapping of source
            variable name to coefficient, defining the new variable as a
            linear combination of loaded variables, e.g.
            ``{"surface_precipitation_rate": {"rainFlux": 1.0, "snowFlux": 1.0}}``.
            Use this when the data splits a field the model wants whole (or
            vice versa, via negative coefficients). Applied after ``rename``
            and ``overwrite``, so unit and sign fixes compose with it. Source
            variables are loaded automatically; those not requested in their
            own right are not returned.
        mask_and_scale: Whether to decode CF ``_FillValue``/``missing_value``
            and ``scale_factor``/``add_offset`` attributes when opening the
            data. Defaults to False. Raw model output (e.g. remapped MPAS
            netCDFs) flags land with a sentinel such as 1e20; leaving this
            False loads the sentinel verbatim rather than as NaN, which
            silently corrupts losses because spatial output masking writes
            NaN over those same points while the target keeps the sentinel.
            Set True for such data, optionally with ``fill_nans``.
        isel: Optional xarray isel arguments to be passed to the dataset. Will
            raise ValueError if time is included here, since the subset argument
            is used specifically for selecting times. Horizontal dimensions are
            also not currently supported.
        labels: Optional list of labels to be returned with the data.

    Examples:
        If data is stored in a directory with multiple netCDF files which can be
        concatenated along the time dimension, use:

        >>> fme.ace.XarrayDataConfig(data_path="/some/directory", file_pattern="*.nc") # doctest: +IGNORE_OUTPUT

        If data is stored in a single zarr store at ``/some/directory/dataset.zarr``,
        use:

        >>> fme.ace.XarrayDataConfig(
        ...     data_path="/some/directory",
        ...     file_pattern="dataset.zarr",
        ...     engine="zarr"
        ... ) # doctest: +IGNORE_OUTPUT
    """  # noqa: E501

    data_path: str
    file_pattern: str = "*.nc"
    n_repeats: int = 1
    engine: Literal["netcdf4", "h5netcdf", "zarr"] = "netcdf4"
    spatial_dimensions: Literal["healpix", "latlon"] = "latlon"
    subset: Slice | TimeSlice | RepeatedInterval = dataclasses.field(
        default_factory=Slice
    )
    infer_timestep: bool = True
    dtype: str | None = "float32"
    rename: Mapping[str, str] = dataclasses.field(default_factory=dict)
    reference_pressure_name: str | None = None
    overwrite: OverwriteConfig = dataclasses.field(default_factory=OverwriteConfig)
    fill_nans: FillNaNsConfig | None = None
    mask_and_scale: bool = False
    combine: Mapping[str, Mapping[str, float]] = dataclasses.field(default_factory=dict)
    isel: Mapping[str, Slice | int] = dataclasses.field(default_factory=dict)
    labels: list[str] | None = None

    def _default_file_pattern_check(self):
        if self.engine == "zarr" and self.file_pattern == "*.nc":
            raise ValueError(
                "The file pattern is set to the default NetCDF file pattern *.nc "
                "but the engine is specified as 'zarr'. Please set "
                "`XarrayDataConfig.file_pattern` to match the zarr filename."
            )

    @property
    def available_labels(self) -> set[str] | None:
        """
        Return the labels that are available in the dataset.
        """
        if self.labels is None:
            return None
        return set(self.labels)

    @property
    def torch_dtype(self) -> torch.dtype | None:
        if self.dtype is None:
            return None
        else:
            try:
                torch_dtype = getattr(torch, self.dtype)
            except AttributeError:
                raise ValueError(f"Invalid dtype '{self.dtype}'")
            if not isinstance(torch_dtype, torch.dtype):
                raise ValueError(f"Invalid dtype '{self.dtype}'")
        return torch_dtype

    def __post_init__(self):
        if self.n_repeats > 1 and not self.infer_timestep:
            raise ValueError(
                "infer_timestep must be True if n_repeats is greater than 1"
            )
        if self.spatial_dimensions not in ["latlon", "healpix"]:
            raise ValueError(
                f"unexpected spatial_dimensions {self.spatial_dimensions},"
                " should be one of 'latlon' or 'healpix'"
            )
        if "time" in set(self.rename) | set(self.rename.values()):
            raise ValueError(
                "XarrayDataConfig.rename cannot rename the time coordinate, "
                f"but got {dict(self.rename)}."
            )
        for target, sources in self.combine.items():
            if not sources:
                raise ValueError(
                    f"XarrayDataConfig.combine entry '{target}' has no source "
                    "variables; provide at least one source and coefficient."
                )
            if target in sources:
                raise ValueError(
                    f"XarrayDataConfig.combine target '{target}' is also one of "
                    "its own sources, which would make the result depend on "
                    "evaluation order."
                )
            chained = set(sources) & set(self.combine)
            if chained:
                raise ValueError(
                    f"XarrayDataConfig.combine target '{target}' draws on "
                    f"{sorted(chained)}, which are themselves combine targets. "
                    "Combining is applied in a single pass over variables read "
                    "from disk, so chained definitions are not supported; write "
                    "the target directly in terms of on-disk variables."
                )
        overwritten_targets = self.overwrite.variables & set(self.combine)
        if overwritten_targets:
            raise ValueError(
                "XarrayDataConfig.overwrite names combine targets "
                f"{sorted(overwritten_targets)}. overwrite is applied to the "
                "variables read from disk, before combine builds its targets, "
                "so these entries would silently do nothing. Apply the scaling "
                "to the combine sources instead."
            )
        if self.engine == "zarr" and self.mask_and_scale:
            raise ValueError(
                "XarrayDataConfig.mask_and_scale is not supported with "
                "engine='zarr'. The zarr read path loads time-dependent "
                "variables directly from the store without CF decoding, so the "
                "flag would decode only the time-invariant variables and leave "
                "raw _FillValue sentinels in everything else."
            )
        renamed_to = list(self.rename.values())
        duplicates = {name for name in renamed_to if renamed_to.count(name) > 1}
        if duplicates:
            raise ValueError(
                "XarrayDataConfig.rename maps multiple variables to the same "
                f"name(s): {sorted(duplicates)}."
            )
        self.torch_dtype  # check it can be retrieved
        self._default_file_pattern_check()

    @property
    def zarr_engine_used(self) -> bool:
        return self.engine == "zarr"

    def update_subset(self, subset: Slice | TimeSlice | RepeatedInterval):
        self.subset = subset

    def build(
        self,
        names: Sequence[str],
        n_timesteps: IntSchedule,
        allow_missing_variables: bool = False,
    ) -> tuple["XarraySubset", DatasetProperties]:
        return get_xarray_dataset(
            self,
            list(names),
            n_timesteps,
            allow_missing_variables=allow_missing_variables,
        )


class XarrayDataset(DatasetABC):
    """Load data from a directory of files matching a pattern using xarray. The
    number of contiguous timesteps to load for each sample is specified by the
    n_timesteps argument.

    For example, if the file(s) have the time coordinate
    (t0, t1, t2, t3, t4) and n_timesteps=3, then this dataset will
    provide three samples: (t0, t1, t2), (t1, t2, t3), and (t2, t3, t4).
    """

    def __init__(
        self,
        config: XarrayDataConfig,
        names: Sequence[str],
        n_timesteps: IntSchedule,
        allow_missing_variables: bool = False,
    ):
        self._horizontal_coordinates: HorizontalCoordinates
        self._requested_names = list(names)
        # Only build targets that were actually asked for, and load their
        # sources in their place.
        self._combine = {
            target: dict(sources)
            for target, sources in config.combine.items()
            if target in self._requested_names
        }
        self._names = _expand_combine_sources(self._requested_names, self._combine)
        self._combine_only_sources = frozenset(self._names) - set(self._requested_names)
        self._allow_missing_variables = allow_missing_variables
        self.path = config.data_path
        self.file_pattern = config.file_pattern
        self.engine = config.engine
        self.dtype = config.torch_dtype
        self.spatial_dimensions = config.spatial_dimensions
        self.fill_nans = config.fill_nans
        self.mask_and_scale = config.mask_and_scale
        self.subset_config = config.subset
        self._rename = dict(config.rename)
        self._rename_inverse = {v: k for k, v in self._rename.items()}
        self._reference_pressure_name = config.reference_pressure_name
        self._raw_paths = get_raw_paths(self.path, self.file_pattern)
        if len(self._raw_paths) == 0:
            raise ValueError(
                f"No files found matching '{self.path}/{self.file_pattern}'."
            )
        self.full_paths = self._raw_paths * config.n_repeats
        self._n_timesteps_schedule = n_timesteps
        self._get_files_stats(
            config.n_repeats,
            config.infer_timestep,
            max_sample_n_times=n_timesteps.max_value,
        )
        first_dataset = self._apply_rename(
            xr.open_dataset(
                self.full_paths[0],
                decode_times=False,
                decode_timedelta=False,
                engine=self.engine,
                chunks=None,
                mask_and_scale=self.mask_and_scale,
            )
        )
        self._spatial_mask_provider = _get_spatial_mask_provider(
            first_dataset, self.dtype
        )
        (
            self._horizontal_coordinates,
            self._static_derived_data,
            _loaded_horizontal_dims,
        ) = self.configure_horizontal_coordinates(first_dataset)
        (
            self._time_dependent_names,
            self._time_invariant_names,
            self._static_derived_names,
        ) = self._group_variable_names_by_time_type()
        loaded_names = self._names
        self._names = (
            list(self._time_dependent_names)
            + list(self._time_invariant_names)
            + list(self._static_derived_names)
        )
        self._missing_names = frozenset(set(loaded_names) - set(self._names))
        # A combine target cannot be built from a source that is not on disk.
        # Without this, allow_missing_variables=True lets construction succeed
        # and then _apply_combine raises an opaque KeyError inside a dataloader
        # worker on the first batch.
        unbuildable = {
            target: sorted(set(sources) & self._missing_names)
            for target, sources in self._combine.items()
            if set(sources) & self._missing_names
        }
        if unbuildable:
            raise ValueError(
                "Cannot build combine target(s) because their source variables "
                f"are not present in the dataset: {unbuildable}. Sources of a "
                "combine target are required even when "
                "allow_missing_variables is True."
            )
        # A target that also exists on disk would be silently shadowed by the
        # computed value, and only for the datasets that request it, so two
        # loads of the same config could disagree about what the name means.
        shadowed = sorted(set(config.combine) & set(first_dataset.variables))
        if shadowed:
            raise ValueError(
                f"XarrayDataConfig.combine target(s) {shadowed} are also "
                f"variables in {self.full_paths[0]}. The computed value would "
                "silently shadow the stored one; rename the target, or drop the "
                "combine entry and read the variable directly."
            )
        # overwrite silently skips names it does not find, which is required
        # because one config may be loaded for several subsets of its names.
        # A name absent from the data entirely can never take effect, though,
        # so treat that as the configuration error it is.
        available = set(first_dataset.variables) | set(StaticDerivedData.names)
        unknown_overwrites = config.overwrite.variables - available
        if unknown_overwrites:
            raise ValueError(
                f"XarrayDataConfig.overwrite names {sorted(unknown_overwrites)}, "
                f"which do not exist in {self.full_paths[0]}. overwrite entries "
                "for variables this dataset does not load are permitted (one "
                "config may be loaded for several subsets of its names), but a "
                "name that is in no file at all is a typo: overwrite is applied "
                "after rename, so use the renamed variable name."
            )
        self._get_variable_metadata(first_dataset)

        self._vertical_coordinate = _get_vertical_coordinate(
            first_dataset, self.dtype, self._reference_pressure_name
        )
        self.overwrite = config.overwrite

        self._nonspacetime_dims = get_nonspacetime_dimensions(
            first_dataset, _loaded_horizontal_dims
        )
        self._shape_excluding_time = [
            first_dataset.sizes[dim]
            for dim in (self._nonspacetime_dims + _loaded_horizontal_dims)
        ]
        self._loaded_dims = ["time"] + self._nonspacetime_dims + _loaded_horizontal_dims
        self.isel = {
            dim: v if isinstance(v, int) else v.slice for dim, v in config.isel.items()
        }
        self._isel_tuple = tuple(
            [self.isel.get(dim, SLICE_NONE) for dim in self._loaded_dims[1:]]
        )
        self._check_isel_dimensions(first_dataset.sizes)
        first_dataset.close()
        self._time_invariant_tensors = self._load_time_invariant_tensors()
        self._apply_sample_n_times(self._n_timesteps_schedule.get_value(0))
        self._labels = set(config.labels) if config.labels is not None else None
        self._infer_timestep = config.infer_timestep
        self._local_epoch: int = -1
        self._global_epoch = torch.tensor(-1)

    def _load_time_invariant_tensors(self) -> dict[str, torch.Tensor]:
        """Load the time-invariant variables into memory.

        These do not vary in time, so they are read once here and broadcast over
        the time dimension of each sample rather than being re-read per sample.
        Values are taken from the first file, consistent with how coordinates,
        vertical coordinate and variable metadata are read in __init__.
        """
        if len(self._time_invariant_names) == 0:
            return {}
        # opened directly rather than via _open_file so that closing this
        # handle cannot close one shared through the file handle cache
        ds = _open_xr_dataset(
            self.full_paths[0],
            engine=self.engine,
            mask_and_scale=self.mask_and_scale,
        )
        # _time_invariant_names are post-rename names, so the rename has to be
        # applied before they are looked up.
        ds = self._apply_rename(ds)
        ds = ds.isel(**self.isel)
        tensors = {}
        for name in self._time_invariant_names:
            variable = ds[name].variable
            if self.fill_nans is not None:
                variable = variable.fillna(self.fill_nans.value)
            tensors[name] = as_alignable_tensor(variable, self.dims)
        ds.close()
        return tensors

    def _ensure_epoch_synchronized(self):
        """Ensure that the local epoch is synchronized with the global epoch.

        This is required for multi-worker data loading, where each worker
        process has its own copy of the dataset object.
        """
        if self._local_epoch != self._global_epoch.item():
            self._local_epoch = self._global_epoch.item()
            sample_n_times = self._n_timesteps_schedule.get_value(self._local_epoch)
            self._apply_sample_n_times(sample_n_times)

    @property
    def _epoch(self) -> int | None:
        self._ensure_epoch_synchronized()
        if self._local_epoch == -1:
            return None
        return self._local_epoch

    def _apply_sample_n_times(self, sample_n_times: int):
        self._sample_n_times = sample_n_times
        logging.info(
            f"Dataset now has {self._n_initial_conditions} samples of "
            f"length {sample_n_times}."
        )

    def _check_isel_dimensions(self, data_dim_sizes):
        # Horizontal dimensions are not currently supported, as the current isel code
        # does not adjust HorizonalCoordinates to match selection.
        if "time" in self.isel:
            raise ValueError("isel cannot be used to select time. Use subset instead.")

        for dim, selection in self.isel.items():
            if dim not in self._nonspacetime_dims:
                raise ValueError(
                    f"isel dimension {dim} must be a non-spacetime dimension "
                    f"of the dataset ({self._nonspacetime_dims})."
                )
            max_isel_index = (
                (selection.start or 0) if isinstance(selection, slice) else selection
            )
            if max_isel_index >= data_dim_sizes[dim]:
                raise ValueError(
                    f"isel index {max_isel_index} is out of bounds for dimension "
                    f"{dim} with size {data_dim_sizes[dim]}."
                )

    @property
    def _shape_excluding_time_after_selection(self):
        final_shape = []
        for orig_size, sel in zip(self._shape_excluding_time, self._isel_tuple):
            # if selecting a single index, dimension is squeezed
            # so it is not included in the final shape
            if isinstance(sel, slice):
                if sel.start is None and sel.stop is None and sel.step is None:
                    final_shape.append(orig_size)
                else:
                    final_shape.append(len(range(*sel.indices(orig_size))))
        return final_shape

    @property
    def dims(self) -> list[str]:
        # Final dimensions of returned data after dims that are selected
        # with a single index are dropped
        final_dims = ["time"]
        for dim, sel in zip(self._loaded_dims[1:], self._isel_tuple):
            if isinstance(sel, slice):
                final_dims.append(dim)
        return final_dims

    @property
    def properties(self) -> DatasetProperties:
        return DatasetProperties(
            self._variable_metadata,
            self._vertical_coordinate,
            self._horizontal_coordinates,
            self._spatial_mask_provider,
            self.timestep,
            self._is_remote,
            self._labels,
        )

    @property
    def _is_remote(self) -> bool:
        protocol = _get_protocol(str(self.path))
        if not protocol or protocol == "file":
            return False
        return True

    def _apply_combine(self, tensors: TensorDict) -> TensorDict:
        """Build each combine target, then drop sources loaded only for it."""
        if not self._combine:
            return tensors
        combined: TensorDict = {}
        for target, sources in self._combine.items():
            total: torch.Tensor | None = None
            for source, coefficient in sources.items():
                data = tensors[source]
                term = data * torch.tensor(
                    coefficient, dtype=data.dtype, device=data.device
                )
                total = term if total is None else total + term
            assert total is not None  # guaranteed by the empty-sources check
            combined[target] = total
        tensors.update(combined)
        for name in self._combine_only_sources:
            tensors.pop(name, None)
        return tensors

    def _get_variable_metadata(self, ds):
        result = {}
        for name in self._names:
            if name in StaticDerivedData.names:
                result[name] = StaticDerivedData.metadata[name]
            else:
                result[name] = VariableMetadata.from_attrs(ds[name].attrs)
        for target, sources in self._combine.items():
            source_metadata = [result[s] for s in sources if s in result]
            # Units only survive if every source agrees on them; a combination
            # of differently-united fields has no meaningful unit to inherit.
            units = source_metadata[0].units if source_metadata else None
            if any(m.units != units for m in source_metadata):
                units = None
            result[target] = VariableMetadata(
                units=units, long_name=_combine_expression(sources)
            )
        for name in self._combine_only_sources:
            result.pop(name, None)
        self._variable_metadata = result

    def _get_files_stats(
        self, n_repeats: int, infer_timestep: bool, max_sample_n_times: int
    ):
        logging.info(f"Opening data at {os.path.join(self.path, self.file_pattern)}")
        raw_times = _get_raw_times(self._raw_paths, engine=self.engine)

        self._timestep: datetime.timedelta | None
        if infer_timestep:
            inferred_timestep = _get_timestep(np.concatenate(raw_times))
            time_coord = _repeat_and_increment_time(
                raw_times, n_repeats, inferred_timestep
            )
            self._timestep = inferred_timestep
        else:
            self._timestep = None
            time_coord = raw_times

        cum_num_timesteps = _get_cumulative_timesteps(time_coord)
        self.start_indices = cum_num_timesteps[:-1]
        self._total_timesteps = cum_num_timesteps[-1]
        self._n_initial_conditions = self._total_timesteps - max_sample_n_times + 1
        self._sample_start_times = xr.CFTimeIndex(
            np.concatenate(time_coord)[: self._n_initial_conditions]
        )
        self._all_times = xr.CFTimeIndex(np.concatenate(time_coord))

        del cum_num_timesteps

    def _group_variable_names_by_time_type(self) -> VariableNames:
        """Returns lists of time-dependent variable names, time-independent
        variable names, and variables which are only present as an initial
        condition.
        """
        (
            time_dependent_names,
            time_invariant_names,
            static_derived_names,
        ) = ([], [], [])
        # Don't use open_mfdataset here, because it will give time-invariant
        # fields a time dimension. We assume that all fields are present in the
        # netcdf file corresponding to the first chunk of time.
        with _open_xr_dataset(
            self.full_paths[0],
            engine=self.engine,
            mask_and_scale=self.mask_and_scale,
        ) as raw_ds:
            ds = self._apply_rename(raw_ds)
            for name in self._names:
                if name in StaticDerivedData.names:
                    static_derived_names.append(name)
                elif name in ds:
                    dims = ds[name].dims
                    if "time" in dims:
                        time_dependent_names.append(name)
                    else:
                        time_invariant_names.append(name)
                elif self._allow_missing_variables:
                    logging.info(
                        f"Variable '{name}' not found in dataset, "
                        "skipping due to allow_missing_variables=True."
                    )
                else:
                    raise ValueError(f"Required variable not found in dataset: {name}.")
        found = time_dependent_names + time_invariant_names + static_derived_names
        logging.info(f"The required variables have been found in the dataset: {found}.")

        return VariableNames(
            time_dependent_names,
            time_invariant_names,
            static_derived_names,
        )

    def configure_horizontal_coordinates(
        self, first_dataset
    ) -> tuple[HorizontalCoordinates, StaticDerivedData, list[str]]:
        horizontal_coordinates: HorizontalCoordinates
        static_derived_data: StaticDerivedData

        horizontal_coordinates, dim_names = get_horizontal_coordinates(
            first_dataset, self.spatial_dimensions, self.dtype
        )
        static_derived_data = StaticDerivedData(horizontal_coordinates)

        coords_sizes = {
            coord_name: len(coord)
            for coord_name, coord in horizontal_coordinates.coords.items()
        }
        logging.info(f"Horizontal coordinate sizes are {coords_sizes}.")
        return horizontal_coordinates, static_derived_data, dim_names

    @property
    def timestep(self) -> datetime.timedelta | None:
        if self._timestep is None:
            if self._infer_timestep is False:
                warnings.warn(
                    "XarrayDataConfig.infer_timestep set to False. "
                    "Timestep was not inferred in the data loader."
                )
                return self._timestep
            else:
                raise ValueError(
                    "Timestep was not inferred in the data loader. Note "
                    "XarrayDataConfig.infer_timestep must be set to True for this "
                    "to occur."
                )
        else:
            return self._timestep

    def _apply_rename(self, ds: xr.Dataset) -> xr.Dataset:
        """Rename on-disk variable names to the names used by FME."""
        if not self._rename:
            return ds
        return ds.rename(self._rename)

    def _open_file(self, idx):
        logger.debug(f"Opening file {self.full_paths[idx]}")
        return self._apply_rename(
            _open_file_fh_cached(
                self.full_paths[idx],
                engine=self.engine,
                mask_and_scale=self.mask_and_scale,
            )
        )

    @property
    def sample_start_times(self) -> xr.CFTimeIndex:
        """Return cftime index corresponding to start time of each sample."""
        self._ensure_epoch_synchronized()
        return self._sample_start_times

    @property
    def all_times(self) -> xr.CFTimeIndex:
        """
        Like sample_start_times, but includes all times in the dataset, including
        final times which are not valid as a start index.

        This is relevant for inference, where we may use get_sample_by_time_slice to
        retrieve time windows directly.

        If this dataset does not support inference,
        this will raise a NotImplementedError.
        """
        return self._all_times

    @property
    def sample_n_times(self) -> int:
        """Number of timesteps in each sample."""
        self._ensure_epoch_synchronized()
        return self._sample_n_times

    def __getitem__(self, idx: int) -> DatasetItem:
        """Return a sample of data spanning the timesteps
        [idx, idx + self.sample_n_times).

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Tuple of a sample's data (i.e. a mapping from names to torch.Tensors) and
            its corresponding time coordinate.
        """
        self._ensure_epoch_synchronized()
        time_slice = slice(idx, idx + self.sample_n_times)
        return self.get_sample_by_time_slice(time_slice)

    def validate_inference_length(self, max_start_index: int, max_window_len: int):
        self._ensure_epoch_synchronized()
        if max_window_len + max_start_index > self._total_timesteps:
            raise ValueError(
                f"The maximum start index {max_start_index} plus window length "
                f"{max_window_len} must be less than or "
                f"equal to the number of steps in the dataset {self._total_timesteps}."
            )

    def get_sample_by_time_slice(self, time_slice: slice) -> DatasetItem:
        self._ensure_epoch_synchronized()
        input_file_idx, input_local_idx = _get_file_local_index(
            time_slice.start, self.start_indices
        )
        output_file_idx, output_local_idx = _get_file_local_index(
            time_slice.stop - 1, self.start_indices
        )

        # get the sequence of observations
        arrays: dict[str, list[torch.Tensor]] = {}
        idxs = range(input_file_idx, output_file_idx + 1)
        total_steps = 0
        for i, file_idx in enumerate(idxs):
            start = input_local_idx if i == 0 else 0
            if i == len(idxs) - 1:
                stop = output_local_idx
            else:
                stop = (
                    self.start_indices[file_idx + 1] - self.start_indices[file_idx] - 1
                )

            n_steps = stop - start + 1
            shape = [n_steps] + self._shape_excluding_time_after_selection
            total_steps += n_steps
            if self.engine == "zarr":
                # this path reads arrays from the store directly, so it must ask
                # for the on-disk names and rename the result itself
                on_disk_names = [
                    self._rename_inverse.get(name, name)
                    for name in self._time_dependent_names
                ]
                loaded = load_series_data_zarr_async(
                    idx=start,
                    n_steps=n_steps,
                    path=self.full_paths[file_idx],
                    names=on_disk_names,
                    final_dims=self.dims,
                    final_shape=shape,
                    fill_nans=self.fill_nans,
                    nontime_selection=self._isel_tuple,
                )
                tensor_dict = {
                    name: loaded[on_disk_name]
                    for name, on_disk_name in zip(
                        self._time_dependent_names, on_disk_names
                    )
                }
            else:
                ds = self._open_file(file_idx)
                ds = ds.isel(**self.isel)
                tensor_dict = load_series_data(
                    idx=start,
                    n_steps=n_steps,
                    ds=ds,
                    names=self._time_dependent_names,
                    final_dims=self.dims,
                    final_shape=shape,
                    fill_nans=self.fill_nans,
                )
                ds.close()
                del ds
            for n in self._time_dependent_names:
                arrays.setdefault(n, []).append(tensor_dict[n])

        tensors: TensorDict = {}
        for n, tensor_list in arrays.items():
            tensors[n] = torch.cat(tensor_list)
        del arrays

        # broadcast the time-invariant variables loaded at construction
        shape = [total_steps] + self._shape_excluding_time_after_selection
        for name, tensor in self._time_invariant_tensors.items():
            tensors[name] = torch.broadcast_to(tensor, shape)

        # load static derived variables
        for name in self._static_derived_names:
            tensor = self._static_derived_data[name]
            horizontal_dims = [1] * tensor.ndim
            tensors[name] = tensor.repeat((total_steps, *horizontal_dims))

        # cast to desired dtype
        tensors = {k: v.to(dtype=self.dtype) for k, v in tensors.items()}

        # Apply field overwrites
        tensors = self.overwrite.apply(tensors)

        # Build combined fields from the (possibly overwritten) sources
        tensors = self._apply_combine(tensors)

        # Fill NaN for missing variables so all samples share the same keys
        missing_names: frozenset[str] | None = None
        if self._allow_missing_variables and self._missing_names:
            fill_shape = [total_steps] + self._shape_excluding_time_after_selection
            fill_dtype = self.dtype if self.dtype is not None else torch.float32
            for name in self._missing_names:
                tensors[name] = torch.full(fill_shape, float("nan"), dtype=fill_dtype)
            missing_names = self._missing_names

        # Create a DataArray of times to return corresponding to the slice that
        # is valid even when n_repeats > 1.
        time = xr.DataArray(self.all_times[time_slice].values, dims=["time"])

        return tensors, time, self._labels, self._epoch, missing_names

    def enable_shared_memory(self):
        """Move epoch counter to shared memory for multi-worker data loading."""
        if not self._global_epoch.is_shared():
            self._global_epoch = self._global_epoch.share_memory_()

    def set_global_epoch_tensor(self, tensor: torch.Tensor):
        """Share a single epoch tensor across multiple datasets."""
        self._global_epoch = tensor

    def set_epoch(self, epoch: int):
        """
        Set the epoch for the dataset. This will update the number of initial
        conditions and the sample start times if the number of timesteps is a schedule.
        """
        self._global_epoch.fill_(epoch)  # values get set lazily based on this


def _get_timestep(time: np.ndarray) -> datetime.timedelta:
    """Computes the timestep of an array of a time coordinate array.

    Raises an error if the times are not separated by a positive constant
    interval, or if the array has one or fewer times.
    """
    assert len(time.shape) == 1, "times must be a 1D array"

    if len(time) > 1:
        timesteps = np.diff(time)
        timestep = timesteps[0]

        if not (timestep > datetime.timedelta(days=0)):
            raise ValueError("Timestep of data must be greater than zero.")

        if not np.all(timesteps == timestep):
            raise ValueError("Time coordinate does not have a uniform timestep.")

        return timestep
    else:
        raise ValueError(
            "Time coordinate does not have enough times to infer a timestep."
        )


def _as_index_selection(
    subset: Slice | TimeSlice | RepeatedInterval, dataset: XarrayDataset
) -> slice | np.ndarray:
    """Converts a subset defined either as a Slice or TimeSlice into an index slice
    based on time coordinate in provided dataset.
    """
    if isinstance(subset, Slice):
        index_selection = subset.slice
    elif isinstance(subset, TimeSlice):
        index_selection = subset.slice(dataset.sample_start_times)
    elif isinstance(subset, RepeatedInterval):
        try:
            index_selection = subset.get_boolean_mask(len(dataset), dataset.timestep)
        except ValueError as e:
            raise ValueError(f"Error when applying RepeatedInterval to dataset: {e}")
    else:
        raise TypeError(f"subset must be Slice or TimeSlice, got {type(subset)}")
    return index_selection


class XarraySubset(DatasetABC):
    def __init__(self, dataset: XarrayDataset, subset: slice | np.ndarray):
        indices = np.arange(len(dataset))[subset]
        logging.info(f"Subsetting dataset samples according to {subset}.")
        self._wrapped_dataset = dataset
        self._dataset = torch.utils.data.Subset(dataset, indices)
        self._sample_start_times = dataset.sample_start_times[indices]
        self._sample_n_times = dataset.sample_n_times
        self._max_timestep_index: int | None = None
        if len(indices) > 0 and np.all(indices[:-1] <= indices[1:]):
            self._max_timestep_index = indices[-1] + dataset.sample_n_times - 1
        self.dims = dataset.dims

    def __getitem__(self, idx: int) -> DatasetItem:
        return self._dataset[idx]

    @property
    def sample_start_times(self):
        return self._sample_start_times

    @property
    def all_times(self) -> xr.CFTimeIndex:
        """
        Like sample_start_times, but includes all times in the dataset, including
        final times which are not valid as a start index.

        This is relevant for inference, where we may use get_sample_by_time_slice to
        retrieve time windows directly.

        If this dataset does not support inference,
        this will raise a NotImplementedError.
        """
        raise NotImplementedError("XarraySubset does not support inference.")

    @property
    def sample_n_times(self) -> int:
        """The length of the time dimension of each sample."""
        return self._sample_n_times

    def get_sample_by_time_slice(self, time_slice: slice) -> DatasetItem:
        raise NotImplementedError(
            "XarraySubset does not support getting samples by time slice, "
            "is this a bug?."
        )

    def validate_inference_length(self, max_start_index: int, max_window_len: int):
        if self._max_timestep_index is None:
            raise ValueError(
                "XarraySubset that does not preserve time ordering of the data "
                "cannot be used for inference."
            )
        if max_start_index + max_window_len - 1 > self._max_timestep_index:
            raise ValueError(
                f"The maximum start index {max_start_index} plus forward steps "
                f"{max_window_len - 1} must be less than or equal to the "
                f"max timestep index in the dataset {self._max_timestep_index}."
            )

    @property
    def properties(self) -> DatasetProperties:
        return self._wrapped_dataset.properties

    def enable_shared_memory(self):
        self._wrapped_dataset.enable_shared_memory()

    def set_global_epoch_tensor(self, tensor: torch.Tensor):
        self._wrapped_dataset.set_global_epoch_tensor(tensor)

    def set_epoch(self, epoch: int):
        self._wrapped_dataset.set_epoch(epoch)


def get_xarray_dataset(
    config: XarrayDataConfig,
    names: Sequence[str],
    n_timesteps: IntSchedule,
    allow_missing_variables: bool = False,
) -> tuple["XarraySubset", DatasetProperties]:
    dataset = XarrayDataset(
        config, names, n_timesteps, allow_missing_variables=allow_missing_variables
    )
    properties = dataset.properties
    index_slice = _as_index_selection(config.subset, dataset)
    return XarraySubset(dataset, index_slice), properties


def get_xarray_datasets(
    dataset_configs: Sequence[XarrayDataConfig],
    names: Sequence[str],
    n_timesteps: IntSchedule,
    strict: bool = True,
    allow_missing_variables: bool = False,
) -> tuple[list[XarraySubset], DatasetProperties]:
    datasets = []
    properties: DatasetProperties | None = None
    for config in dataset_configs:
        dataset, new_properties = get_xarray_dataset(
            config, names, n_timesteps, allow_missing_variables=allow_missing_variables
        )
        datasets.append(dataset)
        if properties is None:
            properties = new_properties
        else:
            properties.update(new_properties, strict=strict)
    if properties is None:
        raise ValueError("At least one dataset must be provided.")

    return datasets, properties
