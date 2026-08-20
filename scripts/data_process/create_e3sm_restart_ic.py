#!/usr/bin/env python
"""Create SamudrACE-E3SMv3 initial condition files from E3SM restart files.

This turns raw E3SM restart output (``eam.i``, ``mpaso.rst``, ``mpassi.rst``) into
the pair of NetCDF files consumed by ``fme.coupled.inference``:

    {prefix}_atmosphere_ic.nc    {prefix}_ocean_ic.nc

Only the *prognostic* variables of each component stepper are required in an
initial condition file (``fme.ace.inference.inference.get_initial_condition``
reads ``in_names & out_names`` and the ``time`` coordinate, nothing else). All
forcing and time-invariant fields (``LANDFRAC``, ``PHIS``, ``SOLIN``, ``ak_*``,
``bk_*``, ``idepth_*``, ``mask_*``, ...) come from the forcing dataset instead,
so they are deliberately not written here.

The processing chain mirrors the training-data pipeline
(``compute_dataset_e3smv2.py`` and ``e3sm_ocean_vertical_coarsen.py`` +
``compute_ocean_dataset_e3sm.py``):

    atmosphere:  eam.i (unique GLL, ncol_d)
                 -> specific total water, pressure thickness from hyai/hybi/PS
                 -> mass-weighted vertical coarsening to 8 layers
                 -> ncremap (conservative) to the shifted-gaussian 1 degree grid

    ocean:       mpaso.rst / mpassi.rst (MPAS mesh, nCells)
                 -> cell-centre velocity reconstruction from normalVelocity
                 -> restingThickness-weighted conservative vertical coarsening
                    to 19 layers
                 -> ncremap -P mpas (conservative) to the same target grid
                 -> NaN outside the training wetmasks

Run ``--help`` for usage. See ``configs/e3smv3-restart-ic.yaml`` for a config
with all defaults spelled out, and the module docstring of
``AtmosphereConfig.near_surface_from_lowest_level`` for the one place where this
script has to approximate a field that E3SM restarts do not carry.
"""

import dataclasses
import glob
import logging
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Sequence

import cftime
import click
import numpy as np
import scipy.signal
import scipy.sparse
import xarray as xr
import yaml

# Prognostic variables of each component of the SamudrACE-E3SMv3 stepper, i.e.
# the intersection of the stepper's in_names and out_names. These are exactly
# the variables an initial condition file has to provide.
N_ATMOSPHERE_LAYERS = 8
N_OCEAN_LAYERS = 19

ATMOSPHERE_PROGNOSTIC_NAMES = (
    ["PS", "TS", "Tat2m", "Qat2m", "Uat10m", "Vat10m"]
    + [f"T_{i}" for i in range(N_ATMOSPHERE_LAYERS)]
    + [f"U_{i}" for i in range(N_ATMOSPHERE_LAYERS)]
    + [f"V_{i}" for i in range(N_ATMOSPHERE_LAYERS)]
    + [f"STW_{i}" for i in range(N_ATMOSPHERE_LAYERS)]
)
OCEAN_PROGNOSTIC_NAMES = ["sst", "ssh", "ocean_sea_ice_fraction", "iceVolumeTotal"] + [
    f"{prefix}_{i}"
    for prefix in (
        "temperatureCoarsened",
        "salinityCoarsened",
        "velocityZonalCoarsened",
        "velocityMeridionalCoarsened",
    )
    for i in range(N_OCEAN_LAYERS)
]

# Mapping from an ocean prognostic variable to the wetmask that defines its
# valid region, matching get_mask_for/ensure_nans_outside_mask in
# compute_ocean_dataset_e3sm.py.
_LEVEL_SUFFIX = re.compile(r"_(\d+)$")

# MPAS writes this fill value into columns below the sea floor.
MPAS_FILL_VALUE_THRESHOLD = 1e30

ZERO_CELSIUS = 273.15

# MPAS-Ocean defaults (config_density0 and gravity), used to convert the sea
# ice load that depresses the restart sea surface back into metres.
SEAWATER_DENSITY = 1026.0
GRAVITY = 9.80616

VARIABLE_ATTRS = {
    "PS": {"units": "Pa", "long_name": "Surface pressure"},
    "TS": {"units": "K", "long_name": "Surface temperature (radiative)"},
    "Tat2m": {
        "units": "K",
        "long_name": "T linearly interpolated to 2.0 m above surface",
    },
    "Qat2m": {
        "units": "kg/kg",
        "long_name": "Q linearly interpolated to 2.0 m above surface",
    },
    "Uat10m": {
        "units": "m/s",
        "long_name": "U linearly interpolated to 10.0 m above surface",
    },
    "Vat10m": {
        "units": "m/s",
        "long_name": "V linearly interpolated to 10.0 m above surface",
    },
    "T": {"units": "K", "long_name": "T"},
    "U": {"units": "m/s", "long_name": "U"},
    "V": {"units": "m/s", "long_name": "V"},
    "STW": {"units": "kg/kg", "long_name": "STW"},
    "sst": {"units": "K", "long_name": "sea surface temperature"},
    "ssh": {"units": "m", "long_name": "sea surface height"},
    "ocean_sea_ice_fraction": {"units": "unitless", "long_name": "sea ice fraction"},
    "iceVolumeTotal": {"units": "m", "long_name": "total sea ice volume per unit area"},
    "temperatureCoarsened": {
        "units": "degC",
        "long_name": "sea water potential temperature",
    },
    "salinityCoarsened": {"units": "PSU", "long_name": "sea water salinity"},
    "velocityZonalCoarsened": {"units": "m s-1", "long_name": "zonal velocity"},
    "velocityMeridionalCoarsened": {
        "units": "m s-1",
        "long_name": "meridional velocity",
    },
}


@dataclasses.dataclass
class MapsConfig:
    """Paths to the ncremap weight files used for horizontal remapping.

    Parameters:
        atmosphere: Map from the EAM unique-GLL grid (ne30np4, ncol_d) to the
            target grid. Generate with::

                ncremap -s ne30np4_pentagons.091226.nc \\
                        -g dst_gaussian_180by360_shifted.scrip.nc \\
                        -m map_ne30np4_to_gaussian_180by360_shifted.nc

        ocean: Map from the MPAS ocean/ice mesh (nCells) to the target grid,
            e.g. map_IcoswISC30E3r5_to_gaussian_180by360_shifted.nc.
    """

    atmosphere: str
    ocean: str

    def validate(self) -> None:
        for name in ("atmosphere", "ocean"):
            path = getattr(self, name)
            if not os.path.exists(path):
                raise ValueError(f"maps.{name} does not exist: {path}")


@dataclasses.dataclass
class MasksConfig:
    """Wetmask / surface-fraction source, used to reproduce the training masks.

    The ocean forcing dataset published with the checkpoint already carries the
    exact masks the model was trained with (``mask_0`` .. ``mask_18``,
    ``mask_2d``, ``mask_ocean_sea_ice_fraction``, ``mask_iceVolumeTotal``) and
    ``sea_surface_fraction``, so it is the natural source.

    Parameters:
        path: NetCDF file (or zarr store) on the target grid holding the masks.
        apply_ocean_masks: Set ocean prognostics to NaN outside their wetmask.
        use_for_surface_blend: Use ``sea_surface_fraction`` from this file when
            blending TS. If False, TS falls back to the lowest-level air
            temperature everywhere.
    """

    path: str | None = None
    apply_ocean_masks: bool = True
    use_for_surface_blend: bool = True

    def validate(self) -> None:
        if self.path is None:
            if self.apply_ocean_masks:
                raise ValueError(
                    "masks.apply_ocean_masks is true but masks.path is not set."
                )
            if self.use_for_surface_blend:
                raise ValueError(
                    "masks.use_for_surface_blend is true but masks.path is not set."
                )
        elif not os.path.exists(self.path):
            raise ValueError(f"masks.path does not exist: {self.path}")


@dataclasses.dataclass
class AtmosphereConfig:
    """Atmosphere-side processing options.

    Parameters:
        vertical_coarsening_indices: Half-open [start, end) index ranges into
            the 80 EAM levels defining the 8 coarse layers. Must match the
            checkpoint; the E3SMv3 values are the default.
        water_species: Fields summed to form specific total water (STW).
        near_surface_from_lowest_level: E3SM restart files do not contain the
            diagnostic near-surface fields ``Tat2m``, ``Qat2m``, ``Uat10m`` and
            ``Vat10m`` (they are computed inside EAM's surface-layer scheme and
            never checkpointed). When True, they are approximated by the lowest
            model level of T, Q, U and V. They are prognostic in the emulator
            but strongly constrained by the rest of the state, so the model
            recovers them within a step or two; set to False to leave them out
            and fail loudly instead.
    """

    vertical_coarsening_indices: Sequence[Sequence[int]] = dataclasses.field(
        default_factory=lambda: [
            [0, 25],
            [25, 38],
            [38, 46],
            [46, 52],
            [52, 56],
            [56, 61],
            [61, 69],
            [69, 80],
        ]
    )
    water_species: Sequence[str] = ("Q", "CLDLIQ", "CLDICE", "RAINQM")
    near_surface_from_lowest_level: bool = True

    def validate(self) -> None:
        if len(self.vertical_coarsening_indices) != N_ATMOSPHERE_LAYERS:
            raise ValueError(
                f"atmosphere.vertical_coarsening_indices must have "
                f"{N_ATMOSPHERE_LAYERS} entries, got "
                f"{len(self.vertical_coarsening_indices)}."
            )
        covered: list[int] = []
        for start, end in self.vertical_coarsening_indices:
            covered.extend(range(start, end))
        if covered != list(range(covered[-1] + 1)):
            raise ValueError(
                "atmosphere.vertical_coarsening_indices must be contiguous, "
                f"non-overlapping and start at 0, got "
                f"{self.vertical_coarsening_indices!r}."
            )


@dataclasses.dataclass
class OceanConfig:
    """Ocean-side processing options.

    Parameters:
        target_interface_levels: The 20 depths (m) bounding the 19 coarse ocean
            layers. Must match the checkpoint (compare with ``idepth_*`` in the
            ocean forcing file); the E3SMv3 values are the default.
        reconstruct_velocity: Reconstruct cell-centre zonal/meridional velocity
            from ``normalVelocity`` on edges by weighted least squares. MPAS
            restarts only store the edge-normal component, so this is required
            for the velocity prognostics.
        exclude_ice_shelf_cavities: Drop cells that sit under an ice shelf,
            identified as ``landIceMask > 0`` or ``landIceDraft < 0``. The
            E3SMv3 ocean mesh (IcoswISC30E3r5) resolves sub-ice-shelf cavities,
            where the sea surface sits at the ice draft and ``ssh`` reaches
            -1700 m. The training data excluded them (its ssh spans roughly
            [-1.3, 1.1] m), so they are excluded here too. ``landIceMask``
            alone is not enough: it marks only the cells where ice-shelf
            pressure is currently applied, while several thousand more cells
            still carry a nonzero ``landIceDraft`` and a correspondingly
            depressed sea surface.
        spatial_filter_scale: If set, apply a boxcar smoothing of this many
            target-grid cells to the remapped 3D ocean fields. The published
            training config (``configs/e3smv3-ocean-1deg.yaml``) enables a
            scale-4 filter while the corresponding dataset is named
            "no-smoothing", so the default here is to not filter. Set to 4 if
            you determine the checkpoint was trained on filtered data.
    """

    target_interface_levels: Sequence[float] = (
        0.0, 20.0, 30.0, 40.0, 50.0, 80.0, 110.0, 140.0, 170.0, 230.0,
        410.0, 530.0, 1020.0, 1080.0, 1720.0, 1980.0, 2820.0, 3380.0,
        4620.0, 6380.0,
    )  # fmt: skip
    reconstruct_velocity: bool = True
    exclude_ice_shelf_cavities: bool = True
    spatial_filter_scale: int | None = None

    def validate(self) -> None:
        if len(self.target_interface_levels) != N_OCEAN_LAYERS + 1:
            raise ValueError(
                f"ocean.target_interface_levels must have {N_OCEAN_LAYERS + 1} "
                f"entries, got {len(self.target_interface_levels)}."
            )
        if any(np.diff(self.target_interface_levels) <= 0):
            raise ValueError("ocean.target_interface_levels must be increasing.")


@dataclasses.dataclass
class TimeConfig:
    """How the ``time`` coordinate of the initial conditions is set.

    ``fme`` requires the initial condition time to be present in the forcing
    dataset's time index, and the coupled loader selects the atmosphere sample
    using the ocean sample's time, so both files always get the same value.

    Parameters:
        source: ``"restart"`` to take the timestamp from the MPAS ``xtime``
            variable, or ``"explicit"`` to use ``timestamps`` instead.
        timestamps: One ISO ``YYYY-MM-DDTHH:MM:SS`` timestamp per restart
            directory, used when ``source == "explicit"``. Useful when the
            forcing dataset's calendar does not line up with the restart dates
            (for example running historical restarts against piControl
            forcing).
        calendar: Calendar of the written time coordinate.
        units: Units of the written time coordinate.
    """

    source: str = "restart"
    timestamps: Sequence[str] = ()
    calendar: str = "noleap"
    units: str = "days since 0001-01-01"

    def validate(self) -> None:
        if self.source not in ("restart", "explicit"):
            raise ValueError(
                f"time.source must be 'restart' or 'explicit', got {self.source!r}."
            )
        if self.source == "explicit" and len(self.timestamps) == 0:
            raise ValueError("time.source is 'explicit' but time.timestamps is empty.")


@dataclasses.dataclass
class CreateRestartICConfig:
    """Configuration for building initial conditions from E3SM restarts.

    Parameters:
        restart_directories: Explicit list of restart directories to process.
        restart_glob: Glob expanded into restart directories, e.g.
            ``"/archive/rest/*"``. Combined with ``restart_directories``.
        output_directory: Where the initial condition files are written.
        output_prefix: Output files are ``{prefix}_atmosphere_ic.nc`` and
            ``{prefix}_ocean_ic.nc``.
        stack: If True, all restart directories are combined into a single pair
            of files with one time per restart (what ``n_initial_conditions``
            in the inference config expects). If False, one pair of files is
            written per restart, suffixed with its timestamp.
        maps: Horizontal remapping weights.
        masks: Wetmask source.
        atmosphere: Atmosphere processing options.
        ocean: Ocean processing options.
        time: Time coordinate handling.
        work_directory: Scratch directory for intermediate files. A temporary
            directory is used when unset.
        keep_intermediate: Keep the pre- and post-remap intermediate files.
        overwrite: Overwrite existing output files.
    """

    output_directory: str
    maps: MapsConfig
    restart_directories: Sequence[str] = ()
    restart_glob: str | None = None
    output_prefix: str = "e3sm-restart"
    stack: bool = True
    masks: MasksConfig = dataclasses.field(default_factory=MasksConfig)
    atmosphere: AtmosphereConfig = dataclasses.field(default_factory=AtmosphereConfig)
    ocean: OceanConfig = dataclasses.field(default_factory=OceanConfig)
    time: TimeConfig = dataclasses.field(default_factory=TimeConfig)
    work_directory: str | None = None
    keep_intermediate: bool = False
    overwrite: bool = False

    def __post_init__(self):
        self.maps.validate()
        self.masks.validate()
        self.atmosphere.validate()
        self.ocean.validate()
        self.time.validate()
        if not self.resolved_restart_directories:
            raise ValueError(
                "No restart directories found; set restart_directories and/or "
                "restart_glob."
            )
        n_dirs = len(self.resolved_restart_directories)
        if self.time.source == "explicit" and len(self.time.timestamps) != n_dirs:
            raise ValueError(
                f"time.timestamps has {len(self.time.timestamps)} entries but "
                f"{n_dirs} restart directories were found."
            )

    @property
    def resolved_restart_directories(self) -> list[str]:
        directories = list(self.restart_directories)
        if self.restart_glob is not None:
            directories.extend(sorted(p for p in glob.glob(self.restart_glob)))
        return [d for d in dict.fromkeys(directories) if os.path.isdir(d)]

    @classmethod
    def from_file(cls, path: str, **overrides) -> "CreateRestartICConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        data.update({k: v for k, v in overrides.items() if v is not None})
        return _from_dict(cls, data)


_NESTED_CONFIGS = {
    "maps": MapsConfig,
    "masks": MasksConfig,
    "atmosphere": AtmosphereConfig,
    "ocean": OceanConfig,
    "time": TimeConfig,
}


def _from_dict(cls, data):
    """Build a (possibly nested) dataclass from a mapping, rejecting typos."""
    field_names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - field_names
    if unknown:
        raise ValueError(
            f"Unknown option(s) for {cls.__name__}: {sorted(unknown)}. "
            f"Valid options are {sorted(field_names)}."
        )
    kwargs = {}
    for name, value in data.items():
        nested = _NESTED_CONFIGS.get(name)
        if nested is not None and isinstance(value, dict):
            kwargs[name] = _from_dict(nested, value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


@dataclasses.dataclass
class RestartFiles:
    """The restart files of a single E3SM restart directory."""

    directory: str
    eam: str
    mpaso: str
    mpassi: str

    @classmethod
    def find(cls, directory: str) -> "RestartFiles":
        def one(pattern: str, description: str) -> str:
            matches = sorted(glob.glob(os.path.join(directory, pattern)))
            if len(matches) != 1:
                raise ValueError(
                    f"Expected exactly one {description} file matching "
                    f"{pattern!r} in {directory}, found {len(matches)}: {matches}"
                )
            return matches[0]

        # The EAM *initial* (inithist) file is used rather than eam.r: the
        # restart file holds the dynamics state on the non-unique GLL grid
        # (ncol_d = 86400 for ne30) with temperature stored as VTheta_dp, while
        # eam.i holds T/U/V/Q/CLDLIQ/CLDICE/RAINQM directly on the unique GLL
        # grid (ncol_d = 48602) that ne30np4_pentagons describes.
        return cls(
            directory=directory,
            eam=one("*.eam.i.*.nc", "EAM initial"),
            mpaso=one("*.mpaso.rst.*.nc", "MPAS-Ocean restart"),
            mpassi=one("*.mpassi.rst.*.nc", "MPAS-Seaice restart"),
        )


def _decode_xtime(dataset: xr.Dataset) -> str:
    """Return the MPAS ``xtime`` timestamp as an ISO string."""
    raw = dataset["xtime"].values.reshape(-1)[0]
    text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
    return text.strip().replace("_", "T")


def _parse_timestamp(timestamp: str, calendar: str) -> cftime.datetime:
    match = re.match(
        r"^(-?\d+)-(\d{2})-(\d{2})[T ](\d{2}):(\d{2}):(\d{2})$", timestamp.strip()
    )
    if not match:
        raise ValueError(f"Timestamp must be YYYY-MM-DDTHH:MM:SS, got {timestamp!r}.")
    year, month, day, hour, minute, second = (int(g) for g in match.groups())
    return cftime.datetime(year, month, day, hour, minute, second, calendar=calendar)


def compute_pressure_thickness(
    surface_pressure: xr.DataArray,
    hyai: xr.DataArray,
    hybi: xr.DataArray,
    reference_pressure: float,
    interface_dim: str,
    level_dim: str,
) -> xr.DataArray:
    """Pressure thickness of each model layer, following compute_dataset_e3smv2."""
    half_level_pressure = reference_pressure * hyai + surface_pressure * hybi
    thickness = (
        half_level_pressure.diff(dim=interface_dim)
        .rename({interface_dim: level_dim})
        # The interface coordinate would be renamed along with the dimension
        # and then clash with the layer-midpoint coordinate of the fields it
        # weights, so it is dropped rather than relabelled.
        .drop_vars(level_dim, errors="ignore")
    )
    thickness.attrs = {"units": "Pa", "long_name": "pressure thickness"}
    return thickness


def vertical_coarsen_atmosphere(
    field: xr.DataArray,
    pressure_thickness: xr.DataArray,
    interface_indices: Sequence[Sequence[int]],
    level_dim: str,
) -> dict[str, xr.DataArray]:
    """Mass-weighted mean of ``field`` over each coarse layer."""
    coarsened = {}
    for i, (start, end) in enumerate(interface_indices):
        layer = {level_dim: slice(start, end)}
        weights = pressure_thickness.isel(layer)
        coarsened[str(i)] = (field.isel(layer) * weights).sum(level_dim) / weights.sum(
            level_dim
        )
    return coarsened


def reconstruct_cell_velocity(
    normal_velocity: np.ndarray,
    angle_edge: np.ndarray,
    dv_edge: np.ndarray,
    edges_on_cell: np.ndarray,
    n_edges_on_cell: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct cell-centre (zonal, meridional) velocity from edge normals.

    MPAS-Ocean restarts store only the edge-normal velocity component, so the
    zonal/meridional velocities the emulator expects have to be reconstructed.
    For each cell and level this solves the edge-length-weighted least squares
    problem

        min_{u, v} sum_e w_e (u cos(theta_e) + v sin(theta_e) - vn_e)^2

    over the cell's edges, where ``theta_e`` is ``angleEdge`` (the angle of the
    edge normal east of north-up local east) and ``w_e`` is ``dvEdge``. Edges
    whose normal velocity is invalid at a given level (below the sea floor of
    either neighbouring cell) are excluded level by level, and cells left with
    fewer than two usable edges get NaN.

    Args:
        normal_velocity: (nEdges, nVertLevels) edge-normal velocity.
        angle_edge: (nEdges,) MPAS ``angleEdge``, radians.
        dv_edge: (nEdges,) MPAS ``dvEdge``, used as the least squares weight.
        edges_on_cell: (nCells, maxEdges) MPAS ``edgesOnCell``, 1-based, 0 pads.
        n_edges_on_cell: (nCells,) MPAS ``nEdgesOnCell``.

    Returns:
        (zonal, meridional) arrays of shape (nCells, nVertLevels).
    """
    n_cells, max_edges = edges_on_cell.shape

    edge_rank = np.arange(max_edges)[None, :]
    present = (edge_rank < n_edges_on_cell[:, None]) & (edges_on_cell > 0)
    cell_index = np.repeat(np.arange(n_cells), max_edges).reshape(n_cells, max_edges)
    rows = cell_index[present]
    cols = edges_on_cell[present] - 1

    valid = np.isfinite(normal_velocity) & (
        np.abs(normal_velocity) < MPAS_FILL_VALUE_THRESHOLD
    )
    velocity = np.where(valid, normal_velocity, 0.0)

    cosine = np.cos(angle_edge)[cols]
    sine = np.sin(angle_edge)[cols]
    weight = dv_edge[cols]
    shape = (n_cells, normal_velocity.shape[0])

    def gather(values: np.ndarray, field: np.ndarray) -> np.ndarray:
        """Sum ``values[e] * field[e, k]`` over the edges of each cell."""
        matrix = scipy.sparse.csr_matrix((values, (rows, cols)), shape=shape)
        return np.asarray(matrix @ field)

    # Normal equations, assembled per level so that edges that are inactive at
    # depth drop out of both the matrix and the right hand side.
    valid_float = valid.astype(normal_velocity.dtype)
    a11 = gather(weight * cosine * cosine, valid_float)
    a12 = gather(weight * cosine * sine, valid_float)
    a22 = gather(weight * sine * sine, valid_float)
    b1 = gather(weight * cosine, velocity)
    b2 = gather(weight * sine, velocity)

    determinant = a11 * a22 - a12 * a12
    n_valid_edges = gather(np.ones_like(weight), valid_float)
    singular = (n_valid_edges < 2) | (
        np.abs(determinant) <= 1e-12 * np.maximum(a11 * a22, 1e-30)
    )
    safe = np.where(singular, 1.0, determinant)
    zonal = np.where(singular, np.nan, (a22 * b1 - a12 * b2) / safe)
    meridional = np.where(singular, np.nan, (a11 * b2 - a12 * b1) / safe)
    return zonal.astype(np.float32), meridional.astype(np.float32)


def _conservative_depth_weights(
    source_interfaces: np.ndarray, target_interfaces: np.ndarray
) -> scipy.sparse.csr_matrix:
    """Fractional overlap of each source layer with each target layer.

    Entry (k, j) is ``overlap(k, j) / dz_k``, reproducing the xgcm conservative
    transform used by ``e3sm_ocean_vertical_coarsen.py``: a source layer's
    depth-integrated content is redistributed to the target layers it overlaps.
    """
    source_top, source_bottom = source_interfaces[:-1], source_interfaces[1:]
    target_top, target_bottom = target_interfaces[:-1], target_interfaces[1:]
    overlap = np.clip(
        np.minimum(source_bottom[:, None], target_bottom[None, :])
        - np.maximum(source_top[:, None], target_top[None, :]),
        0.0,
        None,
    )
    return scipy.sparse.csr_matrix(overlap / (source_bottom - source_top)[:, None])


def vertical_coarsen_ocean(
    field: np.ndarray,
    resting_thickness: np.ndarray,
    weights: scipy.sparse.csr_matrix,
) -> np.ndarray:
    """Thickness-weighted conservative coarsening of an MPAS ocean field.

    ``field`` is (nCells, nVertLevels) with NaN wherever there is no valid
    water (below the sea floor, and under ice shelves when those are
    excluded). Invalid levels are dropped from both the numerator and the
    thickness weights, so the result is a weighted mean over valid water only
    and target layers with no valid water at all come out as NaN -- the wetmask
    convention of the training data. ``restingThickness`` is already exactly
    zero below the sea floor, so this reduces to the training pipeline's
    ``transform(field * restingThickness) / transform(restingThickness)``
    wherever the sea floor is the only source of invalid data.
    """
    valid_thickness = np.where(np.isfinite(field), resting_thickness, 0.0)
    numerator = np.nan_to_num(field * valid_thickness, nan=0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (numerator @ weights) / (valid_thickness @ weights)


def _mask_below_bathymetry(
    field: np.ndarray, min_level_cell: np.ndarray, max_level_cell: np.ndarray
) -> np.ndarray:
    """NaN out levels outside [minLevelCell, maxLevelCell] and MPAS fill values."""
    level = np.arange(field.shape[1])[None, :]
    inside = (level >= (min_level_cell[:, None] - 1)) & (
        level <= (max_level_cell[:, None] - 1)
    )
    finite = np.isfinite(field) & (np.abs(field) < MPAS_FILL_VALUE_THRESHOLD)
    return np.where(inside & finite, field, np.nan)


def build_atmosphere_native(
    files: RestartFiles, config: AtmosphereConfig
) -> xr.Dataset:
    """Vertically coarsened atmosphere fields on the native EAM GLL grid."""
    logging.info("Reading atmosphere state: %s", files.eam)
    needed = list(config.water_species) + ["T", "U", "V", "PS", "hyai", "hybi", "P0"]
    source = xr.open_dataset(files.eam, decode_times=False, decode_timedelta=False)
    missing = [name for name in needed if name not in source]
    if missing:
        raise ValueError(f"{files.eam} is missing required variables: {missing}")
    source = source[needed]

    level_dim, interface_dim = "lev", "ilev"
    n_levels = source.sizes[level_dim]
    expected = config.vertical_coarsening_indices[-1][-1]
    if n_levels != expected:
        raise ValueError(
            f"{files.eam} has {n_levels} levels but "
            f"atmosphere.vertical_coarsening_indices span {expected}."
        )

    surface_pressure = source["PS"].squeeze(drop=True)
    thickness = compute_pressure_thickness(
        surface_pressure,
        source["hyai"],
        source["hybi"],
        float(source["P0"].values),
        interface_dim=interface_dim,
        level_dim=level_dim,
    )

    specific_total_water = sum(
        source[name].squeeze(drop=True) for name in config.water_species
    )
    fields = {
        "T": source["T"].squeeze(drop=True),
        "U": source["U"].squeeze(drop=True),
        "V": source["V"].squeeze(drop=True),
        "STW": specific_total_water,
    }

    output = xr.Dataset()
    for name, field in fields.items():
        coarsened = vertical_coarsen_atmosphere(
            field, thickness, config.vertical_coarsening_indices, level_dim
        )
        for suffix, values in coarsened.items():
            output[f"{name}_{suffix}"] = values
    output["PS"] = surface_pressure

    if config.near_surface_from_lowest_level:
        # See AtmosphereConfig.near_surface_from_lowest_level: E3SM restarts do
        # not checkpoint EAM's diagnostic 2m/10m fields.
        lowest = {level_dim: n_levels - 1}
        output["Tat2m"] = fields["T"].isel(lowest, drop=True)
        output["Qat2m"] = (
            source[config.water_species[0]].squeeze(drop=True).isel(lowest, drop=True)
        )
        output["Uat10m"] = fields["U"].isel(lowest, drop=True)
        output["Vat10m"] = fields["V"].isel(lowest, drop=True)
        # Retained only to blend TS after remapping; dropped from the output.
        output["lowest_level_temperature"] = output["Tat2m"]

    for name in output.data_vars:
        output[name].attrs = dict(VARIABLE_ATTRS.get(_LEVEL_SUFFIX.sub("", name), {}))
    # Vertical coordinates would otherwise be carried along as scalar or
    # level-dimensioned coordinates that ncremap has no map for.
    output = output.drop_vars(
        [name for name in output.coords if name != "ncol_d"], errors="ignore"
    )
    # ncremap matches the horizontal dimension by size; the GLL map's source
    # grid (ne30np4_pentagons) is the unique GLL grid that eam.i calls ncol_d.
    return output.rename({"ncol_d": "ncol"}).astype(np.float32)


def build_ocean_native(files: RestartFiles, config: OceanConfig) -> xr.Dataset:
    """Vertically coarsened ocean and sea ice fields on the native MPAS mesh."""
    logging.info("Reading ocean state: %s", files.mpaso)
    ocean = xr.open_dataset(files.mpaso, decode_times=False, decode_timedelta=False)
    logging.info("Reading sea ice state: %s", files.mpassi)
    ice = xr.open_dataset(files.mpassi, decode_times=False, decode_timedelta=False)

    resting_thickness = ocean["restingThickness"].values.astype(np.float64)
    min_level = ocean["minLevelCell"].values
    max_level = ocean["maxLevelCell"].values

    cavity = np.zeros(ocean.sizes["nCells"], dtype=bool)
    if config.exclude_ice_shelf_cavities:
        missing = [n for n in ("landIceMask", "landIceDraft") if n not in ocean]
        if missing:
            raise ValueError(
                f"{files.mpaso} has no {missing}; set "
                f"ocean.exclude_ice_shelf_cavities to false."
            )

        def flat(name: str) -> np.ndarray:
            return np.squeeze(np.asarray(ocean[name].values)).reshape(-1)

        cavity = (flat("landIceMask") > 0) | (flat("landIceDraft") < 0)
        logging.info("Excluding %d sub-ice-shelf cells", int(cavity.sum()))
    source_interfaces = np.concatenate(
        [[0.0], ocean["refBottomDepth"].values.astype(np.float64)]
    )
    target_interfaces = np.asarray(config.target_interface_levels, dtype=np.float64)
    if target_interfaces[-1] < source_interfaces[-1]:
        raise ValueError(
            f"ocean.target_interface_levels only reach "
            f"{target_interfaces[-1]} m but the mesh reaches "
            f"{source_interfaces[-1]} m."
        )
    weights = _conservative_depth_weights(source_interfaces, target_interfaces)

    native = {
        "temperatureCoarsened": ocean["temperature"].values[0],
        "salinityCoarsened": ocean["salinity"].values[0],
    }
    if config.reconstruct_velocity:
        logging.info("Reconstructing cell-centre velocity from normalVelocity")
        zonal, meridional = reconstruct_cell_velocity(
            ocean["normalVelocity"].values[0],
            ocean["angleEdge"].values,
            ocean["dvEdge"].values,
            ocean["edgesOnCell"].values,
            ocean["nEdgesOnCell"].values,
        )
        native["velocityZonalCoarsened"] = zonal
        native["velocityMeridionalCoarsened"] = meridional

    output = xr.Dataset()
    for name, values in native.items():
        masked = _mask_below_bathymetry(values.astype(np.float64), min_level, max_level)
        masked[cavity, :] = np.nan
        coarse = vertical_coarsen_ocean(masked, resting_thickness, weights)
        for level in range(N_OCEAN_LAYERS):
            output[f"{name}_{level}"] = xr.DataArray(
                coarse[:, level].astype(np.float32), dims=["nCells"]
            )

    # Sea surface temperature, matching add_sst in e3sm_ocean_vertical_coarsen.
    surface_temperature = _mask_below_bathymetry(
        ocean["temperature"].values[0].astype(np.float64), min_level, max_level
    )[:, 0]
    surface_temperature[cavity] = np.nan
    output["sst"] = xr.DataArray(
        (surface_temperature + ZERO_CELSIUS).astype(np.float32), dims=["nCells"]
    )

    # MPAS-Ocean does not checkpoint ssh; it is diagnosed from the column as
    # sum(layerThickness) - bottomDepth, which reproduces the coupler's
    # o2x_ox_So_ssh to round-off. Two loading terms are then removed so that
    # what is left is the dynamic sea surface the emulator was trained on:
    #
    #   landIceDraft        under an ice shelf the raw quantity is the ice
    #                       draft, reaching -1700 m; zero elsewhere.
    #   seaIcePressure      sea ice depresses the surface by up to ~5 m in the
    #                       raw restart. Adding the load back moves the minimum
    #                       from -6.1 m to -1.34 m, against -1.32 m in the
    #                       reference initial conditions published with the
    #                       checkpoint. The atmospheric pressure load is
    #                       deliberately not removed; doing so degrades that
    #                       agreement.
    layer_thickness = _mask_below_bathymetry(
        ocean["layerThickness"].values[0].astype(np.float64), min_level, max_level
    )
    sea_surface_height = np.nansum(layer_thickness, axis=1) - ocean[
        "bottomDepth"
    ].values.astype(np.float64)

    def flat_float(name: str) -> np.ndarray:
        return np.squeeze(np.asarray(ocean[name].values)).reshape(-1).astype(np.float64)

    if "landIceDraft" in ocean:
        sea_surface_height = sea_surface_height - flat_float("landIceDraft")
    if "seaIcePressure" in ocean:
        sea_surface_height = sea_surface_height + flat_float("seaIcePressure") / (
            SEAWATER_DENSITY * GRAVITY
        )
    sea_surface_height[~np.isfinite(surface_temperature)] = np.nan
    output["ssh"] = xr.DataArray(sea_surface_height.astype(np.float32), dims=["nCells"])

    ice_area = ice["iceAreaCategory"].values[0].sum(axis=(1, 2))
    ice_volume = ice["iceVolumeCategory"].values[0].sum(axis=(1, 2))
    ice_area = np.where(cavity, np.nan, ice_area)
    ice_volume = np.where(cavity, np.nan, ice_volume)
    output["ocean_sea_ice_fraction"] = xr.DataArray(
        ice_area.astype(np.float32), dims=["nCells"]
    )
    output["iceVolumeTotal"] = xr.DataArray(
        ice_volume.astype(np.float32), dims=["nCells"]
    )

    # Category-mean ice surface temperature, retained only to blend TS after
    # remapping and dropped from the final output.
    category_area = ice["iceAreaCategory"].values[0][..., 0]
    category_temperature = ice["surfaceTemperature"].values[0][..., 0]
    total_area = category_area.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ice_temperature = np.where(
            total_area > 0,
            (category_area * category_temperature).sum(axis=1) / total_area,
            np.nan,
        )
    output["ice_surface_temperature"] = xr.DataArray(
        (ice_temperature + ZERO_CELSIUS).astype(np.float32), dims=["nCells"]
    )

    for name in output.data_vars:
        output[name].attrs = dict(VARIABLE_ATTRS.get(_LEVEL_SUFFIX.sub("", name), {}))
    output["timestamp"] = _decode_xtime(ocean)
    return output


def remap(
    source_path: str, target_path: str, map_path: str, mpas: bool = False
) -> None:
    """Horizontally remap a NetCDF file with ncremap."""
    command = ["ncremap"]
    if mpas:
        # Matches compute_ocean_dataset_e3sm.sh; handles the MPAS layout and
        # renormalises conservative weights against missing (land) values.
        command += ["-P", "mpas"]
    command += ["-m", map_path, "-i", source_path, "-o", target_path]
    logging.info("Remapping: %s", " ".join(command))
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0 or not os.path.exists(target_path):
        raise RuntimeError(
            f"ncremap failed (exit {result.returncode}).\n"
            f"command: {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


def _boxcar_smooth(field: np.ndarray, scale: int) -> np.ndarray:
    """Periodic-in-longitude boxcar smoothing that ignores NaN."""
    kernel = np.ones((scale, scale))
    valid = np.isfinite(field).astype(np.float64)
    filled = np.where(np.isfinite(field), field, 0.0)
    padded_sum = scipy.signal.convolve2d(filled, kernel, mode="same", boundary="wrap")
    padded_count = scipy.signal.convolve2d(valid, kernel, mode="same", boundary="wrap")
    with np.errstate(invalid="ignore", divide="ignore"):
        smoothed = padded_sum / padded_count
    return np.where(np.isfinite(field), smoothed, np.nan)


def _ocean_mask_name(variable: str, available: set[str]) -> str | None:
    """Wetmask for an ocean variable, following get_mask_for."""
    specific = f"mask_{variable}"
    if specific in available:
        return specific
    match = _LEVEL_SUFFIX.search(variable)
    if match:
        candidate = f"mask_{match.group(1)}"
        return candidate if candidate in available else None
    return "mask_2d" if "mask_2d" in available else None


def finalize(
    atmosphere: xr.Dataset,
    ocean: xr.Dataset,
    masks: xr.Dataset | None,
    config: CreateRestartICConfig,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Blend TS, apply wetmasks and reduce to the prognostic variables."""
    sea_fraction = None
    if masks is not None and config.masks.use_for_surface_blend:
        if "sea_surface_fraction" not in masks:
            raise ValueError(
                f"{config.masks.path} has no 'sea_surface_fraction'; set "
                f"masks.use_for_surface_blend to false."
            )
        sea_fraction = masks["sea_surface_fraction"].astype(np.float32)

    if "lowest_level_temperature" not in atmosphere:
        raise ValueError(
            "TS cannot be built without a near-surface air temperature; set "
            "atmosphere.near_surface_from_lowest_level to true."
        )
    air_temperature = atmosphere["lowest_level_temperature"]
    if sea_fraction is None:
        logging.warning(
            "No surface fractions available; setting TS to the lowest model "
            "level temperature everywhere."
        )
        atmosphere["TS"] = air_temperature
    else:
        # TS is EAM's merged radiative surface temperature. The ocean and sea
        # ice contributions come from the MPAS restarts; E3SM does not
        # checkpoint a land skin temperature in the files this script reads, so
        # the land tile falls back to lowest-level air temperature.
        ice_fraction = ocean["ocean_sea_ice_fraction"].fillna(0.0)
        sea_temperature = xr.where(
            ice_fraction > 0,
            ice_fraction * ocean["ice_surface_temperature"].fillna(ocean["sst"])
            + (1.0 - ice_fraction) * ocean["sst"],
            ocean["sst"],
        )
        sea_temperature = sea_temperature.fillna(air_temperature)
        atmosphere["TS"] = (
            sea_fraction * sea_temperature + (1.0 - sea_fraction) * air_temperature
        )
    atmosphere["TS"].attrs = dict(VARIABLE_ATTRS["TS"])

    if masks is not None and config.masks.apply_ocean_masks:
        available = {str(name) for name in masks.data_vars}
        for name in OCEAN_PROGNOSTIC_NAMES:
            mask_name = _ocean_mask_name(name, available)
            if mask_name is None:
                logging.warning("No wetmask found for %s; leaving unmasked.", name)
                continue
            ocean[name] = ocean[name].where(masks[mask_name] > 0)

    missing_atmosphere = [n for n in ATMOSPHERE_PROGNOSTIC_NAMES if n not in atmosphere]
    missing_ocean = [n for n in OCEAN_PROGNOSTIC_NAMES if n not in ocean]
    if missing_atmosphere or missing_ocean:
        raise ValueError(
            "Prognostic variables are missing from the generated initial "
            f"conditions. atmosphere: {missing_atmosphere}, ocean: {missing_ocean}"
        )
    return atmosphere[ATMOSPHERE_PROGNOSTIC_NAMES], ocean[OCEAN_PROGNOSTIC_NAMES]


def process_restart(
    files: RestartFiles,
    config: CreateRestartICConfig,
    masks: xr.Dataset | None,
    work_directory: str,
    timestamp_override: str | None,
) -> tuple[xr.Dataset, xr.Dataset, cftime.datetime]:
    """Build the remapped, masked prognostic state of a single restart."""
    tag = os.path.basename(os.path.normpath(files.directory))
    paths = {
        key: os.path.join(work_directory, f"{tag}.{key}.nc")
        for key in ("atmosphere_native", "atmosphere", "ocean_native", "ocean")
    }

    ocean_native = build_ocean_native(files, config.ocean)
    timestamp = str(ocean_native["timestamp"].values)
    ocean_native = ocean_native.drop_vars("timestamp")
    ocean_native.to_netcdf(paths["ocean_native"])
    remap(paths["ocean_native"], paths["ocean"], config.maps.ocean, mpas=True)

    atmosphere_native = build_atmosphere_native(files, config.atmosphere)
    atmosphere_native.to_netcdf(paths["atmosphere_native"])
    remap(paths["atmosphere_native"], paths["atmosphere"], config.maps.atmosphere)

    atmosphere = xr.open_dataset(paths["atmosphere"], decode_times=False).load()
    ocean = xr.open_dataset(paths["ocean"], decode_times=False).load()

    if config.ocean.spatial_filter_scale is not None:
        scale = config.ocean.spatial_filter_scale
        logging.info("Applying scale-%d boxcar filter to 3D ocean fields", scale)
        for name in ocean.data_vars:
            if _LEVEL_SUFFIX.search(str(name)) and "Coarsened" in str(name):
                ocean[name] = ocean[name].copy(
                    data=_boxcar_smooth(ocean[name].values, scale)
                )

    atmosphere, ocean = finalize(
        atmosphere.squeeze(drop=True), ocean.squeeze(drop=True), masks, config
    )
    time = _parse_timestamp(timestamp_override or timestamp, config.time.calendar)
    logging.info("%s -> initial condition at %s", tag, time.isoformat())

    if not config.keep_intermediate:
        for path in paths.values():
            if os.path.exists(path):
                os.remove(path)
    return atmosphere, ocean, time


def _write(dataset: xr.Dataset, path: str, config: CreateRestartICConfig) -> None:
    encoding = {
        name: {"_FillValue": np.float32(np.nan), "dtype": "float32"}
        for name in dataset.data_vars
    }
    encoding["time"] = {
        "units": config.time.units,
        "calendar": config.time.calendar,
        "dtype": "float64",
    }
    dataset.to_netcdf(path, encoding=encoding)
    logging.info(
        "Wrote %s (%d variables, %d times)",
        path,
        len(dataset.data_vars),
        dataset.sizes["time"],
    )


def run(config: CreateRestartICConfig) -> None:
    directories = config.resolved_restart_directories
    logging.info("Processing %d restart director(ies)", len(directories))
    os.makedirs(config.output_directory, exist_ok=True)

    masks = None
    if config.masks.path is not None:
        logging.info("Reading masks and surface fractions: %s", config.masks.path)
        masks = xr.open_dataset(config.masks.path, decode_times=False)
        masks = masks.isel(time=0, drop=True) if "time" in masks.dims else masks
        masks = masks.load()

    owns_work_directory = config.work_directory is None
    work_directory = config.work_directory or tempfile.mkdtemp(prefix="e3sm-ic-")
    os.makedirs(work_directory, exist_ok=True)

    atmospheres: list[xr.Dataset] = []
    oceans: list[xr.Dataset] = []
    try:
        for index, directory in enumerate(directories):
            files = RestartFiles.find(directory)
            override = (
                config.time.timestamps[index]
                if config.time.source == "explicit"
                else None
            )
            atmosphere, ocean, time = process_restart(
                files, config, masks, work_directory, override
            )
            time_coordinate = xr.DataArray([time], dims=["time"], name="time")
            atmospheres.append(atmosphere.expand_dims(time=time_coordinate))
            oceans.append(ocean.expand_dims(time=time_coordinate))

            if not config.stack:
                stamp = time.strftime("%Y-%m-%d-%H%M%S")
                prefix = f"{config.output_prefix}-{stamp}"
                _write_pair(atmospheres.pop(), oceans.pop(), prefix, config)
    finally:
        if owns_work_directory and not config.keep_intermediate:
            shutil.rmtree(work_directory, ignore_errors=True)

    if config.stack:
        _write_pair(
            xr.concat(atmospheres, dim="time"),
            xr.concat(oceans, dim="time"),
            config.output_prefix,
            config,
        )


def _write_pair(
    atmosphere: xr.Dataset,
    ocean: xr.Dataset,
    prefix: str,
    config: CreateRestartICConfig,
) -> None:
    for dataset, component in ((atmosphere, "atmosphere"), (ocean, "ocean")):
        path = os.path.join(config.output_directory, f"{prefix}_{component}_ic.nc")
        if os.path.exists(path) and not config.overwrite:
            raise ValueError(f"{path} exists; set overwrite: true to replace it.")
        _write(dataset, path, config)


@click.command()
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to the YAML config, e.g. configs/e3smv3-restart-ic.yaml.",
)
@click.option("--restart-glob", default=None, help="Override restart_glob.")
@click.option("--output-directory", default=None, help="Override output_directory.")
@click.option("--output-prefix", default=None, help="Override output_prefix.")
@click.option("--overwrite/--no-overwrite", default=None, help="Override overwrite.")
def main(
    config_path: str,
    restart_glob: str | None,
    output_directory: str | None,
    output_prefix: str | None,
    overwrite: bool | None,
) -> None:
    """Create SamudrACE-E3SMv3 initial conditions from E3SM restart files."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    config = CreateRestartICConfig.from_file(
        config_path,
        restart_glob=restart_glob,
        output_directory=output_directory,
        output_prefix=output_prefix,
        overwrite=overwrite,
    )
    run(config)


if __name__ == "__main__":
    main()
