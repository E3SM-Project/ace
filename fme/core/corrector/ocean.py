import dataclasses
import datetime
from collections.abc import Mapping
from typing import Any, Literal, Protocol

import torch

from fme.core.atmosphere_data import AtmosphereData
from fme.core.constants import (
    DENSITY_OF_SEA_WATER_CM4,
    FREEZING_TEMPERATURE_KELVIN,
    LATENT_HEAT_OF_VAPORIZATION,
    SPECIFIC_HEAT_OF_SEA_WATER_CM4,
)
from fme.core.coordinates import DepthCoordinate, dz_from_idepth
from fme.core.corrector.registry import (
    Correction,
    CorrectionSequence,
    CorrectorConfigABC,
)
from fme.core.corrector.state import CorrectorState
from fme.core.corrector.utils import ForcePositive, replace_value_keep_gradient
from fme.core.dataset_info import DatasetInfo
from fme.core.gridded_ops import GriddedOperations
from fme.core.ocean_data import HasOceanDepthIntegral, HasOceanLayerGeometry, OceanData
from fme.core.registry.corrector import CorrectorSelector
from fme.core.typing_ import TensorDict, TensorMapping


class AreaWeightedMean(Protocol):
    def __call__(
        self, data: torch.Tensor, keepdim: bool = False, name: str | None = None
    ) -> torch.Tensor: ...


@dataclasses.dataclass
class SeaIceFractionConfig:
    """Correct predicted sea_ice_fraction to ensure it is always in 0-1, and
    land_fraction + sea_ice_fraction + ocean_fraction = 1. After
    sea_ice_fraction is corrected, all variables listed in
    zero_where_ice_free_names will be set to 0 everywhere
    sea_ice_fraction is 0.

    Parameters:
        sea_ice_fraction_name: Name of the sea ice fraction variable.
        land_fraction_name: Name of the land fraction variable.
        zero_where_ice_free_names: List of variable names to set to 0
            wherever sea_ice_fraction is 0.
        remove_negative_ocean_fraction: If True, reduce sea_ice_fraction
            to prevent ocean_fraction (1 - sea_ice_fraction - land_fraction)
            from being negative.
    """

    sea_ice_fraction_name: str
    land_fraction_name: str
    zero_where_ice_free_names: list[str] = dataclasses.field(default_factory=list)
    remove_negative_ocean_fraction: bool = True

    def __call__(
        self,
        gen_data: TensorMapping,
        input_data: TensorMapping,
        keep_gradient: bool = False,
    ) -> TensorDict:
        """
        Returns:
            A ``TensorDict`` containing only the fields modified by this
            correction (the sea ice fraction and the fields zeroed where
            ice-free).
        """
        out: TensorDict = {}
        sif = gen_data[self.sea_ice_fraction_name]
        clamped_sif = torch.clamp(sif, min=0.0, max=1.0)
        if keep_gradient:
            clamped_sif = replace_value_keep_gradient(sif, clamped_sif)
        out[self.sea_ice_fraction_name] = clamped_sif
        if self.remove_negative_ocean_fraction:
            negative_ocean_fraction = (
                1
                - out[self.sea_ice_fraction_name]
                - input_data[self.land_fraction_name]
            )
            negative_ocean_fraction = negative_ocean_fraction.clip(max=0)
            rebalanced_sif = out[self.sea_ice_fraction_name] + negative_ocean_fraction
            if keep_gradient:
                rebalanced_sif = replace_value_keep_gradient(
                    out[self.sea_ice_fraction_name], rebalanced_sif
                )
            out[self.sea_ice_fraction_name] = rebalanced_sif
        for name in self.zero_where_ice_free_names:
            out[name] = gen_data[name] * (out[self.sea_ice_fraction_name] > 0.0)
        return out


OceanHeatContentMethod = Literal["scaled_temperature", "additive_profile"]


@dataclasses.dataclass
class OceanHeatContentBudgetConfig:
    """Configuration for ocean heat content budget correction.

    Both methods make the area-weighted global mean column heat content of the
    prediction equal that of the input plus the time-integrated global mean
    net energy flux into the ocean (plus ``constant_unaccounted_heating``).
    They differ in how the global residual is distributed:

    - "scaled_temperature" multiplies the predicted potential temperature (in
      degrees Celsius) at every level and grid point by one global factor.
      The temperature change is proportional to the temperature itself, so
      it vanishes near 0 degC and changes sign below it.
    - "additive_profile" adds a temperature increment
      ``delta_T[i, k] = R * w[i, k] / sum(C * w)``, where
      ``R = H_input + dt * P_net - H_raw`` is the global heat residual [J],
      ``C[i, k] = density * specific_heat * wet_volume[i, k]`` the heat
      capacity of each wet cell [J/K] and ``w`` a nonnegative, dimensionless
      vertical profile (``profile``). The increment has the same sign as the
      residual everywhere, independent of the predicted temperature. Each
      column receives heat in proportion to its profile-weighted heat
      capacity; per-column surface-flux closure is deliberately not enforced,
      because ocean transport redistributes heat horizontally.

    Ocean heat content here is sensible heat of liquid sea water only. Neither
    method conserves the enthalpy of ocean plus sea ice.

    Parameters:
        method: Method to use for OHC budget correction, "scaled_temperature"
            or "additive_profile" (see above).
        constant_unaccounted_heating: Area-weighted global mean
            column-integrated heating in W/m**2 to be added to the energy flux
            into the ocean when conserving the heat content. This can be useful
            for correcting errors in heat budget in target data. The same
            additional heating is imposed at all time steps and grid cells.
        density: Sea water density in kg/m**3 used for the heat content and
            heat capacity. The default is the CM4 value; E3SM/MPAS-Ocean uses
            1026.
        specific_heat: Specific heat of sea water in J/kg/K used for the
            heat content and heat capacity. The default is the CM4 value;
            E3SM/MPAS-Ocean uses 3996.
        wet_volume: Source of the wet layer thickness. "coordinate" uses the
            depth coordinate's layer thickness, which includes partial bottom
            cells when the dataset provides ``deptho`` (the existing
            behaviour). "full_cells" uses the full interface spacing on every
            wet layer (``idepth`` and the layer mask only), ignoring
            ``deptho``.
        profile: Vertical profile ``w`` for "additive_profile": "exponential"
            (``exp(-z / e_folding_depth)`` averaged over each wet layer between
            its interfaces) or "uniform" (``w = 1`` on every wet layer, i.e. the
            same increment everywhere). Must be None for "scaled_temperature".
        e_folding_depth: E-folding depth in meters of the "exponential"
            profile. Required for it, and must be None otherwise.
    """

    method: OceanHeatContentMethod
    constant_unaccounted_heating: float = 0.0
    density: float = DENSITY_OF_SEA_WATER_CM4
    specific_heat: float = SPECIFIC_HEAT_OF_SEA_WATER_CM4
    wet_volume: Literal["coordinate", "full_cells"] = "coordinate"
    profile: Literal["exponential", "uniform"] | None = None
    e_folding_depth: float | None = None

    def __post_init__(self):
        if self.density <= 0 or self.specific_heat <= 0:
            raise ValueError(
                "density and specific_heat must be positive, got "
                f"{self.density} and {self.specific_heat}"
            )
        if self.method == "scaled_temperature":
            if self.profile is not None or self.e_folding_depth is not None:
                raise ValueError(
                    "profile and e_folding_depth apply only to the "
                    "'additive_profile' method"
                )
        elif self.method == "additive_profile":
            if self.profile is None:
                raise ValueError("the 'additive_profile' method requires a profile")
            if self.profile == "exponential":
                if self.e_folding_depth is None or self.e_folding_depth <= 0:
                    raise ValueError(
                        "the exponential profile requires a positive "
                        f"e_folding_depth, got {self.e_folding_depth}"
                    )
            elif self.e_folding_depth is not None:
                raise ValueError(
                    "e_folding_depth applies only to the exponential profile"
                )
        else:
            raise ValueError(f"unknown ocean heat content method {self.method!r}")


@dataclasses.dataclass
class SurfaceEnergyFluxCorrectionConfig:
    """Configuration for correcting the generated hfds using
    atmosphere-derived surface energy fluxes and ocean_fraction.

    The net_flux is the net surface energy flux computed from atmospheric
    forcing variables and generated SST. The ocean_fraction naturally zeroes
    out the correction on land and reduces it under sea ice.

    Available options are:
      - "residual_prediction": corrected_hfds = gen_hfds + ocean_fraction * net_flux.
        The network predicts a residual that is added to the forcing-derived flux.
      - "prescribed": corrected_hfds = net_flux * ocean_fraction + gen_hfds *
        (1 - ocean_fraction). Open-ocean hfds is prescribed from forcings; the
        network prediction is retained under sea ice and on land.

    Parameters:
        method: Method to use for the correction.

    """

    method: Literal["residual_prediction", "prescribed"]


@dataclasses.dataclass
class SeaIceFractionCorrection:
    """Correction that enforces sea-ice-fraction constraints.

    Wraps ``SeaIceFractionConfig`` so the corrector applies the operation
    without reading config fields. ``forcing_data`` and ``corrector_state`` are
    unused and passed through.

    If ``keep_gradient`` is True, the clamp and rebalance are applied with a
    straight-through estimator so out-of-range cells still get a learning signal.
    """

    config: SeaIceFractionConfig
    keep_gradient: bool = False

    def __call__(
        self,
        input_data: TensorMapping,
        gen_data: TensorMapping,
        forcing_data: TensorMapping,
        corrector_state: CorrectorState | None,
    ) -> tuple[TensorDict, CorrectorState | None]:
        """
        Returns:
            A tuple whose ``TensorDict`` contains only the fields modified by
            this correction (the sea ice fraction and the fields zeroed where
            ice-free). ``SeaIceFractionConfig.__call__`` already returns only
            those fields, preserving the straight-through estimator when
            ``keep_gradient`` is set.
        """
        corrected = self.config(gen_data, input_data, keep_gradient=self.keep_gradient)
        return corrected, corrector_state


@dataclasses.dataclass
class SurfaceEnergyFluxCorrection:
    """Correction that adjusts hfds using atmosphere-derived surface fluxes."""

    method: Literal["residual_prediction", "prescribed"]

    def __call__(
        self,
        input_data: TensorMapping,
        gen_data: TensorMapping,
        forcing_data: TensorMapping,
        corrector_state: CorrectorState | None,
    ) -> tuple[TensorDict, CorrectorState | None]:
        """
        Returns:
            A tuple whose ``TensorDict`` contains only the field modified by this
            correction (the net downward surface heat flux, ``hfds``).
        """
        corrected = _correct_hfds(
            input_data,
            gen_data,
            forcing_data,
            method=self.method,
        )
        return corrected, corrector_state


@dataclasses.dataclass
class OceanHeatContentCorrection:
    """Correction that conserves ocean heat content."""

    area_weighted_mean: AreaWeightedMean
    vertical_coordinate: HasOceanDepthIntegral | None
    timestep_seconds: float
    config: OceanHeatContentBudgetConfig

    def __call__(
        self,
        input_data: TensorMapping,
        gen_data: TensorMapping,
        forcing_data: TensorMapping,
        corrector_state: CorrectorState | None,
    ) -> tuple[TensorDict, CorrectorState | None]:
        """
        Returns:
            A tuple whose ``TensorDict`` contains only the fields modified by
            this correction (the potential temperature at every depth level, and
            the sea surface temperature when present).
        """
        if self.vertical_coordinate is None:
            raise ValueError(
                "Ocean heat content correction is turned on, but no vertical "
                "coordinate is available."
            )
        column_integral = _get_column_integral(
            self.vertical_coordinate, self.config.wet_volume
        )
        if self.config.method == "additive_profile":
            corrected = _additive_profile_ocean_heat_content_correction(
                input_data,
                gen_data,
                forcing_data,
                self.area_weighted_mean,
                column_integral,
                _get_layer_geometry(self.vertical_coordinate, self.config.wet_volume),
                self.timestep_seconds,
                self.config,
            )
        else:
            corrected = _force_conserve_ocean_heat_content(
                input_data,
                gen_data,
                forcing_data,
                self.area_weighted_mean,
                column_integral,
                self.timestep_seconds,
                self.config.method,
                self.config.constant_unaccounted_heating,
                density=self.config.density,
                specific_heat=self.config.specific_heat,
            )
        return corrected, corrector_state


@CorrectorSelector.register("ocean_corrector")
@dataclasses.dataclass
class OceanCorrectorConfig(CorrectorConfigABC):
    """Configuration for corrections applied to generated ocean data.

    Parameters:
        force_positive_names: Names of fields that should be forced to be greater
            than or equal to zero.
        sea_ice_fraction_correction: Optional configuration for a sea-ice-fraction
            correction (bounds sea_ice_fraction to 0-1 and keeps the land, ocean,
            and sea-ice fractions summing to one).
        surface_energy_flux_correction: Optional configuration for a surface energy
            flux correction to the generated hfds.
        ocean_heat_content_correction: Optional configuration for an ocean heat
            content correction.
        keep_gradient_through_clamps: If True, apply the corrector's hard clamps
            (the ``force_positive_names`` clamp and the
            ``sea_ice_fraction_correction`` bound/rebalance) with a straight-through
            estimator: the forward value is still clamped, but gradient flows as if
            the clamp had not happened, so out-of-range cells still get a learning
            signal.
    """

    force_positive_names: list[str] = dataclasses.field(default_factory=list)
    sea_ice_fraction_correction: SeaIceFractionConfig | None = None
    surface_energy_flux_correction: SurfaceEnergyFluxCorrectionConfig | None = None
    ocean_heat_content_correction: OceanHeatContentBudgetConfig | None = None
    keep_gradient_through_clamps: bool = False

    @classmethod
    def remove_deprecated_keys(cls, state: Mapping[str, Any]) -> dict[str, Any]:
        state_copy = dict(state)
        if "masking" in state_copy:
            del state_copy["masking"]
        if "ocean_heat_content_correction" in state_copy and isinstance(
            state_copy["ocean_heat_content_correction"], bool
        ):
            if state_copy["ocean_heat_content_correction"]:
                state_copy["ocean_heat_content_correction"] = (
                    OceanHeatContentBudgetConfig(method="scaled_temperature")
                )
            else:
                state_copy["ocean_heat_content_correction"] = None
        if "sea_ice_fraction_correction" in state_copy:
            sif = state_copy["sea_ice_fraction_correction"]
            if isinstance(sif, dict) and "sea_ice_thickness_name" in sif:
                thickness_name = sif.pop("sea_ice_thickness_name")
                names = sif.setdefault("zero_where_ice_free_names", [])
                # The deprecated key and its replacement coexisted in configs
                # written while the rename was in flight, so appending
                # unconditionally duplicates the name on every such config.
                if thickness_name is not None and thickness_name not in names:
                    names.append(thickness_name)
        return state_copy

    def _get_corrector(
        self,
        dataset_info: DatasetInfo,
    ) -> "OceanCorrector":
        return self._build(
            dataset_info.gridded_operations,
            dataset_info.ocean_vertical_coordinate,
            dataset_info.timestep,
        )

    def _build(
        self,
        gridded_operations: GriddedOperations,
        vertical_coordinate: HasOceanDepthIntegral | None,
        timestep: datetime.timedelta,
    ) -> "OceanCorrector":
        area_weighted_mean = gridded_operations.area_weighted_mean
        timestep_seconds = timestep.total_seconds()
        corrections: list[Correction] = []
        if len(self.force_positive_names) > 0:
            corrections.append(
                ForcePositive(
                    self.force_positive_names,
                    keep_gradient=self.keep_gradient_through_clamps,
                )
            )
        if self.sea_ice_fraction_correction is not None:
            corrections.append(
                SeaIceFractionCorrection(
                    self.sea_ice_fraction_correction,
                    keep_gradient=self.keep_gradient_through_clamps,
                )
            )
        if self.surface_energy_flux_correction is not None:
            corrections.append(
                SurfaceEnergyFluxCorrection(self.surface_energy_flux_correction.method)
            )
        if self.ocean_heat_content_correction is not None:
            corrections.append(
                OceanHeatContentCorrection(
                    area_weighted_mean,
                    vertical_coordinate,
                    timestep_seconds,
                    self.ocean_heat_content_correction,
                )
            )
        return OceanCorrector(corrections)


class OceanCorrector(CorrectionSequence):
    pass


def _compute_ocean_net_surface_energy_flux(
    forcing_data: TensorMapping,
    sst: torch.Tensor,
) -> torch.Tensor:
    """Compute the net surface energy flux into the ocean from atmospheric
    forcing variables and the sea surface temperature.

    This extends the atmosphere net surface energy flux with SST-dependent
    heat transport by precipitation and evaporation.
    """
    atmos = AtmosphereData(forcing_data)
    base_flux = (
        atmos.net_surface_energy_flux
    )  # missing: - calving * LATENT_HEAT_OF_FREEZING
    mass_heat_flux = (
        SPECIFIC_HEAT_OF_SEA_WATER_CM4
        * (
            atmos.precipitation_rate
            + atmos.frozen_precipitation_rate
            - (atmos.latent_heat_flux / LATENT_HEAT_OF_VAPORIZATION)
        )  # missing: + river runoff + calving
        * (sst - FREEZING_TEMPERATURE_KELVIN)
    )
    return base_flux + mass_heat_flux


def _correct_hfds(
    input_data: TensorMapping,
    gen_data: TensorMapping,
    forcing_data: TensorMapping,
    method: Literal["residual_prediction", "prescribed"],
) -> TensorDict:
    """Apply surface energy flux correction to the generated hfds.

    The ocean_fraction naturally zeroes the correction on land and reduces
    it under sea ice.

    Methods:
        residual_prediction: gen_hfds + ocean_fraction * net_flux
        prescribed: net_flux * ocean_fraction + gen_hfds * (1 - ocean_fraction)
    """
    input = OceanData(input_data)
    forcing = OceanData(forcing_data)
    ocean_fraction = input.ocean_fraction
    net_flux = _compute_ocean_net_surface_energy_flux(
        forcing_data, input.sea_surface_temperature
    )
    out: TensorDict = {}
    if "hfds" in gen_data:
        hfds_name = "hfds"
    else:
        hfds_name = "hfds_total_area"
        net_flux = net_flux * forcing.sea_surface_fraction
    gen_hfds = gen_data[hfds_name]
    if method == "residual_prediction":
        out[hfds_name] = net_flux * ocean_fraction + gen_hfds
    elif method == "prescribed":
        out[hfds_name] = net_flux * ocean_fraction + gen_hfds * (1 - ocean_fraction)
    else:
        raise NotImplementedError(
            f"Method {method!r} not implemented for surface energy flux correction"
        )
    return out


def _get_column_integral(
    vertical_coordinate: HasOceanDepthIntegral,
    wet_volume: Literal["coordinate", "full_cells"],
) -> HasOceanDepthIntegral:
    """The depth integral that defines column heat content for the correction."""
    if wet_volume == "coordinate":
        return vertical_coordinate
    idepth, mask, _ = _get_layer_geometry(vertical_coordinate, wet_volume)
    return DepthCoordinate(idepth, mask)


def _get_layer_geometry(
    vertical_coordinate: HasOceanDepthIntegral,
    wet_volume: Literal["coordinate", "full_cells"],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return interface depths, layer mask and wet layer thickness [m].

    The thickness is the coordinate's own (partial bottom cells included when
    it has them) for "coordinate", and the full interface spacing on wet
    layers for "full_cells".
    """
    # The corrector receives its vertical coordinate through the generic
    # HasOceanDepthIntegral protocol of DatasetInfo; only the settings that
    # need layer geometry (the additive profile and the full-cell volumes)
    # narrow it here, so the protocol shared with every other ocean consumer
    # stays unchanged.
    if not isinstance(vertical_coordinate, HasOceanLayerGeometry):
        raise ValueError(
            "This ocean heat content correction needs a depth coordinate with "
            "layer interfaces and a mask (idepth, mask, dz), got "
            f"{type(vertical_coordinate).__name__}."
        )
    idepth = vertical_coordinate.idepth
    mask = vertical_coordinate.mask
    if wet_volume == "coordinate":
        dz = vertical_coordinate.dz
    else:
        dz = dz_from_idepth(idepth, mask)
    return idepth, mask, dz


def _column_heat_content(
    potential_temperature: torch.Tensor,
    column_integral: HasOceanDepthIntegral,
    density: float,
    specific_heat: float,
) -> torch.Tensor:
    """Column-integrated heat content [J/m**2] of a temperature in degC,
    NaN on dry columns. With the CM4 defaults this matches
    ``OceanData.ocean_heat_content`` bit for bit.
    """
    return column_integral.depth_integral(
        potential_temperature * specific_heat * density
    )


def _global_mean_net_energy_flux_into_ocean(
    input: OceanData,
    gen: OceanData,
    forcing: OceanData,
    area_weighted_mean: AreaWeightedMean,
) -> torch.Tensor:
    """Area-weighted global mean net energy flux into the ocean [W/m**2] per
    unit total cell area, applying the sea surface fraction exactly once.

    A flux already normalized by total cell area (``hfds_total_area``) is not
    multiplied by the sea surface fraction again; only the geothermal flux is.
    """
    try:
        # First priority: pre-weighted heat flux in gen_data
        net_energy_flux_into_ocean = (
            gen.net_downward_surface_heat_flux_total_area
            + forcing.geothermal_heat_flux * forcing.sea_surface_fraction
        )
    except KeyError:
        try:
            # Second priority: standard heat flux in gen_data
            net_energy_flux_into_ocean = (
                gen.net_downward_surface_heat_flux + forcing.geothermal_heat_flux
            ) * forcing.sea_surface_fraction
        except KeyError:
            # Third priority: standard heat flux in input_data
            net_energy_flux_into_ocean = (
                input.net_downward_surface_heat_flux + forcing.geothermal_heat_flux
            ) * forcing.sea_surface_fraction
    return area_weighted_mean(
        net_energy_flux_into_ocean,
        keepdim=True,
        name="ocean_heat_content",
    )


def _check_heat_flux_sources(gen_data: TensorMapping, forcing_data: TensorMapping):
    if "hfds" in gen_data and "hfds" in forcing_data:
        raise ValueError(
            "Net downward surface heat flux cannot be present in both gen_data and "
            "forcing_data."
        )


def _force_conserve_ocean_heat_content(
    input_data: TensorMapping,
    gen_data: TensorMapping,
    forcing_data: TensorMapping,
    area_weighted_mean: AreaWeightedMean,
    vertical_coordinate: HasOceanDepthIntegral,
    timestep_seconds: float,
    method: OceanHeatContentMethod = "scaled_temperature",
    unaccounted_heating: float = 0.0,
    density: float = DENSITY_OF_SEA_WATER_CM4,
    specific_heat: float = SPECIFIC_HEAT_OF_SEA_WATER_CM4,
) -> TensorDict:
    if method != "scaled_temperature":
        raise NotImplementedError(
            f"Method {method!r} not implemented for ocean heat content conservation"
        )
    _check_heat_flux_sources(gen_data, forcing_data)
    input = OceanData(input_data, vertical_coordinate)
    gen = OceanData(gen_data, vertical_coordinate)
    forcing = OceanData(forcing_data)
    global_gen_ocean_heat_content = area_weighted_mean(
        _column_heat_content(
            gen.sea_water_potential_temperature,
            vertical_coordinate,
            density,
            specific_heat,
        ),
        keepdim=True,
        name="ocean_heat_content",
    )
    global_input_ocean_heat_content = area_weighted_mean(
        _column_heat_content(
            input.sea_water_potential_temperature,
            vertical_coordinate,
            density,
            specific_heat,
        ),
        keepdim=True,
        name="ocean_heat_content",
    )
    energy_flux_global_mean = _global_mean_net_energy_flux_into_ocean(
        input, gen, forcing, area_weighted_mean
    )
    expected_change_ocean_heat_content = (
        energy_flux_global_mean + unaccounted_heating
    ) * timestep_seconds
    heat_content_correction_ratio = (
        global_input_ocean_heat_content + expected_change_ocean_heat_content
    ) / global_gen_ocean_heat_content
    # apply same temperature correction to all vertical layers
    out: TensorDict = {}
    n_levels = gen.sea_water_potential_temperature.shape[-1]
    for k in range(n_levels):
        name = f"thetao_{k}"
        out[name] = gen.data[name] * heat_content_correction_ratio
    if "sst" in gen.data:
        out["sst"] = (  # assuming sst in Kelvin
            gen.data["sst"] - FREEZING_TEMPERATURE_KELVIN
        ) * heat_content_correction_ratio + FREEZING_TEMPERATURE_KELVIN
    return out


def _vertical_profile(
    idepth: torch.Tensor,
    mask: torch.Tensor,
    dz: torch.Tensor,
    profile: Literal["exponential", "uniform"],
    e_folding_depth: float | None,
) -> torch.Tensor:
    """Dimensionless, nonnegative layer profile ``w`` (vertical last).

    For "exponential" it is the mean of ``exp(-z / e_folding_depth)`` over the
    wet part of each layer, from its upper interface ``idepth[k]`` down by its
    wet thickness ``dz``, i.e. the exact layer integral divided by ``dz``; a
    zero-thickness wet layer takes the value at its upper interface. ``w`` is
    zero on dry layers.
    """
    wet = mask > 0
    if profile == "uniform":
        return wet.to(dz.dtype)
    if e_folding_depth is None:
        raise ValueError("the exponential profile requires an e_folding_depth")
    z_top = idepth[:-1].to(dz.dtype)
    top = torch.exp(-z_top / e_folding_depth)
    bottom = torch.exp(-(z_top + dz) / e_folding_depth)
    safe_dz = torch.where(dz > 0, dz, torch.ones_like(dz))
    w = torch.where(dz > 0, e_folding_depth * (top - bottom) / safe_dz, top)
    return torch.where(wet, w, torch.zeros_like(w))


def _additive_profile_ocean_heat_content_correction(
    input_data: TensorMapping,
    gen_data: TensorMapping,
    forcing_data: TensorMapping,
    area_weighted_mean: AreaWeightedMean,
    column_integral: HasOceanDepthIntegral,
    layer_geometry: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    timestep_seconds: float,
    config: OceanHeatContentBudgetConfig,
) -> TensorDict:
    """Add ``delta_T = R * w / sum(C * w)`` to every wet level.

    All global sums are area-weighted means over the same masked area
    (``name="ocean_heat_content"``), so the normalization cancels in the ratio
    and ``R`` and ``sum(C * w)`` are per unit of that area. The new global mean
    heat content equals ``H_input + dt * (P_net + unaccounted)`` to round-off,
    including under spatial parallelism, where the area-weighted mean is the
    distributed reduction.
    """
    if config.profile is None:
        raise ValueError("the 'additive_profile' method requires a profile")
    _check_heat_flux_sources(gen_data, forcing_data)
    input = OceanData(input_data)
    gen = OceanData(gen_data)
    forcing = OceanData(forcing_data)
    rho, cp = config.density, config.specific_heat
    idepth, mask, dz = layer_geometry
    gen_temperature = gen.sea_water_potential_temperature

    def global_mean(column: torch.Tensor) -> torch.Tensor:
        return area_weighted_mean(column, keepdim=True, name="ocean_heat_content")

    global_input_heat = global_mean(
        _column_heat_content(
            input.sea_water_potential_temperature, column_integral, rho, cp
        )
    )
    global_gen_heat = global_mean(
        _column_heat_content(gen_temperature, column_integral, rho, cp)
    )
    energy_flux_global_mean = _global_mean_net_energy_flux_into_ocean(
        input, gen, forcing, area_weighted_mean
    )
    residual = (
        global_input_heat
        + (energy_flux_global_mean + config.constant_unaccounted_heating)
        * timestep_seconds
        - global_gen_heat
    )  # J/m**2
    w = _vertical_profile(idepth, mask, dz, config.profile, config.e_folding_depth)
    w = w.to(dtype=gen_temperature.dtype, device=gen_temperature.device)
    profile_heat_capacity = global_mean(
        _column_heat_content(w, column_integral, rho, cp)
    )  # J/K/m**2
    delta_t = (residual / profile_heat_capacity).unsqueeze(-1) * w
    out: TensorDict = {}
    for k in range(gen_temperature.shape[-1]):
        name = f"thetao_{k}"
        out[name] = gen.data[name] + delta_t[..., k]
    if "sst" in gen.data:
        # SST (in K) changes with the top level, as for scaled_temperature
        out["sst"] = gen.data["sst"] + delta_t[..., 0]
    return out
