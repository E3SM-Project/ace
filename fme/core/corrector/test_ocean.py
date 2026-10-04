import dataclasses
import datetime
from typing import Literal

import pytest
import torch

from fme import get_device
from fme.core.constants import DENSITY_OF_SEA_WATER_CM4, SPECIFIC_HEAT_OF_SEA_WATER_CM4
from fme.core.coordinates import DepthCoordinate
from fme.core.corrector.ocean import (
    OceanCorrectorConfig,
    OceanHeatContentBudgetConfig,
    OceanHeatContentMethod,
    SeaIceFractionConfig,
    SurfaceEnergyFluxCorrectionConfig,
    _compute_ocean_net_surface_energy_flux,
)
from fme.core.distributed import Distributed
from fme.core.gridded_ops import LatLonOperations
from fme.core.ocean_data import OceanData
from fme.core.spatial_mask_provider import SpatialMaskProvider
from fme.core.typing_ import TensorMapping

DEVICE = get_device()
IMG_SHAPE = (5, 5)
NZ = 2

_MASK = torch.ones(*IMG_SHAPE, NZ, device=DEVICE)
_LAT, _LON = 2, 2
_MASK[_LAT, _LON, :] = 0.0


class _MockDepth:
    def depth_integral(self, integrand: torch.Tensor) -> torch.Tensor:
        idepth = torch.tensor([0, 5, 15], device=DEVICE)
        thickness = idepth.diff(dim=-1)
        return torch.nansum(_MASK * integrand * thickness, dim=-1)


_VERTICAL_COORD = _MockDepth()


def test_ocean_corrector_force_positive():
    """"""
    torch.manual_seed(0)
    config = OceanCorrectorConfig(force_positive_names=["so_0", "so_1"])
    ops = LatLonOperations(torch.ones(size=IMG_SHAPE))
    timestep = datetime.timedelta(seconds=3600)
    corrector = config._build(ops, _VERTICAL_COORD, timestep)
    input_data = {f"so_{i}": torch.randn(IMG_SHAPE, device=DEVICE) for i in range(NZ)}
    input_data["sst"] = torch.randn(IMG_SHAPE, device=DEVICE)
    gen_data = {f"so_{i}": torch.randn(IMG_SHAPE, device=DEVICE) for i in range(NZ)}
    gen_data["sst"] = torch.randn(IMG_SHAPE, device=DEVICE)
    corrected_gen = corrector(input_data, gen_data, {}, None).corrected
    for name in ["so_0", "so_1"]:
        x = corrected_gen[name].clone()
        x[_LAT, _LON] = 0.0
        assert torch.all(x >= 0.0)


def test_sea_ice_fraction_keep_gradient_passes_gradient_through_clamp():
    config = SeaIceFractionConfig(
        sea_ice_fraction_name="sea_ice_fraction",
        land_fraction_name="land_fraction",
        remove_negative_ocean_fraction=False,
    )
    input_data = {"land_fraction": torch.zeros(IMG_SHAPE, device=DEVICE)}
    # values both below 0 and above 1 so the clamp saturates at both ends
    raw = torch.tensor([-0.5, 0.3, 1.5], device=DEVICE)

    sif_plain = raw.clone().requires_grad_(True)
    config({"sea_ice_fraction": sif_plain}, input_data)[
        "sea_ice_fraction"
    ].sum().backward()
    # plain clamp: zero gradient where saturated, one in the interior
    torch.testing.assert_close(
        sif_plain.grad, torch.tensor([0.0, 1.0, 0.0], device=DEVICE)
    )

    sif_ste = raw.clone().requires_grad_(True)
    out = config({"sea_ice_fraction": sif_ste}, input_data, keep_gradient=True)
    # forward value is still clamped to [0, 1]
    torch.testing.assert_close(
        out["sea_ice_fraction"], torch.tensor([0.0, 0.3, 1.0], device=DEVICE)
    )
    out["sea_ice_fraction"].sum().backward()
    torch.testing.assert_close(sif_ste.grad, torch.ones_like(raw))


def test_ocean_corrector_keep_gradient_through_clamps_forward_unchanged():
    # The straight-through flag must not change forward values; only gradients.
    torch.manual_seed(0)
    ops = LatLonOperations(torch.ones(size=IMG_SHAPE))
    timestep = datetime.timedelta(seconds=3600)
    sif = SeaIceFractionConfig(
        sea_ice_fraction_name="sea_ice_fraction",
        land_fraction_name="land_fraction",
    )
    input_data = {
        "land_fraction": torch.ones(IMG_SHAPE, device=DEVICE) * 0.3,
    }
    gen_data = {
        "so_0": torch.randn(IMG_SHAPE, device=DEVICE),
        "sea_ice_fraction": torch.randn(IMG_SHAPE, device=DEVICE),
    }
    baseline = (
        OceanCorrectorConfig(
            force_positive_names=["so_0"], sea_ice_fraction_correction=sif
        )
        ._build(ops, None, timestep)(input_data, gen_data, {}, None)
        .corrected
    )
    ste = (
        OceanCorrectorConfig(
            force_positive_names=["so_0"],
            sea_ice_fraction_correction=sif,
            keep_gradient_through_clamps=True,
        )
        ._build(ops, None, timestep)(input_data, gen_data, {}, None)
        .corrected
    )
    for name in baseline:
        torch.testing.assert_close(baseline[name], ste[name])


def test_ocean_corrector_has_no_negative_ocean_fraction():
    config = OceanCorrectorConfig(
        sea_ice_fraction_correction=SeaIceFractionConfig(
            sea_ice_fraction_name="sea_ice_fraction",
            land_fraction_name="land_fraction",
        ),
    )
    ops = LatLonOperations(torch.ones(size=IMG_SHAPE))
    timestep = datetime.timedelta(seconds=3600)
    input_data = {f"so_{i}": torch.randn(IMG_SHAPE, device=DEVICE) for i in range(NZ)}
    input_data["sst"] = torch.randn(IMG_SHAPE, device=DEVICE)
    input_data["land_fraction"] = torch.ones(IMG_SHAPE, device=DEVICE) * 0.8
    gen_data = {f"so_{i}": torch.randn(IMG_SHAPE, device=DEVICE) for i in range(NZ)}
    gen_data["sst"] = torch.randn(IMG_SHAPE, device=DEVICE)
    gen_data["sea_ice_fraction"] = torch.randn(IMG_SHAPE, device=DEVICE) * 0.5
    gen_data["sea_ice_fraction"][_LAT, _LON] = -0.5
    corrector = config._build(ops, None, timestep)
    violation = (input_data["land_fraction"] + gen_data["sea_ice_fraction"]) > 1.0
    assert violation.any()
    negative_sea_ice_fraction = gen_data["sea_ice_fraction"] < 0.0
    assert negative_sea_ice_fraction.any()

    next_step_input_data: TensorMapping = {}
    gen_data_corrected = corrector(
        input_data, gen_data, next_step_input_data, None
    ).corrected
    corrected_violation = (
        input_data["land_fraction"] + gen_data_corrected["sea_ice_fraction"]
    ) > 1.0
    assert not corrected_violation.any()
    assert not (gen_data_corrected["sea_ice_fraction"] < 0.0).any()


def test_ocean_corrector_has_negative_ocean_fraction():
    config = OceanCorrectorConfig(
        sea_ice_fraction_correction=SeaIceFractionConfig(
            sea_ice_fraction_name="sea_ice_fraction",
            land_fraction_name="land_fraction",
            remove_negative_ocean_fraction=False,
        ),
    )
    ops = LatLonOperations(torch.ones(size=IMG_SHAPE))
    timestep = datetime.timedelta(seconds=3600)
    input_data = {f"so_{i}": torch.randn(IMG_SHAPE, device=DEVICE) for i in range(NZ)}
    input_data["sst"] = torch.randn(IMG_SHAPE, device=DEVICE)
    input_data["land_fraction"] = torch.ones(IMG_SHAPE, device=DEVICE) * 0.8
    gen_data = {f"so_{i}": torch.randn(IMG_SHAPE, device=DEVICE) for i in range(NZ)}
    gen_data["sst"] = torch.randn(IMG_SHAPE, device=DEVICE)
    gen_data["sea_ice_fraction"] = torch.randn(IMG_SHAPE, device=DEVICE) * 0.5
    gen_data["sea_ice_fraction"][_LAT, _LON] = -0.5
    corrector = config._build(ops, None, timestep)
    violation = (input_data["land_fraction"] + gen_data["sea_ice_fraction"]) > 1.0
    assert violation.any()
    negative_sea_ice_fraction = gen_data["sea_ice_fraction"] < 0.0
    assert negative_sea_ice_fraction.any()

    next_step_input_data: TensorMapping = {}
    gen_data_corrected = corrector(
        input_data, gen_data, next_step_input_data, None
    ).corrected
    corrected_violation = (
        input_data["land_fraction"] + gen_data_corrected["sea_ice_fraction"]
    ) > 1.0
    assert corrected_violation.any()
    # sea_ice_fraction values are still clamped to [0, 1]
    assert not (gen_data_corrected["sea_ice_fraction"] < 0.0).any()


def test_zero_where_ice_free_names():
    config = OceanCorrectorConfig(
        sea_ice_fraction_correction=SeaIceFractionConfig(
            sea_ice_fraction_name="sea_ice_fraction",
            land_fraction_name="land_fraction",
            zero_where_ice_free_names=["HI"],
        ),
    )
    ops = LatLonOperations(torch.ones(size=IMG_SHAPE))
    timestep = datetime.timedelta(seconds=3600)
    input_data = {"land_fraction": torch.ones(IMG_SHAPE, device=DEVICE)}
    input_data["land_fraction"][:3, :3] = torch.rand(3, 3, device=DEVICE)
    gen_data = {
        "sea_ice_fraction": torch.rand(IMG_SHAPE, device=DEVICE),
        "HI": torch.rand(IMG_SHAPE, device=DEVICE) * 10,
    }
    corrector = config._build(ops, None, timestep)
    gen_data_corrected = corrector(input_data, gen_data, {}, None).corrected
    sea_ice_zero = gen_data_corrected["sea_ice_fraction"] == 0.0
    thickness = gen_data_corrected["HI"]
    torch.testing.assert_close(
        torch.where(sea_ice_zero, thickness, 0.0), torch.zeros_like(thickness)
    )


def test_zero_where_ice_free_names_multiple_variables():
    config = OceanCorrectorConfig(
        sea_ice_fraction_correction=SeaIceFractionConfig(
            sea_ice_fraction_name="sea_ice_fraction",
            land_fraction_name="land_fraction",
            zero_where_ice_free_names=["HI", "HS"],
        ),
    )
    ops = LatLonOperations(torch.ones(size=IMG_SHAPE))
    timestep = datetime.timedelta(seconds=3600)
    input_data = {"land_fraction": torch.ones(IMG_SHAPE, device=DEVICE)}
    input_data["land_fraction"][:3, :3] = torch.rand(3, 3, device=DEVICE)
    gen_data = {
        "sea_ice_fraction": torch.rand(IMG_SHAPE, device=DEVICE),
        "HI": torch.rand(IMG_SHAPE, device=DEVICE) * 10,
        "HS": torch.rand(IMG_SHAPE, device=DEVICE) * 5,
    }
    corrector = config._build(ops, None, timestep)
    gen_data_corrected = corrector(input_data, gen_data, {}, None).corrected
    sea_ice_zero = gen_data_corrected["sea_ice_fraction"] == 0.0
    for name in ["HI", "HS"]:
        values = gen_data_corrected[name]
        torch.testing.assert_close(
            torch.where(sea_ice_zero, values, 0.0), torch.zeros_like(values)
        )


def test_from_state_migrates_sea_ice_thickness_name():
    state = {
        "sea_ice_fraction_correction": {
            "sea_ice_fraction_name": "ocean_sea_ice_fraction",
            "land_fraction_name": "land_fraction",
            "sea_ice_thickness_name": "HI",
            "remove_negative_ocean_fraction": False,
        },
    }
    config = OceanCorrectorConfig.from_state(state)
    assert config.sea_ice_fraction_correction is not None
    assert config.sea_ice_fraction_correction.zero_where_ice_free_names == ["HI"]


def test_from_state_migrates_sea_ice_thickness_name_already_listed():
    # The deprecated key and its replacement coexisted in the e3sm_hist ocean
    # configs, so the migration must not append a name the list already has.
    state = {
        "sea_ice_fraction_correction": {
            "sea_ice_fraction_name": "ocean_sea_ice_fraction",
            "land_fraction_name": "land_fraction",
            "sea_ice_thickness_name": "HI",
            "zero_where_ice_free_names": ["HI"],
            "remove_negative_ocean_fraction": False,
        },
    }
    config = OceanCorrectorConfig.from_state(state)
    assert config.sea_ice_fraction_correction is not None
    assert config.sea_ice_fraction_correction.zero_where_ice_free_names == ["HI"]


def test_from_state_migrates_sea_ice_thickness_name_none():
    state = {
        "sea_ice_fraction_correction": {
            "sea_ice_fraction_name": "ocean_sea_ice_fraction",
            "land_fraction_name": "land_fraction",
            "sea_ice_thickness_name": None,
            "remove_negative_ocean_fraction": False,
        },
    }
    config = OceanCorrectorConfig.from_state(state)
    assert config.sea_ice_fraction_correction is not None
    assert config.sea_ice_fraction_correction.zero_where_ice_free_names == []


def _make_atmos_forcing_data(shape, device=DEVICE):
    """Build atmosphere forcing tensors needed for the surface energy flux
    correction tests."""
    return {
        "DSWRFsfc": torch.full(shape, 200.0, device=device),
        "USWRFsfc": torch.full(shape, 50.0, device=device),
        "DLWRFsfc": torch.full(shape, 300.0, device=device),
        "ULWRFsfc": torch.full(shape, 350.0, device=device),
        "LHTFLsfc": torch.full(shape, 100.0, device=device),
        "SHTFLsfc": torch.full(shape, 20.0, device=device),
        "PRATEsfc": torch.full(shape, 1e-4, device=device),
        "total_frozen_precipitation_rate": torch.full(shape, 1e-5, device=device),
    }


def test_surface_energy_flux_correction_resid():
    config = OceanCorrectorConfig(
        surface_energy_flux_correction=SurfaceEnergyFluxCorrectionConfig(
            method="residual_prediction"
        ),
    )
    ops = LatLonOperations(torch.ones(size=IMG_SHAPE))
    timestep = datetime.timedelta(seconds=3600)
    corrector = config._build(ops, None, timestep)

    sst = torch.full(IMG_SHAPE, 300.0, device=DEVICE)
    gen_hfds = torch.full(IMG_SHAPE, 5.0, device=DEVICE)
    sea_ice_fraction = torch.zeros(IMG_SHAPE, device=DEVICE)
    sea_ice_fraction[0, :] = 0.3
    land_fraction = torch.zeros(IMG_SHAPE, device=DEVICE)
    land_fraction[-1, :] = 1.0

    gen_data = {
        "sst": sst,
        "hfds": gen_hfds,
        "sea_ice_fraction": sea_ice_fraction,
    }
    forcing_data = {
        "land_fraction": land_fraction,
        **_make_atmos_forcing_data(IMG_SHAPE),
    }
    input_data = {**forcing_data, **gen_data}

    ocean_fraction = 1 - land_fraction - sea_ice_fraction
    expected_net_flux = _compute_ocean_net_surface_energy_flux(input_data, sst)
    expected_hfds = gen_hfds + ocean_fraction * expected_net_flux

    corrected = corrector(input_data, gen_data, forcing_data, None).corrected
    torch.testing.assert_close(corrected["hfds"], expected_hfds)
    # on land ocean_fraction is 0, so hfds is unchanged
    torch.testing.assert_close(corrected["hfds"][-1, :], gen_hfds[-1, :])
    # with sea ice, correction is reduced relative to ice-free rows
    ice_row_correction = (corrected["hfds"][0, 0] - gen_hfds[0, 0]).abs()
    open_row_correction = (corrected["hfds"][1, 0] - gen_hfds[1, 0]).abs()
    assert ice_row_correction < open_row_correction


def test_surface_energy_flux_correction_prescribed():
    config = OceanCorrectorConfig(
        surface_energy_flux_correction=SurfaceEnergyFluxCorrectionConfig(
            method="prescribed"
        ),
    )
    ops = LatLonOperations(torch.ones(size=IMG_SHAPE))
    timestep = datetime.timedelta(seconds=3600)
    corrector = config._build(ops, None, timestep)

    sst = torch.full(IMG_SHAPE, 300.0, device=DEVICE)
    gen_hfds = torch.full(IMG_SHAPE, 5.0, device=DEVICE)
    sea_ice_fraction = torch.zeros(IMG_SHAPE, device=DEVICE)
    sea_ice_fraction[0, :] = 0.3
    land_fraction = torch.zeros(IMG_SHAPE, device=DEVICE)
    land_fraction[-1, :] = 1.0

    gen_data = {
        "sst": sst,
        "hfds": gen_hfds,
        "sea_ice_fraction": sea_ice_fraction,
    }
    forcing_data = {
        "land_fraction": land_fraction,
        **_make_atmos_forcing_data(IMG_SHAPE),
    }
    input_data = {**forcing_data, **gen_data}

    ocean_fraction = 1 - land_fraction - sea_ice_fraction
    net_flux = _compute_ocean_net_surface_energy_flux(input_data, sst)
    expected_hfds = net_flux * ocean_fraction + gen_hfds * (1 - ocean_fraction)

    corrected = corrector(input_data, gen_data, forcing_data, None).corrected
    torch.testing.assert_close(corrected["hfds"], expected_hfds)
    # on land (ocean_fraction=0), hfds equals gen_hfds
    torch.testing.assert_close(corrected["hfds"][-1, :], gen_hfds[-1, :])
    # in open ocean (no ice, no land), hfds equals net_flux
    open_ocean_row = 1
    torch.testing.assert_close(
        corrected["hfds"][open_ocean_row, :], net_flux[open_ocean_row, :]
    )


@pytest.mark.parametrize(
    "hfds_type",
    [
        pytest.param("input", id="hfds_in_input"),
        pytest.param("gen", id="hfds_in_gen"),
        pytest.param("total_area", id="hfds_total_area_in_gen"),
    ],
)
def test_ocean_heat_content_correction(hfds_type):
    config = OceanCorrectorConfig(
        ocean_heat_content_correction=OceanHeatContentBudgetConfig(
            method="scaled_temperature",
            constant_unaccounted_heating=0.1,
        )
    )
    timestep = datetime.timedelta(seconds=5 * 24 * 3600)
    nsamples, nlat, nlon, nlevels = 4, 3, 3, 2
    mask = torch.ones(nsamples, nlat, nlon, nlevels)
    mask[:, 0, 0, 0] = 0.0
    mask[:, 0, 0, 1] = 0.0
    mask[:, 0, 1, 1] = 0.0
    masks = {
        "mask_0": mask[:, :, :, 0],
        "mask_1": mask[:, :, :, 1],
        "mask_2d": mask[:, :, :, 0],
    }
    spatial_mask_provider = SpatialMaskProvider(masks)
    ops = LatLonOperations(torch.ones(size=[3, 3]), spatial_mask_provider)

    idepth = torch.tensor([2.5, 10, 20])
    depth_coordinate = DepthCoordinate(idepth, mask)

    sea_surface_fraction = mask[:, :, :, 0]

    input_data_dict = {
        "thetao_0": torch.ones(nsamples, nlat, nlon),
        "thetao_1": torch.ones(nsamples, nlat, nlon),
        "sst": torch.ones(nsamples, nlat, nlon) + 273.15,
    }
    gen_data_dict = {
        "thetao_0": torch.ones(nsamples, nlat, nlon) * 2,
        "thetao_1": torch.ones(nsamples, nlat, nlon) * 2,
        "sst": torch.ones(nsamples, nlat, nlon) * 2 + 273.15,
    }
    if hfds_type == "gen":
        gen_data_dict["hfds"] = torch.ones(nsamples, nlat, nlon)
    elif hfds_type == "total_area":
        # hfds_total_area is already weighted by sea_surface_fraction also
        # include hfds with a different value to verify hfds_total_area takes
        # priority
        gen_data_dict["hfds"] = (
            torch.ones(nsamples, nlat, nlon) * 100
        )  # should be ignored
        gen_data_dict["hfds_total_area"] = (
            torch.ones(nsamples, nlat, nlon) * sea_surface_fraction
        )
    else:
        input_data_dict["hfds"] = torch.ones(nsamples, nlat, nlon)
    forcing_data_dict = {
        "hfgeou": torch.ones(nsamples, nlat, nlon),
        "sea_surface_fraction": sea_surface_fraction,
    }
    input_data = OceanData(input_data_dict, depth_coordinate)
    gen_data = OceanData(gen_data_dict, depth_coordinate)
    corrector = config._build(ops, depth_coordinate, timestep)
    result = corrector(input_data_dict, gen_data_dict, forcing_data_dict, None)
    gen_data_corrected_dict = result.corrected

    # the OHC correction writes every potential-temperature level and the SST;
    # the heat-flux fields are read but not written, so they stay out of the set
    assert set(result.modified_names) == {"thetao_0", "thetao_1", "sst"}
    for name, delta in result.diagnostics.delta.items():
        torch.testing.assert_close(
            delta, result.corrected[name] - gen_data_dict[name], equal_nan=True
        )

    input_ohc = input_data.ocean_heat_content.nanmean(dim=(-1, -2), keepdim=True)
    gen_ohc = gen_data.ocean_heat_content.nanmean(dim=(-1, -2), keepdim=True)
    torch.testing.assert_close(
        gen_ohc,
        input_ohc * 2,
        equal_nan=True,
    )
    ohc_change = (
        2.1 * timestep.total_seconds()
    )  # 2.1 because of hfds + hfgeou + unaccounted heating
    corrector_ratio = (input_ohc + ohc_change) / gen_ohc
    expected_gen_data_dict = {
        key: value * corrector_ratio if key.startswith("thetao") else value
        for key, value in gen_data_dict.items()
    }
    expected_gen_data_dict["sst"] = (
        gen_data_dict["sst"] - 273.15
    ) * corrector_ratio + 273.15

    torch.testing.assert_close(
        gen_data_corrected_dict["sst"],
        expected_gen_data_dict["sst"],
    )

    expected_gen_data = OceanData(expected_gen_data_dict, depth_coordinate)
    gen_data_corrected = OceanData(gen_data_corrected_dict, depth_coordinate)
    torch.testing.assert_close(
        expected_gen_data.ocean_heat_content,
        gen_data_corrected.ocean_heat_content,
        equal_nan=True,
    )


def test_ocean_corrector_config_fields_are_known():
    # Staleness guard: if a new corrector option is added to
    # OceanCorrectorConfig this fails, flagging that the corrector delta/
    # modified-return tests need to exercise it.
    expected = {
        "force_positive_names",
        "sea_ice_fraction_correction",
        "surface_energy_flux_correction",
        "ocean_heat_content_correction",
        "keep_gradient_through_clamps",
        "corrector_disabled_epochs",  # inherited epoch-scheduling field
    }
    actual = {f.name for f in dataclasses.fields(OceanCorrectorConfig)}
    assert actual == expected, (
        "OceanCorrectorConfig fields changed; update the corrector delta tests "
        f"to cover the new option(s): {actual ^ expected}"
    )


def test_ocean_corrector_delta_matches_modified_returns():
    torch.manual_seed(0)
    config = OceanCorrectorConfig(
        force_positive_names=["so_0"],
        sea_ice_fraction_correction=SeaIceFractionConfig(
            sea_ice_fraction_name="sea_ice_fraction",
            land_fraction_name="land_fraction",
            zero_where_ice_free_names=["HI", "HS"],
        ),
    )
    ops = LatLonOperations(torch.ones(size=IMG_SHAPE))
    timestep = datetime.timedelta(seconds=3600)
    corrector = config._build(ops, None, timestep)
    input_data = {"land_fraction": torch.rand(IMG_SHAPE, device=DEVICE)}
    gen_data = {
        "so_0": torch.randn(IMG_SHAPE, device=DEVICE),
        "so_1": torch.randn(IMG_SHAPE, device=DEVICE),  # uncorrected field
        "sea_ice_fraction": torch.rand(IMG_SHAPE, device=DEVICE),
        "HI": torch.rand(IMG_SHAPE, device=DEVICE) * 10,
        "HS": torch.rand(IMG_SHAPE, device=DEVICE) * 5,
    }
    result = corrector(input_data, gen_data, {}, None)
    # delta keys are exactly the corrector's modified names
    assert set(result.diagnostics.delta) == set(result.modified_names)
    for name, delta in result.diagnostics.delta.items():
        torch.testing.assert_close(delta, result.corrected[name] - gen_data[name])
    assert set(result.modified_names) == {"so_0", "sea_ice_fraction", "HI", "HS"}
    # the uncorrected field passes through unchanged and is absent from the set
    assert "so_1" not in result.modified_names
    torch.testing.assert_close(result.corrected["so_1"], gen_data["so_1"])


def test_ocean_corrector_empty_delta_when_nothing_modified():
    # A corrector with no field-modifying option emits an empty delta and an
    # unchanged copy of gen_data.
    ops = LatLonOperations(torch.ones(size=IMG_SHAPE))
    timestep = datetime.timedelta(seconds=3600)
    corrector = OceanCorrectorConfig()._build(ops, None, timestep)
    gen_data = {"so_0": torch.randn(IMG_SHAPE, device=DEVICE)}
    result = corrector({}, gen_data, {}, None)
    assert dict(result.diagnostics.delta) == {}
    assert set(result.modified_names) == set()
    torch.testing.assert_close(result.corrected["so_0"], gen_data["so_0"])


# --- additive_profile ocean heat content correction ---------------------------

_OHC_IDEPTH = [0.0, 20.0, 50.0, 230.0, 1020.0]
_OHC_TIMESTEP = datetime.timedelta(days=5)
_OHC_RHO, _OHC_CP = 1026.0, 3996.0  # E3SM / MPAS-Ocean


@dataclasses.dataclass
class _OHCCase:
    area: torch.Tensor  # (lat, lon)
    mask: torch.Tensor  # (lat, lon, nz)
    depth: DepthCoordinate
    ops: LatLonOperations
    input_data: dict[str, torch.Tensor]
    gen_data: dict[str, torch.Tensor]
    forcing_data: dict[str, torch.Tensor]


def _make_ohc_case(
    net_flux: float,
    gen_offset: float = 0.0,
    subzero_column: tuple[int, int] | None = None,
    deptho: torch.Tensor | None = None,
    hfds_location: str = "input",
    gen_dry_value: float = float("nan"),
    nlat: int = 4,
    nlon: int = 6,
    seed: int = 0,
) -> _OHCCase:
    """A small float64 ocean with a land column, two shallow columns, a partial
    sea surface fraction and NaN on dry input cells, as in the real data.
    ``gen_dry_value`` fills dry cells of the prediction (the network's raw
    output is finite there before output masking)."""
    generator = torch.Generator().manual_seed(seed)
    dtype = torch.float64
    nz = len(_OHC_IDEPTH) - 1
    mask = torch.ones(nlat, nlon, nz, dtype=dtype)
    mask[0, 0, :] = 0.0  # land
    mask[1, 1, 2:] = 0.0  # 50 m deep
    mask[2, 3, 3:] = 0.0  # 230 m deep
    if deptho is not None:  # no wet layer starts below the sea floor
        mask = mask * (torch.tensor(_OHC_IDEPTH[:-1], dtype=dtype) < deptho[..., None])
    area = torch.linspace(0.5, 1.0, nlat, dtype=dtype)[:, None].expand(nlat, nlon)
    masks = {f"mask_{k}": mask[..., k] for k in range(nz)}
    masks["mask_2d"] = mask[..., 0]
    ops = LatLonOperations(area.clone(), SpatialMaskProvider(masks))
    depth = DepthCoordinate(torch.tensor(_OHC_IDEPTH, dtype=dtype), mask, deptho)
    wet = mask > 0
    input_temperature = 2.0 + 18.0 * torch.rand(
        nlat, nlon, nz, generator=generator, dtype=dtype
    )
    noise = 0.1 * torch.randn(nlat, nlon, nz, generator=generator, dtype=dtype)
    gen_temperature = input_temperature + gen_offset + noise
    if subzero_column is not None:
        input_temperature[subzero_column] = -1.5
        gen_temperature[subzero_column] = -1.5
    nan = torch.tensor(float("nan"), dtype=dtype)
    input_temperature = torch.where(wet, input_temperature, nan)
    gen_temperature = torch.where(
        wet, gen_temperature, torch.tensor(gen_dry_value, dtype=dtype)
    )

    def level_dict(temperature: torch.Tensor) -> dict[str, torch.Tensor]:
        data = {f"thetao_{k}": temperature[None, ..., k] for k in range(nz)}
        data["sst"] = temperature[None, ..., 0] + 273.15
        return data

    input_data = level_dict(input_temperature)
    gen_data = level_dict(gen_temperature)
    sea_surface_fraction = mask[None, ..., 0] * (
        0.5 + 0.5 * torch.rand(1, nlat, nlon, generator=generator, dtype=dtype)
    )
    forcing_data = {
        "hfgeou": torch.full((1, nlat, nlon), 0.05, dtype=dtype),
        "sea_surface_fraction": sea_surface_fraction,
    }
    hfds = torch.full((1, nlat, nlon), net_flux, dtype=dtype)
    if hfds_location == "input":
        input_data["hfds"] = hfds
    elif hfds_location == "gen":
        gen_data["hfds"] = hfds
    else:
        gen_data["hfds_total_area"] = hfds * sea_surface_fraction
    return _OHCCase(area, mask, depth, ops, input_data, gen_data, forcing_data)


def _global_mean_heat(case: _OHCCase, data: TensorMapping, dz: torch.Tensor):
    """Area-weighted global mean column heat content [J/m**2], computed
    independently of the corrector."""
    nz = case.mask.shape[-1]
    temperature = torch.stack([data[f"thetao_{k}"] for k in range(nz)], dim=-1)
    column = (temperature * _OHC_RHO * _OHC_CP * dz).nansum(dim=-1)
    weights = case.area * case.mask[..., 0]
    return (column * weights).sum(dim=(-2, -1)) / weights.sum()


def _expected_flux(case: _OHCCase, net_flux: float) -> torch.Tensor:
    ssf = case.forcing_data["sea_surface_fraction"]
    weights = case.area * case.mask[..., 0]
    flux = (net_flux + 0.05) * ssf
    return (flux * weights).sum(dim=(-2, -1)) / weights.sum()


def _additive_config(
    profile: Literal["exponential", "uniform"] = "exponential",
    e_folding_depth: float = 150.0,
    wet_volume: Literal["coordinate", "full_cells"] = "coordinate",
    unaccounted: float = 0.0,
    method: OceanHeatContentMethod = "additive_profile",
) -> OceanCorrectorConfig:
    additive = method == "additive_profile"
    return OceanCorrectorConfig(
        ocean_heat_content_correction=OceanHeatContentBudgetConfig(
            method=method,
            constant_unaccounted_heating=unaccounted,
            density=_OHC_RHO,
            specific_heat=_OHC_CP,
            wet_volume=wet_volume,
            profile=profile if additive else None,
            e_folding_depth=(
                e_folding_depth if additive and profile == "exponential" else None
            ),
        )
    )


def _apply(config: OceanCorrectorConfig, case: _OHCCase) -> TensorMapping:
    corrector = config._build(case.ops, case.depth, _OHC_TIMESTEP)
    return corrector(case.input_data, case.gen_data, case.forcing_data, None).corrected


@pytest.mark.parametrize(
    "profile, e_folding_depth",
    [
        ("exponential", 75.0),
        ("exponential", 150.0),
        ("exponential", 300.0),
        ("uniform", 0.0),
    ],
)
@pytest.mark.parametrize(
    "net_flux, gen_offset",
    [
        pytest.param(40.0, -0.3, id="heat_added"),
        pytest.param(-40.0, 0.3, id="heat_removed"),
    ],
)
@pytest.mark.parametrize("hfds_location", ["input", "gen", "total_area"])
def test_additive_profile_closes_global_heat_budget(
    profile, e_folding_depth, net_flux, gen_offset, hfds_location
):
    case = _make_ohc_case(net_flux, gen_offset, hfds_location=hfds_location)
    unaccounted = -1.62
    corrected = _apply(
        _additive_config(profile, e_folding_depth, unaccounted=unaccounted), case
    )
    dt = _OHC_TIMESTEP.total_seconds()
    dz = case.depth.dz
    target = _global_mean_heat(case, case.input_data, dz) + dt * (
        _expected_flux(case, net_flux) + unaccounted
    )
    raw = _global_mean_heat(case, case.gen_data, dz)
    residual = target - raw
    assert torch.sign(residual).item() == (1.0 if net_flux > 0 else -1.0)
    corrected_heat = _global_mean_heat(case, {**case.gen_data, **corrected}, dz)
    torch.testing.assert_close(corrected_heat, target, rtol=1e-13, atol=0.0)


def test_additive_profile_warms_subzero_water_when_heat_is_added():
    subzero = (3, 4)
    case = _make_ohc_case(net_flux=60.0, gen_offset=-0.2, subzero_column=subzero)
    additive = _apply(_additive_config("exponential", 150.0), case)
    scaled = _apply(_additive_config(method="scaled_temperature"), case)
    for k in range(case.mask.shape[-1]):
        raw = case.gen_data[f"thetao_{k}"][(0, *subzero)]
        assert raw < 0
        assert additive[f"thetao_{k}"][(0, *subzero)] > raw
        # the multiplicative method cools sub-zero water while adding heat
        assert scaled[f"thetao_{k}"][(0, *subzero)] < raw


@pytest.mark.parametrize("profile", ["exponential", "uniform"])
def test_additive_profile_leaves_dry_cells_unchanged(profile):
    case = _make_ohc_case(net_flux=30.0, gen_offset=-0.5, gen_dry_value=7.0)
    corrected = _apply(_additive_config(profile), case)
    for k in range(case.mask.shape[-1]):
        dry = case.mask[..., k] == 0
        assert dry.any()
        change = (corrected[f"thetao_{k}"] - case.gen_data[f"thetao_{k}"])[0]
        assert (change[dry] == 0).all()
        assert (change[~dry] > 0).all()
    land = case.mask[..., 0] == 0
    torch.testing.assert_close(
        corrected["sst"][0][land], case.gen_data["sst"][0][land], rtol=0, atol=0
    )


def test_additive_profile_keeps_sst_consistent_with_top_level():
    case = _make_ohc_case(net_flux=-25.0, gen_offset=0.4)
    corrected = _apply(_additive_config("exponential", 75.0), case)
    torch.testing.assert_close(
        corrected["sst"] - case.gen_data["sst"],
        corrected["thetao_0"] - case.gen_data["thetao_0"],
        equal_nan=True,
    )


@pytest.mark.parametrize("e_folding_depth", [75.0, 150.0, 300.0])
def test_additive_profile_heat_follows_integrated_exponential(e_folding_depth):
    """Per layer, the added heat in a full-depth column is proportional to the
    integral of exp(-z / L) between the layer interfaces (not its midpoint
    value), so the column total is L * (1 - exp(-H / L))."""
    case = _make_ohc_case(net_flux=50.0)
    corrected = _apply(_additive_config("exponential", e_folding_depth), case)
    idepth = torch.tensor(_OHC_IDEPTH, dtype=torch.float64)
    full_column = (3, 5)
    nz = case.mask.shape[-1]
    delta = torch.stack(
        [
            (corrected[f"thetao_{k}"] - case.gen_data[f"thetao_{k}"])[(0, *full_column)]
            for k in range(nz)
        ]
    )
    layer_heat = delta * idepth.diff()
    exp_integral = e_folding_depth * (
        torch.exp(-idepth[:-1] / e_folding_depth)
        - torch.exp(-idepth[1:] / e_folding_depth)
    )
    torch.testing.assert_close(
        layer_heat / layer_heat.sum(), exp_integral / exp_integral.sum()
    )
    torch.testing.assert_close(
        exp_integral.sum(),
        e_folding_depth * (1 - torch.exp(-idepth[-1] / e_folding_depth)),
    )


def test_additive_uniform_profile_adds_the_same_increment_everywhere():
    case = _make_ohc_case(net_flux=50.0)
    corrected = _apply(_additive_config("uniform"), case)
    increments = torch.cat(
        [
            (corrected[f"thetao_{k}"] - case.gen_data[f"thetao_{k}"])[0][
                case.mask[..., k] > 0
            ]
            for k in range(case.mask.shape[-1])
        ]
    )
    torch.testing.assert_close(increments, increments[:1].expand_as(increments))


@pytest.mark.parametrize("wet_volume", ["coordinate", "full_cells"])
def test_additive_profile_closes_with_partial_bottom_cells(wet_volume):
    deptho = torch.full((4, 6), 1020.0, dtype=torch.float64)
    deptho[3, 2] = 35.0  # partial second layer, layers below dry
    deptho[2, 2] = 600.0  # partial deepest layer
    case = _make_ohc_case(net_flux=20.0, gen_offset=-0.1, deptho=deptho)
    full_cell_dz = DepthCoordinate(case.depth.idepth, case.mask).dz
    assert not torch.equal(full_cell_dz, case.depth.dz)
    dz = case.depth.dz if wet_volume == "coordinate" else full_cell_dz
    corrected = _apply(_additive_config("exponential", 150.0, wet_volume), case)
    dt = _OHC_TIMESTEP.total_seconds()
    target = _global_mean_heat(case, case.input_data, dz) + dt * _expected_flux(
        case, 20.0
    )
    corrected_heat = _global_mean_heat(case, {**case.gen_data, **corrected}, dz)
    torch.testing.assert_close(corrected_heat, target, rtol=1e-13, atol=0.0)


def test_additive_profile_requires_layer_geometry():
    case = _make_ohc_case(net_flux=10.0)
    corrector = _additive_config()._build(case.ops, _VERTICAL_COORD, _OHC_TIMESTEP)
    with pytest.raises(ValueError, match="layer interfaces"):
        corrector(case.input_data, case.gen_data, case.forcing_data, None)


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"method": "additive_profile"}, id="additive_without_profile"),
        pytest.param(
            {"method": "additive_profile", "profile": "exponential"},
            id="exponential_without_depth",
        ),
        pytest.param(
            {
                "method": "additive_profile",
                "profile": "uniform",
                "e_folding_depth": 150.0,
            },
            id="uniform_with_depth",
        ),
        pytest.param(
            {"method": "scaled_temperature", "profile": "uniform"},
            id="scaled_with_profile",
        ),
        pytest.param(
            {"method": "scaled_temperature", "density": 0.0}, id="zero_density"
        ),
    ],
)
def test_ocean_heat_content_config_validation(kwargs):
    with pytest.raises(ValueError):
        OceanHeatContentBudgetConfig(**kwargs)


@pytest.mark.parametrize(
    "state",
    [
        pytest.param(
            {
                "ocean_heat_content_correction": {
                    "method": "scaled_temperature",
                    "constant_unaccounted_heating": -1.62,
                }
            },
            id="dict",
        ),
        pytest.param({"ocean_heat_content_correction": True}, id="deprecated_bool"),
    ],
)
def test_old_ocean_heat_content_config_loads_with_unchanged_behaviour(state):
    config = OceanCorrectorConfig.from_state(state)
    ohc = config.ocean_heat_content_correction
    assert ohc is not None
    assert ohc.method == "scaled_temperature"
    assert ohc.density == DENSITY_OF_SEA_WATER_CM4
    assert ohc.specific_heat == SPECIFIC_HEAT_OF_SEA_WATER_CM4
    assert ohc.wet_volume == "coordinate"
    assert ohc.profile is None and ohc.e_folding_depth is None
    # the scaled_temperature result is the closed form on OceanData's
    # (CM4-constant) heat content, i.e. what the method computed before the
    # constants became configurable
    case = _make_ohc_case(net_flux=10.0, gen_offset=0.3, hfds_location="input")
    corrected = _apply(config, case)
    weights = case.area * case.mask[..., 0]

    def mean_ohc(data):
        column = OceanData(data, case.depth).ocean_heat_content
        column = torch.where(weights > 0, column, 0.0)
        return (column * weights).sum() / weights.sum()

    unaccounted = ohc.constant_unaccounted_heating
    ratio = (
        mean_ohc(case.input_data)
        + _OHC_TIMESTEP.total_seconds() * (_expected_flux(case, 10.0) + unaccounted)
    ) / mean_ohc(case.gen_data)
    torch.testing.assert_close(
        corrected["thetao_2"], case.gen_data["thetao_2"] * ratio, equal_nan=True
    )


@pytest.mark.parallel
def test_additive_profile_matches_global_computation_under_spatial_parallelism():
    """Each rank holds a spatial tile; the distributed area-weighted means must
    give the increment computed from the global arrays."""
    dist = Distributed.get_instance()
    nlat, nlon = 8, 12
    case = _make_ohc_case(net_flux=35.0, gen_offset=-0.2, nlat=nlat, nlon=nlon)
    config = _additive_config("exponential", 150.0)
    # global reference, no distributed reductions
    dt = _OHC_TIMESTEP.total_seconds()
    dz = case.depth.dz
    residual = (
        _global_mean_heat(case, case.input_data, dz)
        + dt * _expected_flux(case, 35.0)
        - _global_mean_heat(case, case.gen_data, dz)
    )
    idepth = case.depth.idepth
    w = (
        150.0
        * (torch.exp(-idepth[:-1] / 150.0) - torch.exp(-idepth[1:] / 150.0))
        / idepth.diff()
    ) * case.mask
    weights = case.area * case.mask[..., 0]
    capacity = ((w * dz).sum(-1) * _OHC_RHO * _OHC_CP * weights).sum() / weights.sum()
    expected_delta = residual[0] / capacity * w  # (lat, lon, nz)
    # local tile
    h, ww = dist.get_local_slices((nlat, nlon))
    local_mask = case.mask[h, ww]
    masks = {f"mask_{k}": local_mask[..., k] for k in range(local_mask.shape[-1])}
    masks["mask_2d"] = local_mask[..., 0]
    device = get_device()
    ops = LatLonOperations(case.area.clone(), SpatialMaskProvider(masks))
    depth = DepthCoordinate(case.depth.idepth, local_mask).to(device)

    def local(data):
        return {k: v[..., h, ww].to(device) for k, v in data.items()}

    corrector = config._build(ops, depth, _OHC_TIMESTEP)
    gen_local = local(case.gen_data)
    corrected = corrector(
        local(case.input_data), gen_local, local(case.forcing_data), None
    ).corrected
    for k in range(case.mask.shape[-1]):
        torch.testing.assert_close(
            (corrected[f"thetao_{k}"] - gen_local[f"thetao_{k}"]).cpu()[0],
            torch.where(local_mask[..., k] > 0, expected_delta[h, ww, k], torch.nan),
            equal_nan=True,
        )
