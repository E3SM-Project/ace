import os

import numpy as np
import pytest
import xarray as xr
import yaml
from create_e3sm_restart_ic import (
    ATMOSPHERE_PROGNOSTIC_NAMES,
    OCEAN_PROGNOSTIC_NAMES,
    AtmosphereConfig,
    CreateRestartICConfig,
    MapsConfig,
    MasksConfig,
    OceanConfig,
    RestartFiles,
    TimeConfig,
    _conservative_depth_weights,
    _fill_horizontal,
    _fill_masked_gaps,
    _mask_below_bathymetry,
    _ocean_mask_name,
    _parse_timestamp,
    compute_pressure_thickness,
    reconstruct_cell_velocity,
    vertical_coarsen_atmosphere,
    vertical_coarsen_ocean,
)

DIRNAME = os.path.abspath(os.path.dirname(__file__))
RESTART_IC_CONFIG_YAML = os.path.join(DIRNAME, "configs", "e3smv3-restart-ic.yaml")


def _hexagonal_cell(n_edges: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One cell whose ``n_edges`` edge normals are evenly spread in azimuth."""
    angle_edge = np.linspace(0.0, np.pi, n_edges, endpoint=False)
    edges_on_cell = np.arange(1, n_edges + 1).reshape(1, n_edges)
    n_edges_on_cell = np.array([n_edges])
    return angle_edge, edges_on_cell, n_edges_on_cell


def _project(zonal: float, meridional: float, angle_edge: np.ndarray) -> np.ndarray:
    return zonal * np.cos(angle_edge) + meridional * np.sin(angle_edge)


def test_prognostic_name_counts():
    # The SamudrACE-E3SMv3 stepper has 38 atmosphere and 80 ocean prognostics.
    assert len(ATMOSPHERE_PROGNOSTIC_NAMES) == 38
    assert len(OCEAN_PROGNOSTIC_NAMES) == 80
    assert len(set(ATMOSPHERE_PROGNOSTIC_NAMES)) == len(ATMOSPHERE_PROGNOSTIC_NAMES)
    assert len(set(OCEAN_PROGNOSTIC_NAMES)) == len(OCEAN_PROGNOSTIC_NAMES)


@pytest.mark.parametrize("n_edges", [5, 6])
@pytest.mark.parametrize("velocity", [(1.5, -0.25), (0.0, 2.0), (-3.0, 0.0)])
def test_reconstruct_cell_velocity_recovers_uniform_flow(n_edges, velocity):
    angle_edge, edges_on_cell, n_edges_on_cell = _hexagonal_cell(n_edges)
    zonal, meridional = velocity
    normal_velocity = _project(zonal, meridional, angle_edge)[:, None]

    result_zonal, result_meridional = reconstruct_cell_velocity(
        normal_velocity,
        angle_edge,
        np.ones(n_edges),
        edges_on_cell,
        n_edges_on_cell,
    )
    np.testing.assert_allclose(result_zonal[0, 0], zonal, atol=1e-5)
    np.testing.assert_allclose(result_meridional[0, 0], meridional, atol=1e-5)


def test_reconstruct_cell_velocity_ignores_invalid_edges_per_level():
    """An edge that is dry at depth must not drag that level's velocity to zero."""
    angle_edge, edges_on_cell, n_edges_on_cell = _hexagonal_cell(6)
    zonal, meridional = 2.0, -1.0
    normal_velocity = np.repeat(
        _project(zonal, meridional, angle_edge)[:, None], 2, axis=1
    )
    # One edge is below the sea floor at the deeper level.
    normal_velocity[0, 1] = -1e34

    result_zonal, result_meridional = reconstruct_cell_velocity(
        normal_velocity,
        angle_edge,
        np.ones(6),
        edges_on_cell,
        n_edges_on_cell,
    )
    np.testing.assert_allclose(result_zonal[0, :], zonal, atol=1e-5)
    np.testing.assert_allclose(result_meridional[0, :], meridional, atol=1e-5)


def test_reconstruct_cell_velocity_nan_when_underdetermined():
    angle_edge, edges_on_cell, n_edges_on_cell = _hexagonal_cell(6)
    normal_velocity = np.full((6, 1), -1e34)
    normal_velocity[0, 0] = 1.0  # a single usable edge cannot determine (u, v)

    result_zonal, result_meridional = reconstruct_cell_velocity(
        normal_velocity, angle_edge, np.ones(6), edges_on_cell, n_edges_on_cell
    )
    assert np.isnan(result_zonal[0, 0])
    assert np.isnan(result_meridional[0, 0])


def test_conservative_depth_weights_partition_source_layers():
    source = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
    target = np.array([0.0, 20.0, 40.0])
    weights = _conservative_depth_weights(source, target).toarray()
    # Every source layer is fully accounted for exactly once.
    np.testing.assert_allclose(weights.sum(axis=1), 1.0)
    # The first two source layers land in the first target layer.
    np.testing.assert_allclose(weights[:, 0], [1.0, 1.0, 0.0, 0.0])
    np.testing.assert_allclose(weights[:, 1], [0.0, 0.0, 1.0, 1.0])


def test_conservative_depth_weights_splits_straddling_layer():
    weights = _conservative_depth_weights(
        np.array([0.0, 10.0, 20.0]), np.array([0.0, 5.0, 20.0])
    ).toarray()
    np.testing.assert_allclose(weights[0], [0.5, 0.5])
    np.testing.assert_allclose(weights[1], [0.0, 1.0])


def test_vertical_coarsen_ocean_is_thickness_weighted_mean():
    field = np.array([[1.0, 3.0]])
    resting_thickness = np.array([[10.0, 30.0]])
    weights = _conservative_depth_weights(
        np.array([0.0, 10.0, 40.0]), np.array([0.0, 40.0])
    )
    coarse = vertical_coarsen_ocean(field, resting_thickness, weights)
    np.testing.assert_allclose(coarse, [[(1.0 * 10.0 + 3.0 * 30.0) / 40.0]])


def test_vertical_coarsen_ocean_excludes_invalid_water_from_weights():
    """A masked level must be dropped, not treated as a zero-valued sample."""
    field = np.array([[np.nan, 3.0]])
    resting_thickness = np.array([[10.0, 30.0]])
    weights = _conservative_depth_weights(
        np.array([0.0, 10.0, 40.0]), np.array([0.0, 40.0])
    )
    coarse = vertical_coarsen_ocean(field, resting_thickness, weights)
    np.testing.assert_allclose(coarse, [[3.0]])


def test_vertical_coarsen_ocean_nan_where_no_valid_water():
    field = np.array([[1.0, np.nan]])
    # restingThickness is zero below the sea floor, as MPAS writes it.
    resting_thickness = np.array([[10.0, 0.0]])
    weights = _conservative_depth_weights(
        np.array([0.0, 10.0, 40.0]), np.array([0.0, 10.0, 40.0])
    )
    coarse = vertical_coarsen_ocean(field, resting_thickness, weights)
    np.testing.assert_allclose(coarse[0, 0], 1.0)
    assert np.isnan(coarse[0, 1])


def test_mask_below_bathymetry():
    field = np.array([[1.0, 2.0, -1e34, 4.0]])
    masked = _mask_below_bathymetry(field, np.array([1]), np.array([2]))
    np.testing.assert_allclose(masked[0, :2], [1.0, 2.0])
    assert np.isnan(masked[0, 2:]).all()


def test_compute_pressure_thickness_sums_to_surface_pressure():
    surface_pressure = xr.DataArray([1.0e5, 9.0e4], dims=["ncol"])
    hyai = xr.DataArray([0.0, 0.2, 0.0], dims=["ilev"])
    hybi = xr.DataArray([0.0, 0.3, 1.0], dims=["ilev"])
    thickness = compute_pressure_thickness(
        surface_pressure,
        hyai,
        hybi,
        reference_pressure=1.0e5,
        interface_dim="ilev",
        level_dim="lev",
    )
    assert thickness.dims == ("lev", "ncol")
    # Interfaces run from the model top (zero pressure) to the surface.
    np.testing.assert_allclose(thickness.sum("lev").values, surface_pressure.values)


def test_vertical_coarsen_atmosphere_is_mass_weighted():
    field = xr.DataArray([[1.0], [3.0], [10.0]], dims=["lev", "ncol"])
    thickness = xr.DataArray([[10.0], [30.0], [100.0]], dims=["lev", "ncol"])
    coarsened = vertical_coarsen_atmosphere(field, thickness, [[0, 2], [2, 3]], "lev")
    np.testing.assert_allclose(coarsened["0"].values, [(10.0 + 90.0) / 40.0])
    np.testing.assert_allclose(coarsened["1"].values, [10.0])


@pytest.mark.parametrize(
    "variable, expected",
    [
        ("temperatureCoarsened_5", "mask_5"),
        ("salinityCoarsened_18", "mask_18"),
        ("sst", "mask_2d"),
        ("ssh", "mask_2d"),
        ("ocean_sea_ice_fraction", "mask_ocean_sea_ice_fraction"),
        ("iceVolumeTotal", "mask_iceVolumeTotal"),
    ],
)
def test_ocean_mask_name(variable, expected):
    available = {"mask_2d", "mask_5", "mask_18"} | {
        "mask_ocean_sea_ice_fraction",
        "mask_iceVolumeTotal",
    }
    assert _ocean_mask_name(variable, available) == expected


def test_parse_timestamp_accepts_mpas_and_iso_forms():
    assert _parse_timestamp("0425-01-03T12:00:00", "noleap").year == 425
    assert _parse_timestamp("1940-01-01 00:00:00", "noleap").day == 1
    with pytest.raises(ValueError, match="YYYY-MM-DDTHH:MM:SS"):
        _parse_timestamp("1940-01-01", "noleap")


def test_atmosphere_config_rejects_gappy_coarsening_indices():
    with pytest.raises(ValueError, match="contiguous"):
        AtmosphereConfig(
            vertical_coarsening_indices=[
                [0, 25],
                [26, 38],  # skips level 25
                [38, 46],
                [46, 52],
                [52, 56],
                [56, 61],
                [61, 69],
                [69, 80],
            ]
        ).validate()


def test_atmosphere_config_rejects_wrong_number_of_layers():
    with pytest.raises(ValueError, match="8 entries"):
        AtmosphereConfig(vertical_coarsening_indices=[[0, 40], [40, 80]]).validate()


def test_ocean_config_rejects_wrong_number_of_interfaces():
    with pytest.raises(ValueError, match="20 entries"):
        OceanConfig(target_interface_levels=[0.0, 100.0]).validate()


def test_ocean_config_rejects_non_increasing_interfaces():
    levels = list(range(20))
    levels[5] = 2
    with pytest.raises(ValueError, match="increasing"):
        OceanConfig(target_interface_levels=levels).validate()


def test_time_config_requires_timestamps_when_explicit():
    with pytest.raises(ValueError, match="time.timestamps is empty"):
        TimeConfig(source="explicit").validate()


def test_time_config_rejects_unknown_source():
    with pytest.raises(ValueError, match="must be 'restart' or 'explicit'"):
        TimeConfig(source="guess").validate()


def _write_maps(tmp_path) -> MapsConfig:
    paths = []
    for name in ("atmosphere_map.nc", "ocean_map.nc"):
        path = tmp_path / name
        path.write_text("not a real map, only its existence is validated")
        paths.append(str(path))
    return MapsConfig(atmosphere=paths[0], ocean=paths[1])


def _masks_off() -> MasksConfig:
    return MasksConfig(
        apply_ocean_masks=False,
        use_for_surface_blend=False,
        fill_masked_gaps=False,
    )


def _make_restart_directory(tmp_path, name: str) -> str:
    directory = tmp_path / name
    directory.mkdir()
    for suffix in ("eam.i.0001-01-01", "mpaso.rst.0001-01-01", "mpassi.rst.0001-01-01"):
        (directory / f"case.{suffix}.nc").write_text("")
    return str(directory)


def test_restart_files_find(tmp_path):
    directory = _make_restart_directory(tmp_path, "0001-01-01-00000")
    files = RestartFiles.find(directory)
    assert files.eam.endswith("eam.i.0001-01-01.nc")
    assert files.mpaso.endswith("mpaso.rst.0001-01-01.nc")
    assert files.mpassi.endswith("mpassi.rst.0001-01-01.nc")


def test_restart_files_find_reports_missing_component(tmp_path):
    directory = tmp_path / "empty"
    directory.mkdir()
    with pytest.raises(ValueError, match="EAM initial"):
        RestartFiles.find(str(directory))


def test_config_rejects_unknown_option(tmp_path):
    path = tmp_path / "config.yaml"
    maps = _write_maps(tmp_path)
    path.write_text(
        yaml.safe_dump(
            {
                "output_directory": str(tmp_path / "out"),
                "maps": {"atmosphere": maps.atmosphere, "ocean": maps.ocean},
                "restart_directories": [_make_restart_directory(tmp_path, "rest")],
                "masks": {"apply_ocean_masks": False, "use_for_surface_blend": False},
                "output_prefx": "typo",
            }
        )
    )
    with pytest.raises(ValueError, match="Unknown option"):
        CreateRestartICConfig.from_file(str(path))


def test_config_requires_a_restart_directory(tmp_path):
    maps = _write_maps(tmp_path)
    with pytest.raises(ValueError, match="No restart directories found"):
        CreateRestartICConfig(
            output_directory=str(tmp_path / "out"),
            maps=maps,
            masks=_masks_off(),
            restart_glob=str(tmp_path / "does-not-exist" / "*"),
        )


def test_config_requires_one_timestamp_per_restart(tmp_path):
    maps = _write_maps(tmp_path)
    with pytest.raises(ValueError, match="1 restart directories"):
        CreateRestartICConfig(
            output_directory=str(tmp_path / "out"),
            maps=maps,
            masks=_masks_off(),
            restart_directories=[_make_restart_directory(tmp_path, "rest")],
            time=TimeConfig(source="explicit", timestamps=["0001-01-01T00:00:00"] * 2),
        )


def test_masks_config_requires_a_path_when_masking(tmp_path):
    maps = _write_maps(tmp_path)
    with pytest.raises(ValueError, match="masks.path is not set"):
        CreateRestartICConfig(
            output_directory=str(tmp_path / "out"),
            maps=maps,
            restart_directories=[_make_restart_directory(tmp_path, "rest")],
        )


def test_shipped_config_parses():
    """The config in configs/ must stay loadable as the options evolve."""
    with open(RESTART_IC_CONFIG_YAML, "r") as f:
        data = yaml.safe_load(f)
    # Paths in the shipped config are examples, so only the option names and
    # option-level validation are exercised here.
    for section, cls in (("atmosphere", AtmosphereConfig), ("ocean", OceanConfig)):
        config = cls(**data[section])
        config.validate()
    TimeConfig(**data["time"]).validate()
    assert set(data) <= {
        field
        for field in CreateRestartICConfig.__dataclass_fields__  # type: ignore[attr-defined]
    }


def _ocean_with_gaps(gaps: dict[str, list[tuple[int, int]]], shape=(6, 8)):
    """Ocean fields that are valid everywhere except at the listed points."""
    ocean = xr.Dataset()
    for name in OCEAN_PROGNOSTIC_NAMES:
        values = np.full(shape, 10.0, dtype=np.float32)
        for row, column in gaps.get(name, []):
            values[row, column] = np.nan
        ocean[name] = xr.DataArray(values, dims=["lat", "lon"])
    return ocean


def _all_wet_masks(shape=(6, 8)):
    masks = xr.Dataset()
    for name in ["mask_2d"] + [f"mask_{level}" for level in range(19)]:
        masks[name] = xr.DataArray(
            np.ones(shape, dtype=np.float32), dims=["lat", "lon"]
        )
    return masks


def test_fill_horizontal_spreads_across_the_longitude_seam():
    """Longitude is periodic, so column 0 may be filled from the last column."""
    values = np.full((3, 4), np.nan)
    values[:, -1] = 4.0
    wanted = np.zeros((3, 4), dtype=bool)
    wanted[:, 0] = True
    filled = _fill_horizontal(values, wanted)
    np.testing.assert_allclose(filled[:, 0], 4.0)


def test_fill_horizontal_leaves_points_outside_the_mask_alone():
    values = np.full((3, 4), np.nan)
    values[1, 1] = 7.0
    wanted = np.zeros((3, 4), dtype=bool)
    wanted[1, 2] = True
    filled = _fill_horizontal(values, wanted)
    assert filled[1, 2] == pytest.approx(7.0)
    assert np.isnan(filled[0, 0])


def test_fill_masked_gaps_takes_a_deep_gap_from_the_layer_above():
    """Masks are nested with depth, so the layer above is wet and is nearest."""
    ocean = _ocean_with_gaps({"temperatureCoarsened_5": [(2, 3)]})
    ocean["temperatureCoarsened_4"][2, 3] = 3.5
    filled = _fill_masked_gaps(ocean, _all_wet_masks())
    assert filled == 1
    assert ocean["temperatureCoarsened_5"].values[2, 3] == pytest.approx(3.5)


def test_fill_masked_gaps_falls_back_to_neighbours_at_the_surface():
    """There is no layer above level 0, so the gap is filled horizontally."""
    ocean = _ocean_with_gaps({"salinityCoarsened_0": [(2, 3)], "sst": [(1, 1)]})
    filled = _fill_masked_gaps(ocean, _all_wet_masks())
    assert filled == 2
    assert ocean["salinityCoarsened_0"].values[2, 3] == pytest.approx(10.0)
    assert ocean["sst"].values[1, 1] == pytest.approx(10.0)


def test_fill_masked_gaps_leaves_dry_points_missing():
    """Only points inside the wetmask are filled; land must stay NaN."""
    ocean = _ocean_with_gaps({"sst": [(1, 1), (4, 4)]})
    masks = _all_wet_masks()
    masks["mask_2d"][4, 4] = 0.0
    _fill_masked_gaps(ocean, masks)
    assert ocean["sst"].values[1, 1] == pytest.approx(10.0)
    assert np.isnan(ocean["sst"].values[4, 4])


def test_fill_masked_gaps_reports_a_field_it_cannot_fill():
    """A mask that covers a field valid nowhere is a configuration error."""
    ocean = _ocean_with_gaps({})
    ocean["sst"] = xr.DataArray(
        np.full((6, 8), np.nan, dtype=np.float32), dims=["lat", "lon"]
    )
    with pytest.raises(ValueError, match="could not be filled"):
        _fill_masked_gaps(ocean, _all_wet_masks())


def test_fill_masked_gaps_needs_the_wetmasks(tmp_path):
    with pytest.raises(ValueError, match="needs masks.apply_ocean_masks"):
        MasksConfig(
            path=None, apply_ocean_masks=False, use_for_surface_blend=False
        ).validate()


def test_ice_shelf_cavities_are_kept_by_default():
    """Excluding them leaves surface points NaN inside mask_2d, which is fatal."""
    assert OceanConfig().exclude_ice_shelf_cavities is False
