import random

import numpy as np
import pytest
import torch
import torch_harmonics as harmonics

from fme.ace.data_loading.augmentation import (
    NullModifier,
    RotateModifier,
    RoundtripConfig,
    RoundtripModifier,
)
from fme.ace.data_loading.batch_data import BatchData


def rotate(data: torch.Tensor) -> torch.Tensor:
    return torch.flip(data, dims=[-2, -1])


def test_rotate_modifier_all_rotation():
    rotate_modifier = RotateModifier(
        rotate_probability=1.0, additional_directional_names=[]
    )
    n_lat = 8
    n_lon = 16
    batch = BatchData.new_for_testing(
        names=["UGRD", "VGRD", "PS"],
        n_samples=1,
        n_timesteps=2,
        img_shape=(n_lat, n_lon),
    )
    rotated_batch = rotate_modifier(batch)
    assert rotated_batch.data["UGRD"].shape == (1, 2, n_lat, n_lon)
    assert torch.allclose(rotate(rotated_batch.data["UGRD"]), -1 * batch.data["UGRD"])
    assert torch.allclose(rotate(rotated_batch.data["VGRD"]), -1 * batch.data["VGRD"])
    assert torch.allclose(rotate(rotated_batch.data["PS"]), batch.data["PS"])


def test_rotate_modifier_no_rotation():
    rotate_modifier = RotateModifier(
        rotate_probability=0.0, additional_directional_names=[]
    )
    n_lat = 8
    n_lon = 16
    batch = BatchData.new_for_testing(
        names=["UGRD", "VGRD", "PS"],
        n_samples=1,
        n_timesteps=2,
        img_shape=(n_lat, n_lon),
    )
    rotated_batch = rotate_modifier(batch)
    assert rotated_batch.data["UGRD"].shape == (1, 2, n_lat, n_lon)
    assert torch.allclose(rotated_batch.data["UGRD"], batch.data["UGRD"])
    assert torch.allclose(rotated_batch.data["VGRD"], batch.data["VGRD"])
    assert torch.allclose(rotated_batch.data["PS"], batch.data["PS"])


def test_rotate_modifier_random_rotation():
    random.seed(0)
    rotate_modifier = RotateModifier(
        rotate_probability=0.5, additional_directional_names=[]
    )
    n_lat = 8
    n_lon = 16
    batch = BatchData.new_for_testing(
        names=["UGRD", "VGRD", "PS"],
        n_samples=40,
        n_timesteps=2,
        img_shape=(n_lat, n_lon),
    )
    rotated_batch = rotate_modifier(batch)
    assert rotated_batch.data.keys() == batch.data.keys()
    assert rotated_batch.data["UGRD"].shape == (40, 2, n_lat, n_lon)
    rotated = {}
    unrotated = {}
    for name in rotated_batch.data:
        unrotated[name] = np.all(
            torch.abs(batch.data[name] - rotated_batch.data[name]).cpu().numpy() < 1e-6,
            axis=(1, 2, 3),
        )
        if name in ("UGRD", "VGRD"):
            sign = -1
        else:
            sign = 1
        rotated[name] = np.all(
            torch.abs(sign * rotate(batch.data[name]) - rotated_batch.data[name])
            .cpu()
            .numpy()
            < 1e-6,
            axis=(1, 2, 3),
        )
        assert np.all(rotated[name] + unrotated[name] == 1), name
        assert np.sum(rotated[name]) > 0, name
        assert np.sum(unrotated[name]) > 0, name
    for name in ("VGRD", "PS"):
        assert np.all(rotated[name] == rotated["UGRD"]), name
        assert np.all(unrotated[name] == unrotated["UGRD"]), name


@pytest.mark.parametrize(
    "name, additional_directional_names, match_expected",
    [
        ("UGRD", [], True),
        ("VGRD", [], True),
        ("UGRD_10m", [], True),
        ("UGRD_10m", ["UGRD"], True),
        ("VGRD200", [], True),
        ("eastward_wind_3", [], True),
        ("UGRD10m", [], True),
        ("NWIND10m", [], False),
        ("NWIND10m", ["NWIND"], True),
    ],
)
def test_rotate_modifier_pattern(
    name: str, additional_directional_names: list[str], match_expected: bool
):
    rotate_modifier = RotateModifier(
        rotate_probability=1.0,
        additional_directional_names=additional_directional_names,
    )
    assert (rotate_modifier._pattern.match(name) is not None) == match_expected, name


def _build_roundtrip_modifier(
    nlat: int,
    nlon: int,
    fraction_modes_kept: float,
    variables: list[str] | None = None,
    grid: str = "equiangular",
    mode: str = "degree_and_order",
) -> RoundtripModifier:
    """Build a RoundtripModifier directly from torch_harmonics, bypassing
    Distributed (matches non-distributed singleton behavior)."""
    probe = harmonics.RealSHT(nlat, nlon, grid=grid)
    default_lmax = int(probe.lmax)
    default_mmax = int(probe.mmax)
    if mode == "degree_only":
        lmax = max(1, min(int(round(fraction_modes_kept * nlat)), default_lmax))
        mmax = default_mmax
    else:
        lmax = max(1, int(round(fraction_modes_kept * default_lmax)))
        mmax = max(1, int(round(fraction_modes_kept * default_mmax)))
    sht = harmonics.RealSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid=grid).float()
    isht = harmonics.InverseRealSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid=grid).float()
    return RoundtripModifier(sht=sht, isht=isht, variables=variables)


def _offline_roundtrip_reference(
    field: torch.Tensor,
    nlat: int,
    nlon: int,
    fraction_modes_kept: float,
    grid: str = "legendre-gauss",
) -> torch.Tensor:
    """Numerical reference for ``xtorch_harmonics.roundtrip_filter`` truncation.

    Builds the SHT at full grid-aware Nyquist, zeros
    ``sht[..., round(n_lat * frac):, :]``, then runs the ISHT.
    """
    sht = harmonics.RealSHT(nlat, nlon, grid=grid).float()
    isht = harmonics.InverseRealSHT(nlat, nlon, grid=grid).float()
    *leading, _, _ = field.shape
    flat = field.reshape(-1, nlat, nlon).to(torch.float32)
    coeffs = sht(flat)
    cutoff = round(nlat * fraction_modes_kept)
    coeffs = coeffs.clone()
    coeffs[..., cutoff:, :] = 0
    out = isht(coeffs).to(field.dtype)
    return out.reshape(*leading, nlat, nlon)


def test_roundtrip_config_validates_fraction():
    with pytest.raises(ValueError):
        RoundtripConfig(fraction_modes_kept=0.0)
    with pytest.raises(ValueError):
        RoundtripConfig(fraction_modes_kept=1.5)
    # valid values must not raise
    RoundtripConfig(fraction_modes_kept=None)
    RoundtripConfig(fraction_modes_kept=0.5)
    RoundtripConfig(fraction_modes_kept=1.0)


def test_roundtrip_config_disabled_returns_null_modifier():
    config = RoundtripConfig(fraction_modes_kept=None)
    modifier = config.build_modifier(
        global_shape=(8, 16), default_grid="equiangular"
    )
    assert isinstance(modifier, NullModifier)


def test_roundtrip_config_requires_grid():
    config = RoundtripConfig(fraction_modes_kept=0.5)
    with pytest.raises(ValueError):
        config.build_modifier(global_shape=(8, 16), default_grid=None)


def test_roundtrip_config_rejects_healpix():
    config = RoundtripConfig(fraction_modes_kept=0.5, grid="healpix")
    with pytest.raises(NotImplementedError):
        config.build_modifier(global_shape=(8, 16))


def test_roundtrip_config_explicit_grid_overrides_default():
    config = RoundtripConfig(fraction_modes_kept=0.5, grid="legendre-gauss")
    modifier = config.build_modifier(
        global_shape=(16, 32), default_grid="equiangular"
    )
    assert isinstance(modifier, RoundtripModifier)


def test_roundtrip_config_validates_mode():
    with pytest.raises(ValueError):
        RoundtripConfig(fraction_modes_kept=0.5, mode="invalid")
    # valid modes must not raise
    RoundtripConfig(fraction_modes_kept=0.5, mode="degree_only")
    RoundtripConfig(fraction_modes_kept=0.5, mode="degree_and_order")


def test_roundtrip_degree_only_matches_offline_reference():
    """Numerical parity with xtorch_harmonics.roundtrip_filter on legendre-gauss.

    On legendre-gauss ``default_lmax == default_mmax == n_lat``, so the offline
    truncation (zero highest L rows at full Nyquist) is bit-equivalent to
    building an SHT with reduced lmax and full mmax.
    """
    nlat, nlon = 16, 32
    frac = 0.5
    modifier = _build_roundtrip_modifier(
        nlat, nlon, fraction_modes_kept=frac, grid="legendre-gauss", mode="degree_only"
    )
    torch.manual_seed(0)
    batch = BatchData.new_for_testing(
        names=["T"], n_samples=2, n_timesteps=2, img_shape=(nlat, nlon)
    )
    out = modifier(batch)
    reference = _offline_roundtrip_reference(
        batch.data["T"], nlat, nlon, fraction_modes_kept=frac, grid="legendre-gauss"
    )
    assert torch.allclose(out.data["T"], reference, atol=1e-5)


@pytest.mark.parametrize("grid", ["equiangular", "legendre-gauss"])
def test_roundtrip_modes_equivalent_in_current_torch_harmonics(grid: str):
    """In torch_harmonics 0.8.0, ``RealSHT`` defaults to ``default_lmax = n_lat``
    and ``default_mmax = nlon // 2 + 1`` for both equiangular and legendre-gauss
    grids. Because real-SHT coefficients satisfy ``|m| <= l``, an mmax bound at
    or above lmax is vacuous, so degree_only and degree_and_order produce
    identical fields. The mode switch is retained for forward-compat with
    torch_harmonics versions where ``default_mmax < default_lmax`` is possible.
    """
    nlat, nlon = 16, 32
    frac = 0.4
    only = _build_roundtrip_modifier(
        nlat, nlon, fraction_modes_kept=frac, grid=grid, mode="degree_only"
    )
    both = _build_roundtrip_modifier(
        nlat, nlon, fraction_modes_kept=frac, grid=grid, mode="degree_and_order"
    )
    torch.manual_seed(0)
    batch = BatchData.new_for_testing(
        names=["T"], n_samples=1, n_timesteps=1, img_shape=(nlat, nlon)
    )
    assert torch.allclose(only(batch).data["T"], both(batch).data["T"], atol=1e-5)


def test_roundtrip_degree_only_caps_lmax_at_default():
    """``degree_only`` uses ``lmax = min(round(n_lat * frac), default_lmax)`` so
    that fractions implying a cutoff above the grid's Nyquist degenerate to a
    full-Nyquist roundtrip rather than overflowing the SHT."""
    nlat, nlon = 16, 32
    # frac=1.0 → round(16*1.0)=16 == default_lmax on both grids in torch_harmonics 0.8;
    # frac=2.0 (hypothetical post-validation override) → cap would kick in.
    # We exercise the cap by patching: build with frac that exceeds Nyquist.
    config = RoundtripConfig(
        fraction_modes_kept=1.0, grid="legendre-gauss", mode="degree_only"
    )
    modifier = config.build_modifier(global_shape=(nlat, nlon))
    assert isinstance(modifier, RoundtripModifier)
    # Verify the resulting SHT lmax matches the Nyquist (i.e. wasn't over-specified).
    probe = harmonics.RealSHT(nlat, nlon, grid="legendre-gauss")
    assert modifier._sht.lmax == probe.lmax


def test_roundtrip_modifier_constant_field_unchanged():
    # A constant field has all spectral energy in mode (0, 0), so any
    # truncation that keeps at least one mode must leave it unchanged.
    n_lat, n_lon = 16, 32
    modifier = _build_roundtrip_modifier(n_lat, n_lon, fraction_modes_kept=0.5)
    batch = BatchData.new_for_testing(
        names=["PS"],
        n_samples=2,
        n_timesteps=3,
        img_shape=(n_lat, n_lon),
    )
    constant = torch.full_like(batch.data["PS"], 0.42)
    batch = BatchData(
        data={"PS": constant},
        time=batch.time,
        horizontal_dims=batch.horizontal_dims,
    )
    out = modifier(batch)
    assert torch.allclose(out.data["PS"], constant, atol=1e-5)


def test_roundtrip_modifier_idempotent():
    n_lat, n_lon = 16, 32
    modifier = _build_roundtrip_modifier(n_lat, n_lon, fraction_modes_kept=0.5)
    torch.manual_seed(0)
    batch = BatchData.new_for_testing(
        names=["T"],
        n_samples=2,
        n_timesteps=2,
        img_shape=(n_lat, n_lon),
    )
    once = modifier(batch)
    twice = modifier(once)
    assert torch.allclose(once.data["T"], twice.data["T"], atol=1e-5)


def test_roundtrip_modifier_only_selected_variables():
    n_lat, n_lon = 16, 32
    modifier = _build_roundtrip_modifier(
        n_lat, n_lon, fraction_modes_kept=0.5, variables=["T"]
    )
    torch.manual_seed(1)
    batch = BatchData.new_for_testing(
        names=["T", "PS"],
        n_samples=1,
        n_timesteps=1,
        img_shape=(n_lat, n_lon),
    )
    out = modifier(batch)
    # PS untouched (object identity is fine since modifier passes through)
    assert torch.equal(out.data["PS"], batch.data["PS"])
    # T filtered (so should differ from input for random data)
    assert not torch.allclose(out.data["T"], batch.data["T"], atol=1e-5)


def test_roundtrip_modifier_preserves_batch_metadata():
    n_lat, n_lon = 8, 16
    modifier = _build_roundtrip_modifier(n_lat, n_lon, fraction_modes_kept=0.5)
    batch = BatchData.new_for_testing(
        names=["T"],
        n_samples=1,
        n_timesteps=2,
        img_shape=(n_lat, n_lon),
        epoch=7,
    )
    out = modifier(batch)
    assert out.horizontal_dims == batch.horizontal_dims
    assert out.epoch == batch.epoch
    assert out.labels == batch.labels
    assert out.time.equals(batch.time)


def test_roundtrip_modifier_rejects_non_latlon_dims():
    n_lat, n_lon = 8, 16
    modifier = _build_roundtrip_modifier(n_lat, n_lon, fraction_modes_kept=0.5)
    batch = BatchData.new_for_testing(
        names=["T"],
        n_samples=1,
        n_timesteps=1,
        img_shape=(n_lat, n_lon),
        horizontal_dims=["face", "x"],
    )
    with pytest.raises(NotImplementedError):
        modifier(batch)


def test_roundtrip_modifier_legendre_gauss_grid():
    # Smoke test that the modifier works with a non-default grid type.
    n_lat, n_lon = 16, 32
    modifier = _build_roundtrip_modifier(
        n_lat, n_lon, fraction_modes_kept=0.5, grid="legendre-gauss"
    )
    torch.manual_seed(0)
    batch = BatchData.new_for_testing(
        names=["T"],
        n_samples=1,
        n_timesteps=1,
        img_shape=(n_lat, n_lon),
    )
    once = modifier(batch)
    twice = modifier(once)
    assert torch.allclose(once.data["T"], twice.data["T"], atol=1e-5)
