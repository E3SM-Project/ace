import gc

import pytest
import torch
import torch_harmonics.distributed as thd

from fme.core.device import get_device
from fme.core.distributed import sht


def _rel_err(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return ((actual - expected).abs().max() / expected.abs().max()).item()


def _assert_matches_torch_harmonics(nlat, nlon, lmax, mmax, grid):
    device = get_device()
    ref_f = thd.DistributedRealSHT(nlat, nlon, lmax, mmax, grid=grid).float()
    ref_i = thd.DistributedInverseRealSHT(nlat, nlon, lmax, mmax, grid=grid).float()
    ref_f, ref_i = ref_f.to(device), ref_i.to(device)
    forward = sht.RealSHT(nlat, nlon, lmax, mmax, grid=grid)
    inverse = sht.InverseRealSHT(nlat, nlon, lmax, mmax, grid=grid)
    assert (forward.lmax, forward.mmax) == (ref_f.lmax, ref_f.mmax)
    assert (inverse.lmax, inverse.mmax) == (ref_i.lmax, ref_i.mmax)

    torch.manual_seed(0)
    x = torch.randn(2, 3, nlat, nlon, device=device, requires_grad=True)
    coeffs = torch.randn(
        2, 3, ref_f.lmax, ref_f.mmax, dtype=torch.complex64, device=device
    ).requires_grad_()

    expected, actual = ref_f(x), forward(x)
    assert _rel_err(actual, expected) < 1e-5
    grad = torch.randn_like(expected)
    (expected_grad,) = torch.autograd.grad(expected, x, grad)
    (actual_grad,) = torch.autograd.grad(actual, x, grad)
    assert _rel_err(actual_grad, expected_grad) < 1e-5

    expected, actual = ref_i(coeffs), inverse(coeffs)
    assert _rel_err(actual, expected) < 1e-5
    grad = torch.randn_like(expected)
    (expected_grad,) = torch.autograd.grad(expected, coeffs, grad)
    (actual_grad,) = torch.autograd.grad(actual, coeffs, grad)
    assert _rel_err(actual_grad, expected_grad) < 1e-5


@pytest.mark.parametrize("grid", ["legendre-gauss", "equiangular", "lobatto"])
@pytest.mark.parametrize(
    "nlat, nlon, lmax, mmax",
    [(16, 32, None, None), (17, 32, 12, 10), (12, 24, 12, 13)],
)
def test_sht_matches_torch_harmonics(nlat, nlon, lmax, mmax, grid):
    _assert_matches_torch_harmonics(nlat, nlon, lmax, mmax, grid)


def test_forward_and_inverse_share_one_table():
    forward = sht.RealSHT(16, 32, grid="legendre-gauss")
    inverse = sht.InverseRealSHT(16, 32, grid="legendre-gauss")
    explicit = sht.RealSHT(16, 32, lmax=16, mmax=17, grid="legendre-gauss")
    other_grid = sht.RealSHT(16, 32, grid="equiangular")
    truncated = sht.InverseRealSHT(16, 32, lmax=8, mmax=8, grid="legendre-gauss")
    assert forward.pct is inverse.pct
    assert explicit.pct is forward.pct
    assert other_grid.pct is not forward.pct
    assert truncated.pct is not forward.pct
    # moving to the device the table was built on keeps it shared
    assert forward.to(get_device()).pct is inverse.pct
    assert forward.state_dict() == {}
    assert inverse.state_dict() == {}


def test_tables_are_counted_once_and_released_with_their_transforms():
    gc.collect()
    n_before = len(sht._TABLES)
    transforms = [
        sht.RealSHT(16, 32, grid="legendre-gauss"),
        sht.InverseRealSHT(16, 32, grid="legendre-gauss"),
        sht.RealSHT(16, 32, grid="legendre-gauss"),
        sht.RealSHT(16, 32, grid="equiangular"),
        sht.InverseRealSHT(16, 32, grid="equiangular"),
    ]
    assert len(sht._TABLES) == n_before + 2
    table_bytes = {
        t.pct.untyped_storage().data_ptr(): t.pct.untyped_storage().nbytes()
        for t in transforms
    }
    assert len(table_bytes) == 2
    assert sum(table_bytes.values()) == 2 * 17 * 16 * 16 * 4
    del transforms
    gc.collect()
    assert len(sht._TABLES) == n_before
