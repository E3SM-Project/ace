import pytest
import torch
import torch_harmonics.sht as th_sht
from torch import nn
from torch_harmonics.distributed import compute_split_shapes

from fme.core.device import get_device
from fme.core.distributed import Distributed, sht
from fme.core.gridded_ops import LatLonOperations
from fme.core.loss import LossConfig
from fme.core.models.conditional_sfno.sfnonet import SFNONetConfig, get_lat_lon_sfnonet


def _local(shapes: list[int], rank: int) -> slice:
    start = sum(shapes[:rank])
    return slice(start, start + shapes[rank])


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, scale: torch.Tensor):
    # a rank's tile may be all zeros (l < m), so scale by the global field
    assert ((actual - expected).abs().max() / scale.abs().max()).item() < 1e-5


@pytest.mark.parallel
@pytest.mark.parametrize("grid", ["legendre-gauss", "equiangular"])
@pytest.mark.parametrize(
    "nlat, nlon, lmax, mmax",
    [(16, 32, None, None), (17, 30, 12, 9), (64, 128, None, None), (48, 96, 30, 41)],
)
def test_spatial_sht_matches_serial_torch_harmonics(nlat, nlon, lmax, mmax, grid):
    dist = Distributed.get_instance()
    device = get_device()
    forward = dist.get_sht(nlat, nlon, lmax=lmax, mmax=mmax, grid=grid)
    inverse = dist.get_isht(nlat, nlon, lmax=lmax, mmax=mmax, grid=grid)
    ref_forward = th_sht.RealSHT(nlat, nlon, lmax, mmax, grid=grid).float().to(device)
    ref_inverse = th_sht.InverseRealSHT(nlat, nlon, lmax, mmax, grid=grid)
    ref_inverse = ref_inverse.float().to(device)

    h, w = dist.get_local_slices((nlat, nlon))
    # the serial backends return transforms without these attributes
    l_local = _local(
        compute_split_shapes(forward.lmax, getattr(forward, "comm_size_polar", 1)),
        getattr(forward, "comm_rank_polar", 0),
    )
    m_local = _local(
        compute_split_shapes(forward.mmax, getattr(forward, "comm_size_azimuth", 1)),
        getattr(forward, "comm_rank_azimuth", 0),
    )

    torch.manual_seed(0)
    x = torch.randn(2, 4, nlat, nlon, device=device, requires_grad=True)
    grad_coeffs = torch.randn(
        2, 4, forward.lmax, forward.mmax, dtype=torch.complex64, device=device
    )
    expected = ref_forward(x)
    (expected_grad,) = torch.autograd.grad(expected, x, grad_coeffs)
    x_local = x.detach()[..., h, w].requires_grad_()
    actual = forward(x_local)
    (actual_grad,) = torch.autograd.grad(
        actual, x_local, grad_coeffs[..., l_local, m_local].contiguous()
    )
    _assert_close(actual, expected[..., l_local, m_local], expected)
    _assert_close(actual_grad, expected_grad[..., h, w], expected_grad)

    coeffs = expected.detach().requires_grad_()
    grad_x = torch.randn(2, 4, nlat, nlon, device=device)
    expected = ref_inverse(coeffs)
    (expected_grad,) = torch.autograd.grad(expected, coeffs, grad_x)
    coeffs_local = coeffs.detach()[..., l_local, m_local].contiguous()
    coeffs_local.requires_grad_()
    actual = inverse(coeffs_local)
    (actual_grad,) = torch.autograd.grad(
        actual, coeffs_local, grad_x[..., h, w].contiguous()
    )
    _assert_close(actual, expected[..., h, w], expected)
    _assert_close(actual_grad, expected_grad[..., l_local, m_local], expected_grad)


def _legendre_tables(*modules: nn.Module) -> set[int]:
    return {
        buffer.untyped_storage().data_ptr()
        for module in modules
        for name, buffer in module.named_buffers()
        if name.split(".")[-1] == "pct"
    }


@pytest.mark.parallel
def test_model_loss_and_diagnostics_share_one_legendre_table():
    nlat, nlon = 16, 32
    if not isinstance(Distributed.get_instance().get_sht(nlat, nlon), sht.RealSHT):
        pytest.skip("this backend's transforms do not share Legendre tables")
    model = get_lat_lon_sfnonet(
        params=SFNONetConfig(embed_dim=8, num_layers=2),
        in_chans=2,
        out_chans=2,
        img_shape=(nlat, nlon),
        data_grid="legendre-gauss",
    ).to(get_device())
    ops = LatLonOperations(torch.ones(nlat, nlon), grid="legendre-gauss")
    loss = LossConfig(type="EnsembleLoss", kwargs={"energy_score_weight": 1.0}).build(
        gridded_operations=ops
    )
    spectrum_sht = ops.get_real_sht()
    assert len(_legendre_tables(model)) == 1
    assert _legendre_tables(model, loss, spectrum_sht) == _legendre_tables(model)
