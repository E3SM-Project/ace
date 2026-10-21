"""Real spherical harmonic transforms that share one Legendre table.

These match torch-harmonics' ``DistributedRealSHT`` and
``DistributedInverseRealSHT`` (``norm="ortho"``, ``csphase=True``) to fp32
roundoff, and reduce to the serial transforms when torch-harmonics' polar and
azimuth groups are not set up.

With orthonormal harmonics the forward table is the inverse table times the
quadrature weights. The forward transform applies the weights (together with
the 2*pi of the longitude integral) to its Fourier coefficients, so both
directions contract with the same table. Tables are cached per configuration
and device, so every transform of one grid in a process (model, loss,
diagnostics) holds the same tensor.

P_l^m is zero for l < m, so the table is stored in blocks of orders, each
holding only the degrees from the block's first order up, and the
contraction skips the rest: about half the table and half the work.
"""

import dataclasses
import math
import weakref

import torch
import torch.nn as nn
import torch_harmonics.distributed as thd
from torch_harmonics.legendre import legpoly
from torch_harmonics.quadrature import (
    clenshaw_curtiss_weights,
    legendre_gauss_weights,
    lobatto_weights,
)

from fme.core.device import get_device

_TABLES: "weakref.WeakValueDictionary[tuple, torch.Tensor]" = (
    weakref.WeakValueDictionary()
)


def _quadrature(nlat: int, grid: str) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return (cos of the nodes, weights, default lmax) for a latitude grid."""
    if grid == "legendre-gauss":
        cost, weights = legendre_gauss_weights(nlat, -1, 1)
        return cost, weights, nlat
    if grid == "lobatto":
        cost, weights = lobatto_weights(nlat, -1, 1)
        return cost, weights, nlat - 1
    if grid == "equiangular":
        cost, weights = clenshaw_curtiss_weights(nlat, -1, 1)
        return cost, weights, nlat
    raise ValueError(f"Unknown quadrature mode {grid!r}")


# Orders per block of the Legendre contraction; the zero triangle inside a
# block is still stored and multiplied.
_M_BLOCK = 32


@dataclasses.dataclass(frozen=True)
class _Block:
    m0: int  # local orders m0:m1
    m1: int
    l0: int  # first stored degree
    offset: int  # into the packed table


@dataclasses.dataclass(frozen=True)
class _LegendrePlan:
    """Layout of a packed table holding P[m, l, k] for l >= l0 of m's block."""

    nlat: int
    lmax: int
    blocks: tuple[_Block, ...]
    numel: int

    @classmethod
    def new(cls, nlat: int, lmax: int, m_start: int, m_stop: int) -> "_LegendrePlan":
        blocks = []
        offset = 0
        m = m_start
        while m < m_stop:
            m_next = min((m // _M_BLOCK + 1) * _M_BLOCK, m_stop)
            l0 = min(m, lmax)
            blocks.append(_Block(m - m_start, m_next - m_start, l0, offset))
            offset += (m_next - m) * (lmax - l0) * nlat
            m = m_next
        return cls(nlat, lmax, tuple(blocks), offset)

    def block_table(self, table: torch.Tensor, block: _Block) -> torch.Tensor:
        shape = (block.m1 - block.m0, self.lmax - block.l0, self.nlat)
        return table[block.offset : block.offset + math.prod(shape)].view(shape)


def _legendre_table(
    nlat: int,
    lmax: int,
    mmax: int,
    grid: str,
    m_start: int,
    m_stop: int,
    device: torch.device,
) -> torch.Tensor:
    """The packed fp32 table of orders m_start:m_stop, shared by every live
    transform with the same arguments.
    """
    key = (nlat, lmax, mmax, grid, m_start, m_stop, str(device))
    table = _TABLES.get(key)
    if table is None:
        cost, _, _ = _quadrature(nlat, grid)
        t = torch.flip(torch.arccos(cost), dims=(0,))
        pct = legpoly(mmax, lmax, torch.cos(t), norm="ortho", csphase=True)
        plan = _LegendrePlan.new(nlat, lmax, m_start, m_stop)
        table = torch.empty(plan.numel, dtype=torch.float32, device=device)
        for block in plan.blocks:
            m0, m1 = m_start + block.m0, m_start + block.m1
            plan.block_table(table, block).copy_(pct[m0:m1, block.l0 :])
        del pct
        _TABLES[key] = table
    return table


def _contract(x: torch.Tensor, table: torch.Tensor, plan: _LegendrePlan):
    """(..., nlat, m, 2) -> (..., lmax, m, 2), summing P[m, l, k] x[k, m]."""
    *lead, nlat, n_m, _ = x.shape
    xt = x.reshape(-1, nlat, n_m, 2).permute(2, 1, 0, 3).reshape(n_m, nlat, -1)
    yt = xt.new_empty(n_m, plan.lmax, xt.shape[-1])
    for block in plan.blocks:
        m0, m1, l0 = block.m0, block.m1, block.l0
        yt[m0:m1, :l0].zero_()
        if l0 < plan.lmax:
            yt[m0:m1, l0:] = torch.bmm(plan.block_table(table, block), xt[m0:m1])
    y = yt.view(n_m, plan.lmax, *lead, 2).movedim((0, 1), (-2, -3))
    # a new tensor, not a view: callers modify the output in place
    return y.clone(memory_format=torch.contiguous_format)


def _contract_transpose(y: torch.Tensor, table: torch.Tensor, plan: _LegendrePlan):
    """(..., lmax, m, 2) -> (..., nlat, m, 2), summing P[m, l, k] y[l, m]."""
    *lead, lmax, n_m, _ = y.shape
    # with l contiguous and the table on the right, the sum over l rounds
    # like torch-harmonics' einsum
    yt = y.reshape(-1, lmax, n_m, 2).permute(2, 0, 3, 1).reshape(n_m, -1, lmax)
    xt = yt.new_empty(n_m, yt.shape[1], plan.nlat)
    for block in plan.blocks:
        m0, m1, l0 = block.m0, block.m1, block.l0
        if l0 < plan.lmax:
            table_block = plan.block_table(table, block)
            torch.bmm(yt[m0:m1, :, l0:], table_block, out=xt[m0:m1])
        else:
            xt[m0:m1].zero_()
    n_lead = len(lead)
    x = xt.view(n_m, *lead, 2, plan.nlat)
    x = x.permute(*range(1, n_lead + 1), n_lead + 2, 0, n_lead + 1)
    return x.clone(memory_format=torch.contiguous_format)


class _Legendre(torch.autograd.Function):
    """Analysis contraction; its adjoint is the synthesis contraction."""

    @staticmethod
    def forward(ctx, x, table, plan):
        ctx.save_for_backward(table)
        ctx.plan = plan
        return _contract(x, table, plan)

    @staticmethod
    def backward(ctx, grad):
        (table,) = ctx.saved_tensors
        return _contract_transpose(grad, table, ctx.plan), None, None


class _LegendreTranspose(torch.autograd.Function):
    """Synthesis contraction; its adjoint is the analysis contraction."""

    @staticmethod
    def forward(ctx, y, table, plan):
        ctx.save_for_backward(table)
        ctx.plan = plan
        return _contract_transpose(y, table, plan)

    @staticmethod
    def backward(ctx, grad):
        (table,) = ctx.saved_tensors
        return _contract(grad, table, ctx.plan), None, None


def _split_start(shapes: list[int], rank: int) -> int:
    return sum(shapes[:rank])


class _SHTBase(nn.Module):
    pct: torch.Tensor

    def __init__(
        self,
        nlat: int,
        nlon: int,
        lmax: int | None,
        mmax: int | None,
        grid: str,
    ):
        super().__init__()
        _, _, default_lmax = _quadrature(nlat, grid)
        self.nlat = nlat
        self.nlon = nlon
        self.grid = grid
        self.norm = "ortho"
        self.csphase = True
        self.lmax = lmax or default_lmax
        self.mmax = mmax or nlon // 2 + 1

        self.comm_size_polar = thd.polar_group_size()
        self.comm_rank_polar = thd.polar_group_rank()
        self.comm_size_azimuth = thd.azimuth_group_size()
        self.comm_rank_azimuth = thd.azimuth_group_rank()
        self.lat_shapes = thd.compute_split_shapes(nlat, self.comm_size_polar)
        self.lon_shapes = thd.compute_split_shapes(nlon, self.comm_size_azimuth)
        self.l_shapes = thd.compute_split_shapes(self.lmax, self.comm_size_polar)
        self.m_shapes = thd.compute_split_shapes(self.mmax, self.comm_size_azimuth)

        m_start = _split_start(self.m_shapes, self.comm_rank_azimuth)
        m_stop = m_start + self.m_shapes[self.comm_rank_azimuth]
        table = _legendre_table(
            nlat, self.lmax, self.mmax, grid, m_start, m_stop, get_device()
        )
        self.register_buffer("pct", table, persistent=False)
        self._plan = _LegendrePlan.new(nlat, self.lmax, m_start, m_stop)

    def extra_repr(self):
        return (
            f"nlat={self.nlat}, nlon={self.nlon},\n lmax={self.lmax}, "
            f"mmax={self.mmax},\n grid={self.grid}, csphase={self.csphase}"
        )


class RealSHT(_SHTBase):
    """Forward real SHT of the last two dimensions, (..., lat, lon) ->
    (..., l, m), with lat/lon and l/m split as in torch-harmonics'
    ``DistributedRealSHT``.
    """

    quadrature_scale: torch.Tensor

    def __init__(
        self,
        nlat: int,
        nlon: int,
        lmax: int | None = None,
        mmax: int | None = None,
        grid: str = "equiangular",
    ):
        super().__init__(nlat, nlon, lmax, mmax, grid)
        # quadrature weights of this rank's latitudes, times the 2*pi of the
        # longitude integral
        _, weights, _ = _quadrature(nlat, grid)
        lat_start = _split_start(self.lat_shapes, self.comm_rank_polar)
        lat_stop = lat_start + self.lat_shapes[self.comm_rank_polar]
        scale = (2.0 * math.pi * weights[lat_start:lat_stop]).float().unsqueeze(-1)
        self.register_buffer(
            "quadrature_scale", scale.to(get_device()), persistent=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() < 3:
            raise ValueError(
                f"Expected tensor with at least 3 dimensions but got {x.dim()} instead"
            )
        num_chans = x.shape[-3]
        if self.comm_size_azimuth > 1:
            x = thd.distributed_transpose_azimuth.apply(x, (-3, -1), self.lon_shapes)
        x = torch.fft.rfft(x, n=self.nlon, dim=-1, norm="forward")
        x = x[..., : self.mmax] * self.quadrature_scale
        if self.comm_size_azimuth > 1:
            chan_shapes = thd.compute_split_shapes(num_chans, self.comm_size_azimuth)
            x = thd.distributed_transpose_azimuth.apply(x, (-1, -3), chan_shapes)
        if self.comm_size_polar > 1:
            x = thd.distributed_transpose_polar.apply(x, (-3, -2), self.lat_shapes)

        with torch.autocast(device_type=x.device.type, enabled=False):
            x = torch.view_as_real(x)
            x = _Legendre.apply(x, self.pct.to(x.dtype), self._plan)
        x = torch.view_as_complex(x)

        if self.comm_size_polar > 1:
            chan_shapes = thd.compute_split_shapes(num_chans, self.comm_size_polar)
            x = thd.distributed_transpose_polar.apply(x, (-2, -3), chan_shapes)
        return x


class InverseRealSHT(_SHTBase):
    """Inverse real SHT, (..., l, m) -> (..., lat, lon), with the splits of
    torch-harmonics' ``DistributedInverseRealSHT``.
    """

    def __init__(
        self,
        nlat: int,
        nlon: int,
        lmax: int | None = None,
        mmax: int | None = None,
        grid: str = "equiangular",
    ):
        super().__init__(nlat, nlon, lmax, mmax, grid)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() < 3:
            raise ValueError(
                f"Expected tensor with at least 3 dimensions but got {x.dim()} instead"
            )
        num_chans = x.shape[-3]
        if self.comm_size_polar > 1:
            x = thd.distributed_transpose_polar.apply(x, (-3, -2), self.l_shapes)

        with torch.autocast(device_type=x.device.type, enabled=False):
            x = torch.view_as_real(x)
            x = _LegendreTranspose.apply(x, self.pct.to(x.dtype), self._plan)
        x = torch.view_as_complex(x)

        if self.comm_size_polar > 1:
            chan_shapes = thd.compute_split_shapes(num_chans, self.comm_size_polar)
            x = thd.distributed_transpose_polar.apply(x, (-2, -3), chan_shapes)
        if self.comm_size_azimuth > 1:
            x = thd.distributed_transpose_azimuth.apply(x, (-3, -1), self.m_shapes)

        # the DC and Nyquist modes of a real signal have no imaginary part
        x[..., 0].imag.zero_()
        if (self.nlon % 2 == 0) and (self.nlon // 2 < x.shape[-1]):
            x[..., self.nlon // 2].imag.zero_()
        x = torch.fft.irfft(x, n=self.nlon, dim=-1, norm="forward")

        if self.comm_size_azimuth > 1:
            chan_shapes = thd.compute_split_shapes(num_chans, self.comm_size_azimuth)
            x = thd.distributed_transpose_azimuth.apply(x, (-1, -3), chan_shapes)
        return x
