import dataclasses
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
    [
        (16, 32, None, None),
        (17, 32, 12, 10),
        (12, 24, 12, 13),
        (64, 128, None, None),  # several blocks of orders
        (48, 96, 30, 41),  # orders beyond lmax
    ],
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


def test_table_skips_the_zero_degrees():
    nlat, mmax = 64, 65
    table = sht.RealSHT(nlat, 2 * (mmax - 1), grid="legendre-gauss").pct
    # one block of 32 orders from l=0, one from l=32, and m=64 from l=64
    assert table.numel() == (32 * 64 + 32 * 32 + 0) * nlat
    assert table.numel() < 0.75 * mmax * nlat * nlat


@pytest.mark.parametrize("k", [1, 2, 3, 4, 8])
def test_m_chunks_cover_the_plan_in_whole_blocks(k):
    plan = sht._LegendrePlan.new(100, 100, 0, 101)
    assert len(plan.blocks) == 4
    chunks = sht._m_chunks(plan, k)
    assert len(chunks) == min(k, 4)
    assert chunks[0][0] == 0 and chunks[-1][1] == 101
    blocks = []
    for (m0, m1, sub), next_chunk in zip(chunks, chunks[1:] + ((101,),)):
        assert m1 == next_chunk[0]
        assert sub.blocks[0].m0 == 0 and sub.blocks[-1].m1 == m1 - m0
        assert (sub.nlat, sub.lmax) == (plan.nlat, plan.lmax)
        blocks += [
            dataclasses.replace(b, m0=b.m0 + m0, m1=b.m1 + m0) for b in sub.blocks
        ]
    assert tuple(blocks) == plan.blocks
    # as even as whole blocks allow, the larger chunks holding the short last block
    n_blocks = [len(sub.blocks) for _, _, sub in chunks]
    assert n_blocks == sorted(n_blocks) and n_blocks[-1] - n_blocks[0] <= 1


def _orders(chunks):
    return [m1 - m0 for m0, m1, _ in chunks]


def test_m_chunks_balance_uneven_blocks():
    # 23 blocks, the last of 16 orders, as in the 0.25 degree grid
    plan = sht._LegendrePlan.new(721, 720, 0, 720)
    assert _orders(sht._m_chunks(plan, 8)) == [64] + [96] * 6 + [80]
    # 6 blocks into 4 chunks: 2, 2, 1, 1 blocks rather than 1, 1, 1, 3
    plan = sht._LegendrePlan.new(181, 180, 0, 180)
    assert _orders(sht._m_chunks(plan, 4)) == [32, 32, 64, 52]


@pytest.mark.parametrize("lmax", [100, 70])  # with lmax < mmax, empty blocks
@pytest.mark.parametrize("k", [2, 3, 4])
def test_chunked_contractions_are_bitwise_the_full_ones(lmax, k):
    nlat, mmax = 100, 101
    device = get_device()
    table = sht._legendre_table(nlat, lmax, mmax, "legendre-gauss", 0, mmax, device)
    plan = sht._LegendrePlan.new(nlat, lmax, 0, mmax)
    chunks = sht._m_chunks(plan, k)
    torch.manual_seed(0)
    x = torch.randn(2, 3, nlat, mmax, 2, device=device)
    y = torch.randn(2, 3, lmax, mmax, 2, device=device)
    for contract, inputs in ((sht._contract, x), (sht._contract_transpose, y)):
        chunked = [
            contract(inputs[..., m0:m1, :], table, sub) for m0, m1, sub in chunks
        ]
        assert torch.equal(torch.cat(chunked, dim=-2), contract(inputs, table, plan))


def test_polar_chunks_default_from_environment_and_serial_fallback(monkeypatch):
    monkeypatch.setenv("FME_SHT_POLAR_CHUNKS", "4")
    monkeypatch.setenv("FME_SHT_POLAR_MIN_CHUNK_MB", "0.5")
    forward = sht.RealSHT(100, 200, grid="legendre-gauss")
    inverse = sht.InverseRealSHT(
        100, 200, grid="legendre-gauss", polar_chunks=2, min_chunk_bytes=0
    )
    assert (forward.polar_chunks, inverse.polar_chunks) == (4, 2)
    assert (forward.min_chunk_bytes, inverse.min_chunk_bytes) == (2**19, 0)
    if forward.comm_size_polar == 1:
        # without a polar split there is nothing to pipeline
        assert forward._chunk_plans == {} and inverse._chunk_plans == {}
        x = torch.zeros(1, 1, 100, 101, dtype=torch.complex64)
        assert forward._select_chunks(x, forward.lat_shapes) is None
    assert forward.state_dict() == {} and inverse.state_dict() == {}
    with pytest.raises(ValueError):
        sht.RealSHT(16, 32, polar_chunks=0)
    with pytest.raises(ValueError):
        sht.RealSHT(16, 32, min_chunk_bytes=-1)


def test_min_chunk_mb_from_environment_defaults_to_4_mib(monkeypatch):
    monkeypatch.delenv("FME_SHT_POLAR_MIN_CHUNK_MB", raising=False)
    assert sht.RealSHT(16, 32).min_chunk_bytes == 4 * 2**20
    for value in ("auto", "-1", "nan", "inf", "1e400"):
        monkeypatch.setenv("FME_SHT_POLAR_MIN_CHUNK_MB", value)
        with pytest.raises(ValueError, match="FME_SHT_POLAR_MIN_CHUNK_MB"):
            sht.RealSHT(16, 32)


@pytest.mark.parametrize(
    "peer_bytes, min_chunk_bytes, expected",
    [
        (101, 0, 4),  # no limit: the most chunks
        (101, 5, 4),  # 4 chunks of 32, 32, 32, 5 orders: exactly 5 bytes
        (101, 6, 3),  # the smallest is short although the average is not
        (101, 32, 3),  # 3 chunks of 32, 32, 37 orders
        (101, 33, 2),  # 2 chunks of 64, 37 orders
        (101, 37, 2),
        (101, 38, 1),  # not even 2 chunks: unchunked
        (0, 1, 1),
    ],
)
def test_choose_chunk_count(peer_bytes, min_chunk_bytes, expected):
    # one byte per order to each peer
    plan = sht._LegendrePlan.new(100, 100, 0, 101)
    plans = {k: sht._m_chunks(plan, k) for k in (2, 3, 4)}
    assert [_orders(plans[k]) for k in (2, 3, 4)] == [
        [64, 37],
        [32, 32, 37],
        [32, 32, 32, 5],
    ]
    assert sht._choose_chunk_count(plans, peer_bytes, min_chunk_bytes) == expected
    assert sht._choose_chunk_count({}, peer_bytes, min_chunk_bytes) == 1


def _polar_rank_modules(monkeypatch, cls, n_ranks, **kwargs):
    """``cls`` as built on each of n_ranks polar ranks, without collectives."""
    modules = []
    monkeypatch.setattr(thd, "polar_group_size", lambda: n_ranks)
    monkeypatch.setattr(thd, "azimuth_group_size", lambda: 1)
    monkeypatch.setattr(thd, "azimuth_group_rank", lambda: 0)
    for rank in range(n_ranks):
        monkeypatch.setattr(thd, "polar_group_rank", lambda rank=rank: rank)
        modules.append(cls(100, 200, grid="legendre-gauss", **kwargs))
    return modules


@pytest.mark.parametrize("offset, expected", [(0, 2), (1, 1)])
@pytest.mark.parametrize("inverse", [False, True])
def test_uneven_splits_choose_the_same_chunks_on_every_rank(
    monkeypatch, inverse, offset, expected
):
    # 100 latitudes and degrees over 3 ranks: rows split as 34, 34, 32
    n_ranks, batch, n_chans, mmax = 3, 2, 4, 101
    bytes_per_row = batch * n_chans * mmax * 8  # complex64
    # at 2 chunks, of 64 and 37 orders, the smaller of the largest rank's
    # messages to each peer is exactly the minimum (offset 0) or a byte short
    # of it (offset 1); the rank with 32 rows alone would always fall short
    min_chunk_bytes = (34 * bytes_per_row // n_ranks) * 37 // mmax + offset
    assert (32 * bytes_per_row // n_ranks) * 37 < min_chunk_bytes * mmax
    cls = sht.InverseRealSHT if inverse else sht.RealSHT
    modules = _polar_rank_modules(
        monkeypatch, cls, n_ranks, polar_chunks=8, min_chunk_bytes=min_chunk_bytes
    )
    counts = []
    for rank, module in enumerate(modules):
        in_shapes = module.l_shapes if inverse else module.lat_shapes
        assert in_shapes == [34, 34, 32]
        assert sorted(module._chunk_plans) == [2, 3, 4]  # 4 blocks of orders
        x = torch.zeros(batch, n_chans, in_shapes[rank], mmax, dtype=torch.complex64)
        chunks = module._select_chunks(x, in_shapes)
        counts.append(1 if chunks is None else len(chunks))
        if chunks is not None:
            assert chunks is module._chunk_plans[len(chunks)]
    assert counts == [expected] * n_ranks


def test_min_chunk_bytes_zero_uses_polar_chunks_and_polar_chunks_one_never_chunks(
    monkeypatch,
):
    x = torch.zeros(1, 1, 34, 101, dtype=torch.complex64)
    for polar_chunks, expected in ((1, None), (2, 2), (3, 3), (8, 4)):
        (module, *_) = _polar_rank_modules(
            monkeypatch, sht.RealSHT, 3, polar_chunks=polar_chunks, min_chunk_bytes=0
        )
        chunks = module._select_chunks(x, module.lat_shapes)
        assert (None if chunks is None else len(chunks)) == expected
    # a single block of orders is never chunked
    x = torch.zeros(1, 1, 6, 17, dtype=torch.complex64)
    for cls in (sht.RealSHT, sht.InverseRealSHT):
        monkeypatch.setattr(thd, "polar_group_rank", lambda: 0)
        module = cls(16, 32, polar_chunks=8, min_chunk_bytes=0)
        assert len(module._plan.blocks) == 1 and module._chunk_plans == {}
        assert module._select_chunks(x, module.lat_shapes) is None


def test_polar_chunks_from_environment_must_be_an_integer(monkeypatch):
    monkeypatch.setenv("FME_SHT_POLAR_CHUNKS", "auto")
    with pytest.raises(ValueError, match="FME_SHT_POLAR_CHUNKS"):
        sht.RealSHT(16, 32)
