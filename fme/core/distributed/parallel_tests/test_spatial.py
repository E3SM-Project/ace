"""
Tests for spatial parallelism primitives.

These tests verify scatter/gather round-trips, reduce_sum_spatial, and
correct area-weighted mean under spatial sharding.  They work both
serially (NonDistributed) and in parallel (torchrun with ModelTorchDistributed).
"""

import pytest
import torch

from fme.core import get_device
from fme.core.distributed import Distributed


# -----------------------------------------------------------------------
# scatter + gather round-trip
# -----------------------------------------------------------------------


@pytest.mark.parallel
def test_scatter_gather_roundtrip_2d():
    """scatter then gather on a simple 2-D tensor recovers the original."""
    dist = Distributed.get_instance()
    h, w = 8, 12
    x = torch.arange(h * w, dtype=torch.float32, device=get_device()).reshape(h, w)
    local = dist.scatter_spatial(x, h_dim=0, w_dim=1)
    reconstructed = dist.gather_spatial(local, h_dim=0, w_dim=1)
    torch.testing.assert_close(reconstructed, x)


@pytest.mark.parallel
def test_scatter_gather_roundtrip_4d():
    """scatter then gather on a (B, C, H, W) tensor recovers the original."""
    dist = Distributed.get_instance()
    B, C, H, W = 2, 3, 8, 12
    x = torch.randn(B, C, H, W, device=get_device())
    local = dist.scatter_spatial(x)  # default h_dim=-2, w_dim=-1
    reconstructed = dist.gather_spatial(local)
    torch.testing.assert_close(reconstructed, x)


@pytest.mark.parallel
def test_scatter_correct_local_shape():
    """Local shard has the expected reduced spatial shape."""
    dist = Distributed.get_instance()
    B, C, H, W = 2, 3, 8, 12
    x = torch.randn(B, C, H, W, device=get_device())
    local = dist.scatter_spatial(x)
    assert local.shape == (B, C, H // dist.h_size, W // dist.w_size)


@pytest.mark.parallel
def test_scatter_spatial_noop_when_single():
    """When spatial parallelism is off (both sizes==1), x is returned as-is."""
    dist = Distributed.get_instance()
    x = torch.randn(4, 4, device=get_device())
    result = dist.scatter_spatial(x, h_dim=0, w_dim=1)
    if dist.h_size == 1 and dist.w_size == 1:
        assert result is x  # exact same object
    else:
        # still correct in shape
        assert result.shape[0] == 4 // dist.h_size


# -----------------------------------------------------------------------
# reduce_sum_spatial
# -----------------------------------------------------------------------


@pytest.mark.parallel
def test_reduce_sum_spatial_all_ones():
    """Summing all-ones tensors across spatial peers scales by h_size * w_size."""
    dist = Distributed.get_instance()
    t = torch.ones(3, 4, device=get_device())
    result = dist.reduce_sum_spatial(t.clone())
    expected = torch.full_like(t, dist.h_size * dist.w_size)
    torch.testing.assert_close(result, expected)


@pytest.mark.parallel
def test_reduce_sum_spatial_noop_when_single():
    """reduce_sum_spatial is identity when no spatial parallelism."""
    dist = Distributed.get_instance()
    t = torch.randn(5, device=get_device())
    result = dist.reduce_sum_spatial(t.clone())
    if dist.h_size == 1 and dist.w_size == 1:
        torch.testing.assert_close(result, t)
    else:
        # When parallel, the sum should differ from the original.
        # Just verify it returned a tensor of the right shape.
        assert result.shape == t.shape


# -----------------------------------------------------------------------
# area-weighted mean under sharding
# -----------------------------------------------------------------------


@pytest.mark.parallel
def test_area_weighted_mean_uniform_weights():
    """With uniform weights, area_weighted_mean == vanilla mean over spatial dims.

    This tests that the separate num/den all-reduce in _spatial_weighted_mean
    produces the correct result even when the data is sharded.
    """
    from fme.core.gridded_ops import _spatial_weighted_mean

    dist = Distributed.get_instance()
    H, W = 8, 12
    x = torch.arange(H * W, dtype=torch.float32, device=get_device()).reshape(1, H, W)
    weights = torch.ones(1, H, W, device=get_device())

    # Shard data + weights to match what training loop would do.
    x_local = dist.scatter_spatial(x, h_dim=-2, w_dim=-1)
    w_local = dist.scatter_spatial(weights, h_dim=-2, w_dim=-1)

    result = _spatial_weighted_mean(x_local, w_local, dim=(-2, -1))

    # Expected: plain mean over all H*W elements.
    expected = x.float().mean(dim=(-2, -1))
    torch.testing.assert_close(result, expected)


@pytest.mark.parallel
def test_area_weighted_mean_nonuniform_weights():
    """Non-uniform (latitude-like) weights must give the correct global mean.

    Construct cosine-latitude weights (vary only in h) and verify the
    weighted mean matches the full-grid computation.
    """
    from fme.core.gridded_ops import _spatial_weighted_mean

    dist = Distributed.get_instance()
    H, W = 8, 12
    # Data: simple latitude gradient.
    data = (
        torch.arange(H, dtype=torch.float32, device=get_device())
        .unsqueeze(-1)
        .expand(H, W)
        .unsqueeze(0)
    )
    # Weights: cosine-like (vary only in lat / h dim).
    lats = torch.linspace(-90, 90, H, device=get_device())
    cos_w = torch.cos(lats * 3.14159265 / 180.0).unsqueeze(-1).expand(H, W)
    cos_w = cos_w.unsqueeze(0)

    # Full-grid reference.
    expected = (data * cos_w).sum(dim=(-2, -1)) / cos_w.sum(dim=(-2, -1))

    # Sharded computation.
    data_local = dist.scatter_spatial(data, h_dim=-2, w_dim=-1)
    w_local = dist.scatter_spatial(cos_w, h_dim=-2, w_dim=-1)
    result = _spatial_weighted_mean(data_local, w_local, dim=(-2, -1))

    torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)


# -----------------------------------------------------------------------
# Sampler uses data-parallel rank / size
# -----------------------------------------------------------------------


@pytest.mark.parallel
def test_sampler_uses_data_parallel_ranks():
    """DistributedSampler should be configured with data-parallel rank/size."""
    dist = Distributed.get_instance()
    ds = torch.utils.data.TensorDataset(torch.randn(100))
    sampler = dist.get_sampler(ds, shuffle=False)
    assert sampler.num_replicas == dist.total_data_parallel_ranks
    assert sampler.rank == dist.data_parallel_rank


# -----------------------------------------------------------------------
# Makani comm.py wiring
# -----------------------------------------------------------------------


@pytest.mark.parallel
def test_comm_get_size_matches_distributed():
    """comm.get_size should agree with the Distributed singleton."""
    from fme.ace.models.makani_fcn3.utils import comm

    dist = Distributed.get_instance()
    assert comm.get_size("h") == dist.h_size
    assert comm.get_size("w") == dist.w_size
    assert comm.get_size("spatial") == dist.h_size * dist.w_size
    assert comm.get_size("matmul") == 1


@pytest.mark.parallel
def test_comm_get_rank_matches_distributed():
    """comm.get_rank should agree with the Distributed singleton."""
    from fme.ace.models.makani_fcn3.utils import comm

    dist = Distributed.get_instance()
    assert comm.get_rank("h") == dist.h_rank
    assert comm.get_rank("w") == dist.w_rank
    assert comm.get_rank("matmul") == 0


@pytest.mark.parallel
def test_comm_get_group_matches_distributed():
    """comm.get_group should return the same groups as Distributed."""
    from fme.ace.models.makani_fcn3.utils import comm

    dist = Distributed.get_instance()
    assert comm.get_group("h") is dist.h_group
    assert comm.get_group("w") is dist.w_group
    assert comm.get_group("spatial") is dist.spatial_group
    assert comm.get_group("matmul") is None


@pytest.mark.parallel
def test_comm_is_distributed_consistent():
    """comm.is_distributed should be True iff size > 1."""
    from fme.ace.models.makani_fcn3.utils import comm

    dist = Distributed.get_instance()
    expected_spatial = (dist.h_size * dist.w_size) > 1
    assert comm.is_distributed("spatial") == expected_spatial
    assert comm.is_distributed("matmul") is False


@pytest.mark.parallel
def test_comm_unknown_name_raises():
    """Requesting an unknown group name should raise ValueError."""
    from fme.ace.models.makani_fcn3.utils import comm

    with pytest.raises(ValueError, match="Unknown comm group name"):
        comm.get_size("nonexistent")
    with pytest.raises(ValueError, match="Unknown comm group name"):
        comm.get_rank("nonexistent")
    with pytest.raises(ValueError, match="Unknown comm group name"):
        comm.get_group("nonexistent")
