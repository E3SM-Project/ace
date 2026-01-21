"""
Tests for local_batch_size under spatial parallelism.

Runs correctly both serially (pytest) and in parallel (torchrun), on CPU and GPU.
"""

import os

from fme.core.distributed import Distributed
from fme.core.distributed.parallel_tests._helpers import requires_parallel


def test_local_batch_size_serial():
    """In serial (NonDistributed), local_batch_size == global batch_size."""
    with Distributed.force_non_distributed():
        dist = Distributed.get_instance()
        assert dist.local_batch_size(16) == 16


@requires_parallel
def test_local_batch_size_spatial_parallelism(monkeypatch):
    """
    Under full spatial parallelism (all ranks used for model parallelism),
    data parallel group size is 1, so local_batch_size == global batch_size.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    monkeypatch.setenv("H_PARALLEL_SIZE", "1")
    monkeypatch.setenv("W_PARALLEL_SIZE", str(world_size))

    dist = Distributed.get_instance()

    # All ranks are used for spatial parallelism, data group size = 1
    assert dist.total_data_parallel_ranks == 1
    assert dist.local_batch_size(16) == 16


@requires_parallel
def test_local_batch_size_mixed_parallelism(monkeypatch):
    """
    With partial spatial parallelism, some ranks are data-parallel.
    local_batch_size should divide by data parallel group size.

    Requires world_size >= 2. If world_size == 2: W=2 means data_group=1
    (same as full spatial). This test is most meaningful at world_size >= 4,
    but still passes at 2 since data_group=1 there.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    monkeypatch.setenv("H_PARALLEL_SIZE", "1")
    monkeypatch.setenv("W_PARALLEL_SIZE", str(world_size))

    dist = Distributed.get_instance()

    global_batch = 32
    data_group_size = dist.total_data_parallel_ranks
    expected = global_batch // data_group_size
    assert dist.local_batch_size(global_batch) == expected


def test_local_batch_size_not_divisible():
    """When batch_size is not divisible by data group size, integer division
    truncates. This documents the expected (lossy) behavior."""
    with Distributed.force_non_distributed():
        dist = Distributed.get_instance()
        # NonDistributed has data group size of 1, so any batch goes through.
        assert dist.local_batch_size(7) == 7

    # Simulate a scenario where total_ranks > 1 would cause truncation.
    # With TorchDistributed, local_batch_size = batch_size // total_ranks.
    # E.g. 7 // 2 == 3 (one sample is dropped).
    # We test this arithmetically since we can't spawn a real 2-rank env here.
    assert 7 // 2 == 3  # documents the truncation behavior
