"""
Test that barrier synchronization works under spatial parallelism.
"""

import os

import torch

from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.distributed.parallel_tests._helpers import requires_parallel


@requires_parallel
def test_barrier_completes(monkeypatch):
    """All ranks should pass through barrier() without deadlock."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    monkeypatch.setenv("H_PARALLEL_SIZE", "1")
    monkeypatch.setenv("W_PARALLEL_SIZE", str(world_size))

    dist = Distributed.get_instance()
    device = get_device()

    # Each rank writes to its own tensor, then barriers, then reads.
    # If barrier works, all ranks survive; if not, torchrun will timeout.
    local_tensor = torch.full(
        (1,), fill_value=float(dist.rank), dtype=torch.float32, device=device
    )
    dist.barrier()
    # After barrier, verify our local tensor is untouched
    assert local_tensor.item() == float(dist.rank)
