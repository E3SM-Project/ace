"""The configured collective timeout must reach every process group of the
spatial backend, not only the world group.

Runs its own process group (gloo, on the CPU), so it is a serial test.
"""

import os
import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed
import torch.multiprocessing as mp

TIMEOUT_MINUTES = 90
WORLD_SIZE = 4


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _timeout(group: torch.distributed.ProcessGroup | None) -> timedelta:
    assert group is not None
    return group._get_backend(torch.device("cpu")).options._timeout


def _check_group_timeouts(rank: int, port: int, h_size: int, w_size: int):
    os.environ.update(
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(WORLD_SIZE),
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        FME_FORCE_CPU="1",
        FME_DIST_TIMEOUT_MINUTES=str(TIMEOUT_MINUTES),
    )
    from fme.core.distributed.model_torch_distributed import ModelTorchDistributed

    backend = ModelTorchDistributed(h_size=h_size, w_size=w_size)
    try:
        groups = {
            "world": torch.distributed.group.WORLD,
            "data": backend._data_group,
            "h": backend._h_group,
            "w": backend._w_group,
            "spatial": backend._spatial_group,
            "data_h": backend._mesh_group(("data", "h")),
            "data_w": backend._mesh_group(("data", "w")),
        }
        timeouts = {name: _timeout(group) for name, group in groups.items()}
        expected = {name: timedelta(minutes=TIMEOUT_MINUTES) for name in groups}
        assert timeouts == expected, timeouts
    finally:
        backend.shutdown()


@pytest.mark.serial
@pytest.mark.parametrize("h_size, w_size", [(2, 1), (2, 2)])
def test_every_process_group_gets_the_configured_timeout(h_size, w_size):
    mp.spawn(
        _check_group_timeouts,
        args=(_free_port(), h_size, w_size),
        nprocs=WORLD_SIZE,
        join=True,
    )
