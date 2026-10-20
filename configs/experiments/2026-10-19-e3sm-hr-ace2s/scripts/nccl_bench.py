"""All-reduce bandwidth across the whole torchrun world (fp32, bus bandwidth)."""

import os
import time

import torch
import torch.distributed as dist

dist.init_process_group("nccl")
r, n = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
for mb in (64, 1024, 8400):
    x = torch.ones(mb * 2**20 // 4, device="cuda")
    for _ in range(2):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    t = time.perf_counter()
    k = 5
    for _ in range(k):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t) / k
    busbw = 2 * (n - 1) / n * x.numel() * 4 / dt / 1e9
    if r == 0:
        print(
            f"RESULT net={os.environ.get('NCCL_NET', 'default')} world={n} "
            f"size={mb}MB time={dt:.3f}s busbw={busbw:.1f}GB/s",
            flush=True,
        )
dist.destroy_process_group()
