"""Time the SHT's distributed transpose at the HR sizes, three ways.

One polar transpose, as in DistributedRealSHT for an H-only split: a
(B, C, H/P, M) complex64 tensor becomes (B, C/P, H, M).

  th      torch-harmonics' path: split -> .contiguous() per chunk ->
          dist.all_to_all(list) -> torch.cat
  single  one permute copy into a send buffer -> all_to_all_single ->
          one permute copy out
  comm    all_to_all_single on preallocated buffers only (the floor)

Prints one RESULT line per (P, size, method) from rank 0. Run with torchrun
over all GPUs that form one transpose group:

    torchrun --nproc-per-node 4 transpose_bench.py
"""

import json
import os
import time

import torch
import torch.distributed as dist


def th_transpose(x, p):
    chunks = [c.contiguous() for c in torch.split(x, x.shape[1] // p, dim=1)]
    recv = [torch.empty_like(chunks[0]) for _ in range(p)]
    dist.all_to_all(recv, chunks)
    return torch.cat(recv, dim=2)


def single_transpose(x, p):
    b, c, h, m = x.shape
    send = x.reshape(b, p, c // p, h, m).permute(1, 0, 2, 3, 4).contiguous()
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send)
    # recv[q] holds rank q's latitude chunk of my channel chunk
    return recv.permute(1, 2, 0, 3, 4).reshape(b, c // p, p * h, m)


def timeit(fn, n=10):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n


def main():
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    p = dist.get_world_size()
    nlat = 720
    for label, b, c, m in [("htf0.5", 2, 384, 360), ("base", 2, 384, 721)]:
        x = torch.randn(b, c, nlat // p, m, dtype=torch.complex64, device="cuda")
        nbytes = x.numel() * x.element_size()
        ref = th_transpose(x, p)
        assert torch.equal(ref, single_transpose(x, p)), "layouts differ"
        send = torch.empty(nbytes // 4, dtype=torch.float32, device="cuda")
        recv = torch.empty_like(send)
        res = {
            "th": timeit(lambda: th_transpose(x, p)),
            "single": timeit(lambda: single_transpose(x, p)),
            "comm": timeit(lambda: dist.all_to_all_single(recv, send)),
        }
        if dist.get_rank() == 0:
            moved = nbytes * (p - 1) / p
            for k, v in res.items():
                out = dict(
                    p=p,
                    model=label,
                    method=k,
                    local_MB=round(nbytes / 1e6),
                    ms=round(v * 1e3, 2),
                    algbw_GBps=round(moved / v / 1e9, 1),
                    ofi="Libfabric" in os.environ.get("NCCL_NET", ""),
                    nnodes=int(os.environ.get("SLURM_NNODES", 1)),
                )
                print("RESULT " + json.dumps(out), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
