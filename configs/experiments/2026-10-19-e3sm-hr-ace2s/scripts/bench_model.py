"""Isolated single-GPU training-step benchmark of NoiseConditionedSFNO.

Builds the ACE2S network exactly as the stepper would (same builder), then
times forward + backward + AdamW on synthetic data at a chosen grid. No data
loading, no stepper, no corrector: this is the compute/memory floor that the
end-to-end runs are compared against.

One configuration per process so an OOM cannot poison the next measurement.
Prints one JSON line.

    python bench_model.py --embed-dim 384 --checkpointing 3 --batch 2
"""

import argparse
import json
import os
import time

import torch

from fme.ace.registry.stochastic_sfno import NoiseConditionedSFNOBuilder
from fme.core.dataset_info import DatasetInfo
from fme.core.distributed import Distributed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nlat", type=int, default=720)
    p.add_argument("--nlon", type=int, default=1440)
    p.add_argument("--embed-dim", type=int, default=384)
    p.add_argument("--noise-embed-dim", type=int, default=32)
    p.add_argument("--num-layers", type=int, default=8)
    p.add_argument("--checkpointing", type=int, default=3)
    p.add_argument("--spectral-ratio", type=float, default=1.0)
    p.add_argument("--filter-num-groups", type=int, default=1)
    p.add_argument("--no-pos-embed", action="store_true")
    p.add_argument("--batch", type=int, default=2, help="local batch x n_ensemble")
    p.add_argument("--n-in", type=int, default=43)
    p.add_argument("--n-out", type=int, default=50)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--steps", type=int, default=6)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument(
        "--htf",
        type=float,
        default=1.0,
        help="hard_thresholding_fraction (builder field since 3c67da976)",
    )
    p.add_argument("--phases", action="store_true", help="sync between fwd/bwd/opt")
    p.add_argument("--profile", default="", help="write a torch.profiler table here")
    p.add_argument("--tag", default="")
    a = p.parse_args()

    with Distributed.context():
        run(a)


def run(a):
    dist = Distributed.get_instance()
    dev = torch.device("cuda", torch.cuda.current_device())
    builder = NoiseConditionedSFNOBuilder(
        embed_dim=a.embed_dim,
        noise_embed_dim=a.noise_embed_dim,
        noise_type="isotropic",
        filter_type="linear",
        use_mlp=True,
        num_layers=a.num_layers,
        checkpointing=a.checkpointing,
        affine_norms=True,
        spectral_ratio=a.spectral_ratio,
        filter_num_groups=a.filter_num_groups,
        pos_embed=not a.no_pos_embed,
        hard_thresholding_fraction=a.htf,
    )
    out = dict(vars(a))
    t0 = time.time()
    model = builder.build(a.n_in, a.n_out, DatasetInfo(img_shape=(a.nlat, a.nlon)))
    model = model.to(dev)
    model = dist.wrap_module(model)
    out["build_s"] = round(time.time() - t0, 1)
    n_total = sum(x.numel() for x in model.parameters())
    n_pos = sum(x.numel() for n, x in model.named_parameters() if "pos_embed" in n)
    n_spec = sum(
        x.numel() for n, x in model.named_parameters() if "filter" in n.lower()
    )
    out.update(params_M=n_total / 1e6, pos_embed_M=n_pos / 1e6, filter_M=n_spec / 1e6)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    torch.cuda.synchronize()
    out["static_GB"] = torch.cuda.memory_allocated() / 1e9

    # each rank holds only its (lat, lon) tile under the spatial backend
    hs, ws = dist.get_local_slices((a.nlat, a.nlon))
    # --batch is global (like FME's batch_size, times n_ensemble): split over
    # the data-parallel ranks only; spatial co-ranks share the same samples.
    nb = a.batch // dist.total_data_parallel_ranks
    assert nb * dist.total_data_parallel_ranks == a.batch, "batch % dp != 0"
    out["local_batch"] = nb
    x = torch.randn(nb, a.n_in, a.nlat, a.nlon)[..., hs, ws].to(dev)
    y = torch.randn(nb, a.n_out, a.nlat, a.nlon)[..., hs, ws].to(dev)
    out["world"] = dist.world_size
    out["fme_h"] = int(os.environ.get("FME_DISTRIBUTED_H", 1))
    out["fme_w"] = int(os.environ.get("FME_DISTRIBUTED_W", 1))
    out["local_shape"] = list(x.shape)
    times = []
    phases: dict[str, list[float]] = {"fwd": [], "bwd": [], "opt": []}

    def step(i):
        torch.cuda.synchronize()
        t = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.bf16):
            pred = model(x)
        loss = torch.nn.functional.mse_loss(pred.float(), y)
        if a.phases:
            torch.cuda.synchronize()
            t1 = time.perf_counter()
        loss.backward()
        if a.phases:
            torch.cuda.synchronize()
            t2 = time.perf_counter()
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        if i >= a.warmup:
            times.append(t3 - t)
            if a.phases:
                phases["fwd"].append(t1 - t)
                phases["bwd"].append(t2 - t1)
                phases["opt"].append(t3 - t2)
        if i == 0:
            torch.cuda.reset_peak_memory_stats()

    try:
        for i in range(a.warmup + a.steps):
            step(i)
        if a.profile:
            from torch.profiler import ProfilerActivity, profile

            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
            ) as prof:
                step(-1)
            if dist.is_root():
                table = prof.key_averages().table(
                    sort_by="cuda_time_total", row_limit=40, max_name_column_width=70
                )
                with open(a.profile, "w") as f:
                    f.write(table)
        out["ok"] = True
        if a.phases:
            for k, v in phases.items():
                out[f"{k}_s"] = sorted(v)[len(v) // 2]
        out["s_per_step_median"] = sorted(times)[len(times) // 2]
        out["s_per_step_min"] = min(times)
        out["peak_GB"] = torch.cuda.max_memory_allocated() / 1e9
        out["reserved_GB"] = torch.cuda.max_memory_reserved() / 1e9
    except torch.OutOfMemoryError as e:
        out["ok"] = False
        out["error"] = "OOM"
        out["peak_GB"] = torch.cuda.max_memory_allocated() / 1e9
        out["msg"] = str(e).splitlines()[0][:200]
    out["grad_sync"] = os.environ.get("FME_SPATIAL_GRAD_SYNC", "ddp")
    out["ofi"] = "Libfabric" in os.environ.get("NCCL_NET", "")
    out["fme_src"] = os.path.dirname(os.path.dirname(__import__("fme").__file__))
    out["host"] = os.uname().nodename
    out["gpu"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if dist.world_size > 1 and out["ok"]:
        # report the worst rank, not rank 0
        for k in ("peak_GB", "s_per_step_median"):
            t = torch.tensor(out[k], device=dev)
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX)
            out[k] = t.item()
    if dist.is_root() or not out["ok"]:
        print("RESULT " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
