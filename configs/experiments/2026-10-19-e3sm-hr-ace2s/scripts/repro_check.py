"""Serial vs spatially parallel reproducibility at 720x1440 on GPU.

Builds the same NoiseConditionedSFNO (same seed, isotropic noise) serially or
under the spatial backend, feeds the same global input (each rank takes its
tile), and records: the forward output, the loss, every parameter gradient
after the spatial gradient sync, and the output after N optimizer steps.

    # 1 GPU: write the reference
    python repro_check.py --save REF.pt
    # 4 GPUs: compare against it (prints max abs/rel errors, worst rank)
    FME_DISTRIBUTED_BACKEND=model FME_DISTRIBUTED_H=4 FME_DISTRIBUTED_W=1 \
        torchrun --nproc-per-node 4 repro_check.py --compare REF.pt

The loss is sum(pred * w) / numel with a fixed random global w, reduced over
the spatial group, so it and its gradients are global quantities.
"""

import argparse
import json
import os

import torch

from fme.ace.registry.stochastic_sfno import NoiseConditionedSFNOBuilder
from fme.core.dataset_info import DatasetInfo
from fme.core.distributed import Distributed


def _isotropic_noise_in_float64():
    """Fme draws isotropic noise coefficients in float32; in a float64 run
    that would put float32 round-off (from the noise iSHT) into the
    comparison, so draw them in float64.
    """
    import math

    import fme.ace.registry.stochastic_sfno as ss

    def isotropic_noise(leading_shape, lmax, mmax, isht, device):
        shape = (*leading_shape, lmax, mmax)
        real = ss.randn(shape, dtype=torch.float64, device=device)
        imag = ss.randn(shape, dtype=torch.float64, device=device)
        imag[..., :, 0] = 0.0
        real[..., :, 1:] /= math.sqrt(2.0)
        imag[..., :, 1:] /= math.sqrt(2.0)
        alm = (real + 1j * imag) * (math.sqrt(4.0 * math.pi) / lmax)
        l_slice, m_slice = Distributed.get_instance().get_local_slices((lmax, mmax))
        return isht(alm[..., l_slice, m_slice])

    ss.isotropic_noise = isotropic_noise


def build(a, dist):
    torch.manual_seed(a.seed)
    model = NoiseConditionedSFNOBuilder(
        embed_dim=a.embed_dim,
        noise_embed_dim=32,
        noise_type="isotropic",
        filter_type="linear",
        use_mlp=True,
        num_layers=a.num_layers,
        checkpointing=3,
        affine_norms=True,
        hard_thresholding_fraction=a.htf,
    ).build(a.n_in, a.n_out, DatasetInfo(img_shape=(a.nlat, a.nlon)))
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nlat", type=int, default=720)
    p.add_argument("--nlon", type=int, default=1440)
    p.add_argument("--embed-dim", type=int, default=256)
    p.add_argument("--num-layers", type=int, default=8)
    p.add_argument("--htf", type=float, default=0.5)
    p.add_argument("--n-in", type=int, default=43)
    p.add_argument("--n-out", type=int, default=50)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--opt", choices=["sgd", "adamw"], default="sgd")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", default="")
    p.add_argument("--compare", default="")
    p.add_argument("--tag", default="")
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    a = p.parse_args()
    # no TF32 anywhere: we want to see partitioning effects, not TF32 noise
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if a.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if a.dtype == "float64":
        _isotropic_noise_in_float64()
    with Distributed.context():
        run(a)


def run(a):
    dist = Distributed.get_instance()
    dev = torch.device("cuda", torch.cuda.current_device())
    dtype = getattr(torch, a.dtype)
    model = build(a, dist).to(dev, dtype)
    names = [n for n, _ in model.named_parameters()]
    wrapped = dist.wrap_module(model)
    if a.opt == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=a.lr)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr * 0.1)

    g = torch.Generator().manual_seed(a.seed + 1)
    shape = (a.batch, a.n_in, a.nlat, a.nlon)
    x_g = torch.randn(shape, generator=g)
    w_g = torch.randn((a.batch, a.n_out, a.nlat, a.nlon), generator=g)
    hs, ws = dist.get_local_slices((a.nlat, a.nlon))
    x = x_g[..., hs, ws].to(dev, dtype)
    w = w_g[..., hs, ws].to(dev, dtype)
    numel = w_g.numel()

    def forward(step):
        torch.manual_seed(1000 + step)  # same noise draw on every rank
        return wrapped(x)

    rec: dict = {"out": [], "loss": [], "grads": None}
    for step in range(a.steps + 1):
        pred = forward(step)
        loss = dist.spatial_reduce_sum((pred * w).sum()) / numel
        rec["out"].append(pred.detach().double().cpu())
        rec["loss"].append(float(loss))
        if step == a.steps:
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if step == 0:
            rec["grads"] = {
                n: q.grad.detach().double().cpu()
                for n, q in zip(names, model.parameters())
                if q.grad is not None
            }
        opt.step()
    rec["params"] = {
        n: q.detach().double().cpu() for n, q in zip(names, model.parameters())
    }

    if a.save:
        torch.save(rec, a.save)
        print("saved", a.save, "losses", rec["loss"])
        return
    ref = torch.load(a.compare)

    def rel(x, y):
        d = (x - y).abs().max().item()
        s = y.abs().max().item()
        return d, d / s if s > 0 else float("nan")

    res = {}
    for i in range(a.steps + 1):
        ro = ref["out"][i][..., hs, ws]
        res[f"out_step{i}"] = rel(rec["out"][i], ro)
        res[f"loss_step{i}"] = (
            abs(rec["loss"][i] - ref["loss"][i]),
            abs(rec["loss"][i] - ref["loss"][i]) / abs(ref["loss"][i]),
        )
    gerr = [rel(rec["grads"][n], ref["grads"][n]) for n in ref["grads"]]
    res["grad_max_rel"] = max(gerr, key=lambda t: t[1])
    perr = [rel(rec["params"][n], ref["params"][n]) for n in ref["params"]]
    res["param_max_rel"] = max(perr, key=lambda t: t[1])
    # worst over ranks
    out = {}
    for k, (ab, re) in res.items():
        t = torch.tensor([ab, re], device=dev)
        if dist.world_size > 1:
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX)
        out[k] = [float(f"{t[0]:.3g}"), float(f"{t[1]:.3g}")]
    if dist.is_root():
        out.update(
            tag=a.tag,
            h=int(os.environ.get("FME_DISTRIBUTED_H", 1)),
            w=int(os.environ.get("FME_DISTRIBUTED_W", 1)),
            opt=a.opt,
            dtype=a.dtype,
            deterministic=a.deterministic,
            nlat=a.nlat,
            ref_losses=ref["loss"],
            losses=rec["loss"],
        )
        print("RESULT " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
