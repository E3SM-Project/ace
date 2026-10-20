"""Compare two training checkpoints' keys, shapes and (optionally) values."""

import sys

import torch


def flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, f"{prefix}{k}."))
        elif torch.is_tensor(v):
            out[f"{prefix}{k}"] = v
    return out


a = flat(torch.load(sys.argv[1], map_location="cpu", mmap=True, weights_only=False))
b = flat(torch.load(sys.argv[2], map_location="cpu", mmap=True, weights_only=False))
print("keys only in A:", sorted(set(a) - set(b))[:10])
print("keys only in B:", sorted(set(b) - set(a))[:10])
bad = [k for k in a if k in b and a[k].shape != b[k].shape]
print(f"{len(a)} tensors; shape mismatches: {len(bad)}", bad[:5])
def worst_of(prefix):
    return max(
        (
            (
                (a[k].double() - b[k].double()).abs().max()
                / (b[k].double().abs().max() + 1e-30)
            ).item(),
            k,
        )
        for k in a
        if k.startswith(prefix)
        and k in b
        and a[k].shape == b[k].shape
        and a[k].is_floating_point()
        and a[k].numel() > 0
    )


for prefix in ("stepper.", "ema.", "optimization."):
    try:
        print(f"max relative difference in {prefix}*:", worst_of(prefix))
    except ValueError:
        print(f"no tensors under {prefix}")
worst = max(
    (
        ((a[k].double() - b[k].double()).abs().max() / (b[k].double().abs().max() + 1e-30)).item(),
        k,
    )
    for k in a
    if k in b and a[k].shape == b[k].shape and a[k].is_floating_point() and a[k].numel() > 0
)
print("max relative difference (after 115 steps):", worst)
