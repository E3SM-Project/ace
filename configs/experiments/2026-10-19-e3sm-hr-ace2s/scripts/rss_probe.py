"""Host RSS of building the HR ACE2S network on CPU, and what the
torch-harmonics lru caches hold afterwards."""

import gc
import resource
import sys

import torch_harmonics.legendre as leg
import torch_harmonics.quadrature as quad

from fme.ace.registry.stochastic_sfno import NoiseConditionedSFNOBuilder
from fme.core.dataset_info import DatasetInfo


def rss():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * 4096 / 1e9


def caches():
    out = []
    for mod in (leg, quad):
        for name in dir(mod):
            f = getattr(mod, name)
            for cell in getattr(f, "__closure__", None) or ():
                c = cell.cell_contents
                if hasattr(c, "cache_info"):
                    out.append((f"{mod.__name__}.{name}", c))
    return out


htf = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
print(f"start rss {rss():.1f} GB")
m = NoiseConditionedSFNOBuilder(
    embed_dim=384, noise_embed_dim=32, noise_type="isotropic", num_layers=8,
    checkpointing=3, affine_norms=True, hard_thresholding_fraction=htf,
).build(43, 50, DatasetInfo(img_shape=(720, 1440)))
params = sum(p.numel() for p in m.parameters()) * 4 / 1e9
bufs = sum(b.numel() * b.element_size() for b in m.buffers()) / 1e9
print(f"after build rss {rss():.1f} GB  params {params:.1f} GB  buffers {bufs:.1f} GB")
for n, c in caches():
    print(" ", n, c.cache_info())
for _, c in caches():
    c.cache_clear()
gc.collect()
print(f"after cache_clear rss {rss():.1f} GB")
print(f"peak rss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB")
