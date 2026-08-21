# Setting up `fme` on OLCF Frontier with `uv`

How to build a working ACE environment on
[Frontier](https://docs.olcf.ornl.gov/systems/frontier_user_guide.html) (AMD MI250X),
using [`uv`](https://docs.astral.sh/uv/) rather than conda. `frontier-env.sh` in this
directory loads the modules and activates the resulting venv.

Scope: **environment only**. The rest of this README's [Launching](README.md#launching),
sizing and timing numbers were measured on Perlmutter (A100-80GB, `$PSCRATCH`) and have
not been reproduced here — no `config-train-*.yaml` in this directory has been run on
Frontier yet. Treat the launch recipes as a starting point that needs re-validating for
8 GCDs per node.

- [The stack, and why](#the-stack-and-why)
- [Creating the environment](#creating-the-environment)
- [Verifying it](#verifying-it)
- [Switching to ROCm 7.2](#switching-to-rocm-72)
- [Troubleshooting](#troubleshooting)

## The stack, and why

| | |
|---|---|
| Hardware | AMD MI250X, target `gfx90a`. 4 MI250X per node = **8 GCDs**; Slurm and the ROCr runtime treat each GCD as a separate GPU. |
| ROCm | `rocm/7.1.1` |
| PyTorch | `2.10.0+rocm7.1` from `https://download.pytorch.org/whl/rocm7.1` |
| Python | 3.12, provided by `uv` (no `miniforge3` needed) |
| venv | `$ACE_ROOT/.venv`, ~13 GB |

Frontier offers ROCm modules from 5.6.0 through 7.13.0, including `7.0.2`, `7.1.1` and
`7.2.0`. **7.1.1** is the pick because two independent constraints agree on it:

1. OLCF's own [PyTorch on Frontier](https://docs.olcf.ornl.gov/software/analytics/pytorch_frontier.html)
   guide validates exactly `rocm/7.1.1` + `torch==2.10.0` + `torchvision==0.25.0`.
2. This repo's `constraints.txt` pins `torch==2.10.0  # version matches torch in Docker
   image`, so the Frontier environment stays in version parity with CI and the Docker
   image.

Both `whl/rocm7.1` and `whl/rocm7.2` wheels do ship `gfx90a` code, which is the thing
worth checking whenever you move to a newer ROCm — AMD periodically drops older targets
from the build matrix, and MI250X is two generations back now. Verify with
`torch.cuda.get_arch_list()`, not with release notes.

> **A note on `pyproject.toml`.** Do not add a `[[tool.uv.index]]` / `[tool.uv.sources]`
> block routing `torch` to a ROCm index in the checked-in `pyproject.toml`. There is no
> platform marker that distinguishes ROCm from CUDA — `sys_platform == 'linux'` is true
> for the CUDA Docker image too, so such a block forces ROCm wheels on every Linux
> consumer of the repo. `uv pip install -e .` *does* honor those tables, so the effect is
> real, not theoretical. If you ever want backend selection in project metadata, use uv's
> [conflicting-extras pattern](https://docs.astral.sh/uv/guides/integration/pytorch/)
> (`[project.optional-dependencies]` + `[tool.uv] conflicts`), not markers. The two-step
> install below keeps the ROCm choice machine-local instead.

## Creating the environment

Frontier login nodes have outbound network access, so all of this runs on a login node.

### 0. One-time: install `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Installs to `~/.local/bin/uv`. Confirm with `uv --version`.

### 1. Pick where things live

`$HOME` on Frontier is a small NFS quota, and the venv is ~13 GB with a ~20 GB build
cache. Put both on a project filesystem:

```bash
export ACE_ROOT=/lustre/orion/cli115/proj-shared/$USER/ace
export UV_CACHE_DIR=/lustre/orion/cli115/proj-shared/$USER/.uv-cache
```

`/lustre/orion` (Orion) is **purged**. For an environment you want to keep across purge
cycles, build it under `/ccs/proj/cli115/$USER` and point `ACE_ROOT` there. OLCF's
guidance is NVMe > Orion >> NFS, and for jobs at scale it is worth `sbcast`-ing the venv
to node-local NVMe.

### 2. Load modules

```bash
cd $ACE_ROOT
source configs/experiments/e3sm_hist_v20260812/frontier-env.sh
```

On a fresh checkout this warns that `.venv` does not exist — expected, the next step
creates it. What the script loads:

```
PrgEnv-gnu/8.7.0            cpe/26.03
rocm/7.1.1                  craype-accel-amd-gfx90a
LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
MPICH_GPU_SUPPORT_ENABLED=1
```

### 3. Create the venv and install PyTorch **first**

The order matters. Installing the ROCm build of `torch` before the project means the
project install sees `torch>=2.4.0` already satisfied and will not reach for the default
(CUDA) wheel on PyPI.

```bash
cd $ACE_ROOT
uv venv --python 3.12 .venv

uv pip install --python .venv/bin/python \
  --index-url https://download.pytorch.org/whl/rocm7.1 \
  torch==2.10.0 torchvision==0.25.0
```

`--index-url` (not `--extra-index-url`) *replaces* PyPI for this command, so the resolver
cannot silently prefer a CUDA build of the same version. The PyTorch index mirrors the
handful of deps torch needs (`numpy`, `sympy`, `pillow`, `triton-rocm`).

Note `triton-rocm`, not `pytorch-triton-rocm`: PyTorch renamed that package around 2.13
and the index carries both. Let the resolver choose; do not pin it yourself.

### 4. Install ACE and its dependencies

```bash
uv pip install --python .venv/bin/python -e ".[dev]"
```

This pulls the rest from PyPI and leaves `torch` alone. `torch-harmonics` is sdist-only,
so it builds here; under build isolation it builds pure-Python (its `setup.py` wraps
`import torch` in a `try`). That covers ACE's single-GPU-per-rank path, which uses the
SHT and quadrature code plus ACE's own vendored pure-PyTorch DISCO in `fme/core/disco/`.
Only the spatial-parallel path (`FME_DISTRIBUTED_H`/`W` > 1) reaches for
torch-harmonics' compiled DISCO kernels, and it falls back with

```
couldn't find CUDA extension, falling back to slow PyTorch implementation
```

which is a performance note, not an error. Building those kernels on Frontier is not
covered here; upstream's HIP path assumes a 32-bit NVIDIA warp mask and does not compile
against ROCm 7.1's 64-thread wavefronts without patching.

Optional HEALPix support:

```bash
uv pip install --python .venv/bin/python -r requirements-healpix.txt
```

Without it you will see `Could not import pad from earth2grid.healpix.` on import. That
is a soft warning, not an error.

## Verifying it

```bash
source configs/experiments/e3sm_hist_v20260812/frontier-env.sh

python -c "
import torch, torchvision, torch_harmonics
print('torch      ', torch.__version__)
print('hip        ', torch.version.hip)
print('gfx90a ok  ', 'gfx90a' in torch.cuda.get_arch_list())
print('torchvision', torchvision.__version__)
print('t-harmonics', torch_harmonics.__version__)
import fme; print('fme        ', fme.__file__)
"
```

Expected:

```
torch       2.10.0+rocm7.1
hip         7.1.25424
gfx90a ok   True
torchvision 0.25.0+rocm7.1
t-harmonics 0.8.0
fme         .../ace/fme/__init__.py
```

Then the fast CPU tests, on a login node:

```bash
python -m pytest fme/test_harmonics.py -q          # ~12 s
make test_very_fast                                # broader, still CPU
```

GPU and RCCL validation needs a compute node. A minimal check — 8 ranks, one per GCD,
all-reducing — is enough to confirm the interconnect and the device binding:

```bash
salloc -A cli115 -p batch -q debug -N 1 -t 00:30:00
# inside the allocation:
source configs/experiments/e3sm_hist_v20260812/frontier-env.sh
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
module load rccl-net-plugin/1.0

# rendezvous through a file on Lustre; it must not already exist
export RDZV_FILE=$PWD/rdzv_$SLURM_JOB_ID && rm -f "$RDZV_FILE"

srun -N 1 -n 8 -c 7 --gpus-per-task=1 --gpu-bind=closest python -c "
import os, torch, torch.distributed as dist
rank, world = int(os.environ['SLURM_PROCID']), int(os.environ['SLURM_NTASKS'])
dev = torch.device('cuda', 0)
torch.cuda.set_device(dev)
dist.init_process_group('nccl', init_method='file://' + os.environ['RDZV_FILE'],
                        rank=rank, world_size=world, device_id=dev)
t = torch.ones(1, device=dev) * rank
dist.all_reduce(t)
assert t.item() == world * (world - 1) / 2
print(f'rank {rank}: allreduce=OK on {torch.cuda.get_device_name(dev)}')
"
```

`-c 7` because Frontier has 64 cores/node with 8 reserved for the OS: 56 allocatable,
7 per GCD. With `--gpus-per-task=1` each rank sees exactly one GCD, so device 0 is the
right device in every rank. Passing `device_id=` binds the process group to that GCD up
front, which avoids a `barrier()` guessing the device. There is no `env://` rendezvous
under a bare `srun` — no `MASTER_ADDR`/`RANK` is set — hence the `file://` path, which is
also what `fme`'s own srun launcher uses.

Expect 8 lines of `allreduce=OK on AMD Instinct MI250X`.

The wider `fme` suite has also been run here as environment validation: the GPU-gated
subset gave 216 passed / 2 skipped, and one configuration of the `-m parallel` RCCL
matrix gave 73 passed / 1 skipped on each of 8 ranks. None of that exercises the configs
in this directory.

## Switching to ROCm 7.2

Verified to work the same way — `torch 2.13.0+rocm7.2`, HIP 7.2.53211, `gfx90a` present.
It moves you off both the OLCF-validated combination and this repo's `torch==2.10.0`
parity with the Docker image, so only do it deliberately.

1. In `frontier-env.sh`: `module load rocm/7.2.0`
2. Redo step 3 with:

```bash
uv pip install --python .venv/bin/python \
  --index-url https://download.pytorch.org/whl/rocm7.2 \
  torch==2.13.0 torchvision
```

`rocm/7.13.0` also exists as a module but has no matching PyTorch wheel index.

## Troubleshooting

**A C/C++ extension fails to build with ancient-compiler errors.** `/usr/bin/c++` on
Frontier is **GCC 7.5.0**, far too old for PyTorch 2.10 headers. The modules provide GCC
14.2.0, but setuptools honors `CC`/`CXX`, not the Cray PrgEnv wrappers, so it never sees
them. Before any `pip install` that compiles:

```bash
export CC="$(command -v gcc)" CXX="$(command -v g++)"   # 14.2.0 from PrgEnv-gnu
```

This applies to anything in this repo that builds a native extension, not just
torch-harmonics.

**`module load` appears to do nothing** (empty `ROCM_PATH`, no module changes). The
module system is only initialized in a login shell. Under a non-interactive shell wrap
it: `bash -lc 'source .../frontier-env.sh && ...'`.

**`ImportError` for a ROCm `.so` at runtime.** The wheels bundle their own ROCm libraries
in `.venv/lib/python3.12/site-packages/torch/lib`. If a module-provided `/opt/rocm-*/lib`
entry shadows them, the ROCm module's major version and the wheel's must match — that is
why the module is `7.1.1` and the wheel is `+rocm7.1`. Do not mix.

**`miopenStatusInternalError`, or sporadic hangs on the first conv.** MIOpen's kernel
database is being shared across ranks or across jobs. Give each job a node-local,
per-job copy in the batch script:

```bash
export MIOPEN_USER_DB_PATH="/tmp/miopen-${USER}-${SLURM_JOB_ID}"
export MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_USER_DB_PATH}"
```

MIOpen JIT-compiles convolution kernels on first use, so a cold cache also makes the
first steps of a run much slower than the steady state — and can blow through
`conftest.py`'s 90-second per-test `SIGALRM` if you are running the test suite. That
looks like a hang or a spurious timeout failure, not a compile.

**Do not run pytest with `FME_USE_SRUN=1` set.** It makes `conftest.py`'s session-scoped
`Distributed.context()` fixture take `TorchDistributed`'s srun branch and call
`init_process_group()` at session start, after which any test building its own
`Distributed()` fails with `ValueError: trying to initialize the default process group
twice!`. The signature is that exactly `test_distributed_gather`, `test_scatter_object`
and `test_non_distributed_gather` fail with an `init_method = 'file://...'` in the
traceback. That is this, not a ROCm problem.

**Environment vanished.** Orion is purged. Rebuild with the steps above, or keep the venv
under `/ccs/proj/cli115/$USER`.

**Slow imports / slow startup at scale.** Python import over Lustre is the usual cause.
`sbcast` the venv to node-local NVMe; see OLCF's
[Python at scale](https://docs.olcf.ornl.gov/software/python/index.html) guidance.
