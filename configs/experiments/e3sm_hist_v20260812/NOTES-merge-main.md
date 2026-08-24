# Relationship to `origin/main`

**Status (2026-08-24):** the merge described in the original version of this
note has happened. The branch is **8 commits ahead of `origin/main` and 3
behind**, with a merge base at `78ea64f15`. The library delta is 9 files,
+1161/−47, all with tests.

## The library work, and the outstanding task

Six of the eight commits are library changes the configs depend on:

| commit | change |
|---|---|
| `0a511e86f` | `get_raw_paths` stdlib-glob fast path; `_get_raw_times` serial and memoized |
| `91b069b44` | `rename` / `combine` / `add_scalar` / `mask_and_scale` in `XarrayDataset`, plus validation rules |
| `83d034b34` | `logging.basicConfig(force=True)` |
| `02caa33e5` | validate that the atmosphere actually forces the ocean |
| `9c264dfae` | restore the bare `frozen_precipitation_rate` alias |
| `3d26128ec` | `DataWriterConfig.prediction_names` (independent of these configs) |

**The outstanding task is to cut this as a reviewable PR against `main`.**
`origin/main` has none of `rename`, `combine`, `add_scalar` or `mask_and_scale`.

## The two hand edits, both now landed

1. **`fme/core/atmosphere_data.py`.** `#1161` rewrote the dict entry and dropped
   the bare `frozen_precipitation_rate` alias, which both the atm config
   (renaming `PRECST`) and the ocn config (renaming `snowFlux`) need. Resolved
   as the union of both names; committed as `9c264dfae`. Main dropping that
   alias looks like an oversight in `#1161` and is worth a one-line PR of its own.

2. **A silent integration bug: `rename` × `#1420`.** That PR added
   `XarrayDataset._load_time_invariant_tensors`, which opens the raw dataset and
   indexes it with `ds[name]` where `name` comes from `_time_invariant_names`
   and is therefore a **post-rename** name. The rename was never applied to that
   dataset, so every renamed time-invariant variable raised `KeyError`.
   `xarray.py` auto-merged cleanly and git reported no conflict — the breakage
   only appeared under test. Fixed by applying `_apply_rename` before the
   lookup; folded into `91b069b44`.

## What `main` brought that was worth having

* `0880e7de0` (#1420) time-invariant variables loaded once, not per sample
* `3abb99c09` (#1421) zarr group/array handles cached per process (little effect
  here — these streams are netCDF)
* `8f506eb8a` (#1364) ocean-sourced sea ice for coupled dataset creation
* `1c62fd132` (#1391) unordered stepper names typed as `set[str]`

`origin/main` has since moved ahead again (`e273d5438`, video diffusion
backbones), so the branch is no longer current with it.

## Running tests in a worktree

`uv run` inside a worktree creates a fresh empty `.venv` and fails with
`No module named pytest`. Use the main repo's interpreter with `PYTHONPATH` set
to the worktree:

```bash
git worktree add --detach /path/to/wt e3sm/exps/hist-v2026.8.0
cd /path/to/wt
PYTHONPATH=$PWD FME_FORCE_CPU=1 /pscratch/sd/m/mahf708/ace/.venv/bin/python \
  -m pytest fme/core/dataset/ fme/coupled/ -q
```

## Test status

On the current branch, `FME_FORCE_CPU=1`, targeted suites
(`fme/core/dataset/`, `fme/coupled/`, `fme/core/models/conditional_sfno/`,
`fme/ace/inference/data_writer/`): **684 passed, 3 skipped, 0 failed**
(2026-08-24).

Full-suite runs on this machine show a handful of environmental failures that
are **not** caused by this work and reproduce on clean `origin/main`:

| cause | tests |
|---|---|
| `ImportError`: GraphCast deps (trimesh, rtree) absent | `test_ice_train.py::test_train_and_inference` |
| `ValueError`: `hpx_padding_mode=earth2grid` needs the earth2grid package | `test_train.py::test_train_and_inference[HEALPix]` |
| `RuntimeError`: device mismatch (AMP on CPU) | `test_optimization.py::test_gradient_clipping_with_amp` |
| `conftest.TimeoutException` under full-suite CPU contention | varies between runs |

Which timeouts trip varies run to run, so exact pass/fail counts from the full
suite are not a stable baseline; compare targeted suites instead.
