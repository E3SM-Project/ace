# Merging `origin/main` into `e3sm/exps/hist`

Recorded 2026-08-13. Nothing here has been committed to the experiment branch --
this is a rehearsal done in a throwaway worktree, written down so it does not
have to be rediscovered.

## Why merge rather than rebase

The branch is **598 commits ahead** of its merge-base with `origin/main`
(`355d51757`, 2026-07-31); `origin/main` is 11 ahead. Rebasing 598 commits,
almost all of which are config and experiment churn, buys nothing.

The actual library delta is small -- `git diff --stat <merge-base> HEAD -- fme/`
is **9 files, +314/-29**. Most of it has already been upstreamed by
`78ea64f15` ("E3SM updated variable names", #1161): `fme/core/ocean_data.py` is
now byte-identical to main, and `fme/core/atmosphere_data.py` differs by a
single hunk.

So the plan is:

1. **Merge** `origin/main` into the experiment branch to keep the experiment
   current.
2. **Separately**, cut a clean branch off `origin/main` carrying only the
   library work -- `rename`, `combine`, `add_scalar`, `mask_and_scale`, plus the
   four review fixes -- for a reviewable PR. `origin/main` has none of these
   (0 hits for `rename`, `combine`, `add_scalar` in `origin/main:xarray.py`).

## The merge needs exactly two hand edits

### 1. Conflict in `fme/core/atmosphere_data.py`

The only conflict. #1161 rewrote the same dict entry and dropped the bare
`frozen_precipitation_rate` alias, which the historical configs need (the atm
config renames `PRECST` to it). Resolve as the union:

```python
    "frozen_precipitation_rate": [
        "total_frozen_precipitation_rate",
        "frozen_precipitation_rate",
    ],
```

Main dropping that alias looks like an oversight in #1161 and is worth a
one-line PR of its own.

### 2. Silent integration bug: `rename` x #1420

`0880e7de0` ("Load time-invariant variables once instead of per sample", #1420)
added `XarrayDataset._load_time_invariant_tensors`, which opens the raw dataset
and indexes it with `ds[name]` -- where `name` comes from
`self._time_invariant_names` and is therefore a **post-rename** name. The
rename is never applied to that dataset, so every renamed time-invariant
variable raises `KeyError`.

`xarray.py` **auto-merges cleanly**; git reports no conflict. The breakage only
shows up when the tests run.

Fix, in `_load_time_invariant_tensors`:

```python
         ds = _open_xr_dataset(self.full_paths[0], engine=self.engine)
+        # _time_invariant_names are post-rename names, so the rename must be
+        # applied before they are looked up.
+        ds = self._apply_rename(ds)
         ds = ds.isel(**self.isel)
```

`overwrite` and `combine` are unaffected -- they are applied to the merged
tensor dict after the time-invariant broadcast, not at the xarray level.

## Test status of the merged state

Symptoms before the fix: `test_rename[mock_monthly_netcdfs-netcdf4-*.nc]` and
`test_rename[mock_monthly_zarr-zarr-*.zarr]` fail with
`KeyError: 'renamed_constant_var'`, plus three cascading
`fme/coupled/test_train.py::test_train_and_inference` failures.

After the fix:

- `fme/core/dataset/` + `fme/coupled/`: **448 passed, 0 failed**.
- Full `fme/` suite under `FME_FORCE_CPU=1`: 12 failed, 3488 passed, 24 skipped.

**None of those 12 are caused by the merge.** Re-running the same node ids on
clean `origin/main` under the same `FME_FORCE_CPU=1` reproduces 11 of them. The
causes are all environmental:

| cause | tests |
| --- | --- |
| `ImportError: GraphCast dependencies (trimesh, rtree) not available` | `test_ice_train.py::test_train_and_inference` |
| `ValueError: hpx_padding_mode=earth2grid requires the earth2grid package` | `test_train.py::test_train_and_inference[HEALPix]` |
| `RuntimeError: Expected all tensors to be on the same device` (AMP on CPU) | `test_optimization.py::test_gradient_clipping_with_amp` |
| `conftest.TimeoutException: Test took too long` | all the rest |

The coupled `test_train` failures are timeouts under full-suite CPU contention,
not defects -- run alone they pass in ~200 s. Which timeouts trip varies
between runs, which is why the merged run showed 12 and the clean-main run 11.

Note when comparing against older numbers in `README.md`: the earlier branch
baseline of "3426 passed, 3 failed" was **not** run with `FME_FORCE_CPU=1`, so
it is not comparable to the counts above.

## Environment

No regression from merging. The venv is already **python 3.13.9 /
torch 2.10.0+cu128**, ahead of main's `083b78e1f` bump (#1402) to
python=3.12 / torch=2.10.

## What main brings that is worth having

- `0880e7de0` (#1420) time-invariant variables loaded once, not per sample
- `3abb99c09` (#1421) zarr group/array handles cached per process (little effect
  here -- these streams are netCDF)
- `8f506eb8a` (#1364) ocean-sourced sea ice for coupled dataset creation
- `1c62fd132` (#1391) unordered stepper names typed as `set[str]`

## Reproducing

```bash
git worktree add --detach /path/to/wt e3sm/exps/hist
cd /path/to/wt
git merge origin/main          # one conflict, in atmosphere_data.py
# resolve as the union above, then apply the _apply_rename fix
PYTHONPATH=$PWD FME_FORCE_CPU=1 /pscratch/sd/m/mahf708/ace/.venv/bin/python \
  -m pytest fme/core/dataset/ fme/coupled/ -q
```

Run pytest with `PYTHONPATH` set to the worktree and the main repo's venv
interpreter -- `uv run` inside a worktree creates a fresh empty `.venv` and
fails with `No module named pytest`.
