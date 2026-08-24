# Working log — E3SMv3 historical configs

Chronological record of what has been run, what changed, and what is still
open. `README.md` is the reference document a colleague reads; this file is the
history, kept so decisions do not have to be rediscovered.

## Guidance for agents working in this directory

* `config-train-cpl.yaml` is **generated**. Edit `config-train-atm.yaml` or
  `config-train-ocn.yaml`, then run `make_cpl_config.py`. Verify with
  `make_cpl_config.py --check`, which exits non-zero with a diff on drift.
* Do not put filesystem paths that are *inputs* under a personal `$PSCRATCH`.
  Use the shared CFS location; `stage-shared-data.sh` moves and repoints them.
* `pre-commit` cannot run on Perlmutter (flock, errno 524). Use the pinned
  `uvx` invocations in the README instead.
* Judge a run by `REAL_EXIT=0` and `DONE ---- rank 0`, never by the log tail —
  successful runs print alarming teardown tracebacks.

## 2026-08-24 — production-readiness pass

Ran on 4 nodes × 4 A100-80GB, Perlmutter's post-update stack.

**Environment revalidated after the Perlmutter software-stack update**
(`cpe/26.03`, `cudatoolkit/13.2`, `cray-mpich/9.1.0`). No rebuild was needed:

* existing venv imports `fme`, sees 4 GPUs, torch 2.10.0+cu128
* `uv sync --frozen` into a clean path takes **55 s** and produces a working env
* `uv lock --check` passes — the lockfile matches `pyproject.toml`
* 8-rank, 2-node NCCL all-reduce is correct (NCCL 2.27.5) in both the existing
  and the freshly built venv

**Runs.**

| test | scale | result |
|---|---|---|
| atm smoke, 6-yr window, 2 epochs | 8 GPU / 2 nodes | **exit 0**, 3169 s, loss 1.18 → 0.32, 61.8 GB/GPU |
| **cpl smoke, 6-yr window, 2 epochs, production rollouts** | 8 GPU / 2 nodes | **exit 0**, 7035 s, 77.1 GB/GPU peak |
| atm, `checkpointing: 3`, 6-yr window | 8 GPU / 2 nodes | **28.4 GB/GPU vs 61.8 GB baseline**, 1.48 s/batch vs 2.11 |
| production launch chain (`requeueable-train.sh`) | 8 GPU / 2 nodes | **exit 0**, `DONE ---- rank 0` |

**The coupled per-epoch cost is now measured, not extrapolated.** That run used
the *production* training rollouts — `n_coupled_steps: 4`, atmosphere `n_steps`
reaching 41 — which the old README recorded as never having completed an epoch.
Steady-state epoch 2 ran 28 batches in 28.6 min = **61.3 s/batch**. `batch_size`
is 8 in both the smoke and production configs, so batches/epoch does not change
with rank count and this scales directly to **~28 h/epoch, ~6 days for the 5
production epochs** — roughly double the ">13 h/epoch" floor previously guessed.
Caveats: another job was competing for the filesystem, and a 6-year window
caches differently than the 90-year record. Expect 25–35 h/epoch.

All three production configs validate (`--config_type train`, exit 0) on the
committed branch with no working-tree changes.

**Bug found and fixed: gradient checkpointing silently dropped encoder
gradients.** The four `torch.utils.checkpoint` call sites in
`conditional_sfno/{sfnonet,layers}.py` omitted `use_reentrant=False`. The
reentrant implementation returns an output with no `grad_fn` when no *input*
requires grad — always true here, the input is data — so `checkpointing >= 1`
left `conditional_model.encoder.{0.weight,0.bias,2.weight}` at `grad=None`,
training a frozen randomly-initialised encoder and raising only a
`UserWarning`. Verified empirically before and after; gradients at levels 1/2/3
are now bit-identical to level 0. Six regression tests added to
`test_sfnonet.py`, confirmed to fail without the fix. **Any earlier run that
set `checkpointing >= 1` is suspect.**

**Other fixes this pass:**

* `make_cpl_config.py`: paths now derived from `__file__` rather than assuming
  the repo root as cwd; added `--out` (so running it cannot silently clobber a
  tracked file) and `--check` (CI-able drift detection); `open()` calls wrapped
  in context managers.
* Coupled atmosphere `n_steps` probabilities summed to **1.045**, not 1.0.
  `TimeLengthProbabilities` renormalizes silently, so the realized distribution
  was not the one written. Corrected to sum to 1.0; this changed
  `config-train-cpl.yaml`, which was regenerated.
* `compute_hist_stats.py` probed partials-writability with `open(path, "wb")`,
  which **truncates**. Running without `--reuse-partials` by accident destroyed
  a 23-minute, three-node, 3.7 TB read. Now `O_CREAT|O_EXCL` with a message
  pointing at `--reuse-partials`.
* `sbatch-scripts/` was untracked, not executable, and its one script called
  `sbatch-train-*.sh` files that did not exist. Added `run-train.sh` (stage,
  validate, submit) and `sbatch-train-{atm,ocn,cpl}.sh`; made everything 755;
  wired the venv's `torchrun` through `FME_TORCHRUN` since this repo uses `uv`
  and there is no conda env to activate on the compute node. Fixed the comment
  citing `e3sm_piControl_v20260602`, which does not exist
  (`e3sm_piControl_v20260507`, on the `e3sm/exps/hist` branch).
* `make_smoke_config.py`: `--full-data` scanned the time coordinate of all
  ~1500 ocean files to build windows it then discarded, costing over two
  minutes (now 5 s), and its summary printed windows and initial conditions
  that were never written into the output config. Also added a note when
  `--batch-size` does not divide plausible rank counts — the script warned
  about initial-condition divisibility but not the parameter it sets itself.
* `stage-shared-data.sh` added, to move the statistics and LANDFRAC inputs off
  personal scratch onto group-readable CFS and repoint the configs. Note the
  destination is group-readable but **not** world-readable: NERSC guidance is to
  share through a project directory rather than by widening permissions on a
  personal one.
* `requeueable-train.sh` sized the torchrun rendezvous from
  `$SLURM_JOB_NUM_NODES`, which is allocation-wide. Under an `salloc` larger
  than the step — or any `srun` with an explicit `--nodes` — torchrun waited
  forever for nodes that never joined, with no error. Now uses
  `SLURM_STEP_NUM_NODES` with a fallback, derives `nproc_per_node` from
  `nvidia-smi` when `SLURM_GPUS_PER_NODE` is unset, and echoes the rendezvous
  parameters. Caught by actually running the chain, not by reading it.
* Added a `12yr_test` held-out inference block to the coupled config. All 8 of
  its existing initial conditions fell inside the training windows, so the
  finetune had no out-of-sample monitoring at all, unlike atm and ocn. The new
  block uses 8 ICs in 2040–2047, verified to lie on the ocean's 5-day axis, with
  a 12-year rollout ending inside the record.

**Documentation.** `README.md` rewritten as a reference document; the
historical log moved here. Corrected stale figures found by audit: coupled
config line count (401 of 765, not 398 of 753), atmosphere outputs (53, not
50), LANDFRAC size (91 MB, not 69), atmosphere stats names referenced (61, not
55/56), and the claim that all three configs use 16 inference initial
conditions (the coupled one uses 8).

Also removed the claim that the `fme/` changes were uncommitted and unreviewed
— they have been committed and the branch pushed. And dropped the
`global_mean_co2` "independent check": the config was later updated to quote
the stats values verbatim, so the comparison is now circular.

**Test status:** `fme/core/ fme/coupled/ fme/ace/inference/` under
`FME_FORCE_CPU=1` → **1921 passed, 10 skipped, 1 failed**. The single failure is
`test_optimization.py::test_gradient_clipping_with_amp`, which is environmental
(AMP on CPU) and reproduces with `origin/main`'s copy of that file, verified
directly.

**Still open:**

* No production-*width* epoch has completed for the atmosphere or the coupled
  model — the coupled rollouts are now production but the data window is not.
  Confirm the ~28 h/epoch figure on the first real job.
* Cut the library work as a PR against `main` (9 files, +1161/−47).
* `time_buffer: 10` (atm) has no coupled equivalent — `CoupledDataLoaderConfig`
  has no such field — so a coupled epoch draws ~11× more samples per unit
  window than an atmosphere epoch over the same period.
* The coupled ocean is given an `EnsembleLoss` with `n_ensemble: 2`, but
  `Samudra.forward` takes no noise input and is deterministic, so both members
  are identical: the energy-score term is identically zero and CRPS collapses
  to MAE, at 2× the ocean forward cost. Decide whether that is intended.
* `use_gradient_accumulation: true` detaches between coupled steps, so no
  gradient crosses the atmosphere↔ocean coupling. Each realm trains against the
  other's forward values only. A design question, not a bug, but it should be a
  deliberate choice.

## 2026-08-14 — CO2 and aerosol channels

Added `global_mean_co2` (renamed from the scalar `co2vmr`), `aerindexall` and
`colccn.3` as inputs; `lwp`, `lcc`, `cdnc` as outputs. All five are
`(time, lat, lon)` in every `eam.h0` file and all six have finite, non-zero
entries in all three stats files. Atmosphere went to 46 in / 53 out; the
prognostic set is unchanged at 38, so residual normalization was untouched.
`config-train-cpl.yaml` was regenerated and its diff was exactly the channel
additions.

`make_cpl_config.py` gained `--atm-ckpt` / `--ocn-ckpt` to inject
`parameter_init.weights_path`, matching the piControl coupled flow, for the
pretrain-then-finetune sequence.

## 2026-08-13 — historical normalization statistics

Replaced the piControl-derived statistics with statistics computed from the
historical run itself, restricted to the training windows. Details in
`NOTES-historical-stats.md`.

An A/B of new versus old stats on an ocean smoke config was **inconclusive by
construction**: repeating the *identical* new-stats run moved the epoch-1
inference error by 0.0035, larger than the entire new-vs-old difference
(0.0025). Runs are not deterministic at this scale, so the two stats sets are
indistinguishable at that scale rather than one being better. The useful finding
was the absence of pathology — loss descends, validation tracks training, no
NaN. Whether historical stats help generalization over a long rollout is not
something two epochs on a smoke config can show.

Because normalization changed, **inference-error numbers recorded before this
date are not comparable** to anything measured after it.

## 2026-08-13 — merge of `origin/main`

Recorded then as a rehearsal in a throwaway worktree; the merge has since
happened and both hand edits landed on the branch:

1. The `fme/core/atmosphere_data.py` conflict — `#1161` dropped the bare
   `frozen_precipitation_rate` alias that both configs rename into. Resolved as
   the union, committed as `9c264dfae`.
2. A silent integration bug: `#1420` added
   `XarrayDataset._load_time_invariant_tensors`, which indexes the raw dataset
   with **post-rename** names without applying the rename, so every renamed
   time-invariant variable raised `KeyError`. `xarray.py` auto-merged cleanly
   and git reported no conflict — the breakage only showed up under test.
   Fixed by applying `_apply_rename` before the lookup; folded into
   `91b069b44`.

The branch has since been rebuilt and is now a clean **8 commits ahead of
`main`, 3 behind**, rather than the 598-commit divergence this note originally
described.

Tip that still applies: run pytest in a worktree with `PYTHONPATH` set to the
worktree and the main repo's venv interpreter — `uv run` inside a worktree
creates a fresh empty `.venv` and fails with `No module named pytest`.

## Earlier — what had been run

All on A100-80GB. "exit 0" means train + validation + inline inference
completed and rank 0 logged `DONE ---- rank 0`. Inference-error numbers here
predate the stats change and are **not comparable** to current runs; they are
kept as a record of what ran.

| test | scale | result |
|---|---|---|
| ocn, full production globs and windows, 1 epoch | 4 GPU | exit 0 — setup 8m41s, epoch 3601s, valid loss 0.277 |
| ocn, 3-yr window, 2 epochs, checkpoints on | 8 GPU / 2 nodes | exit 0 — all three checkpoint files written |
| ocn, resume from checkpoint, epochs 3–4 | 8 GPU / 2 nodes | exit 0 — resumed at "after 2 complete epochs" |
| ocn, 6-yr window, 2 epochs | 4 GPU | exit 0 |
| atm, 6-yr window, 1 epoch | 4 GPU, local batch 1 | exit 0 |
| atm, full production globs, batch 8 | 8 GPU / 2 nodes | setup 20m45s, trained; stopped deliberately |
| cpl, full production globs and windows | 4 GPU | setup 50m57s (1643 train batches, 92 val); stopped deliberately |
| cpl, 2-yr window, reduced rollouts, checkpoints on | 4 GPU | exit 0 — epoch 614 s, `ckpt.tar` 14.1 GB |
| cpl, resume from checkpoint, epoch 2 | 4 GPU | exit 0 — epoch 625 s |
| cpl, 2–4 yr windows, production rollouts | 4 GPU | trains, ranks balanced; no epoch finished in the allocation |

### Bugs fixed while preparing these configs

* **Dataset setup could deadlock a rank during DDP construction.**
  `_get_raw_times` used a `multiprocessing.Pool`, created on a rank that had
  already initialised CUDA and NCCL; forking such a process deadlocks on pool
  teardown. Peers sat in DDP's parameter allgather until the 30-minute NCCL
  watchdog fired, and the aborted collective returned garbage that surfaced as
  a bogus "DDP expects same model across all ranks" error. A
  `ThreadPoolExecutor` was tried and is **worse** — netCDF4/HDF5 is not
  thread-safe, so it survives small runs and then corrupts the heap at
  production width. The reads are now serial and memoized.
* **`get_raw_paths` used fsspec's glob**, which stats every directory entry —
  7.5 s warm / 16.9 s cold per call on this 11,278-file directory versus 0.03 s
  for the stdlib, for an identical result. It runs once per dataset per rank, so
  it added minutes of skew and made the deadlock much easier to hit.
* **The coupled trainer logged nothing after startup.** A `logging.info` in
  `CoupledStepperConfig.__post_init__` runs during `dacite` parsing, before
  `configure_logging`; the root-logger convenience functions implicitly call
  `basicConfig()`, so the later one was a no-op. Fixed both ways — module
  logger, and `force=True`. Observability only; no training result was affected.

### Review points addressed during that work

Each with a test:

* **`overwrite` typos are an error.** `OverwriteConfig.apply` still skips names
  a given load did not request — required, since one config is legitimately
  loaded for several subsets of its names — but a name in no file at all can
  never take effect, so construction raises. Checked after `rename`, so the
  error names the renamed variable.
* **A `combine` target that also exists on disk is an error.** The computed
  value would silently shadow the stored one, and only for datasets requesting
  the target, so two loads of the same config could disagree.
* **A combine target no longer inherits its first source's metadata.** Its
  `long_name` is the definition itself; units are kept only when all sources agree.
* **A mask decoding to NaN is an error.** A `mask_*` variable is 0/1
  everywhere; a NaN appears when the mask carries a `_FillValue` and
  `mask_and_scale` decodes it, inverting the masking there. Checked on values,
  not the attribute. Re-verified on the real data: the 19 `mask_*` fields in
  `fmeDepthCoarsening5D` and `mask_2d` in `fmeDerivedFields5D` decode clean.
