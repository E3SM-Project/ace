# E3SMv3 historical — ACE2S atmosphere, Samudra ocean, SamudrACE coupled

Training configurations for the E3SMv3 historical run
`v3.LR.historical_0101.aigo` (1940–2065, 180×360). Three configs: an
atmosphere, an ocean, and a coupled finetune that composes the two.

All three read the **raw run directory directly** — there is no preprocessing
step. The coupled stepper averages the atmosphere's generated fluxes over the
ocean's step window at runtime, and resolves SST→TS online.

| config | trains | timestep | reads |
|---|---|---|---|
| `config-train-atm.yaml` | ACE2S atmosphere | 6-hourly | `eam.h0.*.nc` |
| `config-train-ocn.yaml` | Samudra ocean | 5-day | `mpaso`/`mpassi` `*5D*.remapped.nc` + LANDFRAC |
| `config-train-cpl.yaml` | coupled finetune | both | both |

**Production order:** train the atmosphere and the ocean separately, then
regenerate the coupled config with both checkpoints injected and finetune.

---

## Quickstart

```bash
# 0. one-time, per clone
uv sync --frozen                      # ~1 min; the venv is not shared between users
./stage-shared-data.sh --check        # confirms the input data is where the configs expect

# 1. sanity-check a config without burning an allocation
uv run python -m fme.ace.validate_config --config_type train config-train-ocn.yaml

# 2. a ~20 minute end-to-end smoke test on 4 GPUs
uv run python make_smoke_config.py config-train-ocn.yaml $PSCRATCH/smoke-ocn.yaml \
    --experiment-dir $PSCRATCH/smoke-out
uv run torchrun --nproc_per_node 4 -m fme.ace.train $PSCRATCH/smoke-ocn.yaml

# 3. a production run (submits to the batch queue, resumes automatically)
./sbatch-scripts/run-train.sh ocn
```

The coupled config uses a **different entry point** — `fme.coupled.train` and
`fme.coupled.validate_config`, not `fme.ace.*`. `run-train.sh` handles this.

---

## What is in this directory

| file | what it is |
|---|---|
| `config-train-atm.yaml` | atmosphere training config |
| `config-train-ocn.yaml` | ocean training config |
| `config-train-cpl.yaml` | coupled config — **generated, do not hand-edit** |
| `make_cpl_config.py` | generates the above from the other two |
| `make_smoke_config.py` | derives a short test config from a production one |
| `make_landfrac_5d.py` | builds the LANDFRAC-on-the-ocean-axis input file |
| `compute_hist_stats.py` | computes the normalization statistics |
| `stage-shared-data.sh` | puts the input data somewhere colleagues can read it |
| `sbatch-scripts/run-train.sh` | **the production entry point** — stage, validate, submit |
| `sbatch-scripts/sbatch-train-{atm,ocn,cpl}.sh` | the batch job for each realm |
| `sbatch-scripts/requeueable-train.sh` | per-node payload; handles preemption and requeue |
| `NOTES-historical-stats.md` | how the normalization statistics were derived |
| `NOTES-frontier-env.md` | building the environment on OLCF Frontier (ROCm) |
| `AGENTS.md` | log of what has been run and what changed when |

### Why the scripts exist rather than being config

* **`make_cpl_config.py`** — 401 of the coupled config's 765 lines (52%) are a
  verbatim mirror of the atm and ocn stepper blocks. YAML anchors do not cross
  files, so this script is the only thing keeping ~400 lines of channel
  definitions in sync. **Regenerate after editing either component config.**
  `--check` exits non-zero with a diff if the committed file has drifted, and
  is the thing to run in CI.
* **`make_smoke_config.py`** — `--override` is a dotlist and cannot index into
  a YAML list, so it cannot reach the `inference` blocks or `concat`/`merge`
  members. The script also keeps inference initial conditions on the ocean's
  5-day axis and keeps the coupled realms' first timestamps equal; both are
  enforced at runtime and fail the run otherwise.
* **`make_landfrac_5d.py`** — produces a data file no config can synthesise.

---

## Prerequisites

### Data

The raw model output is on the Community File System and is readable by the
`e3sm` group:

    /global/cfs/cdirs/e3smdata/simulations/v3.LR.historical_0101.aigo/run/

1501 files per stream, four streams, ~3.7 TB.

Two smaller **inputs** are derived products: the normalization statistics and
`LANDFRAC` resampled onto the ocean's 5-day axis (~115 MB together). These were
originally produced into personal scratch. Before sharing or running this
branch, relocate them:

    ./stage-shared-data.sh

This copies both trees to
`/global/cfs/cdirs/e3smdata/emulator/SamudrACE-E3SMv3/historical/`, sets
group-read (not world-read) permissions, and rewrites the paths in the two
component configs, then regenerates the coupled one. It is idempotent and never
clobbers an existing copy; `--check` reports without changing anything.

Do this rather than leaving the configs pointing at `$PSCRATCH`: scratch is
purged on an inactivity policy and is not backed up, so a colleague cloning
this branch months from now would hit `FileNotFoundError`. NERSC's guidance is
to share through a project directory with group permissions and to
[avoid world-readable data](https://docs.nersc.gov/filesystems/sharing/).

The three `experiment_dir` values are **outputs** and intentionally stay on
personal scratch — the batch scripts override them per job.

### Environment

The repo uses `uv`, not conda. From the repo root:

    uv sync --frozen

Roughly one minute. Verified on Perlmutter's current stack (`cpe/26.03`,
`cudatoolkit/13.2`) as of 2026-08-24: torch 2.10.0+cu128, NCCL 2.27.5,
8-rank multi-node all-reduce correct. The venv is per-clone; each user builds
their own.

On OLCF Frontier the environment is ROCm-based and set up differently — see
`NOTES-frontier-env.md` and `frontier-env.sh`. Nothing here has been *trained*
on Frontier, and the sizing and timing below are Perlmutter measurements.

**`pre-commit` cannot run on Perlmutter** — it takes an `flock` under `$HOME`
and that filesystem returns `OSError: [Errno 524]`. Run the pinned hooks
directly:

    export UV_TOOL_DIR=/tmp/$USER/uvtools UV_CACHE_DIR=/tmp/$USER/uvcache
    uvx ruff@0.8.1 check <files>
    uvx ruff@0.8.1 format --check <files>
    uvx --with types-PyYaml==5.4.3 mypy@1.15.0 \
        --ignore-missing-imports --check-untyped-defs <files>

Pin the versions — a newer ruff reformats untouched lines and adds diff noise.

---

## Sizing: how many GPUs

Two divisibility rules must both hold, and they decide the rank count for you:

1. **`batch_size` must be divisible by the number of ranks.** Reported clearly.
2. **Each inference block's initial-condition count must be divisible by the
   rank count.** Reported as `UnionMatchError: can not match type "list"`,
   which says nothing useful — re-validate at the same world size to see the
   real message.

| config | train `batch_size` | val `batch_size` | inference ICs | **valid rank counts ≤ 16** |
|---|---|---|---|---|
| atm | 8 | 32 | 16, 16 | 1, 2, 4, 8 |
| ocn | 16 | 16 | 16, 16 | 1, 2, 4, 8, 16 |
| cpl | 8 | 8 | 8 | 1, 2, 4, 8 |

Measured peak memory per GPU on `A100-SXM4-80GB`:

| config | local batch | peak/GPU | |
|---|---|---|---|
| atm | 1 | **61.8 GB** | runs |
| atm | 2 | 77.4 GB | **OOM even at 80 GB** |
| ocn | 4 | 15.5 GB | runs |
| cpl | 1 | 37.8 GB steady, **77.1 GB peak** | runs, but see below |

**As shipped, all three configs need 80 GB cards.** The atmosphere wants local
batch 1, so `batch_size` must equal the rank count — **8 ranks (2 nodes) is the
intended atmosphere configuration**. The coupled config looks modest most of the
time but peaks near 77 GB whenever the atmosphere's `n_steps` distribution draws
its 41-step rollout, which is 96% of an 80 GB card.

**This is fixable — see "Making the coupled config fit" below.** Enabling
gradient checkpointing drops the atmosphere to 28.4 GB with bit-identical
gradients, which removes the coupled memory cliff entirely.

To go wider, raise `batch_size` to the rank count and add initial conditions so
their count stays divisible too. Per-GPU memory does not change with rank count
at fixed local batch, so **scaling out is the walltime lever** — the atmosphere
at 16 ranks costs the same GPU-hours and finishes in half the wall time.

---

## Launching

### Batch (production)

    ./sbatch-scripts/run-train.sh atm|ocn|cpl

The driver stages the config to `$PSCRATCH/fme-config/<uuid>` so edits between
submit and start cannot change what runs, validates it (catching the
divisibility errors before the allocation starts), then submits. The job
handles preemption and walltime: `SIGTERM` stops cleanly, `USR1` requeues, and
training resumes from the last checkpoint. A copy of exactly what ran is left
in `<experiment_dir>/job_config/`.

Resume a requeued or preempted run explicitly:

    RESUME_JOB_ID=<job id> ./sbatch-scripts/run-train.sh atm

### Interactive

    # single node, up to 4 GPUs
    uv run torchrun --nproc_per_node 4 -m fme.ace.train config-train-ocn.yaml

    # multiple nodes, from inside an salloc
    MASTER=$(scontrol show hostnames $SLURM_NODELIST | head -1)
    srun --nodes=2 --ntasks-per-node=1 --gpus-per-node=4 \
      bash -c "uv run torchrun --nnodes=2 --nproc_per_node=4 \
                 --node_rank=\$SLURM_NODEID --master_addr=$MASTER --master_port=29517 \
                 -m fme.ace.train config-train-atm.yaml"

Pass an explicit `--master_port`: `torchrun` defaults to 29500 and two runs on
one node collide with `EADDRINUSE`.

**Do not use `FME_USE_SRUN=1` on Perlmutter.** That launcher hardcodes
`torch.cuda.set_device(0)` assuming `--gpus-per-task=1`; with the
`--gpus-per-node=4` binding used here every rank dies at the first collective
with `ncclUnhandledCudaError ... invalid device ordinal`. `torchrun` sets the
device from `LOCAL_RANK`, which is correct. On Frontier the srun path *is* the
right one — one rank per GCD with `--gpus-per-task=1 --gpu-bind=closest`.

### Checkpointing and resuming

`save_checkpoint: true` in all three configs. Checkpoints land in
`<experiment_dir>/training_checkpoints/`:

| file | ocean | coupled |
|---|---|---|
| `ckpt.tar` (full training state) | 1.3 GB | 14.1 GB |
| `best_ckpt.tar` | 341 MB | 3.5 GB |
| `best_inference_ckpt.tar` | 341 MB | 3.5 GB |

Resuming is automatic: relaunch against the same `experiment_dir` and it logs
`Resuming training from ...` / `Beginning epoch after N complete epochs`.
Checkpoints are per-epoch, so a requeue mid-epoch repeats that epoch. Budget
disk: the coupled `ckpt.tar` is 14 GB and is rewritten every interval.

---

## Timing and planning

Measured on A100-80GB. Dataset setup happens per rank before any training.

| | dataset setup | 1 train epoch | validation + inference |
|---|---|---|---|
| **ocn, production width, 4 GPU** | **8m41s** | **3601s (60 min)**, 410 batches | ~4 min |
| **atm, production width, 8 GPU** | **20m45s** | not measured (far longer) | — |
| **cpl, production width, 4 GPU** | **50m57s** | not measured; 1643 batches/epoch | — |
| ocn, 6-yr window, 4 GPU | 1m11s | 3m51s, 13 batches | 2m40s |
| atm, 6-yr window, 8 GPU | 2m00s | 2.11 s/batch | 8m56s |
| **cpl, 6-yr window, 8 GPU, production rollouts** | **24m54s** | **28m36s**, 28 batches | ~11 min |

Extrapolated to production:

| config | ranks | batches/epoch | s/batch | **h/epoch** | `max_epochs` | **total** |
|---|---|---|---|---|---|---|
| ocn | 4 | 410 | 8.8 | **1.0** (measured) | 150 | **~6 days** |
| atm | 8 | 16434 | 1.4 | **6.3** | 30 | **~8 days** |
| cpl (production rollouts) | 8 | 1643 | **61.3** (measured) | **~28** | 5 | **~6 days** |

The ocean row is the sanity check: 410 × 8.8 s predicts 1.0 h and the measured
epoch was 3601 s, so the method holds.

**The coupled figure is now measured, not extrapolated** (2026-08-24): a 6-year
window on 8 ranks with the *production* training rollouts — `n_coupled_steps: 4`
and the atmosphere `n_steps` distribution reaching 41 — completed a full
two-epoch cycle, exit 0, at 61.3 s/batch in the steady-state epoch. `batch_size`
is 8 in both cases, so batches/epoch is unchanged by rank count and this scales
directly: **~28 h/epoch, ~6 days for 5 epochs**, plus 51 minutes of setup per
job.

That is roughly double the ">13 h/epoch" floor previously guessed, so budget
accordingly. Two caveats: the measurement ran with another job competing for the
filesystem, and a 6-year window has different caching behaviour than the full
90-year record. Re-measure on the first production job and expect the real
number to sit in the 25–35 h/epoch range.

### Planning rules

* **Do not submit any of these to a 4-hour queue.** The cheapest full epoch
  (ocean) is an hour on top of a 9-minute build.
* **Request `setup + n_epochs × h_per_epoch`, then round up.** Setup is paid
  again on every requeue — 51 minutes for the coupled config.
* **Lean on checkpoint/resume** rather than fitting a whole run in one job.
* **Scale the atmosphere by rank count, not batch size** (it needs local batch
  1). Every inference block's IC count must divide the rank count too.
* **Watch `training_samples_per_second_on_rank_0`.** A large shortfall usually
  means filesystem contention, not a model problem.

Setup cost is dominated by reading every file's time coordinate. That is
memoized per distinct file list, so a config opening the same stream for
train/validation/inference pays it once. It still scales with the number of
distinct streams, which is why the coupled config (five streams plus the
atmosphere, ~1500 files each) takes 51 minutes before the first batch. **A long
silent startup is not a hang** — check for growing log lines or `py-spy dump`.

---

## Making the coupled config fit, and making it cheaper

The coupled config peaks at 77.1 GB on an 80 GB card. Three approaches were
considered; the measurements below say which one to use.

### Gradient checkpointing — recommended, and already implemented

`SFNONetConfig.checkpointing` (0=off, 1=encoder/decoder, 2=+block MLPs,
3=every block) trades recompute for activation memory. **Measured on 8 A100s,
same nodes, same config, only the flag changed:**

| | peak/GPU | s/batch |
|---|---|---|
| atmosphere, `checkpointing: 0` | 61.8 GB | 2.11 |
| atmosphere, `checkpointing: 3` | **28.4 GB** | 1.48 |

A 54% memory reduction, with gradients verified **bit-identical** to the
uncheckpointed path (it is exact recompute, not an approximation). Throughput
did not regress in this measurement — plausibly because the baseline runs near
capacity where the allocator thrashes — but the two runs had different
filesystem contention, so treat the speedup as unconfirmed and the memory
saving as solid.

This makes the coupled model fit with wide headroom, and it is the reason not
to spend scientific capital shrinking the model.

> **This required a bug fix, included on this branch.** The four
> `torch.utils.checkpoint` call sites omitted `use_reentrant=False`. The
> reentrant implementation returns an output with no `grad_fn` when no *input*
> requires grad — which is always the case, since the network input is data —
> so `checkpointing >= 1` silently left the encoder's parameters at
> `grad=None`, training a frozen randomly-initialised encoder while raising
> nothing louder than a `UserWarning`. **Any earlier run that set
> `checkpointing >= 1` is suspect.** Regression tests are in
> `fme/core/models/conditional_sfno/test_sfnonet.py`.

To enable it, add to the atmosphere's builder config (then regenerate the
coupled config):

    stepper.step.config.builder.config.checkpointing: 3

### Where the memory actually goes

Worth knowing before reaching for other levers. The atmosphere is 800.3M
parameters (94% of them in the spectral filter, which scales as
`num_layers × embed_dim² × lmax`); the ocean is 82.8M. At local batch 1 with
`n_ensemble: 2`, roughly **67% of the atmosphere's memory is activations**.

Crucially, **activation memory does not scale with rollout length here.** The
atmosphere pretrain uses `n_forward_steps: 1`, and `optimize_last_step_only`
wraps every non-optimized step in `torch.no_grad()`, so exactly one step
carries gradients regardless of whether 1 or 41 are drawn. The knob is network
width and depth, not rollout — which is exactly why checkpointing works so well.

### Other levers, and why they rank lower

* **bf16 autocast** (`enable_automatic_mixed_precision: true`) — ~12–16 GB
  saving. The spectral path already forces fp32, so the risky part is
  protected. No recorded reason for the current `false`; would need a loss-curve
  A/B. Try this *after* checkpointing so effects are attributable.
* **Shrinking `embed_dim`/`num_layers`** — real capacity cut requiring a full
  retrain to evaluate. Only if the two above fail. Note that
  `filter_num_groups`/`spectral_ratio` cut parameters 3.4× but leave
  activations *unchanged*, so they are a poor memory lever.
* **Dropping the 41-step atmosphere outcome** — removes the peak, but costs the
  only ≥10-day coupled horizon, and buys only ~3.5% walltime. Emergency use only.
* **Spatial/model parallelism** — not viable for this campaign. The backward
  pass is wrong on `main` (`_AutogradAllReduce.backward` returns `grad_output`
  unchanged); the fix lives on an unmerged branch, the aggregators and data
  writers are not shard-aware, and checkpointing under `H/W > 1` is untested.
  Forward-only equivalence is mature; training is not.
* **`n_ensemble: 1`** — halves activations, but CRPS at one member degenerates
  to MAE, so this switches ACE2S to a deterministic model. That is a
  model-class decision, not a memory optimisation.

### Reducing walltime

* **Ocean I/O first.** An epoch is 6560 samples through an 83M-parameter,
  already-checkpointed model at 8.8 s/batch — far more time than the FLOPs
  justify. It is almost certainly I/O-bound on the three-way netCDF merge.
  Raising `num_data_workers` (currently 2) and `prefetch_factor` (1) is free to
  try; preprocessing to zarr is the fallback.
* **Scale out.** Per-GPU memory is independent of rank count at fixed local
  batch, so the atmosphere at 16 ranks is ~4 days instead of 8 for the same
  GPU-hours.
* **Cut the *ocean's* `n_steps` distribution, not the atmosphere's.** Expected
  outer steps per coupled batch is 2.34, driven by the ocean's `steps: 4`
  outcome (p=0.3). Moving that mass to `steps: 2` is **−24% walltime**; to
  `steps: 1`, −35%. Dropping the atmosphere's 41-step outcome buys only −3.5%.
* **Halving the coupled finetune record** (your 100→50 year idea) is exactly
  linear: 1643 → ~820 batches/epoch, **−50%**. It is the largest single
  walltime win and it costs no memory. The caveat: the config concatenates
  1940–1990 and 2000–2040, so halving drops one forcing regime rather than
  merely thinning samples. Prefer the ocean-rollout cut first since that is
  free; combine the two if you need more.

---

## Scientific configuration

### CO2 and aerosol channels

Training on historical rather than piControl is about the forced trend, so the
atmosphere carries it explicitly:

* **Inputs** (forcings): `global_mean_co2` (renamed from the scalar `co2vmr`,
  which the loader broadcasts to the grid), `aerindexall` (aerosol index),
  `colccn.3` (column CCN at S=0.3%).
* **Outputs** (diagnostics, loss weight 1.0): `lwp`, `lcc`, `cdnc`, all also in
  `force_positive_names`.

The atmosphere is **46 in / 53 out**, 38 prognostic. `colccn.3` contains a
literal dot — it is used only as a list entry and dict key, never through
dotlist overrides, which would misparse it.

`make_cpl_config.py` takes `--atm-ckpt` / `--ocn-ckpt` to inject
`stepper_training.<realm>.parameter_init.weights_path` for the finetune. Pass
each component's `best_ckpt.tar` (a stepper checkpoint, not the training-state
`ckpt.tar`). With no flags the realms train from scratch.

### Inference blocks

Each config carries a `weight: 1.0` block scored during training and a
`weight: 0.0` `12yr_test` block that is monitored but never selects the best
checkpoint. The coupled `12yr_test` initial conditions start in 2040, after the
second training window closes, so the finetune has genuine out-of-sample
monitoring; a 876-step (12-year) rollout from the last of them ends in 2059,
inside the 2065 record.

### Ocean forcing: EAM names, MPAS data

The ocean is forced from the MPAS streams but under **EAM variable names**,
because atmosphere→ocean coupling is resolved by intersecting the ocean's
input-only names with the atmosphere's output names. MPAS-native names give an
empty intersection, which used to train happily as a silently one-way coupled
model. `_validate_atmosphere_to_ocean_coupling` now **raises** on an empty
intersection when the ocean declares next-step forcings, and **warns** on a
partial match (a next-step forcing may legitimately come from the ocean's own
forcing window, and this code also runs when loading a trained checkpoint).

The mapping is config-only:

| ocean input | from MPAS | transform |
|---|---|---|
| `TAUX`, `TAUY` | `windStress{Zonal,Meridional}` | `rename` + `multiply_scalar: -1` |
| `FSNS` | `shortWaveHeatFlux` | `rename` |
| `FLDS` | `longWaveHeatFluxDown` | `rename` |
| `FLUS`, `LHFLX`, `SHFLX` | `longWaveHeatFluxUp`, `latentHeatFlux`, `sensibleHeatFlux` | `rename` + `multiply_scalar: -1` |
| `frozen_precipitation_rate` | `snowFlux` | `rename` |
| `surface_precipitation_rate` | `rainFlux + snowFlux` | `combine` |
| `sst` | `sst` | `add_scalar: 273.15` (MPAS is °C, stats are K) |

The sign flips are measured, not assumed: on open ice-free ocean the flipped
MPAS fields match EAM to 0.3–4.6% of each field's standard deviation. Without
the wind-stress flip, `TAUX` disagrees by 2.06 sigma.

`FSNS` replaces the `FSDS`/`FSUS` pair because MPAS carries only net shortwave,
so the atmosphere predicts `FSNS` (53 outputs, not 54). To revert, put `FSDS`
and `FSUS` back in `out_names`, loss weights and `force_positive_names`, and
split the ocean's shortwave input.

**`mask_and_scale: true` is required on every MPAS stream.** Those files flag
land with `_FillValue = 1e20` rather than NaN. Without it, land loads as a
literal 1e20 in the targets while output masking writes NaN over the same
points; the loss only zeroes points where the *target* is NaN, so training dies
with `Loss is NaN-valued`.

`icebergHeatFlux` is excluded — identically zero across the run, so its stats
scale is 0 and normalizing gives 0/0.

### LANDFRAC on the ocean axis

`LANDFRAC` and `sea_surface_fraction` are EAM fields absent from the MPAS
streams, but the coupled ocean needs them: a cell's sea ice fraction is
`ocean_sea_ice_fraction * (1 - LANDFRAC)`, and `mask_2d` is binary so it cannot
substitute (~20% of ocean cells are coastal with fractional land). Merge members
must share `sample_start_times`, so LANDFRAC is materialised on the 5-day axis:

    uv run python make_landfrac_5d.py <output-dir>

126 year-files, 91 MB.

### Normalization

All three configs use statistics computed **from the historical run itself**,
restricted to the training windows (`train-only/`). This replaced piControl-derived
stats, which were a different climate with no CO2 trend.

    stats-2026-08-13/
        train-only/atmosphere/   <- config-train-atm and -cpl
        train-only/ocean/        <- config-train-ocn and -cpl
        atmosphere/, ocean/      <- full record; leaks into validation, see below
        _partials/               <- per-file partials; re-aggregate any window

Headlines (details in `NOTES-historical-stats.md`):

* All 1501 files of every stream, no subsampling; exact streaming algorithm
  (per-file count/mean/M2, Chan combination), so it is not an approximation.
* Computed on the field **as the loader delivers it** — transforms applied in
  loader order, so the sign flips, the `sst` Kelvin offset and
  `surface_precipitation_rate = rainFlux + snowFlux` are baked in. A sum of two
  fields cannot be reconstructed from the two fields' separate statistics.
* **Unweighted** over (time, lat, lon), matching `get_stats.py` and the
  existing files. When sanity-checking a value: unweighted global-mean `TS` is
  **279.5 K**, not the 287 K you get area-weighted. That is the convention.
* Coverage is a deliberate **superset** — 83 atmosphere and 127 ocean entries,
  versus 61 and 91 the configs reference. So reverting the `FSNS` split, or
  adding a channel like `TREFHT` or `sss`, needs no new statistics. Adding a
  *variable* does, and that re-reads the run (~23 min on three nodes).
* **Why `train-only/`**: it covers 1940–1990 and 2000–2040, exactly the training
  windows. The full-record set spans 1940–2065 and has therefore seen the
  validation window and the `12yr_test` period, which is leakage. Restricting
  is nearly free — of 350 numbers, all temperatures, winds, fluxes, salinities,
  velocities, precipitation, `PS`, `TS` and `LANDFRAC` move under 2%.

Three fields were dropped for a zero scale, each for a real reason: **`sol_tsi`
is `-1.0` in all 1501 files** (a sentinel — this EAM configuration never
diagnoses total solar irradiance), `icebergHeatFlux` is identically zero, and
`layerThicknessCoarsened_0` is exactly 20.0 m everywhere.

**Known wrinkle, `STW_0`.** Its full-field scale is 4.3× the piControl value
even in the train-only set, because the secular stratospheric-water trend spans
the training windows too; no choice of window fixes it. Its *residual* scale —
what the loss uses — is unchanged to 0.13%, so loss weighting is fine and only
the network input is ~4× more compressed. Worth watching if the stratosphere
misbehaves; not obviously wrong for a trending field.

To point at a different stats set without editing configs, override at launch.
The key prefix differs per config: `stepper.step.config.normalization` for atm
and ocn, but `stepper.<realm>.stepper.step.config.normalization` for cpl. The
ocean has only a `network` block; the atmosphere has `network` and `residual`,
and `residual` takes the *same* centering file with `scaling-residual.nc`:

    S=<stats-root>/train-only
    N=stepper.step.config.normalization

    --override \
      $N.network.global_means_path=$S/atmosphere/centering.nc \
      $N.network.global_stds_path=$S/atmosphere/scaling-full-field.nc \
      $N.residual.global_means_path=$S/atmosphere/centering.nc \
      $N.residual.global_stds_path=$S/atmosphere/scaling-residual.nc

---

## Short test runs

    uv run python make_smoke_config.py config-train-ocn.yaml \
        $PSCRATCH/smoke-ocn.yaml --experiment-dir $PSCRATCH/smoke-out

    uv run torchrun --nproc_per_node 4 -m fme.ace.train $PSCRATCH/smoke-ocn.yaml
    # coupled uses a different entry point:
    uv run torchrun --nproc_per_node 4 -m fme.coupled.train $PSCRATCH/smoke-cpl.yaml

Defaults to 6 years of data, 2 epochs, batch size 4, no checkpointing, no
wandb. `--years`, `--epochs`, `--batch-size` adjust it; **`--batch-size` must
divide your rank count**, and the default 4 will fail on 8 ranks.

`--full-data` keeps the production globs and windows and shrinks only the epoch
count and inference block — use it to rehearse a production launch, since
dataset construction over ~1500 files per stream is the expensive and
historically fragile part.

Do not shrink by hand with `--override`: dotlist overrides cannot index into a
list, so they cannot reach the `inference` blocks or `concat`/`merge` members.

**Check the real exit code, not the log tail.** A `time_buffer` teardown emits
`Bad file descriptor` and `AssertionError: can only test a child process` on a
*successful* run; conversely a trailing `echo` can mask a rank failure. Look
for `REAL_EXIT=0` and `DONE ---- rank 0`.

---

## Library changes this branch depends on

These configs need `fme/` changes that are **committed on this branch but not
yet merged to `main`**, so a checkout of `main` cannot run them. The branch is
8 commits ahead of `main` and 3 behind.

| commit | change | why |
|---|---|---|
| `91b069b44` | `mask_and_scale`, `add_scalar`, `combine`, plus seven validation rules and generated metadata for combine targets | raw MPAS files flag land with `_FillValue = 1e20`; `sst` is degC while stats are K; MPAS has `rainFlux`+`snowFlux` but no total precipitation |
| `0a511e86f` | `get_raw_paths` stdlib-glob fast path; `_get_raw_times` serial and memoized | fsspec's glob is ~250× slower here; the old fork pool deadlocked ranks and a thread pool corrupted the heap |
| `83d034b34` | `logging.basicConfig(force=True)` | a stray root-logger call before configuration otherwise silences INFO for the whole run |
| `02caa33e5` | raise when no atmosphere output feeds the ocean | catches the MPAS/EAM naming mistake, which otherwise trains as silently one-way coupled |
| `9c264dfae` | restore the bare `frozen_precipitation_rate` alias | both the atm and ocn configs rename into it; `#1161` dropped it |
| *(this work)* | `use_reentrant=False` at the four checkpoint call sites, with regression tests | `checkpointing >= 1` silently dropped encoder gradients |

`3d26128ec` (`DataWriterConfig.prediction_names`) is also on the branch but no
config here uses it; it belongs to a separate PR.

The upstreamable library work is a clean, reviewable delta — 9 files,
+1161/−47, all with tests — and cutting it as a PR off `main` is the main
outstanding task.

---

## Gotchas

These all cost real debugging time; the symptoms are misleading.

* **Misleading DDP parameter-mismatch errors mean a straggling rank.**
  `DDP expects same model across all ranks, but Rank 1 has 70 params, while
  rank 0 has inconsistent 0 params` does *not* indicate a model mismatch. A
  rank stuck in dataset setup leaves its peers in DDP's allgather until the
  30-minute NCCL watchdog fires, and the aborted collective returns garbage.
  Run `py-spy dump --pid <pid>` on every rank — the odd one out is the cause.
  The watchdog output also names the straggler: the rank with the lower
  `Last enqueued NCCL work`.
* **An empty `out.log` with a run that is training normally** means a
  `logging.info` ran during `dacite` parsing, before `configure_logging`. The
  root-logger convenience functions implicitly call `basicConfig()`, so the
  later one is a no-op. If you add diagnostics to any `__post_init__`, use a
  module logger, never `logging.info`.
* **Inference start indices need room for the rollout.** `max_start_index +
  window_length` must not exceed the dataset length.
* **The inference aggregator defaults to a metric at step 20**, so a run with
  fewer than 21 forward steps dies with `MetricNotSupportedError`. Use ≥21
  steps or set `aggregator.log_step_means: []`.
* **Keep generated configs on a shared filesystem, never `/tmp`.** `/tmp` is
  node-local, so a config written there and launched with `srun` works only
  when the job lands on the writing node — it fails *intermittently*, and
  `torchrun` buries the `FileNotFoundError` ~80 lines above a `ChildFailedError`
  that reports only `exitcode: 1`. Write to `$PSCRATCH`. (`run-train.sh` does.)
* **Do not `git stash` while a run is starting.** Config parsing is strict, so
  if the tree briefly loses the `combine`/`mask_and_scale` fields, a launching
  run fails with `UnionMatchError: can not match type "dict" to any type of
  "train_loader.dataset"` — pointing at the dataset rather than the missing
  dataclass field.

---

`AGENTS.md` in this directory records what has actually been run, when, and at
what scale, along with the history of how these configs reached their current
state.
