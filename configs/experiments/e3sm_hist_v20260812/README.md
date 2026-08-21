# E3SMv3 historical (v3.LR.historical_0101.aigo) — ACE2S / SamudrACE configs

Three configs, all reading the **raw run directory directly**. There is no
`create_coupled_datasets.py` step: the coupled stepper averages the atmosphere's
generated fluxes over the ocean's step window at runtime
(`fme/coupled/stepper.py`), and SST→TS is likewise resolved online.

| file | trains | reads |
|---|---|---|
| `config-train-atm.yaml` | ACE2S atmosphere, 6-hourly | `eam.h0.*.nc` |
| `config-train-ocn.yaml` | Samudra ocean, 5-day | `mpaso`/`mpassi` `*5D*.remapped.nc` + LANDFRAC |
| `config-train-cpl.yaml` | coupled finetune | both |

Setting up the environment on OLCF Frontier (AMD MI250X, ROCm + `uv` instead of conda)
is documented in `NOTES-frontier-env.md`; `frontier-env.sh` loads the modules and
activates the venv. That covers the environment only — nothing in this directory has
been trained on Frontier, and the sizing and timing below are Perlmutter measurements.

### The scripts, and why each one exists

| script | why it can't just be config |
|---|---|
| `make_cpl_config.py` | 53% of `config-train-cpl.yaml` (398 of 753 lines) is a verbatim mirror of the atm and ocn stepper blocks. YAML anchors don't cross files, so this is the only thing keeping ~400 lines of channel definitions in sync. **Regenerate after editing either component config**, and don't hand-edit the coupled one. |
| `make_landfrac_5d.py` | produces a data file (LANDFRAC on the ocean's 5-day axis) that no config can synthesise |
| `make_smoke_config.py` | `--override` is a dotlist and cannot index into a list, so it cannot reach the `inference` blocks or `concat`/`merge` members; it also has to keep inference ICs on the ocean's 5-day axis and the coupled realms' first timestamps equal |
| `compute_hist_stats.py` | computes normalization stats from the raw run. `scripts/data_process/get_stats.py` reads a single zarr with a known dim convention; this reads 5 raw netCDF streams, must apply the loader's `overwrite`/`combine` transforms before taking any statistic, and must decode the MPAS fill value. See [Normalization](#normalization) and `NOTES-historical-stats.md`. |

There is deliberately no `make_ocn_config.py`. It was a one-shot that lifted
Elynn's 92 channel names and swapped `FSDS`/`FSUS` for `FSNS`; once
`config-train-ocn.yaml` existed the script was a second source of truth for the
same file, so it was deleted. Edit the YAML directly.

## CO2 and aerosol channels (added 2026-08-14)

The point of training on historical rather than piControl is the forced trend,
so the atmosphere now carries it explicitly:

* **Inputs** (forcings, current-step): `global_mean_co2` (renamed from the
  scalar `co2vmr`, which the loader broadcasts to the grid), `aerindexall`
  (aerosol index) and `colccn.3` (column CCN at S=0.3%).
* **Outputs** (diagnostics, loss weight 1.0 by default): `lwp`, `lcc`, `cdnc`,
  all also in `force_positive_names`.

All five aerosol/cloud fields are `(time, lat, lon)` in every `eam.h0` file,
and all six names have finite, non-zero entries in all three stats files
(verified programmatically: 61 atmosphere names present, no zero scales; the
prognostic set is unchanged at 38, so residual normalization is untouched).
The atmosphere is now 46 in / 53 out. `config-train-cpl.yaml` was regenerated
with `make_cpl_config.py`, and its diff was exactly the channel additions.

Note `colccn.3` contains a literal dot; it is used only as a list entry and
dict key, never through dotlist overrides (which would misparse it).

`make_cpl_config.py` now also takes `--atm-ckpt` / `--ocn-ckpt` to inject
`stepper_training.<realm>.parameter_init.weights_path` — the same mechanism
Elynn's piControl coupled config used — for the production sequence of
atm-only and ocn-only pretraining followed by a coupled finetune. Pass each
component's `best_ckpt.tar` (a stepper checkpoint, not the training-state
`ckpt.tar`). With no flags the output is unchanged and the coupled realms
train from scratch.

## What has actually been run

All on A100-**80GB** nodes. "exit 0" means the full train + validation +
inline-inference cycle completed and rank 0 logged `DONE ---- rank 0`.

| test | scale | result |
|---|---|---|
| **ocn, full production globs and windows, 1 epoch** | 4 GPU | **exit 0** — setup 8m41s, epoch 3601s, valid loss 0.277, inference error 0.175 |
| ocn, 3-yr window, 2 epochs, **checkpoints on** | **8 GPU / 2 nodes** | **exit 0** — `ckpt.tar` 1.3 GB + `best_ckpt.tar` + `best_inference_ckpt.tar` written; inference error 0.535 |
| ocn, **resume from checkpoint**, epochs 3-4 | **8 GPU / 2 nodes** | **exit 0** — resumed correctly at "Beginning epoch after 2 complete epochs" |
| ocn, 4-yr window, 1 epoch | **8 GPU / 2 nodes** | **exit 0**, inference error 0.652 |
| ocn, 6-yr window, 2 epochs | 4 GPU | **exit 0**, 0.639 -> 0.550 |
| atm, 6-yr window, 1 epoch | 4 GPU, local batch 1 | **exit 0**, inference error 0.2285 |
| atm, **full production globs**, production batch 8 | **8 GPU / 2 nodes** | setup 20m45s, reached "Beginning epoch" and trained; stopped deliberately (a production epoch is far longer than an interactive allocation) |
| cpl, **full production globs and windows** | 4 GPU | **setup 50m57s** (train 1643 batches, val 92), entered training; stopped deliberately |
| **cpl, 2-yr window, reduced rollouts, checkpoints on** | 4 GPU, local batch 1 | **exit 0** — epoch 614s, valid loss 5.29, inference error 0.918, `ckpt.tar` 14.1 GB written |
| cpl, 2-4 yr windows, **production rollouts** | 4 GPU, local batch 1 | trains, all ranks balanced; no epoch completed within the allocation |
| **cpl, resume from checkpoint**, epoch 2 | 4 GPU | **exit 0** — resumed at "Beginning epoch after 1 complete epochs", epoch 625s |
| multi-node NCCL allreduce | 8 ranks / 2 nodes | correct |
| **ocn smoke, 2 epochs, on the new historical stats** | 4 GPU | **exit 0** — 0.609 -> 0.449 train, inference error 0.414 -> 0.335 |
| ocn smoke, same config, **old piControl-era ocean stats** | 4 GPU | **exit 0** — inference error 0.417 -> 0.338 |
| ocn smoke, **repeat of the new-stats run** | 4 GPU | **exit 0** — inference error 0.418 -> 0.336 |
| **atm smoke, on the new historical stats** | 4 GPU | epoch 1 completed cleanly — train 0.318, valid 0.340, **inference error 0.156**, 1489 s. Epoch 2 was killed by allocation expiry (`SignalException: got signal: 15`), not by any fault of the run |

Note the last three. Repeating the *identical* configuration moves the epoch-1
inference error by 0.0035 (0.4144 vs 0.4179), which is larger than the entire
new-stats-vs-old-stats difference (0.0025). **Runs are not deterministic at
this scale, so that A/B shows the two stats sets to be indistinguishable, not
one of them better.** The useful finding is the absence of pathology: loss
descends, validation tracks training, no NaN. Whether historical stats help
generalization over a long rollout — the actual reason to prefer them — is not
something two epochs on a smoke config can show.

### What is still NOT proven

* **The coupled config has completed a full cycle only with reduced
  rollouts** (`stepper_training.n_coupled_steps: 1` and single-step `n_steps`
  for both realms). That proves the whole pipeline end to end — training,
  validation, inline inference and checkpoint write — but at production
  rollout lengths (`n_coupled_steps: 4`, atmosphere `n_steps` up to 41) no
  epoch has finished inside an interactive allocation. Nothing suggests it
  would fail, only that it is slow; budget for it rather than assuming.
* Checkpointing is verified end to end for **both** the ocean (8 GPUs, 2 nodes)
  and the coupled model (4 GPUs): save, then relaunch and resume into later
  epochs.
* The atmosphere has never completed an epoch at production width either — only
  at a 6-year window.

## Ocean forcing: EAM names, MPAS data

The ocean is forced from the MPAS streams but under **EAM variable names**,
because atmosphere→ocean coupling is resolved by intersecting the ocean's
input-only names with the atmosphere's output names. MPAS-native names produce
an empty intersection, which used to train happily as a silently one-way
coupled model. `_validate_atmosphere_to_ocean_coupling` now **raises** when the
intersection is empty and the ocean declares next-step forcings, and **warns**
when only some of them match — a warning rather than an error because a
next-step forcing may legitimately come from the ocean's own forcing window,
and because this code also runs when loading a trained checkpoint.

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

`FSNS` replaces the `FSDS`/`FSUS` pair because MPAS only carries net shortwave.
ACE therefore predicts `FSNS` (50 outputs, not 51). To revert, put `FSDS` and
`FSUS` back in the atmosphere's `out_names`, loss weights and
`force_positive_names`, and split the ocean's shortwave input accordingly.

**`mask_and_scale: true` is required on every MPAS stream.** Those files flag
land with `_FillValue = 1e20` rather than NaN. Without it land loads as a
literal 1e20 in the targets while output masking writes NaN over the same
points; the loss only zeroes points where the *target* is NaN, so training dies
with `Loss is NaN-valued`.

`icebergHeatFlux` is excluded: identically zero across the run, and its stats
scale is 0, so normalizing gives 0/0.

## LANDFRAC

`LANDFRAC` and `sea_surface_fraction` are EAM fields absent from the MPAS
streams, but the coupled ocean needs them (a cell's sea ice fraction is
`ocean_sea_ice_fraction * (1 - LANDFRAC)`; `mask_2d` is binary and cannot
substitute — ~20% of ocean cells are coastal with fractional land). Merge
members must share `sample_start_times`, so they are materialised on the ocean's
5-day axis:

    uv run python make_landfrac_5d.py /pscratch/sd/m/mahf708/e3sm-hist-aux/landfrac5d

126 year-files, 69 MB (the field is constant in time, so it compresses hard).

## Normalization

All three configs now use stats computed **from the historical run itself**,
restricted to the training windows:

    /pscratch/sd/m/mahf708/2026-08-13-E3SMv3-historical-stats/
        train-only/atmosphere/   <- in use by config-train-atm and -cpl
        train-only/ocean/        <- in use by config-train-ocn and -cpl
        atmosphere/              <- full record; leaks, see below
        ocean/
        _partials/               <- per-file partials; re-aggregate any window

This replaced the piControl-derived stats, which were a different climate with
no CO2 trend. **The inference-error numbers recorded elsewhere in this README
were all measured against the old stats and are not comparable to runs made
now.** They are kept as a record of what ran, not as a baseline to beat.

The old paths, if you need to A/B against them:

    atmosphere  .../2026-06-02-E3SMv3-piControl-105yr-coupled-stats/uncoupled_atmosphere
    coupled atm .../2026-06-02-E3SMv3-piControl-105yr-coupled-stats/coupled_atmosphere
    ocean       .../2026-08-12-E3SMv3-hist-ocean-stats-with-FSNS

Note the piControl set distinguished `coupled_atmosphere` from
`uncoupled_atmosphere`, and `make_cpl_config.py` used to rewrite the path so
the coupled config picked the former. The historical stats make no such
distinction — they come from one run, in which the atmosphere is coupled — so
that rewrite is gone. Reinstate it if you ever point back at a split set.

The ocean's previous `FSNS` entry (centre 152.24, scale 105.75 over ocean
points) was already computed from this run, because the piControl ocean stats
predate the `FSDS`/`FSUS` -> `FSNS` change. It is superseded.

`compute_hist_stats.py` and `NOTES-historical-stats.md` in this directory
document how they were made. Headlines:

* All 1501 files of every stream, no subsampling. Exact streaming algorithm
  (per-file count/mean/M2, Chan combination), so it is not an approximation.
* Computed on the field **as the loader delivers it** — transforms applied
  pointwise in loader order, so the sign flips, the `sst` Kelvin offset and
  `surface_precipitation_rate = rainFlux + snowFlux` are all baked in. A sum
  of two fields cannot be built from the two fields' separate statistics.
* **Unweighted** over (time, lat, lon), matching `get_stats.py` and therefore
  the existing files. Confirmed numerically: piControl `LANDFRAC` centering is
  0.3426250 and this run's unweighted `LANDFRAC` mean is 0.3426344, while the
  cos(lat)-weighted mean is 0.2934. Consequence when sanity-checking a value:
  unweighted global-mean `TS` is **279.5 K**, not the 287 K you get
  area-weighted. That is the convention, not a bug.
* `global_mean_co2` is included, so CO2-as-input (now enabled in
  `config-train-atm.yaml`, see "CO2 and aerosol channels" above) needs no
  hand-patched stats file.

### Coverage: 83 atmosphere, 127 ocean entries

Deliberately a **superset** of what the configs use, as the piControl files
were. Each realm file holds the union of:

* the transformed loader names the configs actually reference (56 atmosphere,
  91 ocean), and
* every other field in the raw streams that is not already a source of a
  `rename`/`overwrite`/`combine` used to build one of those (27 atmosphere,
  36 ocean) — `FSDS`, `FSUS`, `FLNS`, `FLNT`, `FSNTOA`, `TREFHT`, `QREFHT`,
  `STWat2m`, `Q_0..Q_7`, the GHG scalars, `lwp`/`lcc`/`cdnc`, `sss`,
  `riverRunoffFlux`, `evaporationFlux`, `surfaceHeatFluxTotal`,
  `layerThicknessCoarsened_1..18`, and so on.

The second rule's exclusion is what keeps a name meaning one thing:
`windStressZonal` must not sit beside `TAUX`, nor `sst` in degC beside `sst` in
Kelvin. Superseded raw names, with their replacements, are tabulated in
`NOTES-historical-stats.md`. Auxiliaries `get_stats.py` drops (`mask_*`,
`idepth_*`, `ak_*`, `bk_*`, `P0`) are excluded too.

Practical effect: reverting the `FSNS` -> `FSDS`/`FSUS` split, or adding a
channel like `TREFHT` or `sss`, needs no new stats. `FSDS` comes out 161.7 and
`FSUS` 33.0 against piControl's 164.1 and 34.5.

Three fields were dropped for a zero scale, each for a real reason worth
knowing: **`sol_tsi` is `-1.0` in all 1501 files** (a sentinel — this EAM
configuration never diagnoses total solar irradiance), `icebergHeatFlux` is
identically zero, and `layerThicknessCoarsened_0` is exactly 20.0 m
everywhere, the top coarse bin being a fixed thickness.

`_partials/` now holds every covered variable, so re-aggregating a different
time window is a seconds-long operation. Adding a *variable* is not — that
re-reads the run (3.7 TB, ~23 min on three nodes).

**Caveat on the residual file.** `f11vmr`, `f12vmr`, `n2ovmr` and `ch4vmr` have
true residual scales of 6e-15 to 6e-12, because they are slowly-varying
scalars. Residual normalization only applies to prognostic names and these are
forcings, so the values are inert — the smallest residual scale over the 38
actually-prognostic names is 2.97e-09 (`STW_0`/`Q_0`). But if you ever make one
of them prognostic, substitute its full-field scale first, the way
`global_mean_co2` already does.

**Why `train-only/` and not the full record.** It covers 1940-1990 and
2000-2040, exactly the training windows, 131399 atmosphere and 6569 ocean samples (72.0% of the
record = 90 of 125 years). The full-record set spans 1940-2065 and has
therefore seen the validation window *and* the `12yr_test` period, which is
leakage. Restricting the window is nearly free: of 350 numbers, all
temperatures, winds, fluxes, salinities, velocities, precipitation, `PS`, `TS`
and `LANDFRAC` move under 2%. Only `global_mean_co2` (scale -31%), the `STW_*`
scales, `ICEFRAC` and `iceVolumeTotal` (+11% mean) move meaningfully.

Independent check that the window is right: `global_mean_co2` restricted to the
training years comes out 3.658244e-04 / 5.128173e-05, reproducing to four
significant figures the values quoted in `config-train-atm.yaml`, which were
derived by a completely different route.

**Known wrinkle, `STW_0`.** Its full-field scale is 4.3x the piControl value
(9.69e-08 vs 2.27e-08) even in the train-only set, because the secular
stratospheric-water trend spans the training windows too; no choice of window
fixes it. Its *residual* scale — what the loss uses — is 2.974e-09 against
piControl's 2.978e-09, i.e. unchanged to 0.13%. So loss weighting is fine and
only the network input is ~4x more compressed than it was under piControl.
Worth watching if the stratosphere misbehaves; it is not obviously wrong for a
trending field.

To point a run at a different stats set without editing the configs — the
full-record set, or the old piControl set — override the paths at launch. The key
prefix differs per config: `stepper.step.config.normalization` in
`config-train-atm.yaml` and `config-train-ocn.yaml`, but
`stepper.atmosphere.stepper.step.config.normalization` and
`stepper.ocean.stepper.step.config.normalization` in `config-train-cpl.yaml`.
The ocean has only a `network` block; the atmosphere has `network` and
`residual`, and `residual` takes the *same* centering file with the
`scaling-residual.nc` scaling.

    S=/pscratch/sd/m/mahf708/2026-08-13-E3SMv3-historical-stats/train-only
    N=stepper.step.config.normalization        # atm/ocn; see above for cpl

    # atmosphere
    --override \
      $N.network.global_means_path=$S/atmosphere/centering.nc \
      $N.network.global_stds_path=$S/atmosphere/scaling-full-field.nc \
      $N.residual.global_means_path=$S/atmosphere/centering.nc \
      $N.residual.global_stds_path=$S/atmosphere/scaling-residual.nc

    # ocean
    --override \
      $N.network.global_means_path=$S/ocean/centering.nc \
      $N.network.global_stds_path=$S/ocean/scaling-full-field.nc

### What was checked when the configs were switched over

* All three configs parse (`fme.ace.validate_config` / `fme.coupled.validate_config`,
  `--config_type train`, exit 0).
* Every normalizer builds from the paths now in the configs: atmosphere network
  and residual (55 names), ocean network (91 names), and both realms of the
  coupled config. That is what catches a missing name or a zero scale.
* `config-train-cpl.yaml` was regenerated with `make_cpl_config.py`, not
  hand-edited, and its diff is exactly the six normalization paths — nothing
  else drifted.

**Not verified: no training run has been done against these stats.** The checks
above are config-level. Expect the first epoch's loss to sit at a different
absolute level than the numbers recorded in this README, because the
normalization changed; that is not a regression by itself.

## Launching

### Single node (up to 4 GPUs)

    uv run torchrun --nproc_per_node 4 -m fme.ace.train config-train-ocn.yaml
    uv run torchrun --nproc_per_node 4 -m fme.coupled.train config-train-cpl.yaml   # note the entry point

### Multiple nodes

Use `torchrun` with an explicit rendezvous, one task per node. **Verified**:
8 GPUs across 2 nodes, full train + validation + inference, exit 0.

    MASTER=$(scontrol show hostnames $SLURM_NODELIST | head -1)
    # pick an unused port: torchrun defaults to 29500 and two runs on one node collide
    srun --nodes=2 --ntasks-per-node=1 --gpus-per-node=4 \
      bash -c "uv run torchrun --nnodes=2 --nproc_per_node=4 \
                 --node_rank=\$SLURM_NODEID --master_addr=$MASTER --master_port=29517 \
                 -m fme.ace.train config-train-atm.yaml"

**Do not use the `FME_USE_SRUN=1` path on this system.** `fme` supports an
srun-native launcher (`SLURM_PROCID` + `SRUN_DIST_FILE_PATH`), but it hardcodes
`torch.cuda.set_device(0)` on the assumption of `--gpus-per-task=1`. With that
binding here every rank dies at the first collective with:

    ncclUnhandledCudaError ... Cuda failure 101 'invalid device ordinal'

Multi-node NCCL itself is fine — a minimal 8-rank allreduce succeeds when all
four GPUs are visible per node and the device is set from `SLURM_LOCALID`,
which is exactly what `torchrun` does via `LOCAL_RANK`.

That is a property of the `--gpus-per-node=4` binding used here, not of the launcher.
On Frontier the srun path is the right one: one rank per GCD with
`--gpus-per-task=1 --gpu-bind=closest` makes device 0 correct in every rank.

## Checkpointing and resuming

Set `save_checkpoint: true` (the production configs already do). Checkpoints
land in `<experiment_dir>/training_checkpoints/`:

| file | ocean | coupled |
|---|---|---|
| `ckpt.tar` (full training state) | 1.3 GB | 14.1 GB |
| `best_ckpt.tar` | 341 MB | 3.5 GB |
| `best_inference_ckpt.tar` | 341 MB | 3.5 GB |

**Resuming is automatic**: relaunch the same config against the same
`experiment_dir` and it picks up where it left off, logging

    Resuming training from <experiment_dir>/training_checkpoints/ckpt.tar
    Beginning epoch after N complete epochs

Verified for the ocean on 8 GPUs / 2 nodes (a 2-epoch run extended to 4 epochs
restarted correctly at epoch 3) and for the coupled model on 4 GPUs (1 epoch
extended to 2). Budget disk accordingly — the coupled `ckpt.tar` is 14 GB and
is rewritten at every checkpoint interval.

## Sizing: ranks, batch_size and GPU memory

Three constraints interact. Two are enforced, though only the first reports
itself clearly:

1. **`batch_size` is global** and must be divisible by the number of ranks.
   Enforced: `ValueError: batch_size must be divisible by the number of
   data-parallel workers, got 2 and 4`.
2. **The number of inference initial conditions must be divisible by the rank
   count.** Enforced in `train_config.py`, but note that when it fires inside
   the `inference` list, `dacite` reports it as an unhelpful
   `UnionMatchError: can not match type "list"`. If you see that, re-validate
   the config at the same world size to get the real message. All three
   production configs use 16 ICs, which divides 1/2/4/8/16.
3. **The atmosphere needs ~59 GB per sample**, so local batch must be 1.

Combining these: `config-train-atm.yaml` has `batch_size: 8`, which means it is
built for **8 ranks (2 nodes)**. Running it on 4 GPUs gives local batch 2 and
OOMs even on an 80 GB card. To run it on N GPUs, set `batch_size: N`.

Rank counts each config accepts as shipped (`batch_size` and every inference
block's IC count must both divide the rank count):

| config | batch_size | ICs per inference block | valid rank counts up to 16 |
|---|---|---|---|
| atm | 8 | 16, 16 | 1, 2, 4, 8 |
| ocn | 16 | 16, 16 | 1, 2, 4, 8, 16 |
| cpl | 8 | 8 | 1, 2, 4, 8 |

The atmosphere wants local batch 1, so **8 ranks is the intended atmosphere
configuration**. To go wider, raise `batch_size` to the rank count and add
initial conditions so their count stays divisible too.

Measured peak memory per GPU on `A100-SXM4-80GB`:

| config | local batch | peak/GPU | result |
|---|---|---|---|
| atm | 1 | 59.4 GB | runs |
| atm | 2 | 77.4 GB | **OOM even at 80 GB** |
| ocn | 4 | 15.5 GB | runs |
| cpl | 1 | 37.8 GB steady, **75.8 GB peak** | runs |

**All three configs require 80 GB GPUs.** The atmosphere cannot fit on a 40 GB
card at any batch size. The coupled model looks modest at 37.8 GB most of the
time but peaks near 76 GB whenever its atmosphere `n_steps` distribution draws
the 41-step rollout, so the headroom is smaller than the steady state suggests.

`n_ensemble` is not a lever for reducing memory: the energy score is
implemented for exactly 2 members and raises `NotImplementedError` for 1.

## Short test run

    uv run python configs/experiments/e3sm_hist_v20260812/make_smoke_config.py \
        configs/experiments/e3sm_hist_v20260812/config-train-ocn.yaml \
        $PSCRATCH/smoke-ocn.yaml --experiment-dir $PSCRATCH/smoke-out

    uv run torchrun --nproc_per_node 4 -m fme.ace.train $PSCRATCH/smoke-ocn.yaml
    # coupled uses a different entry point:
    uv run torchrun --nproc_per_node 4 -m fme.coupled.train $PSCRATCH/smoke-cpl.yaml

Defaults to 6 years of data, 2 epochs, batch size 4, no checkpointing, no wandb.
`--years`, `--epochs`, `--batch-size` adjust it.

Do not shrink by hand with `--override`: dotlist overrides cannot index into a
list, so they cannot reach the `inference` blocks or the `concat`/`merge`
members. The script also keeps inference start times on the ocean's 5-day axis
and keeps the coupled realms' first timestamps equal — both are enforced at
runtime.

**Check the real exit code, not the log tail.** A `time_buffer` teardown emits
`Bad file descriptor` and semaphore tracebacks on a successful run; conversely a
trailing `echo` can mask a rank failure. Look for `REAL_EXIT=0` and
`DONE ---- rank 0`.

## Timing

Measured on A100-80GB. Dataset setup is per rank and happens before any
training; a production epoch is the dominant cost after that.

| | dataset setup | 1 epoch (train) | validation + inference |
|---|---|---|---|
| **ocn, production width, 4 GPU** | **8m41s** | **3601s (60 min)**, 410 batches | ~4 min |
| **atm, production width, 8 GPU** | **20m45s** | not measured (far longer) | - |
| **cpl, production width, 4 GPU** | **50m57s** (34m to train+val, rest inference datasets) | not measured; 1643 batches/epoch | - |
| ocn, 6-yr window, 4 GPU | 1m11s | 3m51s, 13 batches | 2m40s |
| atm, 6-yr window, 4 GPU | 2m00s | 17m50s, 1100 batches | 8m56s |
| cpl, 4-yr window, 4 GPU | 4m52s | **>55 min, did not finish** | - |

### Measured throughput at production width

From `training_samples_per_second_on_rank_0` and step timestamps in real runs
over the **full** globs and training windows. Loss was decreasing in all three,
so these are healthy runs, not stalled ones.

| config | ranks | batches/epoch | s/batch | **h/epoch** | `max_epochs` | **total** |
|---|---|---|---|---|---|---|
| ocn | 4 | 410 | 8.8 | **1.0** (measured 3601s) | 150 | **~6 days** |
| atm | 8 | 16434 | 1.4 | **6.3** | 30 | **~8 days** |
| cpl (reduced rollouts) | 4 | 1643 | 7.1 | **3.2** | 5 | ~16 h |
| cpl (production rollouts) | 4 | 1643 | not measured | **>13 (est.)** | 5 | **>3 days** |

The ocean row is the sanity check: 410 x 8.8s predicts 1.0 h and the measured
epoch was 3601 s, so the method is sound.

The coupled production-rollout row is an extrapolation, not a measurement.
Production uses `n_coupled_steps: 4` rather than 1, and each coupled step is 20
atmosphere steps, so the work per batch is at least 4x the measured row; the
atmosphere `n_steps` distribution reaching 41 adds backprop depth on top. Treat
">13 h/epoch" as a floor.

A direct measurement was attempted — production rollouts on a 2-year window
with `log_train_every_n_batches: 2` — and did not emit a single step timing in
roughly 8 minutes of training. That is consistent with the floor above being
too optimistic, but it is not a clean number (the run may still have been in
inference-dataset construction), so it is recorded as an open question rather
than a figure. **Measure this on your first real coupled run** before committing
to a walltime; it is the single largest unknown in this configuration.

Add the dataset-setup cost once per job: 9 min (ocn), 21 min (atm, 8 ranks),
51 min (cpl). A job that is requeued repeatedly pays it every time.

### Planning a production run

* **Do not submit any of these to a 4-hour queue.** Even the cheapest full
  epoch (ocean) is an hour on top of a 9-minute build; the atmosphere and
  coupled configs need many hours per epoch.
* **Request walltime as `setup + n_epochs x h_per_epoch`, then round up.**
  Setup is paid again on every requeue, and it is 51 minutes for the coupled
  config.
* **Enable checkpointing and lean on resume** rather than trying to fit a whole
  training run in one job. Resume is verified for both realms: relaunch the
  same config against the same `experiment_dir`. Keep an eye on disk — the
  coupled `ckpt.tar` is 14 GB and is rewritten at each interval.
* **Scale the atmosphere by rank count, not batch size.** It needs local batch
  1, so `batch_size` must equal the number of ranks, and every inference block's
  IC count must divide it too (see Sizing).
* **Watch `training_samples_per_second_on_rank_0`** in the log to confirm the
  rate matches the table above; a large shortfall usually means filesystem
  contention rather than a model problem.

Setup cost is dominated by reading every file's time coordinate. That is now
memoized per distinct file list, so a config that opens the same stream for
train/validation/inference pays it once rather than ~20 times. It still scales
with the number of distinct streams, which is why the coupled config (five
streams plus the atmosphere, over ~1500 files each) takes **51 minutes** before
the first batch — 34 minutes to build train and validation, the rest for the
inference datasets. Plan for that in your walltime request, and do not
interpret a long silent startup as a hang — check for growing log lines or use
`py-spy dump` on a rank.

## Uncommitted changes to `fme/` that these configs depend on

These configs will not run against a clean checkout. The following live in the
working tree and still need review before committing (nothing has been
committed or pushed).

| file | change | why |
|---|---|---|
| `core/dataset/xarray.py` | `mask_and_scale` option | raw MPAS files flag land with `_FillValue = 1e20`; without decoding it, land reaches the loss as 1e20 |
| | `add_scalar` on `OverwriteConfig` | MPAS `sst` is degC, the stats are Kelvin |
| | `combine` (linear combination of loaded fields) | MPAS has `rainFlux` + `snowFlux` but no total precipitation |
| | `get_raw_paths` stdlib-glob fast path | fsspec's glob is ~250x slower on this directory |
| | `_get_raw_times` serial + memoized | replaces a fork pool that deadlocked ranks, and a thread pool that corrupted the heap |
| | validation: chained `combine`, `overwrite` on a combine target, `overwrite` on an unknown name, `combine` source missing, `combine` target shadowing an on-disk variable, `mask_and_scale` with zarr, a mask decoding to NaN | each was a silent no-op, a silent wrong answer, or a late opaque `KeyError` |
| | combine targets get generated metadata | inheriting the first source's `long_name`/units mislabels the result |
| `core/dataset/merged.py` | advertise `combine` targets when routing merge members | otherwise the target routes to no member |
| `core/logging_utils.py` | `basicConfig(..., force=True)` | otherwise a stray root-logger call before configuration silences INFO for the whole run |
| `coupled/stepper.py` | raise when no atmosphere output feeds the ocean; warn on partial mismatch; use a module logger | catches the MPAS/EAM naming mistake, which otherwise trains as silently one-way coupled |

Test status, `python -m pytest fme -q -p no:randomly` on the final tree:
**3426 passed, 24 skipped, 3 failed**, and all three failures are verified
pre-existing (see below). The fourth known pre-existing failure,
`parallel_tests/test_step.py::test_step_regression`, only runs under `torchrun`
with `-m parallel` and so is not collected by that command.

`pre-commit` cannot run on Perlmutter: it takes an `flock` on a lockfile under
`$HOME`, and that filesystem returns `OSError: [Errno 524]`. Pointing
`PRE_COMMIT_HOME` at node-local storage does not help, because virtualenv
creation locks too. Run the same hooks at their pinned versions directly:

    export UV_TOOL_DIR=/tmp/$USER/uvtools UV_CACHE_DIR=/tmp/$USER/uvcache
    uvx ruff@0.8.1 check <files>
    uvx ruff@0.8.1 format --check <files>
    uvx --with types-PyYaml==5.4.3 mypy@1.15.0 \
        --ignore-missing-imports --check-untyped-defs <files>

Pin the versions. A newer ruff reflows `assert x, "msg"` differently and will
reformat untouched lines, adding noise to the diff.

The four review points raised against these changes have since been addressed,
each with a test:

* **`overwrite` typos are now an error.** `OverwriteConfig.apply` still skips
  names a given load did not request — that is required, since one config is
  legitimately loaded for several subsets of its names — but a name that is in
  no file at all can never take effect, so dataset construction raises. The
  check runs against the first file *after* `rename`, so it names the renamed
  variable in the error.
* **A `combine` target that also exists on disk is now an error.** The computed
  value would silently shadow the stored one, and only for the datasets that
  request the target, so two loads of the same config could disagree about what
  the name means.
* **A combine target no longer inherits its first source's metadata.** Its
  `long_name` is now the definition itself (`rainFlux + snowFlux`,
  `foo - bar`), and units are kept only when every source agrees on them.
* **A mask that decodes to NaN is now an error.** A `mask_*` variable is 0/1
  everywhere, so a NaN never belongs there; it appears when the mask carries a
  `_FillValue` and `mask_and_scale` decodes it, which inverts the masking at
  those points. Checked directly on the mask values rather than on the
  attribute, so it holds whatever the backend does. Re-verified against the
  real data: the 19 `mask_*` fields in `fmeDepthCoarsening5D` and `mask_2d` in
  `fmeDerivedFields5D` carry no `_FillValue` and decode clean.

All three configs were re-checked against the first file of every stream under
the stricter rules: no unknown `overwrite` name, no shadowed `combine` target,
no missing `combine` source.

## Known issues and gotchas

These all cost real debugging time. Several were fixed in `fme/` while
preparing these configs; they are recorded because the symptoms are misleading.

### Fixed: dataset setup could deadlock a rank during DDP construction

`_get_raw_times` used a `multiprocessing.Pool` to read each file's time
coordinate. That pool is created on a rank that has already initialised CUDA
and NCCL, and forking such a process deadlocks on pool teardown. One rank would
wedge forever:

    rank 3:      _get_raw_times -> Pool.__exit__ -> _terminate_pool -> join -> poll
    rank 0,1,2:  _verify_param_shape_across_processes        (DDP allgather)

Its peers sat in DDP's parameter allgather until the **30-minute NCCL
watchdog** fired, and the aborted collective returned garbage that surfaced as:

    DDP expects same model across all ranks, but Rank 1 has 70 params,
      while rank 0 has inconsistent 0 params
    value cannot be converted to type int without overflow

**Neither message indicates a model mismatch.** If you see them, run
`py-spy dump --pid <pid>` on every rank — the odd one out is the cause. The
watchdog output also names the straggler: the rank with the lower
`Last enqueued NCCL work` never reached the collective.

A `ThreadPoolExecutor` was tried as the fix and is **worse**: netCDF4/HDF5 is
not thread-safe, so it survives small runs and then corrupts the heap at
production width (`corrupted size vs. prev_size`, SIGSEGV, exit -11) partway
through dataset construction. The reads are now serial and memoized instead.

Related: `get_raw_paths` used fsspec's glob, which stats every directory entry
— 7.5s warm / 16.9s cold per call on this 11,278-file run directory versus
0.03s for the stdlib, for an identical result. It runs once per dataset per
rank, so it added minutes of skew and made the deadlock much easier to hit.

### Fixed: the coupled trainer logged nothing after startup

A `logging.info` call in `CoupledStepperConfig.__post_init__` runs during
`dacite` parsing, i.e. **before** `LoggingConfig.configure_logging`. The
root-logger convenience functions implicitly call `basicConfig()`, installing a
handler; the later `basicConfig(level=INFO)` is then a **no-op**, because
CPython only applies the level when the root logger has no handlers. The result
was a 15-50 minute run with an empty `out.log`, no progress output, and no
`DONE ---- rank` line — while training ran perfectly normally.

Fixed two ways: the coupled config now uses a module logger, and
`configure_logging` passes `force=True` so it is authoritative regardless of
prior state. If you add diagnostics to any `__post_init__`, use a module
logger (`logging.getLogger(__name__)`), never `logging.info`.

This was observability only — no training result was affected.

### Divisibility rules that fail in confusing ways

* `batch_size` must be divisible by the rank count. Clear error.
* The number of inference initial conditions must be divisible by the rank
  count. The error is raised in a `__post_init__` inside the `inference` list,
  so `dacite` reports it as `UnionMatchError: can not match type "list"`, which
  says nothing useful. Re-validate the config at the same world size to see the
  real message.

### Other

* **Inference start indices need room for the rollout.** `max_start_index +
  window_length` must not exceed the dataset length, or you get
  `The maximum start index N plus window length M must be less than or equal
  to the number of steps in the dataset`.
* **The inference aggregator defaults to a metric at step 20**, so any run with
  fewer than 21 forward steps dies with `MetricNotSupportedError: step_mean
  step 20 exceeds n_forward_steps`. Either use >= 21 steps or set
  `aggregator.log_step_means: []`.
* **`torchrun` defaults to port 29500.** Two runs on one node collide with
  `EADDRINUSE`; pass `--master_port`.
* **Check the real exit code, not the log tail.** A `time_buffer` teardown
  emits `Bad file descriptor` and semaphore tracebacks on a *successful* run,
  and a trailing `echo` can mask a rank failure. Look for `REAL_EXIT=0` and
  `DONE ---- rank 0`.

### Keep the config on a shared filesystem, not `/tmp`

`/tmp` is node-local. A config written to `/tmp` and launched with `srun` works
only when the scheduler happens to place the job on the node that wrote it, so
this fails **intermittently** — the same command can succeed and then fail
minutes later on a different node. All four ranks die instantly with

    FileNotFoundError: [Errno 2] No such file or directory: '/tmp/.../smoke-ocn.yaml'

but `torchrun` buries that ~80 lines above a `ChildFailedError` summary that
reports only `exitcode: 1` per rank and says nothing about the cause. Write
generated configs to `$PSCRATCH`. The same applies to anything else a job
reads: `NOTES-historical-stats.md` records that the stats run lost an entire
atmosphere shard this way, because the per-node `/tmp` partials the shards
wrote were invisible to the aggregating process.

### Do not `git stash` while a run is starting

Config parsing is strict (`dacite.Config(strict=True)`), so if the working tree
briefly loses the `combine` / `mask_and_scale` fields these configs use, a
launching run fails with a misleading

    UnionMatchError: can not match type "dict" to any type of
      "train_loader.dataset" union

pointing at the dataset rather than at the missing dataclass field. Cost me a
run. The same masking applies to any `__post_init__` `ValueError` raised inside
a union-typed field.

### Pre-existing, unrelated to these configs

Four tests fail on this checkout, all verified identical with every local
change stashed, so none are caused by this work:

    fme/core/distributed/parallel_tests/test_step.py::test_step_regression[sm_with_atmos_corr-...]
    fme/ace/registry/test_stochastic_sfno.py::test_isotropic_noise[8-16]
    fme/ace/test_ice_train.py::test_train_and_inference          (ImportError)
    fme/ace/test_train.py::test_train_and_inference[HEALPix]     (ValueError)

Everything else passes; see the counts above.
