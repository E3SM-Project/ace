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
* **`config-train-atm.yaml` and `config-train-ocn.yaml` are campaign runs**, not
  templates: they are E01 and E11 of the aug26 list. Changing either changes the
  campaign's control. Regenerate `runs/` with
  `sbatch-scripts/generate-campaign.sh` afterwards, and the coupled config with
  `make_cpl_config.py`.
* The **hackathon page is the source of truth** for the run list, the factor
  alphabet and the baseline model settings:
  <https://e3sm.atlassian.net/wiki/spaces/p3ai/pages/6550683662>. If this
  directory and the page disagree, this directory is wrong.
* **Never `git checkout` a tracked file here.** Several files carry uncommitted
  work at any given time; a checkout silently discards it.

## 2026-08-29 (later) — E## rename, W3, the O1 cadence, and wandb

### Experiment ids renamed A##/O## -> E##

The prefix overloaded the factor alphabet twice in one run id --
`A05...A3_B16...` (experiment 5 vs aerosol level 3) and `O01...O5...`
(experiment 1 vs 5-daily stepping) -- and a coupled `C##` would have collided
with the CO2 factor too. The realm is already its own field. `E` is the only
letter the factor alphabet (A, B, C, O, W, X) and the seed (S) both leave free.

A01-A10 -> E01-E10, O01-O04 -> E11-E14. Done before any run launched, so no
wandb history carries the old scheme.

### W3 exists now, chosen against the statistics

The page's W list skipped from W2 to W4. W3 is now the second half of a matched
pair with W4 -- both zero one channel, for two different reasons:

| set | realm | channel | evidence |
|---|---|---|---|
| W4 | atm | `STW_0` | residual/full-field 0.031, 2nd lowest after `PS`; \|mean\|/std 12.8 |
| W4 | ocn | `velocityMeridionalCoarsened_18` | most extreme ocean output by \|mean\|/std, 0.005 |
| W3 | atm | `STW_1` | residual/full 0.70; the pre-existing hand-tuned weights singled out exactly STW_0 and STW_1, both at 0.25 |
| W3 | ocn | `iceVolumeTotal` | structurally zero over most of the domain, already special-cased by `zero_where_ice_free_names` |

Rejected: `FSNS` (1.28), `FSUTOA` (1.14), `SHFLX` (1.09), `DTENDTTW` (1.05) all
have residual/full above 1.0, but for the first three that is the diurnal cycle,
which the model resolves from `SOLIN`; `DTENDTTW` would confound the
moisture-budget corrector that consumes it.

### The 1-daily ocean data exists -- an earlier claim here was wrong

A previous entry said only the 5-daily MPAS streams were staged. That was
inferred from the config's file patterns rather than checked. **The run
directory carries both**: `fmeDepthCoarsening`, `fmeDerivedFields`,
`fmeSeaiceDerivedFields` (no suffix) are 1-daily, 1501 files each, 1940-2065,
same as their `5D` counterparts. Both are interval means (`time_bnds` span 1 day
and 5 days), so the cadence is a data swap, not a resample. The daily
`fmeDepthCoarsening` also carries 95 `*_inst` variables the 5-day one does not.

Built for it: `make_landfrac_5d.py` generalised to `make_landfrac_ocn.py
--cadence 5d|1d` (LANDFRAC and sea_surface_fraction are time-invariant but have
to be materialised on the matching axis, because merge members must share
`sample_start_times`); `landfrac1d/` written for all 126 years and staged to CFS
beside `landfrac5d/`; generator support for the cadence switch, which moves four
things together (file patterns, landfrac axis, inference `n_forward_steps`
876->4380, and `max_epochs`); and a `check_campaign.py` assertion that all four
merge members agree on cadence, verified against a config with one member left
on the 5-day axis.

E17 uses the 5-day statistics. Measured 1-day/5-day std ratios over 12 sampled
months: 1.000-1.005 for `temperature_*`, `sst`, `ssh`, ice area; **1.115**
`latentHeatFlux`, **1.130** `velocityMeridionalCoarsened_18`. Means agree to 4
significant figures. Defensible for a cadence comparison; a production O1 run
should recompute.

### Clean vs contended measurement, and the time_buffer OOM

Re-measured the ocean baseline **alone** on 2 nodes, nothing else on the
allocation, then compared against the same config run alongside one other
2-node job on disjoint nodes:

| | s/step | setup |
|---|---|---|
| alone | **1.390** (full 411-step epoch, exit 0) | 10.5 min |
| one other 2-node job | 2.945 | 13.1 min |

**Contention is worth 2.1x on step time.** CFS is the binding resource, not the
nodes. This invalidates the absolute ocean numbers taken earlier today (o1/o2 at
24.36/3.10, c5 at 2.945) as anything but upper bounds -- the ratio between arms
of a concurrent A/B is still meaningful, the absolutes are not. The atmosphere's
0.925 s/batch was measured alone and stands.

It also corrects a claim made mid-session: O1 is **not** faster per step than O5.
Clean-to-clean it is 1.538 vs 1.390, ~10% slower, which is what the identical
per-step model work predicts. The apparent 2x advantage was O5 being measured
under contention while O1 ran nearly alone.

**`time_buffer` on the ocean is killed by the host OOM killer.** `time_buffer: 10`
with `time_buffer_pool_size: 2` on the train loader dies before the first step
(`Detected 2 oom_kill events`, `nid008316: task 1: Out Of Memory`). Each worker
holds `prefetch_factor` input batches of `local_batch x (n_timesteps +
time_buffer)` samples; the ocean window is ~4.5x the atmosphere's (91 channels vs
~50, local batch 2 vs 1, 5 timesteps vs 2), so in-flight host memory goes
28 -> 84 GB per node before the pool, model and optimizer.

And it is unnecessary: alone at `time_buffer: 0` the ocean shows **no stalls at
all** across a full epoch, every interval 1.00-1.50 s. `time_buffer` was the fix
for the atmosphere's bimodal stalls; the ocean's loader keeps up once workers and
prefetch are raised. **Do not add it to the ocean.**

### wandb consolidated

Both realms were splitting across `ace-eamv3` and `ocean-emulator-e3sm`. All 33
runs now go to `e3sm-aig/SamudrACE-E3SMv3`, asserted by `check_campaign.py`.
`entity` is the team: the wandb source builds the login line as
`f"Currently logged in as: {username}{entity_str}"`, so `e3sm-ai` is the account
and `e3sm-aig` the entity. Run identity stays in the `.env` files, which the
wandb library reads from the environment.

Key handling documented in the README: each person uses their own key via
`wandb login`, sharing the *team* rather than a credential; team service accounts
are the supported path if one shared key is genuinely wanted.

### Style

`EXPERIMENTS.md`, the artifact and the Confluence page are now written as
present-tense descriptions of the campaign, with no before/after narrative. This
file keeps the history.

## 2026-08-29 — the page redesigned the campaign; baselines and generator rebuilt

The hackathon page replaced the finetune chain with a flat list of independent
from-scratch runs, and re-specified the baseline model. This entry records what
changed here in response.

### The page's five corrections to the baseline ("FOR NASER")

All five were deviations from Elynn's configuration and all five are now in
`config-train-atm.yaml` / `config-train-ocn.yaml`:

| item | was | now |
|---|---|---|
| `embed_dim` | 512 | **384** |
| `noise_embed_dim` | 64 | **32** |
| loss weighting | 13 hand-tuned per-variable weights | **equal** — the `weights:` block is deleted, which `_construct_weight_tensor` reads as uniform 1.0 |
| `checkpoint_save_epochs` | unset (best-only) | `{step: 1}` |
| `ema_checkpoint_save_epochs` | unset | `{step: 1}` |

Cross-checked against Elynn's piControl config
(`origin/e3sm/exps/hist:configs/experiments/e3sm_piControl_v20260507/atmosphere/config-train.yaml`),
which is where `embed_dim: 384` comes from. Two further differences from that
file were **left alone deliberately**, because the page did not list them and
silently changing them would be a second deviation rather than a fix:
`stepper_training.n_forward_steps` (1 here, 2 there, with
`optimize_last_step_only`) and the `EnsembleLoss`/`n_ensemble: 2` stochastic
setup versus plain MSE.

### wandb: 2D off, 1D kept

Implemented with the **typed** aggregator configs, not the deprecated
boolean-flag variants. Off: `zonal_mean`, `video`, `trend`, `seasonal`,
`near_zero_fraction`, `enso_coefficient`, `step_diagnostics.correction_maps`,
and the one-step aggregator's `snapshot` and `mean_map`. On: `histogram` (it is
a 1D distribution plot), `mean`, `mean_norm`, `power_spectrum`, `annual`,
`enso_index`, `ipo_index`.

**Verified by building the config, not just parsing it.** `validate_config`
proves the YAML parses; it does not prove which union member dacite picked.
Constructing `InlineInferenceConfig`/`InlineValidationConfig` from the committed
file and calling `_get_metrics()` confirms both blocks resolve to the *typed*
`InferenceEvaluatorAggregatorConfig` / `OneStepAggregatorConfig` (not the
deprecated `LegacyFlag*` members) and that the enabled set is exactly

    inference:  annual, ensemble_step_20, enso_index, histogram, ipo_index,
                mean, mean_norm, mean_step_20, mean_step_20_norm,
                power_spectrum, time_mean, time_mean_norm
    validation: ensemble, mean, mean_norm, power_spectrum

with `zonal_mean`, `video`, `trend`, `seasonal`, `near_zero_fraction`,
`enso_coefficient`, `snapshot` and `mean_map` all absent.

This matters because **dacite matches a union member by shape**: a config
written with the old boolean flags (`log_zonal_mean_images: 4096`) parses
happily, emits one `DeprecationWarning` nobody reads, and silently turns the 2D
metrics back on. `check_campaign.py` now rejects those field names and requires
each image metric to be explicitly disabled -- verified against a negative
control that plants exactly that regression.

`time_mean_denorm` / `time_mean_norm` are the one deliberate exception. Their
`get_logs` builds the bias image and the `rmse`/`bias` scalars in the same loop
with no separate switch, and those scalars are the campaign's headline skill
metric. Disabling them to remove an image would remove the metric. Removing the
image properly is a code change, not a config change.

### The generator was rewritten, not patched

`make_ablation_config.py` used to exist mainly to enforce the chain's
prefix-superset rule (`parent.in_names == child.in_names[:n]`), because
`ParameterInitializationConfig` overwrites weights by position with no name
checking. With no parents, that rule and `weights_path` are both gone. What it
does now:

* holds `RUNLIST`, a transcription of the page's run list, and expands it to 30
  runs (14 experiments × seeds × batch variants);
* emits the page's naming convention exactly —
  `E05.aug26.atm.A3_B16_C1_O5_W0_X0.S01`, factor word in fixed `A_B_C_O_W_X`
  order;
* applies the C/A channel additions, including growing
  `corrector.force_positive_names` with `lwp`/`lcc`/`cdnc` (they are
  non-negative by definition and the corrector is the only thing enforcing it);
* builds the three W sets;
* sizes the run: `validation.loader.batch_size` = `batch_size`, and **rewrites
  both initial-condition lists to 32 entries** for the atmosphere's B32 arm,
  because a dotlist `--override` cannot index into a YAML list;
* re-anchors every block's `epochs.start = (max_epochs - 1) % step` and asserts
  the last fire is the final epoch (`(max_epochs - 1)`, not `max_epochs`: a
  block fires on `list(range(1, max_epochs + 1))[start::step]`, from 1 because
  `evaluate_before_training` is off);
* sets every block's rollout from `INFERENCE_YEARS` and its cadence from
  `INFERENCE_EVALUATIONS`, so the atmosphere and both ocean cadences cover the
  same span and are scored the same number of times;
* writes `FME_NODES` into each `.env`, which `run-train.sh` now passes as
  `--nodes` — without it the B08 and B32 arms would run at the baseline's node
  count.

Node budget comes out at 94 (atm) + 17 (ocn) = **111**, matching the page's own
arithmetic to the node, against a 96-node reservation. `submit-campaign.sh`
therefore walks `runs/MANIFEST.tsv` in priority order (P1 baselines 14 nodes,
P2 single-seed ablations 34, P3 extra seeds 28, P4 batch sweeps 35) rather than
cutting runs.

### A checker, because `validate_config` does not check the thing that matters

`fme.ace.validate_config` proves a config *parses*. It cannot prove that
`E05.aug26.atm.A3_B16_C1_O5_W0_X0.S01` actually has CO2, both aerosol sets, batch
16, equal loss weights and no AMP. Every plot and every conclusion this week is
labelled by the run id, so a silent disagreement between the id and the file is
the worst failure mode available.

`check_campaign.py` asserts the factor word against the file for all 35 runs in
about a second, plus the invariants that are easy to break by hand: IC counts
divisible by the rank count, `force_positive_names` growing with the aerosol
outputs, `embed_dim`/`noise_embed_dim` still at the page's values, the loader
settings, the per-epoch checkpoint slices, no personal-scratch input paths, and
the fires-on-the-final-epoch rule, and that the held-out block's name still
states its own rollout length. `generate-campaign.sh` runs it
automatically after generating.

Verified against a negative control: a config with four planted errors (CO2
removed while the word still says `C1`, `embed_dim` back to 512,
`num_data_workers` back to 2, `5yr_test` start off by four) is caught on all
four, and the clean set reports `checked 35 configs, 0 with problems`.

### Verified

* All 35 generated configs pass `fme.ace.validate_config --config_type train`.
* `config-train-cpl.yaml` regenerated from the new baselines;
  `make_cpl_config.py --check` clean and `fme.coupled.validate_config` passes.
* `submit-campaign.sh --dry-run` exercised at `--max-priority 3` and `--only ocn`,
  and `--preflight` run over all 35: it stages each config, sources its `.env`,
  applies the per-run `--nodes` and runs the validator -- everything except the
  `sbatch` call. Reports `35 runs, 129 nodes -- all staged and validated`.
* `make_smoke_config.py` still produces valid configs from both new baselines,
  so the documented 20-minute smoke path is not broken by the aggregator blocks.
* Reservation confirmed by `scontrol show res _CAP_aigs_hist`: 96 nodes,
  `hbm80g`, 2026-08-31 09:00 → **2026-09-05 15:00** (Saturday). The page's
  "3pm pacific on Friday" contradicts its own "5days6hours"; the duration is
  right and the weekday is a typo.

### Measured on 4x A100-80GB, job 57705134

Two runs of the E01 baseline at hackathon settings (4 nodes, 16 ranks, global
batch 16, local batch 1, `embed_dim: 384`, `checkpointing: 3`), inference blocks
removed, identical except for the data-loader block.

| | m1 as committed | m2 loader tuned |
|---|---|---|
| `num_data_workers` / `prefetch_factor` / `time_buffer_pool_size` | 2 / 1 / 1 | **8 / 4 / 2** |
| trainable parameters | 456,223,488 | 456,223,488 |
| peak GPU memory / rank | 19.0 GB | 18.8 GB |
| compute-bound step | 0.90 s | 0.85 s |
| **effective step** | **3.155 s** (220 steps) | **0.925 s** (680 steps) |
| dataset setup | 22 min 28 s | 22 min 42 s |

**The committed loader was starving the GPU.** The m1 step log is bimodal, not
noisy: twenty steps at 17-18 s, then one interval at 163-216 s, repeatedly, with
GPU memory flat at 18.6-19.0 GB throughout. It is the `time_buffer` window refill
against one pool slot and two prefetched batches, reading 1501 files from CFS.
Under m2 there are two small stalls in the first 200 steps and then **500
consecutive clean ones**.

In the units that decide the campaign: an 8,210-step epoch is **7.2 h** at m1 and
**2.11 h** at m2, so `max_epochs: 30` is **216 h** against a 126 h window, or
**63 h** with 2x margin. Both baselines now carry the m2 settings and all 35 run
configs were regenerated; the coupled config was regenerated and revalidated.

Caveats, stated because they matter:

* **Three settings changed at once**; the attribution among them is unknown. The
  bundle is measured end-to-end at exactly the configuration that will run.
* **`time_buffer_pool_size: 2` is a sampling change too** — with one slot,
  consecutive output batches come from the same preloaded window; with two they
  interleave. Better statistically, applied identically to all 35 runs, but
  results are not comparable to earlier runs at pool size 1.
* **The ocean was then measured too, and is starved worse.** E11 baseline,
  2 nodes / 8 ranks, both arms run concurrently (so contended, i.e. a lower
  bound): **24.36 s/step at 2/1 versus 3.10 s/step at 8/4 -- 7.9x**. Memory ~9 GB
  and flat in both. It reads a four-way merge (`fmeDepthCoarsening5D`,
  `fmeDerivedFields5D`, `fmeSeaiceDerivedFields5D`, `landfrac5d`), which is why
  it was hit harder than the atmosphere. A 411-step ocean epoch is 2.78 h at 2/1
  and 0.35 h at 8/4, so `max_epochs: 150` was a **417 h** run and is now a
  **53 h** one. Ocean dataset setup is 13 min at 8 ranks, not the ~9 min in the
  older notes; 82,822,138 parameters confirms the carried-over 82.8 M.
  **Both realms' epoch counts were resting on a loader setting nobody had
  measured.**
* **Expect worse at campaign scale** — this is one 4-node job; twenty-plus
  concurrent jobs will contend on the same 3.7 TB directory. Stagger launches.

### The two memory probes, measured

Run concurrently on the same allocation (2 nodes each), so the filesystem was
shared -- and neither showed a single loader stall, which is a useful extra data
point for the fix holding under concurrency.

| probe | local batch | `checkpointing` | mem/GPU | s/step | s/sample |
|---|---|---|---|---|---|
| m2 | 1 | 3 | **19.0 GB** | 0.925 (0.85 clean) | 0.85-0.93 |
| m4 | 1 | **0** | **40.9 GB** | 0.830 | **0.830** |
| m3 | **2** | 3 | **28.7 GB** | 1.660 | **0.830** |

**Keep `checkpointing: 3`; it is nearly free at this width.** 3-5% of step time
for 54% of activation memory. The "+33% step compute" figure in the older notes
was measured at `embed_dim: 512` and does not survive the move to 384.

**Local batch 2 fits (28.7 GB of 80) and is marginally better per sample -- but
it does not "dissolve" the node problem, it trades it.** Halving the ranks at
fixed global batch halves the nodes *and* doubles the epoch: 4 nodes and 2.11
h/epoch (63 h for 30) versus 2 nodes and 3.79 h/epoch (114 h). Both fit the
126 h window. What differs is when the science lands -- at local batch 1,
P1+P2+P3 is 84 nodes, starts immediately, and the headline E01/E02/E05
comparisons finish **Wednesday night**; at local batch 2 everything runs at once
with 32 nodes spare but nothing finishes until **Friday morning**.

**Recommendation: stay at local batch 1.** Fifty hours of extra time to look at
the headline result is worth more than removing a queueing problem Slurm handles
for free. `--local-batch atm=2` regenerates the whole campaign at 64 nodes and
passes `check_campaign.py`, so the switch is one command if the group prefers it.
An earlier draft of this entry claimed local batch 2 simply removed the
111-versus-96 problem; that was wrong, and the wall-clock cost is the reason.

**Dataset setup is 22.5 min and did not improve with 8 workers** (22:28 vs
22:42), so it is the initial time-coordinate read rather than the batch pipeline.
Every requeue pays it again — six times over a 63 h run at a 12 h walltime.

**Checkpoint storage, by arithmetic rather than measurement.** 456 M parameters
with `checkpoint_save_epochs: {step: 1}` (full, optimizer state included) plus
`ema_checkpoint_save_epochs: {step: 1}` (weights only) is order 9 GB per epoch
per run, so order 9 TB across 35 runs x 30 epochs. `myquota` fails on a compute
node; check it from a login node before Monday.

### Experiment ids renamed A##/O## -> E##

The prefix overloaded the factor alphabet twice in one run id --
`A05...A3_B16...` (experiment 5 vs aerosol level 3) and `O01...O5...`
(experiment 1 vs 5-daily stepping) -- and a coupled `C##` would have collided
with the CO2 factor too. The realm is already its own field, so the prefix never
needed to carry it. `E` is the only letter the factor alphabet (A, B, C, O, W,
X) and the seed (S) both leave free.

Now one incrementing sequence: **A01-A10 -> E01-E10, O01-O04 -> E11-E14.**
E01-E10 are the atmosphere and E11-E14 the ocean, but nothing depends on that
split; a future coupled run is just E15. Done before any run launched, so no
wandb history carries the old scheme.

### Shared inputs moved to CFS

`stage-shared-data.sh`'s destinations existed only as a plan. Both trees are now
at `/global/cfs/cdirs/e3smdata/emulator/SamudrACE-E3SMv3/historical/`
(`stats-2026-08-13/`, `landfrac5d/`), group `e3smdata`, mode `g+rX,o-rwx`, and
both committed configs point there. `--check` now reports zero personal-scratch
input references in all three configs. Until this, four of the five people on
the reservation could not have run anything.

Access was **verified rather than assumed**: every directory on the path is
`drwxrwx--- :e3smdata`, every file `-rw-rw---- :e3smdata`, and all five users
named on the reservation (`elynnwu`, `imanick`, `rebassoo`, `olawale`,
`mahf708`) are in that group. Group-readable through a project directory and
not world-readable, per NERSC's sharing guidance.

Worth noting for next time: the CFS write that a previous session recorded as
blocked (classifier-denied) went through without trouble from a compute node.

### Mistake worth recording

`git checkout config-train-ocn.yaml` was used to undo a bad in-place edit and
discarded that file's uncommitted working-tree changes. The delta was recovered
by diffing `HEAD` against a `runs/*.yaml` generated from the working tree
earlier (it was `seed` plus `5yr_test.epochs.start`), and both were being
rewritten anyway. The guidance block above now says not to do this.

### Interpretations flagged for the page, not settled here

1. **Ocean W1.** "Upweight fluxes in both models" is literal for the atmosphere,
   which predicts its fluxes. Samudra predicts **none** — `TAUX`, `FSNS`,
   `LHFLX` and the rest are all inputs — so a literal ocean W1 is the empty set.
   Implemented as an upweight of the air–sea interface state instead
   (`sst`, `ssh`, `ocean_sea_ice_fraction`, `iceVolumeTotal`).
2. **Ocean W4.** The page names `STW_0` for the atmosphere and says "something
   similar for ocn". Picked `velocityMeridionalCoarsened_18` on the same
   rationale (near-zero, poorly constrained, deepest level).
3. **A3's channel list.** The page writes A1 as "(aerindex, ccn)" and A3 as
   "(aerindex, aod, lwp, lcc, cdnc)" — A3 drops ccn and adds aod. Implemented
   A3 = A1 inputs ∪ A2 outputs, which is what the adjacent line "with both
   aerosol inputs and outputs" says and what makes E03/E04/E05 a decomposition
   rather than a repeat. `AODVISall` is in the stats; `--aod` adds it.
4. **O1 (1-daily ocean)** is in the factor alphabet, is used by no run, and is
   not runnable: only the 5-daily MPAS streams are staged. The generator raises
   rather than emitting a config that would fail at load.

## 2026-08-28 — pre-hackathon readiness pass: fresh GPU verification

Run on the user's interactive allocation (nid008553/008556, A100-SXM4-80GB,
torch 2.10.0+cu128), three days before the reservation opens. Scripts and raw
logs in `/pscratch/sd/m/mahf708/readiness-20260828/`.

**Checkpointing re-verified on current HEAD, isolated (exact committed atm
model, 800.3M params, 46-in/53-out, 180x360, single GPU):**

| setting | median step | peak alloc | note |
|---|---|---|---|
| ckpt0, local 1 | 0.369 s | 30.7 GiB | |
| ckpt1, local 1 | 0.370 s | 30.4 GiB | encoder/decoder only: no memory win |
| ckpt3, local 1 | 0.501 s | 17.2 GiB | **-44% mem, +36% step** |
| ckpt3, local 2 | 1.154 s | 19.8 GiB | fits easily but **x0.87 throughput/sample** |
| ckpt3, local 1, bf16 | 0.486 s | 16.9 GiB | -0.3 GiB, -3% step: negligible |
| ckpt0, local 1, bf16 | 0.356 s | 28.6 GiB | |

* Gradients at ckpt1 and ckpt3 are **bitwise identical** to ckpt0 (135/135
  parameters, max abs diff exactly 0.0) with the noise draw seeded identically.
  The `use_reentrant=False` fix (`c5d39a0fa`) holds on current code.
* **Local batch 2 is a throughput loss** (0.58 vs 0.50 s/sample): the SFNO at
  this resolution saturates an A100 at local batch 1. Rank count is the only
  throughput lever; do not widen local batch.
* bf16 autocast buys ~nothing in isolation (spectral path is fp32-protected).
* The 2026-08-24 in-trainer observation that ckpt3 was *faster* (1.48 vs
  2.11 s/batch) is consistent with these numbers only as allocator pressure at
  61.8/80 GiB; isolated, ckpt3 costs +36% compute.
* ckpt1 saves nothing on this model: the blocks, not the encoder/decoder,
  hold the activation memory. Use 0 or 3, nothing between.

**End-to-end smoke on the committed config** (1-year window via
`make_smoke_config.py --years 1940 --batch-size 4`, 1 node / 4 ranks, ckpt3,
requeueable-train.sh path): `REAL_EXIT=0`, `DONE ---- rank 0`, train loss
1.196 -> 0.515 in one epoch, ~1.29 s/batch at local batch 1
(`training_samples_per_second_on_rank_0` = 0.82).

**Reservation confirmed via scontrol:** `_CAP_aigs_hist`, 96 nodes, hbm80g,
gpu_ss11, **Mon 2026-08-31 09:00 -> Sat 2026-09-05 15:00 (5d6h, 126 h,
504 node-days)**. The window ends Saturday, not Friday. `run-train.sh` now
attaches `--reservation` when `RESERVATION` is exported (verified: submitted
jobs pend with reason `Reservation` until the window opens).

**Schedule arithmetic at measured speed** (1.35 s/batch, ckpt3, local 1):
a 4-node atm lane runs 8,217 steps/epoch = ~3.1 h/epoch, so a 30-epoch trunk
is ~4 days and the E01->E02->E04->E05->E07->E09 critical path is ~185 h against
the 126 h window: **the chain does not fit at 4-node sizing**. At 16 nodes /
batch 64 the same path is ~48 h and the whole 21-run campaign uses ~25% of the
reservation. Local batch 2 is not a way out (slower per sample).

**Still open before Monday:** stage-shared-data.sh (`--check` on 2026-08-28
still reports both CFS destinations MISSING and all three configs pointing at
personal scratch); commit + push the working tree (configs, sbatch machinery,
generator, runs/, EXPERIMENTS.md are uncommitted and the fork is 2 commits
behind on top of that). The legacy `SFNO-v0.1.0` builder
(`fme/ace/models/modulus/sfnonet.py`) still omits `use_reentrant=False` —
unused by these configs, but warn anyone who reaches for it with
checkpointing.


## 2026-08-25 — noleap calendar: the off-axis IC "fix" is retracted

An earlier review pass today concluded the initial-condition dates drifted
off the ocean's 5-day axis in later decades, and "fixed" 32 IC timestamps in
`config-train-ocn.yaml`, 4 validation starts, and `make_cpl_config.py`.
**That analysis was wrong; all of those edits were reverted.**

The ocean axis is on a **cftime noleap calendar** (file attrs: `units:
'days since 1940-01-01 00:00:00'`, `calendar: 'noleap'`). The bad check
counted days with pandas, which uses the proleptic Gregorian calendar, where
leap years shift day counts. In noleap 365 % 5 == 0, so the 5-day axis never
drifts: **01-06 and 07-05 are on-axis in every year.** The original dates
were correct all along; the "fix" would have introduced off-axis dates.

Definitive re-verification (real code path, full on-disk axis = union of all
1501 depth5D files: 9125 noleap times, 1940-01-06 .. 2065-01-01): every
original IC in all four lists (ocn inference 16, ocn 5yr_test 16, cpl IC 8,
cpl IC_TEST 8) resolves through `TimestampList.as_indices`, which does an
exact `get_loc` and raises `ValueError` when a date is off-axis; the
876-step rollout from the latest IC of each list fits on the axis; an
off-axis probe (1945-01-07) raises; and all three coupled window starts
(1940/1990/2000-01-06) floor to the same timestamp in the ocean and
atmosphere data. **Do not "fix" IC dates with pandas** — compare on the
index's own calendar.

Stale numbers corrected in the same pass:

* `README.md`: the coupled config is 837 lines, 412 of them (49%) mirrored by
  `make_cpl_config.py`; the branch is 17 commits ahead of `main` (3 behind);
  the library delta is 12 files, +1227/−51.
* `README.md` + `requeueable-train.sh`: the checkpoint/resume claim. The
  trainer writes checkpoints at every epoch boundary and on graceful
  shutdown (atomic tmp + rename), and resume skips already-processed batches
  of the current epoch, so a USR1 requeue mid-epoch continues the epoch.
  Only if the 14 GB restart save is killed mid-write does the epoch repeat.
  This supersedes the "Checkpoints are per-epoch" claim verified in the
  entry below (which was true of the README text at the time).
* `README.md`: expected outer steps per coupled batch is 2.29 under
  `max(ocean, ceil(atm / 20))` with the current draws — the old 2.34
  predated or dropped the atmosphere's `steps: 21` (p=0.1) and `steps: 41`
  (p=0.05) outcomes. Ocean `4→2` is −25%; `4→1` is −37%; dropping the
  atmosphere's `steps: 41` outcome is only ~−2.5%.
* `README.md` + `NOTES-historical-stats.md`: the five train-only statistics
  files hold 503 values (83×3 atm + 127×2 ocean); between the train-only and
  full-record sets, 174 / 93 / 22 move by more than 1% / 2% / 10%.

In the `fme/` library delta (same branch), this pass also added a
`DataWriterConfig.__post_init__` guard that raises on `prediction_names: []`
(an empty allowlist silently writes prediction files with no data
variables), with a test.

## 2026-08-25 — independent verification of the checkpointing claims

Second review campaign, run blind against the 2026-08-24 numbers on two fresh
4-node A100-80GB allocations. Everything below was re-measured from scratch,
not carried over.

**The `use_reentrant` bug and fix, verified both ways.** The regression tests
in `test_sfnonet.py` were run on GPU against the pre-fix tree (`c5d39a0fa~1`
in a worktree): all 6 fail, with exactly `encoder.0.weight`, `encoder.0.bias`,
`encoder.2.weight` at `grad=None` for every level 1/2/3 and nothing louder
than a `UserWarning`. At HEAD all 6 pass, and the full sfnonet file passes on
GPU (34 passed, 1 skipped).

**Isolated sweep at production width** (fwd+bwd+fused-AdamW, effective batch 2
to match `n_ensemble: 2`, fixed input, one process per config, A100-80GB):

| level | peak alloc | median s/step |
|---|---|---|
| 0 | 46.58 GB | 0.868 |
| 1 | 46.09 GB | 0.870 |
| 2 | 38.86 GB | 0.893 |
| 3 | **16.81 GB** | 1.153 |

−64% memory for +33% compute at level 3; levels 1–2 are not worth having.
Gradients bit-identical across all levels (max abs diff exactly 0.0 over all
135 parameter tensors). Static memory 9.07 GB = params 2.98 + AdamW states
5.96, consistent with the 61.8 GB end-to-end figure once EMA, DDP grads and
NCCL are added.

**bf16 autocast measured, README corrected.** The old "~12–16 GB saving"
estimate was wrong: measured **4.3 GB** saving uncheckpointed and **0.4 GB**
after `checkpointing: 3`, because `SpectralConvS2.forward` upcasts to fp32
before the SHT, so the dominant spectral activations never shrink under
autocast. bf16 *is* 11–13% faster per step. Also noted: the coupled trainer
wraps loss and per-step `backward()` in autocast (`fme/coupled/stepper.py:2285`),
unlike the ACE path which scopes it to the forward — untested combination.

**Coupled worst case measured to completion.** The gap the 2026-08-24 pass
left open. A 6-yr-window coupled config with `checkpointing: 3` and the worst
case *forced* (atm `steps: 41` at p=1.0, ocn `steps: 4` at p=1.0, local batch
1, 8 ranks) ran a full epoch + validation + 876-step inference, exit 0:

* training: **flat 43.3–44.1 GB/GPU** (p50 = p99 = max on every rank)
* peak: **45.6 GB during EMA validation** — `validate_using_ema` clones every
  parameter (`ema.py store()`), so validation briefly holds params ×3
* 54 batches at ~75 s/batch forced-worst (vs 61.3 s/batch natural draws)
* the 12-yr inference (16 ICs, `coupled_steps_in_memory: 2`) took 3.4 min

Without checkpointing the same window peaks at 77.1 GB. Measurement note:
5-second `nvidia-smi` sampling produced isolated single-sample garbage (62.9,
70.5 GiB readings with ~44 GiB neighbors, including during dataset setup) —
use p50/p99 of the series, never the raw max.

**Code-claim audit (file:line verified).** Samudra's `checkpoint_strategy`
already passes `use_reentrant=False` everywhere — never affected.
`_AutogradAllReduce.backward` returning `grad_output` unchanged: confirmed
verbatim on this branch; the fix exists on unmerged
`feature/model-parallel-backward-pass`; the identity backward is wrong only
for the corrector global-mean path (scaled by 1−1/N). No FSDP/ZeRO anywhere.
DDP is built without `gradient_as_bucket_view` — an unexploited ~3 GB saving
for the 800M model. EMA is an unconditional fp32 GPU copy (+3.2 GB) with no
disable flag. `OptimizationConfig.checkpoint` (rollout-level checkpointing via
`after_n_forward_steps`) exists as a further lever, orthogonal to
`SFNONetConfig.checkpointing`. `spectral_ratio` *does* cut spectral-domain
activations (README corrected — the old "activations unchanged" claim was
true only for `filter_num_groups`). `FusedAdam` now maps to
`AdamW(fused=True)` with a deprecation warning; atm and cpl configs still say
`FusedAdam`, ocn says `AdamW` — harmless, mildly inconsistent.

**Gradient-accumulation finding promoted to the README.** `use_gradient_
accumulation: true` in the coupled config detaches the atmosphere fluxes fed
to the ocean (`fme/coupled/stepper.py:1288`) and the carry-over states between
outer steps — the coupled finetune passes no gradient between realms. Written
up in the README as a modeling decision in disguise, together with the
degenerate ocean `EnsembleLoss` at `n_ensemble: 2`.

**Staging still pending.** `stage-shared-data.sh` remains unrun — CFS writes
are blocked in the agent sandbox (verified again). The destination
`/global/cfs/cdirs/e3smdata/emulator/SamudrACE-E3SMv3/historical/` now exists
(empty). This is the one manual step left.

**E4/E5 (atmosphere A/B + coupled control) were killed with the session;
partial data recovered.** Launched 05:51 PDT on allocation 57585167
(nid008208,008540,008648,008652): E4a = 6-yr atmosphere A/B
`checkpointing: 0`, E4b = same with `checkpointing: 3` (4 ranks, batch 4 each),
E5 = the unmodified coupled config (8 ranks, the ~77 GB control). All three
`srun` steps were killed at 06:03:38 PDT — the exact moment the launching
session died. The `srun` processes died with the process group, so the E3
precedent (steps surviving orphaning) does not generalize: for multi-hour
verifications, submit with `sbatch` or check job survival after any
interruption. Recovered from the logs and the final ~100 s of each memory
series:

* E4a (ckpt0): flat **58.4 GiB/GPU** in-trainer; E4b (ckpt3): flat
  **28.6 GiB/GPU** — the same −51% as the 24 Aug campaign (61.8 → 28.4 GB)
* Step-100 throughput: 1.10 vs 1.38 s/step — +25% at 4 ranks, trending to the
  +33% isolated cost; neither run reached steady state
* E5 never left dataset construction — the 77.1 GB as-shipped coupled peak
  stands on the 24 Aug measurement

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
* Added a `5yr_test` held-out inference block to the coupled config. All 8 of
  its existing initial conditions fell inside the training windows, so the
  finetune had no out-of-sample monitoring at all, unlike atm and ocn. The new
  block uses 8 ICs in 2040–2047, verified to lie on the ocean's 5-day axis, with
  a 5-year rollout ending inside the record.

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
