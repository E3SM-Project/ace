# Experiments — E3SMv3 historical, 2026-08-31 hackathon

**Source of truth is the hackathon page**
<https://e3sm.atlassian.net/wiki/spaces/p3ai/pages/6550683662>. Everything here
is downstream of it. Where this file and the page disagree, the page wins.

This file describes the campaign as it stands. `AGENTS.md` is the working log
and holds the history.

Not tracked in git. The backup of record is the last section of the *Historical
Ablation Campaign* artifact
(<https://claude.ai/code/artifact/ccd2b27e-fae3-4090-83f7-a07cf351664b>), which
embeds this file verbatim inside `<pre class="source" id="md-source">`. The
publish pipeline strips HTML comments, so find that block by its `id`.

---

## The baselines

Two committed files, and they *are* runs E01 and E11 — not templates:

    config-train-atm.yaml  =  E01.aug26.atm.A0_B16_C0_O5_W0_X0.S01
    config-train-ocn.yaml  =  E11.aug26.ocn.A0_B16_C0_O5_W0_X0.S01

Everything else is generated from them by `make_ablation_config.py`, and
`check_campaign.py` asserts that each emitted config matches its run id.

Model settings, per the page:

| item | value | note |
|---|---|---|
| `embed_dim` | 384 | atmosphere only |
| `noise_embed_dim` | 32 | atmosphere only |
| loss weighting | equal | no `weights` block; `weights.get(key, 1.0)` reads that as uniform |
| `checkpoint_save_epochs` | `{step: 1}` | full checkpoint, optimizer state included |
| `ema_checkpoint_save_epochs` | `{step: 1}` | weights only |

Samudra has no `embed_dim`/`noise_embed_dim`, and its loss is unweighted MSE.

### wandb: 1D only

2D image metrics are off, via the **typed** aggregator configs — not the
deprecated boolean-flag variants, which `dacite` matches by shape and which
silently re-enable the images:

| aggregator | off | on |
|---|---|---|
| inference | `zonal_mean`, `video`, `trend`, `seasonal`, `near_zero_fraction`, `enso_coefficient`, `step_diagnostics.correction_maps` | `histogram`, `mean`, `mean_norm`, `power_spectrum`, `annual`, `enso_index`, `ipo_index` |
| validation (one-step) | `snapshot`, `mean_map` | `mean`, `mean_norm`, `power_spectrum`, `ensemble` |

**`time_mean_denorm` / `time_mean_norm` stay on.** They emit
`time_mean/rmse/<var>` and `time_mean/bias/<var>` — the scalars every run is
compared on — and build the bias map in the same call, with no separate switch
(`fme/ace/aggregator/inference/time_mean.py`, `get_logs`). Disabling them to
remove the image removes the metric. Removing the image alone is a code change.

---

## Naming

    <exp>.<hackathon_date>.<realm>.<tuning_set>.S<seed>
    E05  .aug26          .atm   .A3_B16_C1_O5_W0_X0.S01

The experiment number is one incrementing `E` sequence — E01–E10 atmosphere,
E11–E17 ocean, and a coupled run would be the next number. `E` is the one letter
the factor alphabet (A, B, C, O, W, X) and the seed (S) both leave free; an
`A##` or `O##` prefix collides with the aerosol and ocean-cadence factors inside
the same run id, and `C##` collides with CO₂.

The tuning set is a **fixed-order** factor word `A?_B??_C?_O?_W?_X?`:

| pos | levels |
|---|---|
| `A` | `A0` none · `A1` aerosol **inputs** (`aerindexall`, `colccn.3`) · `A2` aerosol **outputs** (`lwp`, `lcc`, `cdnc`) · `A3` both |
| `B` | `B08` · `B16` · `B32` — global batch size |
| `C` | `C0` no CO₂ · `C1` `global_mean_co2` as an input |
| `O` | `O1` 1-daily ocean step · `O5` 5-daily |
| `W` | `W0` equal · `W1` flux upweight · `W2` away-from-surface dilution · `W3`/`W4` zero one poor channel |
| `X` | `X0` baseline · `X1` AMP (bf16 autocast) |

Each factor is a separate `WANDB_TAG` as well as being inside
`WANDB_JOB_TYPE`, so "every C1 run" is a filter rather than a regex.

---

## Weights & Biases

All 33 runs go to one project so both realms share a workspace:

| | |
|---|---|
| entity | `e3sm-aig` |
| project | `SamudrACE-E3SMv3` |

`entity` is the team, not the account — `wandb login` prints
`Currently logged in as: <username> (<entity>)`, so `e3sm-ai` is the account.
`check_campaign.py` asserts both on every generated config.

Run identity comes from the environment, not the yaml: `WANDB_NAME`,
`WANDB_RUN_GROUP`, `WANDB_JOB_TYPE`, `WANDB_TAGS` and `WANDB_NOTES` are read
directly by the wandb library. `make_ablation_config.py` writes them into
`runs/<runid>.env`, `run-train.sh` exports them. Seeds collapse into one
`WANDB_RUN_GROUP`; the factor word is the job type; every factor is also its own
tag, so "every C1 run" is a filter.

**Each person logs in with their own key.** A W&B API key is a personal
credential — whoever holds it can read, edit and delete anything that account
can, and runs made with it are attributed to that account. Share the *team*
instead: an admin adds people to `e3sm-aig`, each runs `wandb login` once with
their own key, and runs still land in the shared project because entity and
project come from the config. For unattended jobs that should not carry a
person's identity, use a team **service account**. Keys live in `~/.netrc` or
`WANDB_API_KEY`, never in a config or the wiki.

---

## The run list

33 runs, 119 nodes. `make_ablation_config.py --list` prints it.

### Atmosphere — E01–E10, E15

| exp | factors | what it adds | seeds | nodes each |
|---|---|---|---|---|
| **E01** | `A0_B16_C0_O5_W0_X0` | **baseline** | 3 (+B08 +B32) | 4 |
| **E02** | `A0_B16_C1_O5_W0_X0` | + CO₂ | 3 (+B08 +B32) | 4 |
| E03 | `A1_B16_C1_O5_W0_X0` | + aerosol input | 1 | 4 |
| E04 | `A2_B16_C1_O5_W0_X0` | − aerosol input + aerosol output | 1 | 4 |
| **E05** | `A3_B16_C1_O5_W0_X0` | both aerosol inputs and outputs | 3 (+B08 +B32) | 4 |
| E06 | `A3_B16_C0_O5_W0_X0` | − CO₂ (vs E05: aerosol/GHG interplay) | 1 | 4 |
| E07 | `A3_B16_C1_O5_W1_X0` | flux-upweighted loss | 1 | 4 |
| E08 | `A3_B16_C1_O5_W2_X0` | away-from-surface dilution | 1 | 4 |
| E09 | `A3_B16_C1_O5_W4_X0` | zero `STW_0` | 1 | 4 |
| E10 | `A3_B16_C1_O5_W0_X1` | AMP | 1 | 4 |
| E15 | `A3_B16_C1_O5_W3_X0` | zero `STW_1` | 1 | 4 |

E03 → E04 → E05 is cumulative: E03 adds the aerosol *inputs*, E04 swaps them for
the *outputs*, E05 has both. The three decompose the aerosol question.

### Ocean — E11–E14, E16, E17

| exp | factors | what it adds | seeds | nodes each |
|---|---|---|---|---|
| **E11** | `A0_B16_C0_O5_W0_X0` | **baseline** | 3 (+B08 +B32) | 2 |
| E12 | `A0_B16_C0_O5_W1_X0` | interface-upweighted loss | 1 | 2 |
| E13 | `A0_B16_C0_O5_W2_X0` | away-from-surface dilution | 1 | 2 |
| E14 | `A0_B16_C0_O5_W4_X0` | zero deepest meridional velocity | 1 | 2 |
| E16 | `A0_B16_C0_O5_W3_X0` | zero `iceVolumeTotal` | 1 | 2 |
| E17 | `A0_B16_C0_O1_W0_X0` | 1-daily stepping (vs E11's 5-daily) | 1 | 2 |

### Node budget

atm 98 + ocn 21 = **119 nodes against 96 reserved**. In aggregate the campaign
is roughly 7,400 node-hours against 12,100 available — about 61% — so this is a
concurrency limit, not a capacity one. It drains in priority order:

| pri | nodes | cumulative | what |
|---|---|---|---|
| P1 | 14 | 14 | the four bolded baselines at B16 S01 — E01 E02 E05 E11 |
| P2 | 42 | 56 | the single-seed science ablations, including E15/E16/E17 |
| P3 | 28 | 84 | seeds S02/S03 of the bolded four |
| P4 | 35 | 119 | the B08/B32 batch sweeps |

P1+P2+P3 = 84 nodes, all of which start immediately; only P4 queues. A
single-seed ablation is the only measurement of its factor that exists, whereas
a third seed refines an error bar there are already two samples of, and a batch
sweep answers an optimizer question rather than a science one.

---

## The loss reweightings

`weights.get(key, 1.0)` means an omitted variable is 1.0, so each set below is
the complete delta from W0.

**W1 — upweight fluxes.** Atmosphere: `LHFLX SHFLX FLUS FLUT FLDS FSNS FSUTOA
TAUX TAUY DTENDTTW surface_precipitation_rate frozen_precipitation_rate` → 2.0.

Samudra predicts **no** fluxes — `TAUX`, `FSNS`, `LHFLX` and the rest are inputs
in `next_step_forcing_names`, never in `out_names`, and loss weights apply only
to `out_names`. A literal ocean W1 is therefore the empty set, which would make
E12 a duplicate of E11. The ocean W1 upweights the air–sea interface state it
does predict: `sst ssh ocean_sea_ice_fraction iceVolumeTotal` → 2.0.

**W2 — away-from-surface dilution.** A monotone, surface-heaviest profile over
each vertical family, **mean-normalized to 1.0**. The realms index the vertical
in opposite directions, verified against the centering statistics:

| realm | index 0 is | evidence | weights |
|---|---|---|---|
| atm | top of atmosphere | `T_0` = 220.9 K, `T_7` = 277.5 K | `T_0` 0.40 → `T_7` 1.60 |
| ocn | the surface | `temperature_0` = 13.6 °C, `_18` = 0.7 °C | `_0` 1.60 → `_18` 0.40 |

Mean-normalization is load-bearing: a profile that does not average to 1.0
changes the total loss magnitude as well as its shape, confounding W2 with an
effective learning-rate change.

**W3 and W4 — zero one poor channel.** A matched pair probing two different
reasons a channel is a poor loss citizen, each picked against the training
statistics:

| set | realm | channel | why |
|---|---|---|---|
| **W4** | atm | `STW_0` | residual/full-field scale **0.031**, second lowest of any output after `PS`, and \|mean\|/std 12.8 — almost all its spread is the secular stratospheric-water trend, not anything predictable step-to-step |
| **W4** | ocn | `velocityMeridionalCoarsened_18` | most extreme ocean output by \|mean\|/std (**0.005**) |
| **W3** | atm | `STW_1` | residual/full **0.70**, the level below `STW_0`; the hand-tuned weight set that predates this campaign singled out exactly `STW_0` and `STW_1`, both at 0.25 |
| **W3** | ocn | `iceVolumeTotal` | structurally zero across most of the domain and already special-cased by the corrector's `zero_where_ice_free_names`, so most of its loss is trivially satisfiable and the rest concentrates at the ice edge |

Deliberately not used: `FSNS` (residual/full 1.28), `FSUTOA` (1.14), `SHFLX`
(1.09) and `DTENDTTW` (1.05) all look unpredictable by that ratio, but for the
first three it is the diurnal cycle, which the model can resolve from `SOLIN`.
`DTENDTTW` would additionally confound the moisture-budget corrector, which
consumes it.

---

## Ocean cadence: O5 and O1

Both MPAS cadences exist in the run directory, 1501 files each, 1940–2065:

| | streams | timestep | records/month |
|---|---|---|---|
| **O5** | `fmeDepthCoarsening5D`, `fmeDerivedFields5D`, `fmeSeaiceDerivedFields5D` | 5 days | 6 |
| **O1** | the same names **without** the `5D` suffix | 1 day | 30 |

Both are interval **means** — `time_bnds` span 5 days and 1 day respectively —
so switching cadence is a data swap, not a resample. The daily
`fmeDepthCoarsening` additionally carries 95 `*_inst` variables the 5-day stream
does not; nothing in the campaign uses them.

Switching cadence changes four things together, which is why
`make_ablation_config.py` does it rather than a `sed`:

1. all three MPAS file patterns lose the `5D`;
2. `LANDFRAC`/`sea_surface_fraction` must be materialised on the matching axis —
   merge members have to share `sample_start_times`. `make_landfrac_ocn.py
   --cadence 1d` writes `landfrac1d.<year>.nc`;
3. every inference block's `n_forward_steps` scales ×5, 876 → **4380**, to cover
   the same 12-year rollout;
4. an epoch holds 5× the samples, so `max_epochs` comes down —
   `DEFAULT_EPOCHS["ocn-O1"]` is 30 against O5's 150.

Inference initial conditions need no change: the 5-day timestamps are a subset
of the 1-day axis.

`check_campaign.py` enforces that all four merge members agree on cadence. A
mixed config either fails at load on time alignment or, worse, aligns on the
intersection and silently trains on a fraction of the record.

### Measured cost of O1

| | O5 (clean) | O1 (near-clean) | ratio |
|---|---|---|---|
| s/step | 1.390 | 1.538 | 1.11 |
| steps/epoch | 411 | 2,053 | 5.00 |
| **h/epoch** | **0.16** | **0.88** | **5.5** |
| dataset setup | 10.5 min | **50.7 min** | 4.8 |

Per step the two cadences are within ~10% — the model work is identical and only
the file layout differs — so the epoch cost is essentially the 5x sample count.

**Setup is the surprise.** The config builds **12 datasets** (4 merge members x 2
concat blocks for training, plus 4 for validation) and each globs and opens all
1501 files to read time coordinates, serially. The daily files hold 30 records
each against the 5-day files' 6, so there is ~5x the time-coordinate metadata to
decode per file: 10.5 min becomes 50.7. Throughout, the ranks sit at ~2.4% CPU
with flat memory — it is metadata I/O, not compute. **That cost is paid on every
job start and every requeue.**

At `DEFAULT_EPOCHS` — O5 150, O1 30 — including setup on 12 h segments:

| | training | setup | total |
|---|---|---|---|
| E11 (O5), 150 epochs | 23.8 h | 1.1 h (3 starts) | **24.9 h** |
| E17 (O1), 30 epochs | 26.3 h | 2.5 h (3 starts) | **28.8 h** |

Comparable, and both comfortably inside the window. For an equal-wall-clock
comparison against E11's 150 epochs, O1 gets **27 epochs**.

### Statistics

The committed ocean statistics were computed on the 5-day stream. Measured
comparison of 1-day against 5-day over 12 sampled months across 1950/1975/2000:

| variable | std ratio 1D/5D |
|---|---|
| `temperatureCoarsened_0`, `_18`, `sst`, `ssh`, `iceAreaTotal` | 1.000 – 1.005 |
| `latentHeatFlux` | **1.115** |
| `velocityMeridionalCoarsened_18` | **1.130** |

Means agree to four significant figures throughout. Slow fields are unaffected
by the averaging window; flux-like and deep-velocity channels are ~12% wider at
daily resolution. **E17 uses the 5-day statistics**, which is defensible for a
cadence comparison and leaves a ~12% normalization error confined to those
channels. A production O1 run should get its own from `compute_hist_stats.py`.

### What the cadence means for coupling

The coupled stepper **derives** the ratio; it is not configured
(`fme/coupled/requirements.py`, `_compute_n_steps_fast`):

    n_steps_fast = ocean_timestep / atmosphere_timestep

and requires the atmosphere timestep to divide the ocean's. With a 6-hourly
atmosphere:

| ocean | atm steps per ocean step | atm timepoints per sample at `n_coupled_steps: 4` | physical window |
|---|---|---|---|
| **O5** | **20** | (4 × 20) + 1 = **81** | 20 days |
| **O1** | **4** | (4 × 4) + 1 = **17** | 4 days |

Both are integers, so O1 is structurally supported with no code change. Two
consequences for a coupled run at O1:

- `n_coupled_steps: 4` covers 4 days rather than 20. To keep the same physical
  horizon, `n_coupled_steps` goes to **20**, which restores the same 81
  atmosphere timepoints per sample — the same atmosphere cost — while running 5×
  the ocean steps.
- the ocean `n_steps` outcome distribution `{0, 1, 2, 4}` spans 0–20 days at O5
  and 0–4 days at O1. Scale it ×5 for an equivalent rollout horizon.

---

## Sizing, and the two rules that bite

`batch_size` is global and divided across ranks by `dist.local_batch_size`
(`fme/ace/data_loading/getters.py:120`). Both baselines run at a fixed local
batch, so the rank count follows:

| realm | local batch | ranks | nodes | B08 | B16 | B32 |
|---|---|---|---|---|---|---|
| atm | 1 | = batch | ranks / 4 | 2 | **4** | 8 |
| ocn | 2 | batch / 2 | ranks / 4 | 1 | **2** | 4 |

Two divisibility rules must hold, both enforced by the generator on a login node
rather than surfacing minutes into an allocation as
`UnionMatchError: can not match type "list"`:

1. `validation.loader.batch_size` must divide the rank count — set equal to
   `batch_size`.
2. **every** inference block's initial-condition count must divide the rank
   count. The baselines ship 16 ICs, covering 4/8/16 ranks; the atmosphere's B32
   run is 32 ranks, so the generator rewrites both IC lists. A dotlist
   `--override` cannot index into a yaml list.

**The node count travels with the run.** `#SBATCH --nodes=` is only a default
for ad-hoc runs; the generator writes `FME_NODES` into each `.env` and
`run-train.sh` passes `--nodes` to `sbatch`. Without it the B08 and B32 arms run
at the baseline's node count.

`--local-batch atm=2` regenerates the whole campaign at 2 samples per rank,
halving every atmosphere run's node count. It fits (28.7 GB/GPU measured) — see
"Measurements" for why the campaign does not use it.

### `12yr_test` and the final epoch

`12yr_test` fires on `range(max_epochs + 1)[start::step]`, so a `start` chosen
for one run length silently stops scoring the final epoch at another. The
generator solves `start = max_epochs % step` and asserts the last fire equals
`max_epochs`.

---

## Schedule

    ReservationName=_CAP_aigs_hist
    StartTime=2026-08-31T09:00:00   EndTime=2026-09-05T15:00:00
    Duration=5-06:00:00   NodeCnt=96   Features=hbm80g   PartitionName=gpu_ss11
    Users=elynnwu,imanick,rebassoo,olawale,mahf708

126 hours, 504 node-days, ending **Saturday** 2026-09-05.

At the measured 2.11 h/epoch, a 30-epoch atmosphere run is 63 h, so P1+P2+P3 all
start Monday morning and the headline E01/E02/E05 comparisons complete Wednesday
night. P4 runs in the freed nodes.

### Two operational musts

1. **`RESERVATION=_CAP_aigs_hist`.** `run-train.sh` passes `--reservation` when
   that variable is set, and nothing sets it for you. Without it jobs sit in the
   regular queue while 96 reserved nodes idle.
2. **Drop the flag for anything crossing Saturday 15:00.** A 12 h segment that
   cannot finish before the reservation ends will not start inside it; a requeued
   continuation runs on the normal allocation.

### Steps per epoch

Atmosphere: 1940–1990 plus 2000–2040 = 90 years at 6-hourly = 131,400 samples,
so at global batch 16 an epoch is **≈ 8,210 steps**.

`time_buffer: 10` does not reduce that. It subsamples the dataset to every 11th
start index (`getters.py:98`) **and** makes each loaded window yield 11 output
batches (`dataloader.py:183`). The two cancel: it is an I/O optimization, not a
subsample. Where it does bite is a short split — a window needs roughly
`11 × batch_size` timesteps to yield one batch, about 45 days at 6-hourly and
batch 16, below which the loader fails with "No batches in dataloader".

Ocean at O5: 90 years × 73 records = 6,570 samples, **411 steps** per epoch. At
O1, 90 years × 365 = 32,850 samples, **2,053 steps**.

---

## Measurements — 2026-08-29, A100-80GB

### Measurement hygiene: contention is worth 2x

Two 2-node jobs on disjoint nodes still share CFS, and for the ocean that is the
binding resource. The **same config** measured alone and alongside one other job:

| | s/step | setup |
|---|---|---|
| alone | **1.390** | 10.5 min |
| one other 2-node job on the same filesystem | 2.945 | 13.1 min |

**2.1x on step time, 1.25x on setup.** Every ocean figure below is marked clean
or contended; treat contended ones as upper bounds. The atmosphere's
0.925 s/batch was measured alone.

### Data-loader settings

Atmosphere baseline, 4 nodes / 16 ranks / global batch 16 / local batch 1,
inference removed, measured alone:

| | `num_data_workers` / `prefetch_factor` / `time_buffer_pool_size` | effective s/batch |
|---|---|---|
| lowered | 2 / 1 / 1 | 3.155 (220 steps) |
| **committed** | **8 / 4 / 2** | **0.925** (680 steps) |

The step log is bimodal, not noisy: twenty steps at 17-18 s, then one interval
at 163-216 s, with GPU memory flat at 18.6-19.0 GB throughout — the `time_buffer`
window refill starving the GPU against 1501 files on CFS.

The ocean moves the same direction, measured with both arms concurrent, so both
contended and the ratio is the meaningful part:

| | workers / prefetch | effective s/step |
|---|---|---|
| lowered | 2 / 1 | 24.36 |
| **committed** | **8 / 4** | **3.10** |

### What it costs

| | s/step | h/epoch | full run | fits 126 h? |
|---|---|---|---|---|
| atm, lowered | 3.155 | 7.2 h | 30 ep = 216 h | no |
| **atm, committed** (clean) | **0.925** | **2.11 h** | **30 ep = 63 h** | yes |
| ocn, lowered (contended) | 24.36 | 2.78 h | 150 ep = 417 h | no |
| ocn, committed (contended) | 3.10 | 0.35 h | 150 ep = 53 h | yes |
| **ocn, committed (clean)** | **1.390** | **0.16 h** | **150 ep = 24 h** | yes, easily |

**Do not lower the worker settings without re-measuring.** For the atmosphere
three settings changed together, so the attribution among them is unknown, and
`time_buffer_pool_size: 2` is also a sampling change — with one pool slot
consecutive output batches come from the same preloaded window, with two they
interleave. It is applied identically to all runs, so within-campaign comparisons
hold.

### `time_buffer` is for the atmosphere only — it OOMs the ocean

The ocean sets no `time_buffer`, and it must stay that way at the committed
worker settings. Measured: `time_buffer: 10` with `time_buffer_pool_size: 2` on
the ocean train loader is **killed by the host OOM killer** before the first step
(`Detected 2 oom_kill events`, `nid008316: task 1: Out Of Memory`).

The cause is in-flight host memory. Each worker holds `prefetch_factor` input
batches, and an input batch is `local_batch x (n_timesteps + time_buffer)`
samples:

| | window | channels | local batch | GB/batch | 8x4 in flight | per node |
|---|---|---|---|---|---|---|
| ocean, `time_buffer: 0` | 5 | 91 | 2 | 0.22 | 7.0 GB/rank | **28 GB** |
| ocean, `time_buffer: 10` | 15 | 91 | 2 | 0.66 | 21.1 GB/rank | **84 GB** |
| atmosphere, `time_buffer: 10` | 12 | 50 | 1 | 0.14 | 4.6 GB/rank | 19 GB |

The ocean's per-sample window is ~4.5x the atmosphere's — 91 channels against
~50, local batch 2 against 1, and `n_forward_steps: 4` needing 5 timesteps
against the atmosphere's 2 — so a `time_buffer` costing the atmosphere 19 GB per
node costs the ocean 84 GB, before the pool, the model and the optimizer.

**And the ocean does not need it.** Measured alone at `time_buffer: 0`, a full
411-step epoch runs at **1.390 s/step with no stalls at all** — every interval
between 1.00 and 1.50 s. The atmosphere needed `time_buffer` because its loader
was starving the GPU; the ocean's keeps up. Raising workers and prefetch is the
fix that mattered for both realms.

### Memory, and the two levers

| local batch | `checkpointing` | mem/GPU | s/step | s/sample |
|---|---|---|---|---|
| 1 | 3 | **19.0 GB** | 0.925 | 0.925 |
| 1 | **0** | **40.9 GB** | 0.830 | 0.830 |
| **2** | 3 | **28.7 GB** | 1.660 | 0.830 |

**Keep `checkpointing: 3`.** It costs 3–5% of step time for 54% of activation
memory at `embed_dim: 384`. The "+33% step compute" figure elsewhere in the notes
is for the 512-wide model and does not hold here.

**Local batch 2 fits and is marginally better per sample; the campaign uses
local batch 1.** Halving the ranks at fixed global batch halves the nodes *and*
doubles the epoch:

| | nodes/run (B16) | epoch | 30 epochs | campaign nodes |
|---|---|---|---|---|
| **local batch 1** | 4 | **2.11 h** | **63 h** | 119 |
| local batch 2 | 2 | 3.79 h | 114 h | ~72 |

Both fit the window. At local batch 1 the headline comparisons land Wednesday
night; at local batch 2 nothing finishes before Friday morning. Fifty hours of
time to look at the result is worth more than removing a queueing problem Slurm
handles for free.

### Fixed costs

| | value |
|---|---|
| atmosphere parameters | 456,223,488 |
| ocean parameters | 82,822,138 |
| atmosphere dataset setup | 22.5 min — unchanged by worker count; it is the initial time-coordinate read |
| ocean dataset setup, O5 | 10.5 min at 8 ranks, alone (13.1 min contended) |
| ocean dataset setup, O1 | 50.7 min — 12 dataset opens x 1501 files, 5x the time records each |

Setup is paid again on **every requeue** — six times over a 63 h run at a 12 h
walltime, about 2 h of window per run.

### Checkpoint storage

`checkpoint_save_epochs` writes the full checkpoint including optimizer state
(`fme/core/generics/trainer.py:775`); `ema_checkpoint_save_epochs` writes weights
only. At `{step: 1}` that is one of each per epoch per run. With 456 M
parameters this is order **9 GB per epoch per run**, so order **8 TB** across the
campaign. *That is arithmetic, not a measurement.*

Check `myquota` from a **login** node — it fails on a compute node. If it does
not fit, keep the EMA save every epoch and back the full save off to `{step: 5}`:
the EMA weights are what gets evaluated, and the optimizer state only matters for
resuming.

---

## Status

Ready:

- Shared inputs on CFS at
  `/global/cfs/cdirs/e3smdata/emulator/SamudrACE-E3SMv3/historical/` —
  `stats-2026-08-13/`, `landfrac5d/`, `landfrac1d/`. Group `e3smdata`, mode
  `g+rX,o-rwx`; all five users named on the reservation are in that group and the
  path is group-traversable end to end. No config references personal `$PSCRATCH`.
- All 33 configs pass `fme.ace.validate_config` and `check_campaign.py`, and all
  pass the real submit path via `submit-campaign.sh --preflight` — staging,
  `.env`, per-run `--nodes` and the validator, everything but the `sbatch` call.
- `config-train-cpl.yaml` regenerated from the baselines; `make_cpl_config.py
  --check` clean, `fme.coupled.validate_config` passes.

- **Committed and pushed** to `e3sm/exps/hist-v2026.8.0`. A clone of the branch
  the page names gets the baselines, the generator, the checker, all 33 run
  configs and their `.env` files.
- **wandb verified end to end.** A run reaches
  `e3sm-aig/SamudrACE-E3SMv3` with name, group, job type and all eleven tags
  populated from the generated `.env`.

Open:

1. **Scratch quota vs per-epoch checkpoints** — order 8 TB; check `myquota` from
   a login node.
2. **E17 uses the 5-day ocean statistics.** Fine for a cadence comparison; a
   production O1 run wants its own.
3. **Soil moisture** is item 5 of the page's preamble and is not implemented —
   no `H2OSOI`/`TSOI` in any config, and the stats cover only atmosphere and
   ocean. Needs the ELM history stream plus a stats recompute.
4. **The clock control.** `global_mean_co2` is a `(time,)` scalar with no spatial
   structure, monotone over the record, so a model can use it as a clock rather
   than a forcing. `make_time_ramp.py` builds a matched physics-free ramp and is
   one command from being another run. It separates "the CO₂ channel works" from
   "the model learned to read a clock". Not on the page's list.
5. **Checkpoint selection is in-sample.** `save_all_checkpoints(valid_loss,
   inference_error)` selects on `valid_loss` over the 1990–95 window — an
   interpolation window between the two training blocks — and on
   `inference_error`, the weighted sum over inference blocks
   (`train_config.py:284`). Only the `inference` block carries weight 1.0 and all
   16 of its initial conditions are inside the training window; `12yr_test`, the
   held-out 2040s rollout, is weight 0.0 and influences nothing. Selecting on the
   held-out set would contaminate it, so this is the right setup — but every
   reported number has to name which of the two it came from.

---

## Decision rules

| claim | what would have to be true |
|---|---|
| **CO₂ helps** | E02 beats E01 on `12yr_test` `time_mean/rmse` for `TS`, `T_*`, `PS`, outside E01's S01–S03 spread |
| **Aerosols help** | E05 beats E02 on the same metric; E06 then says whether the two forcings are separable |
| **A weight set wins** | it improves its target variables without degrading `time_mean/rmse/channel_mean`. W2 is mean-normalized so this comparison means something |
| **AMP is worth it** | s/batch and whether the loss curve tracks E05's — nothing else. At `checkpointing: 3` the bf16 memory saving is 0.4 GB, and checkpointing itself costs only 3–5%, so E10 has to beat an efficient baseline |
| **A batch size wins** | samples per second, and whether the validation curve at equal *sample count* — not equal step count — matches B16 |
| **1-daily ocean is worth it** | E17 beats E11 on 12-year drift at equal **wall clock**, not equal epochs. It sees 5× the samples per epoch, so an equal-epoch comparison flatters it |

**Report the epoch a number came from.** With `checkpoint_save_epochs: 1` every
epoch is on disk, so it is easy to compare epoch 28 of one run with epoch 30 of
another.

---

## Known gotchas

- **Gradient checkpointing and `use_reentrant`.** The conditional SFNO's three
  `torch.utils.checkpoint` call sites must pass `use_reentrant=False`; the
  reentrant variant builds no backward graph when none of the segment's inputs
  require grad, and the encoder's input is raw data. With `checkpointing >= 1`
  that silently trains a frozen, randomly initialized encoder — no error on one
  GPU, a DDP unused-parameter error on several. Fixed at `c5d39a0fa`; **any run
  predating it with `checkpointing >= 1` is invalid.**
- **The legacy SFNO-v0.1.0 builder still has that bug** —
  `fme/ace/models/modulus/sfnonet.py`, three call sites. This campaign does not
  use it, but reaching for the old builder with checkpointing reproduces it.
- `fme/core/optimization.py`, behind the ocean's `checkpoint_strategy: all`,
  passes `use_reentrant=False` — the ocean is unaffected.
- **Judge a run by `REAL_EXIT=0` and `DONE ---- rank 0`, not the log tail.**
  `time_buffer` teardown prints alarming but harmless tracebacks on success.
- **Do not use the `FME_USE_SRUN=1` launcher on Perlmutter** — it hardcodes cuda
  device 0 and every rank dies with `invalid device ordinal`.
- **`srun` dies with the launching session.** Multi-hour work goes through
  `sbatch`.
- **Expect I/O contention at campaign scale.** The measurements above are single
  jobs; twenty-plus concurrent runs read the same 3.7 TB directory. Stagger
  Monday's launches by a few minutes.

---

## Reference facts

**Ocean model.** 91 in / 80 out, all 80 outputs prognostic (no diagnostics), 11
forcing-only inputs, 19 depth levels × 4 fields plus `sst`, `ssh`,
`ocean_sea_ice_fraction`, `iceVolumeTotal`. Unweighted MSE.
`n_forward_steps: 4` with **no** `optimize_last_step_only`, so gradients flow
through all four steps — rollout length is a cost knob as well as a skill knob.
The atmosphere is the opposite: `n_forward_steps: 1` *with*
`optimize_last_step_only`.

**Aerosol is not monotone.** It peaks and declines over the record, unlike CO₂ —
which is why E03/E05 can explain mid-century structure a CO₂-only model reads as
noise, and why E06 is not automatically free of the clock problem.

**piControl checkpoints as a warm start.** `in_names` is a clean prefix match
against both piControl generations on `origin/e3sm/exps/hist`, and both are
`embed_dim: 384`, matching this campaign. `out_names` is not: historical predicts
`FSNS` where piControl predicts the `FSDS`/`FSUS` pair, so the lists diverge at
index 41 against `e3sm_piControl_v20260527` (36 against `_v20260507`) and every
output channel after that shifts by one. `ParameterInitializationConfig`
overwrites by position with no name checking, so it would load silently and train
a decoder whose channels mean the wrong things. Restoring `FSDS`/`FSUS` and
appending new names at the end would make historical a strict superset. Nothing
in this campaign needs it — every run trains from scratch.

**The coupled ocean's `n_ensemble: 2` buys nothing.** `Samudra.forward` takes no
noise input and is deterministic, so both members are identical: the
`energy_score_weight` term is identically zero and CRPS degenerates to MAE, at
double the ocean forward cost. The atmosphere's `EnsembleLoss` is meaningful
because the SFNO is noise-conditioned.

**`CoupledDataLoaderConfig` has no `time_buffer` field** while the atmosphere
sets `time_buffer: 10`, so a coupled epoch draws ~11× more samples per unit
window. A code change, not a config one; it affects how epoch counts compare
across realms.

**Seed spread is not optional.** The 2026-08-13 ocean statistics A/B was
inconclusive by construction: repeating the identical run moved epoch-1
inference error by 0.0035 while the effect being measured was 0.0025. Three seeds
on E01/E02/E05/E11 is what makes a result falsifiable.

**Coupled dataset setup is ~51 min**, and the coupled config is capped at 8 ranks
by its 8 inference ICs. **Never submit any of this to a 4-hour queue.**

**MPAS sign flips and the `sst` Kelvin offset** are validated against EAM to
0.3–4.6% of each field's standard deviation.

---

## Files

| file | what |
|---|---|
| `config-train-atm.yaml` | baseline = E01 |
| `config-train-ocn.yaml` | baseline = E11 |
| `config-train-cpl.yaml` | coupled; not in the aug26 list, regenerate with `make_cpl_config.py` after any baseline change |
| `make_ablation_config.py` | the generator; `RUNLIST` transcribes the page |
| `check_campaign.py` | asserts every emitted config matches its run id; run by `generate-campaign.sh` |
| `runs/*.yaml`, `runs/*.env` | 33 generated runs plus wandb/sizing provenance |
| `runs/MANIFEST.tsv` | priority, runid, realm, nodes, ranks, batch, seed, note |
| `sbatch-scripts/generate-campaign.sh` | regenerates `runs/` and checks it |
| `sbatch-scripts/submit-campaign.sh` | walks the manifest in priority order; `--dry-run`, `--preflight` |
| `sbatch-scripts/run-train.sh` | stages, validates, sizes and submits one run; `--no-submit` |
| `stage-shared-data.sh` | moves aux inputs to CFS; re-run is a no-op |
| `make_landfrac_ocn.py` | LANDFRAC/sea_surface_fraction on the ocean axis, `--cadence 5d\|1d` |
| `compute_hist_stats.py` | normalization statistics |
| `make_time_ramp.py` | the clock control, built and unscheduled |
| `make_smoke_config.py` | short test config from a production one |
| `README.md` | launch recipes, verified numbers, gotchas |
| `AGENTS.md` | working log and history |
| `NOTES-historical-stats.md` | how the statistics were produced |
