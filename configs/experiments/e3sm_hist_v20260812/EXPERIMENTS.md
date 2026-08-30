# Experiments — E3SMv3 historical, 2026-08-31 hackathon

**Source of truth is the hackathon page**
<https://e3sm.atlassian.net/wiki/spaces/p3ai/pages/6550683662>. Everything here
is downstream of it. Where this file and the page disagree, the page wins.

This file describes the campaign as it stands. `AGENTS.md` is the working log
and holds the history.

Tracked on `e3sm/exps/hist-v2026.8.0`. The *Historical Ablation Campaign*
artifact (<https://claude.ai/code/artifact/ccd2b27e-fae3-4090-83f7-a07cf351664b>)
renders the same material and embeds this file verbatim inside
`<pre class="source" id="md-source">`; the publish pipeline strips HTML comments,
so find that block by its `id`. Update both together or they drift.

---

## The baselines

Two committed files, and they *are* runs E01 and E11 — not templates:

    config-train-atm.yaml  =  E01.aug26.atm.A0_B16_C0_L0_O5_W0_X0.S01
    config-train-ocn.yaml  =  E11.aug26.ocn.A0_B16_C0_L0_O5_W0_X0.S01

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

### wandb: 1D logs, and what gets a picture

2D image metrics are off, via the **typed** aggregator configs — not the
deprecated boolean-flag variants, which `dacite` matches by shape and which
silently re-enable the images:

| aggregator | off | on |
|---|---|---|
| inference | `zonal_mean`, `video`, `trend`, `seasonal`, `near_zero_fraction`, `enso_coefficient`, `ipo_index`, `step_diagnostics.correction_maps` | `histogram`, `mean`, `mean_norm`, `power_spectrum`, `annual`, `enso_index`, `time_mean_denorm`, `time_mean_norm` |
| validation (one-step) | `snapshot`, `mean_map`, `ensemble_denorm.log_mean_maps` | `mean`, `mean_norm`, `power_spectrum`, `ensemble` scalars |

`ipo_index` is off because it can never build here — it needs >80 years and the
scored rollout is 12, so every run logged four "metric not supported" warnings
and nothing else.

### The upload budget: 100 GB, and one PNG per channel

MEASURED 2026-08-30 by counting the PNGs W&B actually uploaded:

| | images per epoch | MB per epoch |
|---|---|---|
| atmosphere (both blocks fired) | 506 | 55 |
| ocean (both blocks fired) | 484 | 24 |

Where they came from, per epoch:

| source | atm | ocn |
|---|---|---|
| `time_mean/gen_map/<var>` + `time_mean/bias_map/<var>` | 232 | 324 |
| `time_mean_norm/gen_map/<var>` | 100 | 160 |
| `val/ensemble/{crps,ssr_bias,ensemble_mean_rmse}/mean_map/<var>` | 174 | — |

Extrapolated to the campaign at production cadence that is order **50 GB** in
maps alone — half the account — plus roughly 9 GB of per-variable spectrum
figures, 7 GB of annual-mean figures and 5 GB of histogram payloads. About 70%
of the account for one week of runs, leaving nothing for the fine-tuning and
coupled work the plan says comes next.

**The lever is which channels get plotted, not whether.** Every scalar is
untouched either way, so nothing that a comparison depends on is at stake — only
the pictures.

| | plotted | of | dropped |
|---|---|---|---|
| atmosphere | **38** | 50 | interior levels 2, 3, 5 of `T`/`STW`/`U`/`V` |
| ocean | **28** | 80 | interior levels 3–8 and 10–16 of the four coarsened stacks |

Every flux and every surface/2m/10m field is plotted. The atmosphere keeps the
two top and two bottom levels of each stack plus one mid-column reference
(index 0 is TOA, 7 is the surface); the ocean keeps the top three, one mid-depth
reference and the two deepest, which is where W4's zeroed channel lives.

Three metrics share the one list — `time_mean_denorm`, `time_mean_norm` and
`power_spectrum` `plot_variables`, plus `histogram.variables` — so one screen
shows the same channels as maps, spectra and distributions. `check_campaign.py`
asserts the four lists are identical, are a strict subset of `out_names`, and
contain no name that is not an output: a typo there fails silently, it simply
never plots.

**The mechanism.** `TimeMeanMetricConfig` gained two fields on this branch,
modelled on `PowerSpectrumMetricConfig`'s existing toggles:

* `report_plot: bool = True` — master switch. `False` emits no map images at
  all, and changes no scalar.
* `plot_variables: list[str] | None = None` — narrows what the master allows.

Both gate image emission only: `rmse/<var>`, `bias/<var>`, `ref_bias/<var>`,
`ref_rmse/<var>` and `rmse/channel_mean` are emitted for every channel
regardless, and `get_dataset` still returns `gen_map-<var>` and
`bias_map-<var>` as DataArrays for every channel.

The one image metric with no per-variable control is the validation ensemble's
`crps` / `ssr_bias` / `ensemble_mean_rmse` mean maps —
`OneStepEnsembleMetricConfig` has only `log_mean_maps`, a bool. That is 174 PNGs
per atmosphere epoch, 47% of the atmosphere's map bytes, duplicating information
the inference time-means already carry, so it is **off**. Flip
`ensemble_denorm.log_mean_maps: true` if someone wants it back; it is
all-or-nothing.

`save_per_epoch_diagnostics: true` is the safety net: every aggregator flushes
its full fields to `experiment_dir` as netCDF each epoch (~66 MB/epoch, ~2 GB
per atmosphere run, 0.5% of what the checkpoints cost). Any channel can be
plotted from there at whatever colour scale and projection you want, including
the interior levels that are not uploaded.

### Verified: what one epoch actually uploads

W&B run `bench.2026-08-30.atm-plots-4380` (`ejvnljzy`) — the production
atmosphere config with only the training window shortened, and a 3-year rollout
so `annual` and `enso_index` build (they need >2 years and are silently skipped
below that, which is why a short benchmark looks emptier than production).
One epoch, both inference blocks:

| | files | MB |
|---|---|---|
| PNG (maps) | **228** | **27.04** |
| plotly (1D charts) | **272** | **5.73** |
| scalars | 2,675 keys | — |

Every one of the 228 is accounted for: 38 plotted channels x 3 map metrics
(`time_mean/gen_map`, `time_mean/bias_map`, `time_mean_norm/gen_map`) x 2 blocks.
No other metric emits a PNG.

The 1D charts are plotly, not images — `annual` 58 per block, `power_spectrum`
38, `histogram` 38, `enso_index` 2. `power_spectrum` and `histogram` follow the
plot list; `annual` and `enso_index` are unrestricted, and at 21 KB per chart
they are not worth restricting.

At production cadence — `inference` every epoch, `12yr_test` every fifth:

| | per epoch | per run | campaign |
|---|---|---|---|
| atmosphere | 19.7 MB | 0.59 GB | **14.8 GB** (25 runs) |
| ocean | ~8.2 MB | 1.2 GB | **11 GB** (10 runs) |
| | | | **~26 GB of 100** |

Against ~70 GB if every channel were plotted, and against ~0 if none were. The
epoch cost is unchanged within noise: 2,039 s at a 219-window rollout, with the
two block summaries at 109 s and 27 s — plotting 38 channels costs about what
plotting 58 did.

**Pin `time_mean_norm.target: norm` when you write that block.** dacite builds
the field from the yaml alone, so the dataclass default (`denorm`) wins over the
`default_factory`, and the config then fails `__post_init__` with a
`UnionMatchError` on `inference` that names neither the field nor the reason.

---

## Naming

    <exp>.<hackathon_date>.<realm>.<tuning_set>.S<seed>
    E05  .aug26          .atm   .A3_B16_C1_L0_O5_W0_X0.S01

The experiment number is one incrementing `E` sequence — E01–E10 atmosphere,
E11–E17 ocean, and a coupled run would be the next number. `E` is the one letter
the factor alphabet (A, B, C, O, W, X) and the seed (S) both leave free; an
`A##` or `O##` prefix collides with the aerosol and ocean-cadence factors inside
the same run id, and `C##` collides with CO₂.

The tuning set is a **fixed-order** factor word `A?_B??_C?_L?_O?_W?_X?`,
alphabetical by position:

| pos | levels |
|---|---|
| `A` | `A0` none · `A1` aerosol **inputs** (`aerindexall`, `colccn.3`) · `A2` aerosol **outputs** (`lwp`, `lcc`, `cdnc`) · `A3` both |
| `B` | `B08` · `B16` · `B32` — global batch size |
| `C` | `C0` no CO₂ · `C1` `global_mean_co2` as an input |
| `L` | `L0` baseline lr · `L1` lr × √(batch / 16) |
| `O` | `O1` 1-daily ocean step · `O5` 5-daily |
| `W` | `W0` equal · `W1` flux upweight · `W2` away-from-surface dilution · `W3`/`W4` zero one poor channel |
| `X` | `X0` baseline · `X1` AMP (bf16 autocast) |

`C1 − C0` measures whether the model uses the channel, not whether it responds to CO₂. `co2vmr` in the h0 stream is strictly increasing at every 6-hourly timestep from 311.4 ppm (1940) to 551.0 ppm (2065) with no seasonal cycle, so Spearman(`co2vmr`, time) = 1.000 exactly: over this record CO₂ *is* a clock, up to a monotone warp the first layer absorbs. Separating forcing response from time-indexing needs CO₂ varied against a fixed time axis, which is an inference-time counterfactual on a trained checkpoint, not another training arm.

Each factor is a separate `WANDB_TAG` as well as being inside
`WANDB_JOB_TYPE`, so "every C1 run" is a filter rather than a regex.

---

## Weights & Biases

All 35 runs go to one project so both realms share a workspace:

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

35 runs, 129 nodes. `make_ablation_config.py --list` prints it.

### Atmosphere — E01–E10, E15

| exp | factors | what it adds | seeds | nodes each |
|---|---|---|---|---|
| **E01** | `A0_B16_C0_L0_O5_W0_X0` | **baseline** | 3 (+B08 +B32) | 4 |
| **E02** | `A0_B16_C1_L0_O5_W0_X0` | + CO₂ | 3 (+B08 +B32) | 4 |
| E03 | `A1_B16_C1_L0_O5_W0_X0` | + aerosol input | 1 | 4 |
| E04 | `A2_B16_C1_L0_O5_W0_X0` | − aerosol input + aerosol output | 1 | 4 |
| **E05** | `A3_B16_C1_L0_O5_W0_X0` | both aerosol inputs and outputs | 3 (+B08 +B32) | 4 |
| E06 | `A3_B16_C0_L0_O5_W0_X0` | − CO₂ (vs E05: aerosol/GHG interplay) | 1 | 4 |
| E07 | `A3_B16_C1_L0_O5_W1_X0` | flux-upweighted loss | 1 | 4 |
| E08 | `A3_B16_C1_L0_O5_W2_X0` | away-from-surface dilution | 1 | 4 |
| E09 | `A3_B16_C1_L0_O5_W4_X0` | zero `STW_0` | 1 | 4 |
| E10 | `A3_B16_C1_L0_O5_W0_X1` | AMP | 1 | 4 |
| E15 | `A3_B16_C1_L0_O5_W3_X0` | zero `STW_1` | 1 | 4 |

The aerosol arms are a **2x2 factorial**, not a chain. E04 removes the inputs
that E03 added, so "E03 then E04 then E05" is not cumulative:

| | no aerosol outputs | aerosol outputs |
|---|---|---|
| **no aerosol inputs** | E02 (`A0`) | E04 (`A2`) |
| **aerosol inputs** | E03 (`A1`) | E05 (`A3`) |

Analyse it as a factorial: the input main effect is (E03-E02) and (E05-E04), the
output main effect is (E04-E02) and (E05-E03), and the interaction is
(E05-E03)-(E04-E02). E05-E02 on its own is not "the aerosol effect" -- it moves
the predictors, the target dimensionality and the training objective at once.
All four cells share `C1`, so CO2 is held fixed across the square; E06 is the
separate `A3_C0` cell that opens the aerosol/GHG interaction.

### Ocean — E11–E14, E16, E17

| exp | factors | what it adds | seeds | nodes each |
|---|---|---|---|---|
| **E11** | `A0_B16_C0_L0_O5_W0_X0` | **baseline** | 3 (+B08 +B32) | 2 |
| E12 | `A0_B16_C0_L0_O5_W1_X0` | interface-upweighted loss | 1 | 2 |
| E13 | `A0_B16_C0_L0_O5_W2_X0` | away-from-surface dilution | 1 | 2 |
| E14 | `A0_B16_C0_L0_O5_W4_X0` | zero deepest meridional velocity | 1 | 2 |
| E16 | `A0_B16_C0_L0_O5_W3_X0` | zero `iceVolumeTotal` | 1 | 2 |
| E17 | `A0_B16_C0_L0_O1_W0_X0` | 1-daily stepping (vs E11's 5-daily) | 1 | 2 |

### Node budget

atm 108 + ocn 21 = **129 nodes against 96 reserved**. In aggregate the campaign
is roughly 10,000 node-hours against 12,100 available — about **83%** once
inline inference is counted — so this is a concurrency limit that is also close
to a capacity one. It drains in priority order:

| pri | nodes | cumulative | what |
|---|---|---|---|
| P1 | 14 | 14 | the four bolded baselines at B16 S01 — E01 E02 E05 E11 |
| P2 | 42 | 56 | the single-seed science ablations, including E15/E16/E17 |
| P3 | 28 | 84 | seeds S02/S03 of the bolded four |
| P4 | 45 | 129 | the B08/B32 batch sweeps, at both L0 and L1 |

P1+P2+P3 = 84 nodes, all of which start immediately; only P4 queues. A
single-seed ablation is the only measurement of its factor that exists, whereas
a third seed refines an error bar there are already two samples of, and a batch
sweep answers an optimizer question rather than a science one.

**Expect P4 not to finish.** At 83% utilisation the campaign has no room for a
45-node tail, which is exactly why the batch sweep is the thing at the back.

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

## Learning rate and batch size

`L0` holds the base learning rate (1e-4) at every batch size. `L1` scales it by
`sqrt(batch / 16)`:

| batch | L0 | L1 |
|---|---|---|
| B08 | 1.000e-4 | **7.071e-5** |
| B16 | 1.000e-4 | *(identical to L0 — the generator rejects it)* |
| B32 | 1.000e-4 | **1.414e-4** |

Neither is "correct", which is why both run. Linear scaling, lr ∝ batch (Goyal
et al. 2017), is derived for SGD with momentum and rests on one large step
approximating *k* small ones. Square-root scaling (Krizhevsky 2014; Hoffer et
al. 2017) instead keeps the gradient-noise level fixed, since the update
variance goes as lr² / batch. **Both realms use an Adam-family optimizer** —
`FusedAdam` for the atmosphere, `AdamW` for the ocean — which normalizes by the
gradient's second moment and so is already partly invariant to gradient scale;
√B is the better prior there than linear. There is also a ceiling: above a
critical batch size (McCandlish et al. 2018) more batch buys little whatever the
learning rate, but B08–B32 on a 456 M-parameter model is far below it.

**L0 is the campaign default.** With one seed per batch arm, changing batch and
learning rate together produces a result nobody can attribute. The cost is that
a null at B32-L0 is ambiguous — "B32 does not help" and "B32 needed a bigger
step" look identical. E01 therefore carries both: `B08_L0`, `B08_L1`, `B32_L0`,
`B32_L1`, all on the baseline experiment so nothing else varies. Three-way
comparison against `B16_L0` separates the two explanations.

`check_campaign.py` verifies the learning rate numerically against the factor
word rather than trusting a flag, because a wrong lr is invisible in a run id.

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

**Both totals are training-only, and neither is a run total.** Inline
inference runs every epoch and is ~31% of an ocean epoch. Priced with it, E11 is
~49 h and E17 is ~49 h — see "Ocean — measured at production rollout length"
below for the arithmetic. They are equal-wall-clock at the shipped
`DEFAULT_EPOCHS["ocn-O1"] = 30`, which is why that number stands.

Earlier drafts said 27 epochs here and ~24 on the published page. Both came from
cost models that left inference out of one side or the other. **The generator is
the number of record; this file follows it.**

**What E17 actually varies.** `apply_ocean_cadence` swaps the streams, the
LANDFRAC axis, the statistics, `max_epochs` and the scored inference horizon,
but it leaves `stepper_training.n_forward_steps: 4` alone. The training rollout
is therefore 4 *steps* in both arms, which is 20 physical days at O5 and 4 at
O1. It also points at its own normalization statistics (see "Statistics"), whose
surface-velocity and flux scales differ from the 5-day set's by up to ~1.3–1.5x,
which re-weights the per-channel MSE relative to E11. So E17 is "daily cadence
**and** a 5x shorter physical training rollout **and** a different effective loss
weighting", not a cadence ablation with everything else held fixed. Two ways to
read it:

* **As shipped** — report it under that longer name. It is still the honest
  answer to "should we train the ocean on daily data", because the 4-step
  rollout is what a daily-cadence run would naturally use.
* **Matched horizon** — set O1 to `n_forward_steps: 20` so both arms train
  across 20 physical days. That isolates cadence, and costs roughly 5x the
  training step time, which does not fit E17's ~49 h budget.

The shipped choice is the first. Do not describe E17 as a clean cadence
ablation in a figure caption.

### Statistics

**E17 has its own.** `train-only/ocean-1d/` holds 221 variables against the
5-day set's 127, a strict superset, computed over 1940–1989 and 2000–2039 from
4 sharded passes over 2.42 TiB. `apply_ocean_cadence` points every O1 run there
and raises if a 5-day stats path survives the switch.

Means agree throughout and the median standard-deviation ratio is **1.0004**,
but the spread is not uniform:

| variable | std ratio 1D/5D |
|---|---|
| `temperature*`, `salinity*`, `sst`, `ssh` | 1.000 |
| `LHFLX` (`latentHeatFlux`) | **1.117** |
| `velocityMeridionalCoarsened_*` (surface) | 1.15 – 1.30 |
| `SHFLX` (`sensibleHeatFlux`) | **1.207** |
| `TAUX` (`windStressZonal`) | **1.216** |
| `surface_precipitation_rate` | **1.401** |
| `frozen_precipitation_rate` (`snowFlux`) | **1.432** |
| `TAUY` (`windStressMeridional`) | **1.481** |
| `airStressZonal`, `airStressMeridional` | **0.842**, **0.823** |

The statistics are keyed by the **FME** channel name, not the MPAS one — the
config renames `windStressZonal`/`windStressMeridional` to `TAUX`/`TAUY`,
`latentHeatFlux`/`sensibleHeatFlux` to `LHFLX`/`SHFLX`, and `snowFlux` to
`frozen_precipitation_rate`. Grepping the stats for an MPAS name returns
nothing and does not mean the channel is absent.

The state channels are insensitive to the averaging window and the
high-frequency ones are not, which is why borrowing the 5-day scales would have
mis-normalized by up to 48% exactly the channels a cadence experiment is about.
The two `airStress*` ratios run the other way, so the 1-day and 5-day streams do
not carry the same averaging for those two; do not read a cadence signal in them.

`layerThicknessCoarsened_0`, `layerThicknessCoarsened_0_inst` and
`icebergHeatFlux` are dropped as constant over the window in both sets, so no
config gains or loses a channel by switching.

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

At the measured **2.82–2.96 h/epoch** — inline inference included, see the
2026-08-30 measurements — a 30-epoch atmosphere run is **88–92 h**, not the 63 h
a training-only figure suggests. P1+P2+P3 still all start Monday morning and the
headline E01/E02/E05 comparisons land Thursday night, but the margin on the
window is **1.4x** rather than 2x, and P4 should not be expected to finish.

### Two operational musts

1. **`RESERVATION=_CAP_aigs_hist`.** `run-train.sh` passes `--reservation` when
   that variable is set, and nothing sets it for you. Without it jobs sit in the
   regular queue while 96 reserved nodes idle.
2. **Drop the flag for anything crossing Saturday 15:00.** A 12 h segment that
   cannot finish before the reservation ends will not start inside it; a requeued
   continuation runs on the normal allocation.

### Launch ramp — the Monday morning procedure

**This is the risk that costs the campaign rather than a run.** The margin on
the window is 1.4x; I/O contention was measured at up to **2.1x** with a single
competing job; Monday's plan puts ~25 jobs, each 8 workers x 16 ranks, on the
same 3.7 TB directory, plus a rolling 14–22 min setup storm on every 12 h
requeue. Do not release 129 nodes at once.

1. **Queue P1 only** — the four bolded baselines, 14 nodes:

       RESERVATION=_CAP_aigs_hist ./sbatch-scripts/submit-campaign.sh --max-priority 1

2. **Let E01 log two epochs** and read `epoch_total_seconds` in wandb. One epoch
   is not enough: epoch 1 carries the setup and the first checkpoint write.
3. **Then decide:**

   | `epoch_total_seconds` | what it means | what to do |
   |---|---|---|
   | under ~10,600 s (2.95 h) | contention is not biting | release P2+P3: `--max-priority 3`, then P4 as nodes free |
   | ~10,600–11,900 s (2.95–3.3 h) | tight but survivable | release P2+P3, skip P4 |
   | over ~11,900 s (3.3 h) | 30 epochs does not fit | pull a lever from "The levers, if the budget gets tight" **before** queueing anything else |

The ramp costs nothing. P1 is the four runs everything else is compared against,
so they have to go first regardless; the only thing being deferred is the
release of runs that cannot start until nodes free anyway.

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

> **Every full-run figure in this section is training-only.** Inline inference
> was excluded, so the `full run`, `30 epochs` and `63 h` numbers below
> understate reality by 25–30%. The per-step and per-epoch *measurements* stand;
> the extrapolations from them are superseded by "Measurements — 2026-08-30".

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

The `full run` column is **training only**. With inference the committed rows are
88–92 h and ~49 h, and the `fits 126 h?` answers become "yes, at 1.4x margin" and
"yes, at 2.6x".

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

Training-only again: with inference the two columns are ~2.9 h / 88–92 h and
~4.6 h / ~140 h, and **local batch 2 no longer fits the window at all** — which
only strengthens the choice below.

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

Setup is paid again on **every requeue** — eight times over an 88–92 h run at a
12 h walltime, about 3 h of window per run.

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

## Measurements — 2026-08-30: a whole production epoch

Everything above measures **training**, with inference removed. That is not what
an epoch costs. The committed atmosphere config runs a 16-IC, 12-year weighted
rollout every epoch — `InlineInferenceConfig.epochs` defaults to
`Slice()`, which is "every epoch" (`train_config.py:120`) — plus the held-out
12-year block every fifth. This section prices the whole thing.

Method: the production config with **only** the training window shortened
(1940–1943, 275 batches instead of 8,212) and the inference blocks shortened
(365 forward steps instead of 17,520), both firing every epoch. Validation
window, aggregators, checkpoint cadence, loader settings, model and node count
are all at production values. Two epochs, 4 nodes / 16 ranks, no other job on
the allocation. Configs and logs under
`$PSCRATCH/fme-bench/2026-08-30/`; W&B run `bench.2026-08-30.atm-infer-365`
(`fmn11h7u`) in group `bench.2026-08-30`.

### Atmosphere — where an epoch actually goes

| phase | epoch 1 | epoch 2 | scales with |
|---|---|---|---|
| training (275 batches) + train-evaluation | 332.1 s | 311.5 s | batches |
| validation, 1990–95, production window | 201.7 s | 200.5 s | **fixed** |
| `inference` block, 365 steps | 284.8 s | 271.6 s | windows (see below) |
| ├ rollout | 118.2 s | 91.6 s | windows |
| └ aggregator summary + flush + next-block setup | 166.6 s | 180.0 s | **fixed** |
| `epoch_total_seconds` | **818.6** | **783.7** | |
| wandb log | 26.9 s | | fixed |
| checkpoint write, ~20 GB | 31.1 s | | fixed |

Repeatable to 4.3% between the two epochs. The two numbers that extrapolate:

* **Steady training rate 0.887 s/batch** (steps 40→260, 220 batches), against
  0.925 measured on 2026-08-29. The production epoch is 8,212 batches.
* **Steady inference rate 2.17–2.25 s per 20-step window**, i.e. **0.109 s per
  forward step**, flat across 19 windows in four separate blocks. A production
  block is 17,520 steps = 876 windows.

### The extrapolation is 48x, so it was tested

A second run at `n_forward_steps: 2920` — 146 windows per block, 7.7x the
rollout — checks whether the per-window cost drifts as the rollout grows
(aggregator state, deeper reads into the record). W&B run
`bench.2026-08-30.atm-infer-2920`. Per-window cost by position in the rollout:

| windows | n | mean | median | worst |
|---|---|---|---|---|
| 1–20 | 19 | 2.444 s | 2.325 s | 3.72 s |
| 21–50 | 30 | 2.351 s | 2.325 s | 2.77 s |
| 51–80 | 30 | 2.808 s | 2.319 s | **15.91 s** |
| 81–110 | 30 | 3.957 s | 2.320 s | **36.11 s** |
| 111–146 | 36 | 2.320 s | 2.317 s | 2.38 s |
| all | 145 | **2.782 s** | **2.32 s** | |

**The median is flat to within 0.4% across the whole rollout** — there is no
drift, so the linear model holds. The mean is 20% higher because of two
filesystem stalls (15.9 s and 36.1 s) in 146 windows: the same bimodal
CFS-metadata signature as the training loader, not a property of inference.

Setup and the other phases reproduced: 21.7 min setup (against 22.4), 329.4 s
training for the same 275 batches (against 332.1 and 311.5), 203.6 s validation
(against 201.7 and 200.5). `epoch_total_seconds` 1570.3.

So the production block is bounded, not point-estimated:

| basis | s / block | per 30-epoch run |
|---|---|---|
| median, no stalls | 876 x 2.32 + 90 = **2,120 s** | |
| mean, stalls included | 876 x 2.78 + 90 = **2,530 s** | |

### Atmosphere — the production epoch

| phase | seconds | note |
|---|---|---|
| training, 8,212 batches at 0.887 s | 7,284 | |
| train-evaluation pass | 68 | fixed |
| validation | 202 | measured directly, production window |
| `inference`, weight 1.0, **every epoch** | 2,120 – 2,530 | 876 windows, median / mean-with-stalls |
| `12yr_test`, weight 0.0, every 5th epoch | 424 – 506 | amortised |
| wandb log + checkpoint write | 58 | |
| **total** | **10,156 – 10,648 s = 2.82 – 2.96 h** | |

**A 30-epoch atmosphere run is 85–89 h of epochs, not 63 h.** The inline
inference the old figure omitted is 25–29% of an epoch.

### Dataset setup is 22 minutes, not 9

Measured on the same run, 4 nodes / 16 ranks, clean:

| | seconds |
|---|---|
| train loader open | 1,286 |
| validation loaders (warm page cache) | 0.9 |
| two inference loaders | 42.4 |
| stepper construction | 16 |
| **total, process start to first batch** | **1,346 s = 22.4 min** |

The train loader's cost is **independent of the training window** — this run
subset to three years and still paid 21.4 minutes, because the glob
`eam.h0.*.nc` matches all 1,501 files and the subset is applied after the time
coordinates are decoded. The ranks sit in `D` state at ~10% CPU throughout: it
is metadata I/O, not compute.

**It is paid on every job start and every requeue.** At a 12 h walltime an
85–89 h run is 8 starts, so **3.0 h of setup**, and the run is **88–92 h** end
to end. Against the 126 h reservation window that is a **1.37–1.44x margin**,
not the 2x the 63 h figure implied.

### Ocean — measured at production rollout length, no extrapolation

The ocean's scored rollout is 876 steps, short enough to run at full length in a
benchmark, so these are direct measurements. E11 baseline, 2 nodes / 8 ranks,
alone on the allocation, training window cut to 1940–1960 (91 batches); W&B run
`bench.2026-08-30.ocn-infer-876`.

| phase | epoch 1 | epoch 2 | production scaling |
|---|---|---|---|
| training (91 batches) + train-evaluation | 264.3 s | 280.7 s | 150 s fixed + 1.34 s/batch |
| validation, 1990–95 | 32.3 s | 32.2 s | fixed |
| both 876-step inference blocks | 543.5 s | 631.4 s | **already production length** |
| `epoch_total_seconds` | **840.1** | **944.4** | |
| wandb log | 58.7 s | | fixed |
| checkpoint write | 8.3 s | | fixed |

Setup, clean: train loader 676 s, everything else warm, **837 s = 14.0 min**
total — close to the 10.5 min measured on 2026-08-29.

Per-window cost is noisier than the atmosphere's: 3.52, 3.71 and 5.01 s per
20-step window across the four blocks, against the atmosphere's 2.17–2.32. The
ocean reads a four-way `merge`, which is why.

| phase | seconds | note |
|---|---|---|
| training, 411 batches | 701 | 150 fixed + 411 x 1.34 |
| validation | 32 | |
| `inference`, weight 1.0, **every epoch** | 294 | mean of four measured blocks |
| `12yr_test`, every 5th epoch | 59 | amortised |
| wandb log + checkpoint write | 67 | |
| **total** | **1,153 s = 0.32 h** | |

**A 150-epoch ocean run is 48 h of epochs, not 24 h**, plus 1.2 h of setup over
five 12 h segments: **~49 h**. Inference is 31% of an ocean epoch — a larger
share than the atmosphere's, because the ocean's training epoch is 10x shorter.

**E17 (O1), priced from the same measurements.** O1 has 5.00x the samples and a
5.00x longer scored rollout (4,380 steps against 876), at the measured 1.11x
per-step penalty — so every sample-proportional phase scales by 5.55x while the
fixed costs do not:

| phase | O5 | scaling | O1 |
|---|---|---|---|
| training | 701 s | 150 s fixed + 2,053 x 1.487 | 3,203 s |
| validation | 32 s | x5.55 | 178 s |
| `inference`, weight 1.0 | 294 s | x5.55 | 1,633 s |
| `12yr_test`, amortised | 59 s | x5.55 | 327 s |
| wandb log + checkpoint write | 67 s | fixed | 67 s |
| **epoch total** | **1,153 s = 0.32 h** | | **5,408 s = 1.50 h** |

30 epochs is 45.1 h, plus 4.2 h of setup over five 12 h segments at 50.7 min a
start: **~49 h**, against E11's ~49 h. **The two arms are equal-wall-clock at the
shipped `DEFAULT_EPOCHS`**, so O1 keeps 30 epochs and the decision rule's "equal
wall clock, not equal epochs" is satisfied without changing anything.

*Caveat on this row.* Only O1's **training** step was measured at 1-day cadence;
the 1.11x per-step penalty is applied to O5's measured inference and validation
costs rather than measured there. If O1 inference turns out disproportionately
slower — plausible, since it holds 219 windows in memory against O5's 44 — E17
runs long and epochs come off the end. `epoch_total_seconds` on E17's first two
epochs settles it; above ~1.65 h/epoch, cut E17 to 27.

### Ocean checkpoint sizes

| file | size |
|---|---|
| `ckpt_NNNN.tar` (weights + optimizer) | 1.335 GB |
| `ema_ckpt_NNNN.tar` | 0.341 GB |
| `best_ckpt.tar`, `best_inference_ckpt.tar` | 0.341 GB each |

**1.68 GB accumulates per epoch**, so ~250 GB per 150-epoch ocean run and
~2.3 TB across the ten ocean runs.

### Checkpoint I/O

Written every epoch by the atmosphere baseline:

| file | size | every epoch? |
|---|---|---|
| `ckpt_NNNN.tar` (weights + optimizer) | 7.30 GB | yes, accumulates |
| `ema_ckpt_NNNN.tar` (weights) | 1.82 GB | yes, accumulates |
| `ckpt.tar` (restart) | 7.30 GB | yes, overwritten |
| `best_ckpt.tar`, `best_inference_ckpt.tar` | 1.82 GB each | when improved |

**~20 GB written per epoch, of which 9.12 GB accumulates.** 31 s of wall clock,
so 0.3% of an epoch — the cost is capacity, not time: **~275 GB per atmosphere
run**, 6.8 TB across the 25 atmosphere runs, 2.3 TB across the ten ocean runs,
**~9.2 TB** for the campaign.

**`CAMPAIGN_ROOT` stays at `$PSCRATCH/aug26` — per submitter, decided 2026-08-30.**
Each person's runs land in their own scratch, which spreads ~9.2 TB across three
quotas instead of one and means no one can purge or overwrite anyone else's
outputs. The cost is that the two submission guards — the checkpoint-resume guard
and the config-identity guard — see only their own submitter, as does the
queued-duplicate guard, because `squeue -u $USER` has the same scope. **Nothing
in the tooling can catch two people submitting the same run id.** What prevents
that is the ownership rule: every run id has exactly one owner, and only its
owner submits it. Post the split before Monday and treat it as binding.

Two practical consequences. Checkpoints stay where they were written, so the
person who owns a run is the person who reads it back; and wandb is the shared
surface, since all 35 runs report to one project regardless of whose scratch
they live in.

### The levers, if the budget gets tight

Priced against the 10,011 s epoch, in the order they cost least science:

| change | saves per epoch | saves per 30-epoch atm run | costs |
|---|---|---|---|
| `12yr_test` every 10th epoch instead of 5th | ~230 s | ~1.9 h | half as many held-out scores |
| weighted `inference` every 2nd epoch | ~1,150 s | ~9.6 h | checkpoint selection sees every other epoch |
| weighted `inference` rollout 12 yr → 6 yr | ~1,060 s | ~8.8 h | selection horizon no longer matches the reported one |
| first two together | ~1,380 s | ~11.5 h | |

**None is applied.** At 88–92 h the run fits the window, and changing the
selection metric mid-design is worse than a 1.4x margin. These are what to pull
if P1 comes in slower than this benchmark, not before. Watch
`epoch_total_seconds` on E01 for the first three epochs: above ~3.3 h/epoch the
run does not finish inside the reservation and the second lever is the one to
pull.

## Status

Ready:

- Shared inputs on CFS at
  `/global/cfs/cdirs/e3smdata/emulator/SamudrACE-E3SMv3/historical/` —
  `stats-2026-08-13/`, `landfrac5d/`, `landfrac1d/`. Group `e3smdata`, mode
  `g+rX,o-rwx`; all five users named on the reservation are in that group and the
  path is group-traversable end to end. No config references personal `$PSCRATCH`.
- All 35 configs pass `fme.ace.validate_config` and `check_campaign.py`, and all
  pass the real submit path via `submit-campaign.sh --preflight` — staging,
  `.env`, per-run `--nodes` and the validator, everything but the `sbatch` call.
- `config-train-cpl.yaml` regenerated from the baselines; `make_cpl_config.py
  --check` clean, `fme.coupled.validate_config` passes.

- **Committed and pushed** to `e3sm/exps/hist-v2026.8.0`. A clone of the branch
  the page names gets the baselines, the generator, the checker, all 35 run
  configs and their `.env` files.
- **wandb verified end to end.** A run reaches
  `e3sm-aig/SamudrACE-E3SMv3` with name, group, job type and all eleven tags
  populated from the generated `.env`.

- **Scratch quota is not a constraint.** `myquota` reports 47.39 of 120 TiB used
  and 2.73 of 10 M inodes. The campaign's per-epoch checkpoints are **8.0 TiB**
  (5.7 atmosphere + 2.3 ocean) in about 4,400 files, against 72.6 TiB and 7.27 M
  inodes free.

- **E17 has its own 1-day ocean statistics.** 221 variables against the 5-day
  set's 127, a strict superset. The median 1-day/5-day standard-deviation ratio
  is 1.0004 and the state channels (`temperature*`, `salinity*`, `sst`, `ssh`)
  are 1.000, but the high-frequency channels are not: `TAUY` 1.481,
  `frozen_precipitation_rate` 1.432, `surface_precipitation_rate` 1.401, `TAUX`
  1.216, `SHFLX` 1.207, surface velocities 1.15–1.30. `airStressZonal` and
  `airStressMeridional` go the other way, 0.842 and 0.823. Borrowing the 5-day
  set would have mis-scaled the forcing channels by up to 48%.

Open:

1. **Checkpoint selection is in-sample.** `save_all_checkpoints(valid_loss,
   inference_error)` selects on `valid_loss` over the 1990–95 window — an
   interpolation window between the two training blocks — and on
   `inference_error`, the weighted sum over inference blocks
   (`train_config.py:284`). Only the `inference` block carries weight 1.0;
   `12yr_test`, the held-out 2040s rollout, is weight 0.0 and influences nothing.
   Selecting on the held-out set would contaminate it, so this is the right
   setup — but every reported number has to name which of the two it came from.

   **The whole trajectory has to be in-sample, not just the start.** The scored
   rollout is 12 years long, so an initial condition three years before the end
   of a training block runs nine years past it. The `1980` and `2030` starts did
   exactly that — `1980 → 1992` reached into the 1990–95 validation split and
   `2030 → 2042` into the held-out test period — so both moved back three years,
   to `1977 → 1989` and `2027 → 2039`. `check_campaign.py` now recomputes every
   weighted block's end date against the training windows and fails the config if
   any trajectory leaves them, in both realms and at both ocean cadences.

---

## Decision rules

| claim | what would have to be true |
|---|---|
| **CO₂ helps** | E02 beats E01 on `12yr_test` `time_mean/rmse` for `TS`, `T_*`, `PS`, outside E01's S01–S03 spread |
| **Aerosols help** | E05 beats E02 on the same metric; E06 then says whether the two forcings are separable |
| **A weight set wins** | it improves its target variables without degrading `time_mean/rmse/channel_mean`, by more than E05's S01–S03 spread on the same metric. W2 is mean-normalized so this comparison means something |
| **AMP is worth it** | s/batch and whether the loss curve tracks E05's — nothing else. At `checkpointing: 3` the bf16 memory saving is 0.4 GB, and checkpointing itself costs only 3–5%, so E10 has to beat an efficient baseline |
| **A batch size wins** | samples per second, and whether the validation curve at equal *sample count* — not equal step count — matches B16. Use `valid_loss`, not `inference_error`: the B32 arms need 32 ICs to divide their rank count, so they score inference on a different IC ensemble (Jan **and** Jul starts) than the B08/B16 arms' 16 January starts, and the two are not IC-matched |
| **A learning rate scaling wins** | B32-L1 beats B32-L0 on validation at equal sample count. If it does, a B32-L0 deficit against B16 was a step-size problem rather than a batch-size one; if it does not, the batch-size reading stands |
| **1-daily ocean is worth it** | E17 beats E11 on 12-year drift at equal **wall clock**, not equal epochs. It sees 5× the samples per epoch, so an equal-epoch comparison flatters it |

### What counts as a result, for the arms with one seed

**Pre-registered 2026-08-30, before any campaign run started.** Only E01, E02,
E05 and E11 have three seeds. Every other arm has one, and the measured
repeat-run noise floor is **0.0035** on `time_mean/rmse/channel_mean` against
effects as small as **0.0025** — so a single-seed number read on its own is not
readable.

**The rule: a single-seed arm counts as a result only if it falls outside the
three-seed spread of its parent on the same metric and the same epoch.** Every
arm has one, by construction:

| arms | differ only in | parent, which has S01–S03 |
|---|---|---|
| E03, E04 | `A` (aerosol) | E02 `A0_B16_C1` |
| E06 | `C` (CO₂) | E05 `A3_B16_C1` |
| E07 E08 E09 E15 | `W` (loss weights) | E05 `A3_B16_C1_W0_X0` |
| E10 | `X` (AMP) | E05 `A3_B16_C1_W0_X0` |
| E12 E13 E14 E16 | `W` | E11 `A0_B16_C0_O5_W0` |
| E17 | `O` (cadence) | E11 `A0_B16_C0_O5_W0` |
| every B08/B32 and L1 arm | `B`, `L` | the same experiment at B16 — E01, E02, E05 or E11, all three-seeded |

Inside the parent's spread, report the arm as **"no effect resolvable at one
seed"** — which is a real finding about effect size, not a failure — rather than
as a direction. Do not read a sign off a difference smaller than the spread, and
do not spend a seed to break a tie unless the arm is on the critical path: a
fourth run of an arm whose effect is below the noise floor buys almost nothing.

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
| `runs/*.yaml`, `runs/*.env` | 35 generated runs plus wandb/sizing provenance |
| `runs/MANIFEST.tsv` | priority, runid, realm, nodes, ranks, batch, seed, note |
| `sbatch-scripts/generate-campaign.sh` | regenerates `runs/` and checks it |
| `sbatch-scripts/submit-campaign.sh` | walks the manifest in priority order; `--dry-run`, `--preflight` |
| `sbatch-scripts/run-train.sh` | stages, validates, sizes and submits one run; `--no-submit` |
| `stage-shared-data.sh` | moves aux inputs to CFS; re-run is a no-op |
| `make_landfrac_ocn.py` | LANDFRAC/sea_surface_fraction on the ocean axis, `--cadence 5d\|1d` |
| `compute_hist_stats.py` | normalization statistics |
| `make_smoke_config.py` | short test config from a production one |
| `README.md` | launch recipes, verified numbers, gotchas |
| `AGENTS.md` | working log and history |
| `NOTES-historical-stats.md` | how the statistics were produced |
