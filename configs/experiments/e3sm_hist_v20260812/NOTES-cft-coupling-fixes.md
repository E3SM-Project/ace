# CFT coupling fixes: findings, fixes, verification, and the plan to resume

Status as of 2026-09-10 05:00 (allocation 58140126 ended 00:56; the
full-config CFT completed epoch 1 with its inference and was cut in epoch 2). Diagnostics, configs and rollouts live in
`$PSCRATCH/cft-diag/`; the working log is the 2026-09-09 entry in
`AGENTS.md`. Everything below is uncommitted on `e3sm/exps/hist-v2026.8.0`
(see "Commits to make").

## 1. Findings

### Bug 1 (dominant): the ocean's flux forcings change *side* at coupling time

`config-train-ocn.yaml` forces the ocean with `fmeDerivedFields5D` fluxes,
which are MPAS-Ocean-side: heat and freshwater through the ice-free part of
the sea surface, per ocean cell. Under full ice they are ~0. The coupled
stepper averages the ACE atmosphere's cell-mean outputs over the ocean window
and hands them over unchanged. Measured 1990-01, 5-day EAM means vs MPAS:

| cells | FLDS atm−ocn | FLUS atm−ocn | FSNS | LHFLX rms |
|---|---|---|---|---|
| open ocean | −0.1 | +0.1 | +5.7 | 20 |
| coast (0.05<OCNFRAC<0.95) | +3.4 | −1.0 | −12.6 | 43 |
| ice edge | **+150** | **+177** | +20 | 9 |
| full ice | **+169** | **+207** | +0.8 | 1.3 |

That is 1–1.5 std of the ocean's normalizer for the longwave channels. Wind
stress and precipitation match; the fraction/mask plumbing in `fme/coupled`
is correct (coupled ICEFRAC vs EAM: mean |diff| 0.007). Upstream CM4 baselines
never hit this because their ocean is trained on the atmosphere's 5-daily
surface fluxes.

Isolation, uncoupled E11-FT ocean, 4 ICs × 5 years, nothing from `fme/coupled`:

| forcing | sst bias / rmse (K) | polar ice rmse | ice area / target, month 60 |
|---|---|---|---|
| MPAS ocean-side (as trained) | −0.02 / 0.12 | 0.007 | 1.00 |
| EAM 5-day cell means (what coupling supplies) | **+1.15 / 1.47** | **0.455** | **0.41** (0.23 at month 1) |
| EAM × (1 − ice fraction) | +0.05 / 0.41 | 0.041 | 1.00 |

### Bug 2 (secondary, coastal/polar TS): the atmosphere's TS blend at partial cells

The atmosphere's `ocean: {interpolate: true}` writes
`TS = w·target + (1−w)·gen`, `w = OCNFRAC`. In E01/E01-FT the target is EAM's
cell-mean TS, so wherever `w < 1` the network learned `gen ≈ TS_cell`. Coupled,
the target is SST, and the same blend errs by `w(1−w)(SST − T_nonocean)`:
zero over open water and land, ~4.5 K at a half-ocean polar cell. Measured
mean TS bias at polar, MPAS-wet, partial-land cells:

| rollout | TS bias there | over open polar water |
|---|---|---|
| raw pair, flux fix only | +4.4 K | +0.1 K |
| E18-CFT, epoch 19 (the campaign run) | +4.2 K | +0.5 K |

The warm strip along both polar coastlines in every CFT TS map is this term.

### What the campaign runs show

E18-CFT at epoch 19 vs its uncoupled parents (heldout 1990s, time-mean RMSE):
TS 1.37 vs 0.37 K, sst 0.70 vs 0.11 K, sea-ice fraction 0.082 vs 0.0025.
E18-CFT's own pre-training validation (`evaluate_before_training`) is
sst 2.88 / ice 0.199: the raw pair loses the ice pack within a month, and
20 epochs of fine-tuning walk it part of the way back.

### Not bugs, but recorded

* EAM and MPAS wet masks disagree in 4692 coastal cells (0.56% of ocean area);
  ~950 cells per step lose their SST prescription. Small.
* `E11-FT` configs list `iceVolumeTotal` twice in `zero_where_ice_free_names`
  (harmless; the generator now inherits it).
* `check_campaign.py` had never handled a stage-3 config (`KeyError: 'step'`).
* `config-train-cpl.yaml` had drifted from its generator (365 vs 364 steps).

## 2. Fixes

### Fix 1: open-water flux scaling (`fme/coupled/stepper.py`)

`CoupledStepperConfig.open_water_flux_scaling: OpenWaterFluxScalingConfig`
with `names` (the atmosphere-to-ocean heat and freshwater fluxes; stress is
left alone) and a **required** `ice_free_sst_threshold`. The listed fluxes are
multiplied by the ice-free fraction of the sea surface, from the ocean's own
sea-ice fraction at the start of the coupled step, except where the ocean's
SST exceeds the threshold (275.15 K; 99.8% of E3SM ice area lies below it).
The threshold is required because without it a sliver of predicted ice over a
cold blob zeroes the fluxes, the water cools, the ice grows, and sea ice
reaches the subtropics within months (measured: ice area 4× target by month
12). Requires `ocean_fraction_prediction`. Old checkpoints load with it off.
Also threaded through `StandaloneComponentCheckpointsConfig`.

### Fix 2: TS blend exponent (`fme/core/ocean.py`, `fme/core/prescriber.py`)

`OceanConfig.interpolate_weight_power` (default 1.0 = old behaviour): the
prescription weight becomes `OCNFRAC**power` while the network input stays
`OCNFRAC`. Error becomes `w**p (1−w)(SST − T_nonocean)`; open water stays
fully prescribed. Applied through the CFT config (and
`atmosphere_stepper_override` for standalone inference); E01-FT is not
retrained.

### Fix 3: atmosphere-flux fraction under ice shelves (data + `fme/coupled`)

`OpenWaterFluxScalingConfig.atmosphere_flux_fraction_name`: a static,
data-only field in the ocean forcing data (never an ocean network input)
multiplied into the scaling. `make_landfrac_ocn.py` derives it as the
1990–91 ratio of MPAS ocean-side FLDS to EAM FLDS × (1−ice), writes it into
the `landfrac5d` files (the 126 files on `$PSCRATCH` are augmented; the CFS
copy needs `stage-shared-data.sh`). It is 1.000 everywhere except 1090
Antarctic cells (ocean under ice shelves). Details in 3d item 5.

### Plumbing

* `make_cpl_config.py` emits all three (`--ts-blend-power`, default 8;
  `--ice-free-sst-threshold`, default 275.15; `--no-open-water-scaling`;
  `--no-flux-fraction`).
  `config-train-cpl.yaml` regenerated, `--check` clean.
* `regen_cft_configs.sh` regenerates the five `runs/E18–E22-CFT` yamls from
  their parents through the generator; done.
* `check_campaign.py` now checks stage-3 configs: all three fixes present,
  names match the ocean's heat/freshwater next-step forcings, the fraction
  field exists in the landfrac data, parents are existing
  `best_inference_ckpt.tar`.
* Tests: `fme/coupled/test_stepper.py` (+6), `fme/core/test_prescriber.py`
  (+2), `fme/core/test_ocean.py` (+2); `fme/coupled` + the two core files pass;
  ruff/mypy clean (pinned uvx, see AGENTS.md guidance).

## 3. Verification

### 3a. Coupled validation, untouched E01-FT × E11-FT B32, no training

The campaign's own metric (`evaluate_before_training`, 2 coupled steps):

| | sst rmse | ice frac rmse | ice volume rmse |
|---|---|---|---|
| unscaled (= E18-CFT epoch 0) | 2.93 | 0.201 | 0.290 |
| flux fix | **0.50** | **0.028** | **0.050** |

E18-CFT needed ~3000 steps to get sst below 1.1.

### 3b. 1-year coupled rollouts, 2 ICs, no fine-tuning unless stated

| run | sst bias / rmse | polar ice rmse | open-ocean ice rmse | ice area m12 | TS at polar partial cells |
|---|---|---|---|---|---|
| raw pair | +1.58 / 3.86 | 0.569 | 0.017 | 0.02 | +11.3 |
| + flux scaling, no guard | −1.28 / 4.82 | 0.202 | 0.191 | 4.11 | — |
| + flux scaling + guard | +0.04 / 0.87 | 0.199 | 0.019 | 1.01 | +4.4 |
| E18-CFT epoch 19 (20 epochs, no fixes) | +0.56 / 0.98 | 0.329 | 0.014 | 0.25 at m7 | +4.2 |

The trained campaign model is worse than the untrained pair with the flux
fix on every ice and polar number: E18 loses three quarters of the ice pack
within seven months of a 1-year rollout.

TS blend exponent sweep (raw pair, flux fix + guard, first ~5 months of the
same rollouts, TS bias at polar MPAS-wet partial-land cells; open-ocean sst
rmse 0.79–0.91 across the sweep, i.e. unchanged):

| `interpolate_weight_power` | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| TS bias, polar partial cells (K) | +4.4 | +3.9 | +2.9 | +2.3 | +1.9 |
| TS rmse, global (K) | 1.65 | 1.70 | 1.69 | 1.59 | 1.48 |

Monotone in the power, as the `w**p (1−w)` form predicts; the residual is
the network's own error at cells where `w` is near 1 and its loss weight
`1−w` was small. p = 8 is the default; the CFT loss sees TS at these cells
and is expected to close the rest.

### 3c. Full campaign config with both fixes (`cft-full-fixed`, this allocation)

E18-CFT's exact config (B32, `n_coupled_steps 4`, same parents) plus
`open_water_flux_scaling` (275.15 K) and `interpolate_weight_power 8`, run on
16 ranks at local batch 2 (48 GB/GPU; local batch 4 on 8 ranks OOMs at the
pre-training evaluation), with a 16-IC 1-year inline inference after every
epoch and the campaign's own validation before training and each epoch.

Pre-training validation (same val set and metric as E18-CFT's wandb `val/`):

| | sst rmse | ice frac rmse | ice volume | TS rmse |
|---|---|---|---|---|
| E18-CFT epoch 0 (no fixes) | 2.88 | 0.199 | — | — |
| both fixes, epoch 0 | **0.60** | **0.028** | 0.050 | 0.73 |
| E18-CFT epoch 2 (412 steps) | 1.96 | 0.159 | — | — |
| all three fixes, epoch 0 (`cft-full-all3-ema`) | **0.56** | **0.026** | 0.035 | 0.73 |

Per epoch (205 steps each), 16-IC 1-year inline inference, time-mean over
the rollouts; the E18 row is the trained campaign checkpoint rolled out the
same way (2 ICs):

| | sst bias / rmse | polar sst rmse | polar ice bias / rmse | open-ocean ice rmse | TS bias / rmse | polar TS rmse | TS at polar partial cells | val sst / ice |
|---|---|---|---|---|---|---|---|---|
| E18-CFT epoch 19 (no fixes) | +0.56 / 0.98 | 1.12 | −0.271 / 0.329 | 0.014 | +0.74 / 1.75 | 4.16 | +4.2 | — |
| both fixes, epoch 0 (no training) | — | — | — | — | — | — | — | 0.598 / 0.028 |
| two fixes, epoch 1 (EMA-lagged) | −0.07 / 0.76 | 1.00 | +0.055 / 0.147 | 0.005 | +0.03 / 0.71 | 1.12 | +0.27 | 0.564 / 0.027 |
| **all three fixes + EMA warm-up, epoch 1** (`cft-full-all3-ema`) | **−0.14 / 0.49** | **0.46** | **+0.010 / 0.091** | 0.010 | **−0.17 / 0.60** | **1.06** | **−0.11** | **0.382 / 0.022** |

The last row is the configuration the campaign will run: E18's config with
the three interface fixes and the EMA warm-up, one real epoch (the EMA now
tracks training). Against E18 after 20 epochs: sst RMSE 0.49 vs 0.98, polar
sst 0.46 vs 1.12, polar ice RMSE 0.091 vs 0.329 with the ice pack kept,
polar TS RMSE 1.06 vs 4.16, and at the Antarctic coastal cells sst −0.02 K
(E18 +0.74, two-fix run +1.38) and TS −0.11 K (E18 +4.79). Its own
validation is sst 0.38 / ice 0.022 against E18's 2.38 / 0.181 at epoch 1.
The −90° TS rows read +0.2 K in this run (4.2 K in the two-fix run, 6.9 K
in E18): one epoch of coupled fine-tuning with a tracking EMA suppressed the
B32 parent's pole-row drift, though whether that holds over 20 epochs is
for Phase B. Figure: `$PSCRATCH/cft-diag/fig_all3_ep1_gen.png`.

After one epoch the fixed CFT beats the campaign's 20-epoch E18 on every
polar and ice number by 2–4x, holds the ice pack (a small surplus rather
than a loss), has no spurious ice anywhere, and the coastal TS strip is gone
(+0.27 K at the polar partial-land cells vs +4.2). The remaining sst bias
map is the ordinary mid-latitude coupled drift that fine-tuning is for, plus
a warm strip along the Antarctic coast in sst that is the land-contaminated
coastal flux (open item 2). Figure: `$PSCRATCH/cft-diag/fig_final.png`.

## 3d. Audit of the CFT setup against early results (2026-09-10)

Things in the E18–E22 configs that distort *early* stage-3 results or waste
budget, found while checking why the one-epoch numbers look the way they do:

1. **EMA lag makes every early-epoch metric mostly the untrained model.**
   `ema: {decay: 0.9995, faster_decay_at_start: false}` with 205 steps per
   epoch: validation and inline inference use the EMA weights, and after N
   steps the EMA still holds `0.9995**N` of the initial weights: 0.90 at
   epoch 1, 0.65 at epoch 4 (E18's first inference), 0.40 at epoch 9, 0.14 at
   epoch 19. E18's "improvement over epochs" is partly the EMA catching up,
   and `best_inference_ckpt.tar` is selected on lagged weights. My
   `cft-full-fixed` epoch-1 numbers are therefore ~90% the raw pair with the
   fixes, which is why they sit so close to the untrained rollout. Fix:
   `faster_decay_at_start: true` (decay ramps as (1+n)/(10+n)); the
   generator now sets it, and adds a weight-0 `first_year` inference block
   (73 coupled steps, the selection ICs) every epoch as the early warning.
   Section 3c reports the EMA-corrected run.
2. **The ocean's `n_ensemble: 2` is wasted compute** (already in
   EXPERIMENTS.md): Samudra takes no noise, both members are identical, the
   energy-score term is zero and CRPS degenerates to MAE at 2× ocean cost.
3. **`use_gradient_accumulation: true` severs the cross-realm gradient**
   (see the 2026-08-25 note): the atmosphere never receives a gradient
   through the ocean's response to its fluxes and vice versa. Each realm
   fine-tunes on its own loss given the other's detached output. That is a
   modeling decision, but it means the CFT cannot learn to *compensate* an
   interface error, which is why the interface has to be right by
   construction (this note) rather than trained away.
4. **First inference at epoch 4, then every 5** (`epochs: {start: 4, step:
   5}`) combined with (1) meant nothing informative before ~1 GPU-day. The
   generator now adds the cheap 1-year block every epoch (weight 0, so
   selection still uses the 5-year block on its cadence).
5. **Coastal flux contamination (Antarctic strip).** Every Antarctic ocean
   cell adjacent to the coast has EAM land in it: 364 partial cells and 129
   cells where EAM is all land but MPAS is ocean (ice-shelf margins). The
   atmosphere's cell-mean flux there is mostly ice-sheet flux. Measured sst
   bias at those cells, Antarctic only:

   | | coastal, 0<LF<1 (364) | coastal, LF==1 (129) | interior, 0<LF<1 (1586) |
   |---|---|---|---|
   | uncoupled ocean, MPAS fluxes | −0.01 | −0.16 | +0.03 |
   | uncoupled ocean, EAM fluxes × (1−ice) | **+1.45** | **+3.75** | +1.16 |
   | raw pair + both fixes | +1.57 | +0.53 | +0.74 |
   | cft-full-fixed epoch 1 | +1.38 | +0.35 | +0.91 |

   It appears in the uncoupled ocean as soon as the fluxes are EAM's, so it
   is not a coupling-code effect. It is also not land contamination:
   replacing the coastal flux by nearby open-ocean values made it worse
   (coastal sst rmse 0.68 → 1.03), and scaling by the cell open-water
   fraction fixed Antarctica but broke every other coast (−5.4 K). The
   flux-by-flux comparison settles it: at Antarctic coastal cells MPAS's
   FLDS is 64 W/m² against 209 for EAM × (1−ice), and *exactly zero* in the
   129 cells where EAM is all land — everywhere outside Antarctica the two
   agree to a few W/m². That is ocean under floating ice shelves: MPAS-Ocean
   has cavities there, the atmosphere sees land, and the ocean-side
   atmosphere flux is zero. Neither LANDFRAC nor the wet mask can see it.

   **Fix 3 (data + `fme/coupled`):** a static `atmosphere_flux_fraction`
   field — the 1990–91 ratio of MPAS ocean-side FLDS to EAM FLDS × (1−ice) —
   is exactly 1.000 at all 44 540 ocean cells outside Antarctica and below
   0.9 in 1090 cells, all Antarctic. It now lives in the `landfrac5d` files
   (generated by `make_landfrac_ocn.py`, 126 years augmented in place) and
   `OpenWaterFluxScalingConfig.atmosphere_flux_fraction_name` multiplies it
   into the scaling; it is data-only and never reaches the ocean network.
   Uncoupled ocean with EAM fluxes × (1−ice) × fraction, Antarctic cells:
   coastal partial **−0.18 K** (was +1.45), ice-shelf cells **−0.14** (was
   +3.75), interior partial −0.04 (was +1.16); global sst rmse 0.41 → 0.28,
   polar ice rmse 0.041 → 0.016, ice area 1.01. 

   Raw coupled 1-year rollout (2 ICs), flux fix + guard + p=8, with and
   without the fraction:

   | | Antarctic coastal partial sst | ice-shelf cells sst | polar partial-cell TS | polar TS rmse | global TS rmse | global sst rmse | polar ice rmse |
   |---|---|---|---|---|---|---|---|
   | without | +1.57 | +0.53 | +2.31 | 3.32 | 1.87 | 1.10 | 0.214 |
   | with | **−0.38** | **−0.23** | **+0.39** | **1.69** | **0.97** | **0.73** | 0.173 |

   The coastal warm ring was feeding the polar atmosphere: removing it
   halves the polar TS RMSE of the untrained pair.
6. **Warm pole row in TS.** The TS bias at 89.5°S is +4.2 K (epoch 1),
   +6.9 K in E18, decaying to zero by 86.5°S. The uncoupled E01-FT atmosphere, rolled out one year with TS prescribed
   from EAM (`cft-diag/atm-ctrl`), has **+9.6 K at 89.5°S, +6.7 at 88.5°S,
   +2.7 at 87.5°S**, and −2.6 K at 89.5°N. Month by month it is
   seasonal, not steady: near zero through austral summer and autumn, then
   +23 / +31 / +31 K at 89.5°S in June–August and decaying through spring,
   for both ICs, and absent in the first two months of either the stage-1
   E01 or E01-FT. The atmosphere loses the polar-night surface cooling over
   the Antarctic plateau once a rollout has run for months; the +9.6 K
   annual mean is that averaged down. Every variable shares it (Tat2m +9.8,
   T_7 +7, PS +884 Pa, FLDS +40 in the annual mean), so it is a column-wide
   pole-row failure of the atmosphere model, halved rather than created by
   coupling. Not a CFT issue; a stage-1/2 item for the atmosphere side. It is
   also parent-specific: the wandb `heldout_1990s` TS bias map of E01-FT
   **B32** (the CFT parent) has the saturated red bottom rows, while the
   E01-FT **B16 S01** map has none (its pole rows are slightly cold on a
   ±1.2 K scale). Another reason to reseed stage 3 from the B16 parents.
   The pole rows stay in every map and metric: nothing is masked.
   The uncoupled atmosphere has no coastal TS strip (−0.3 K at the Antarctic
   coastal cells), confirming that strip is a coupling-interface effect.

## 4. Readiness verdict

**Yes for the CFT stage, with two conditions.** The three mechanisms behind
the polar and coastal biases are identified, isolated outside the coupling
code, fixed at the interface, and the fixes verified three ways: the
uncoupled ocean recovers, the campaign's own validation metric improves
6x before any training, and one epoch of the campaign CFT config with the
fixes beats 20 epochs without them on every ice and polar metric while
introducing no new failure mode (no subtropical ice, open-ocean sst rmse
unchanged, mid-latitude drift the same or smaller). The component stages
are untouched and nothing upstream needs rerunning.

Conditions:

1. Commit the code and config changes first (5.2) and launch from the
   commit; `check_campaign.py` now refuses a stage-3 config without the
   three fixes or without the flux-fraction field in the data. Re-stage the
   augmented `landfrac5d` files to CFS (`stage-shared-data.sh`) and set
   `ema.faster_decay_at_start: true` in the CFT configs (3d item 1).
2. Re-seed stage 3 from better parents than the B32 pair (5.1, 5.3). The
   fixes make a CFT viable; the parents decide how good it gets, and the
   only finished atmosphere parents are the two weakest.

Not covered by this verification: epochs beyond 1 (the allocation ended
during epoch 2), the 5-year `heldout_1990s` block (only the 1-year block was run),
and the `future_2040` block. None of those change the interface; they are
what Phase B is for.

Test status (2026-09-10, final code): `fme/core` + `fme/coupled` serially
without the per-test timeout, 1772 passed, 10 skipped, 1 failed:
`test_gradient_clipping_with_amp`, a cuda/cpu device mismatch in untouched
code when `FME_FORCE_CPU` is set on a GPU node, i.e. environmental. The
torchrun parallel tests were not run. ruff and mypy are clean on every
changed file.

## 5. Plan to resume the campaign

### 5.1 Where the campaign stands (wandb + disk, 2026-09-09)

* Stage 1: complete for every arm that has a stage-2 child (E01–E08, E11–E13);
  E17 (O1) stopped at epoch 15 of 30, no child.
* Stage 2 ocean: all seven `-FT` runs finished 40/40.
* Stage 2 atmosphere: only the two **B32** runs finished 20/20; every B16 arm
  was preempted between epoch 5 and 18 when the reservation closed. That is
  why all five CFTs were seeded from B32 parents, and the B32 parents are the
  weakest stage-2 atmospheres by rollout skill (`best_inference_error`,
  lower is better):

| atm parent | best_inference_error | last TS rmse (K) | epochs |
|---|---|---|---|
| E05-FT B16 S01 (A3 C1) | **0.0320** | 0.21 | 18/20 |
| E02-FT B16 S03 (C1) | 0.0323 | 0.20 | 14/20 |
| E01-FT B16 S01 (baseline) | 0.0325 | 0.17 | 14/20 |
| E02-FT B16 S02 (C1) | 0.0335 | 0.20 | 7/20 |
| E07-FT B16 (A3 C1 W1) | 0.0339 | 0.20 | 9/20 |
| E01-FT B32 (E18/E20/E22 parent) | 0.0746 | 0.37 | 20/20 |
| E02-FT B32 (E19/E21 parent) | **0.1698** | 1.74 | 20/20 |

| ocn parent | best_inference_error | sst rmse | ice rmse |
|---|---|---|---|
| E12-FT B16 W1 | 0.0487 | **0.078** | **0.0011** |
| E11-FT B08 | **0.0475** | 0.094 | 0.0014 |
| E11-FT B16 S01–S03 | 0.0488–0.0493 | 0.101–0.104 | 0.0018–0.0020 |
| E13-FT B16 W2 | 0.0528 | 0.087 | 0.0021 |
| E11-FT B32 (E18/E19 parent) | 0.0509 | 0.112 | 0.0025 |

* Stage 3: E18–E22 ran 4–20 of 50 epochs on an interface that was wrong on
  both sides. Their checkpoints are not worth continuing: the ocean has been
  fine-tuned for 20 epochs to expect cell-mean fluxes under ice, which the
  fix removes again. Restart from the parents.

### 5.2 Commits to make (in this order, each reviewable alone)

1. `fix/coupled-open-water-flux-scaling` (fme/coupled): `OpenWaterFluxScalingConfig`
   with the required SST guard and the optional static flux-fraction field,
   `StandaloneComponentCheckpointsConfig` passthrough, tests.
2. `fix/ocean-interpolate-weight-power` (fme/core): `interpolate_weight_power`
   on `OceanConfig`/`Prescriber`, tests.
3. `config/e3sm-hist-cft-interface-fixes`: `make_cpl_config.py` flags,
   `make_landfrac_ocn.py` flux fraction, regenerated `config-train-cpl.yaml`
   and `runs/E18–E22-CFT`, `regen_cft_configs.sh`, `check_campaign.py`
   stage-3 branch, the `AGENTS.md` entries and this note.
The campaign scripts record the commit in `job_config/COMMIT`; do not launch
a CFT from an uncommitted tree.

### 5.3 Execution plan

**Phase A — finish stage 2 (atmosphere), ~2 days on 4 nodes per run.**
Resume the preempted B16 `-FT` runs; `run-train.sh` resumes from
`ckpt.tar` automatically. Priority by rollout skill and seeds:
E01-FT B16 S01/S02/S03 (baseline triplet, gives error bars), E02-FT B16
S02/S03 (C1), E05-FT B16 S01 (A3 C1), E07-FT (W1). E03/E06/E08-FT are 5–9
epochs in and low priority. Skip E01-FT B32 L1 and E08-FT (worst arms).

**Phase B — a single pilot CFT first, 1 day.** Before spending the stage-3
budget, run one CFT with both fixes to 10 epochs at the campaign config
(8 nodes) and confirm the polar/coastal maps and the heldout ice RMSE stay
where `cft-full-fixed` puts them (section 3c). Use E01-FT B16 S01 ×
E12-FT B16 W1 (best parents available now, both `best_inference_ckpt.tar`).
If B16 parents are used the coupled global batch should track them:
`make_cpl_config.py --nodes 4` (B16) — record it in the run id as B16.

**Phase C — stage 3 proper, 2×3 grid plus seeds.** Atmosphere axis: E01-FT
B16 (baseline) and E02-FT B16 (C1), each at its best seed; add E05-FT B16
(A3 C1) as a third column if Phase A finishes it — aerosols are the
scientific point of the campaign. Ocean axis: E11-FT B16 (W0), E12-FT B16
(W1), E13-FT B16 (W2), all finished. Keep the documented rule of "siblings
differ in exactly one line" (parents at the same batch, only the
`parameter_init` paths change) so the grid stays interpretable; the B32
parents are dropped, which also removes the E20 B16-vs-B32 confound noted in
AGENTS.md. Seeds: three CFTs of the control cell (E01×E11) from the three
E01-FT seeds, one each elsewhere. That is 3 + 5 = 8 CFTs, 50 epochs, 4 nodes
each at B16.

**Phase D — evaluation.** The campaign's `heldout_1990s` block (364 steps,
32 ICs) is right for selection; add the 1-year, 16-IC block from
`cft-diag/make_cft_test.py` at every epoch — it is cheap and it is where
the ice-loss mode shows first. Judge a CFT against its parents' uncoupled
heldout numbers (E11-FT ice 0.0025, E01-FT TS 0.37), not against E18.

### 5.4 Open items, in order of risk

1. The blend exponent is a heuristic. p=8 removes about half of the
   partial-cell TS error before fine-tuning; the CFT is expected to close
   the rest because the loss sees TS at those cells. If Phase B's TS map
   still carries the coastal strip, raise p (16 was better still in the
   sweep) rather than retrain E01.
2. Land contamination of the atmosphere's coastal fluxes (30–60 W/m² rms)
   has no fix at the interface. It is a stage-1 design choice (train the
   ocean on EAM 5-daily fluxes, as CM4 does) and belongs in the next
   campaign, not this one.
3. `E11-FT` configs carry a duplicated `iceVolumeTotal` in
   `zero_where_ice_free_names`; harmless, inherited by the regenerated CFTs.
4. `check_campaign.py` reports 23 pre-existing complaints (personal-scratch
   paths in stage-2 configs); none on the CFT configs.
