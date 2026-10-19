# E3SMv3 F2010 — Samudra-style land emulator (ELM)

**Status: planning.** No configs, scripts or statistics exist yet. This README
is the plan of record for agents picking this up; update it as decisions are
made, and keep a dated working log in an `AGENTS.md` beside it (same
convention as `../e3sm_hist_v20260812/AGENTS.md`).

Branch: `e3sm/exps/land`, cut from `e3sm/exps/hist-v2026.8.0` at `4a48f902`.

## Goal

Build a land-model emulator for E3SM's land component (ELM) by **duplicating
the ocean recipe** in `../e3sm_hist_v20260812/`: the same
`single_module` stepper with the same **`Samudra` builder** (the M2Lines
ConvNeXt U-Net, `fme/ace/models/ocean/m2lines/samudra.py`, registered as
`type: Samudra` in `fme/ace/registry/m2lines.py`), the same masking and
normalization conventions, and the same ablation/launch tooling. The
reference is `config-train-ocn.yaml` (run `E11` of the 2026-08-31 campaign).
Read that directory's `README.md` and `AGENTS.md` before starting. They record
many decisions and bugs that apply here too.

Out of scope for the first pass: coupling the land into the atmosphere
(the land analogue of the `config-train-cpl.yaml` CFT runs). Design choices
should not rule it out, though; see "Coupling later" below.

## Data

```python
DATA_ROOT = "/global/cfs/cdirs/e3smdata/simulations"
CASES = {
    "PD": "v3.LR.F2010.aigo.PD.compy",   # present-day forcing
    "PI": "v3.LR.F2010.aigo.PI.compy",   # pre-industrial forcing
}
# files: f"{DATA_ROOT}/{case}/run/{case}.<stream>.<YYYY-MM>.nc"
```

What was verified on 2026-10-19 (PD, `0005-06`; PI has the same layout):

* **F2010 compset.** Atmosphere + land with a *data* ocean (`docn`, prescribed
  SST/ice), not a coupled run. The only difference between the cases is the
  PD vs PI forcing.
* **Grid and cadence.** Every stream below is regridded to **180×360**
  (`lat`, `lon`), the same grid as the historical run. Files are **monthly**, with
  **120 steps = 6-hourly**, `days since 0001-01-01`, noleap. Each case has
  361 files per stream: `0000-12` plus years 0001–0030. `elm.h0.0000-12`
  is ~19 MB against ~2.2 GB for the others, so it is a spin-up stub: confirm
  and exclude it.
* PD also contains `discarded_782295_nodefail/` (a node failure that was
  rerun). Do not read from it.

Streams in `run/`:

| stream | contents | relevance |
|---|---|---|
| `elm.h0` | 81 ELM history fields (below), 2.2 GB/month | **primary source**: states, fluxes, forcings |
| `cpl.fme.x2l` | coupler → land: `Sa_*` atm state, `Faxa_*` precip/radiation, `Flrr_*` river | alternative forcing source, exactly what ELM received |
| `cpl.fme.l2x` | land → coupler: `Sl_*` albedo/T/q, `Fall_*` fluxes, `Flrl_*` runoff | what the atmosphere sees, the targets for later coupling |
| `mosart.h0` | river routing (MOSART) storage and discharge | not in the first pass |
| `eam.h0`, `cpl.fme.a2x`/`x2a`, ocean/ice streams | atmosphere, data ocean, sea ice | not needed for an offline land model |

`elm.h0` fields:

* **2D:** `landfrac` (static, 0–1; 27,139 land cells, 8,768 of them partial).
* **Prognostic state candidates (instantaneous, `time: point`):**
  `TSOI_0/1/2`, `TSOI_10CM`, `H2OSOI_0/1/2`, `SOILWATER_10CM`, `H2OSNO`,
  `SNOWDP`, `SNOWICE`, `SNOWLIQ`, `FSNO`, `FH2OSFC`, `TG`, `TWS`, `ZWT`,
  `TSA`, `Q2M`, `RH2M`. The `_0/_1/_2` suffixes are thickness-weighted layer
  means (`_0` covers 0–0.07 m); check the other bounds in `long_name`.
* **Atmospheric forcing (atm → land):** `TBOT`, `THBOT`, `QBOT`, `PBOT`,
  `ZBOT`, `WIND`, `RH`, `FLDS`, `FSDS`, the four `FSDS[VN][DI]` components,
  `RAIN`, `SNOW`, and the depositions `BCDEP`, `DSTDEP`, `OCDEP`.
* **Land → atmosphere fluxes and diagnostics:** `FSH`, `EFLX_LH_TOT` (with
  `FCEV`, `FCTR`; no `FGEV` in this stream), `FIRE`, `FIRA`, `FSA`, `FGR`,
  `TAUX`, `TAUY`, `NEE`, `NEM` (`time: mean`); `ALB[DI]_{vis,nir}` and `U10`
  (instantaneous, `time: point`).
* **Hydrology fluxes:** `QRUNOFF`, `QOVER`, `QDRAI`, `QDRAI_PERCH`, `QINFL`,
  `QSNOMELT`, `QSOIL`, `QVEGE`, `QVEGT`, `QINTR`, `QCHARGE`, `QH2OSFC`, `QRGWL`,
  `QSNWCPICE`.
* **Vegetation:** `ELAI`, `TLAI`, `TSAI`. Check whether F2010 runs ELM with
  prescribed phenology (SP mode). If it does, these are forcings, not states.

### Masking: the fill value is not the declared one

**In `elm.h0` only, non-land cells hold `1e20` but the declared `_FillValue`
is `1e36`**, so `mask_and_scale: true` does **not** mask them. The `cpl.fme.*`
and `mosart.h0` streams declare `1e20` and are fine. Details, verification and
the requested upstream fix are in
[`ISSUE-elm-fillvalue.md`](ISSUE-elm-fillvalue.md). If the files get fixed,
`mask_and_scale: true` works as it does for the ocean, and options 1–2 below
are only needed for the in-land soil gaps (if at all). Within land cells, some soil
fields (`H2OSOI_0`, `TSOI_0`, `ZWT`, ...) also hold `1e20`, presumably at
glacier/lake columns that have no soil. Any stats or training code that does
not handle this will be ruined by `1e20` values. Options, in order of
preference:

1. Treat `|x| >= 1e19` as NaN in the loader (check whether
   `fme/core/dataset` has a hook for this; otherwise add one). Then use
   `normalization.fill_nans_on_normalize` and `input_masking` as the ocean
   does.
2. Preprocess to NaN-masked files or zarr. This breaks the "read the raw run
   directory, no preprocessing" property the historical configs have.

The masking mechanism needs a static mask from `landfrac`, analogous to the
ocean's `mask_2d`/`LANDFRAC` handling (`input_masking` with
`exclude_names_and_prefixes`). The per-variable mask will be needed too,
because soil fields are missing where `landfrac > 0`.

## Plan

Each step is meant to be a reviewable commit on `e3sm/exps/land`.

1. **Data audit and dashboard** (`audit/`). `audit/prep.py` reads every
   `elm.h0` file of both cases, plus `cpl.fme.x2l` for the forcing comparison,
   and checks time axis continuity, the `0000-12` stub, `1e20` coverage per
   variable, cells where `landfrac > 0` but a field is `1e20`, value ranges,
   whether `elm.h0` forcings match `x2l`, and whether vegetation is
   prescribed. It writes a static dashboard (modelled on
   `/pscratch/sd/i/imanick/sep26v3-inference/dashboard`) to
   `/pscratch/sd/i/imanick/land-inference/dashboard/`. The data lives there;
   the code stays in this repo. The dashboard shows PD and PI now, and
   inference runs are added later as further sources. Record the findings in
   `AGENTS.md`.
2. **Variable selection.** Fix `in_names` / `out_names` /
   `next_step_forcing_names` and record the reasoning here. Starting proposal:
   * prognostic (in+out): `TSOI_0..2`, `H2OSOI_0..2`, `H2OSNO`, `SNOWDP`,
     `TG`, `TWS` (optionally `ZWT`, `FSNO`)
   * diagnostic (out only): `FSH`, `EFLX_LH_TOT`, `FIRE`, `ALB*`, `TSA`,
     `Q2M`, `QRUNOFF` (these are what an atmosphere coupling needs)
   * forcing (in only, `next_step_forcing_names`): `TBOT`, `QBOT`, `PBOT`,
     `WIND`, `FLDS`, `FSDS`, `RAIN`, `SNOW`, plus the depositions if they
     differ between PD and PI (the model must see whatever forcing changes
     between the cases, or the cross-case test cannot succeed)
   * static: `landfrac`; consider topography from `x2l` `Sa_topo`
3. **Masking support.** Make `1e20` become NaN (step 1 of the masking section),
   then add a `land_corrector` analogous to `ocean_corrector`:
   `force_positive_names` for snow/water, and zeroing over non-land cells.
   Check whether the older `soil_moisture_corrector` from
   `../2025-04-20-lsm-offline/` (a CM4 offline LSM built on `LandNet`/MLP) is
   still in `fme` and can be reused. That experiment is a useful reference for
   which land variables were chosen and why.
4. **Normalization statistics.** Adapt `../e3sm_hist_v20260812/compute_hist_stats.py`
   (streaming, exact, unweighted, training-window-only, computed on the field
   *as the loader delivers it*) to the `elm.h0` stream with the `1e20` masking,
   over the training case's training years only.
   Write the result to CFS under
   `/global/cfs/cdirs/e3smdata/emulator/...`, next to the SamudrACE stats.
5. **Baseline config** `config-train-lnd.yaml`, cloned from
   `config-train-ocn.yaml` and keeping its optimizer, EMA, `n_forward_steps: 4`,
   MSE loss, Samudra widths/dilations and `checkpoint_strategy`. Then
   `validate_config`, a smoke test with `make_smoke_config.py`, and a 4-GPU
   timing run.
6. **Campaign.** Reuse `make_ablation_config.py` and the `sbatch-scripts/`
   pattern (copy, or generalise in place). Candidate axes: training case
   (PD→PI vs PI→PD), variable set, Samudra width.

### Decisions

Settled (2026-10-19, by the user):

* **Timestep: 6 hours**, the native cadence of the data. This matches the
  atmosphere, so the land can later be coupled to ACE, and keeps the diurnal
  cycle that dominates `TG`/`TSA`. One year of inference is 1460 steps.
* **PD vs PI is an extrapolation test.** Train on **one** case, with some of
  its years held out for validation, then run inference on the **other** case
  to measure how well the model extrapolates to a different forcing climate.
  The case not trained on is never used for training, normalization
  statistics or model selection. Which case trains, and which years are held
  out, is decided **after the audit** (step 1). Normalization statistics
  (step 4) must therefore come from the training case only.

Still open:

* **Which case to train on**, and the train/validation years within it.
  Decide from the audit (step 1): how far PD and PI differ, per variable.
* **Forcing source.** `elm.h0` (one stream, easy) or `cpl.fme.x2l` (exactly
  what ELM saw, and what a coupled atmosphere would produce). The audit
  compares the two.
* **`TG`/`TSA` as state or diagnostic.**
* **Inference length** for the cross-case test (up to the full 30 years,
  about 43,800 steps).

### Coupling later

Keep variable names compatible with what ACE's atmosphere emits and consumes
(the ocean config renames MPAS names to EAM names: `TAUX`, `FLDS`, `LHFLX`,
...), so a coupled land–atmosphere step can be added later. `l2x` and `x2l`
are the ground truth for that interface.

## Where things live

| what | where |
|---|---|
| reference ocean config | `../e3sm_hist_v20260812/config-train-ocn.yaml` |
| stats, ablation and launch tooling to adapt | `../e3sm_hist_v20260812/*.py`, `sbatch-scripts/` |
| previous offline LSM (CM4, MLP) | `../2025-04-20-lsm-offline/` |
| Samudra / ConvNeXt model | `fme/ace/models/ocean/m2lines/`, `fme/ace/registry/m2lines.py` |
| ocean corrector (template for a land corrector) | grep `ocean_corrector` under `fme/` |
| raw data | `/global/cfs/cdirs/e3smdata/simulations/v3.LR.F2010.aigo.{PD,PI}.compy/run` |
