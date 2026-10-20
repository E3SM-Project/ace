# Working log: E3SMv3 F2010 land emulator

Dated entries, newest first. The plan of record is `README.md`; this file
records what was found and why decisions were made. Follow the conventions of
`../e3sm_hist_v20260812/AGENTS.md`.

## 2026-10-20: cold start, BGC mode, spin-up drift, glacier check

From `run/lnd_in` (same in both cases) and the rebuilt dashboard data.

* **ELM cold-started.** `finidat = ' '`, so there is no initial-condition
  file. This explains the spin-up drift seen in the annual land means, which
  is the same in PD and PI. ZWT falls from 8.7 m to 5.85 m, TWS rises by about
  760 mm, H2OSOI_2 rises from 0.214 to 0.275, and ELAI rises from 0.02 to
  0.25 by year 5, then decays to 0.12. Hydrology states settle (annual mean
  within 2σ of the years 21–30 mean) around years 15–25. ZWT, TWS and
  ELAI/TLAI/TSAI still trend in years 10–30. It also explains the ~0.004 ELAI
  in the first months noted on 2026-10-19. Drop at least years 1–10 from
  training, validation and normalization statistics.
* **ELM ran with prognostic BGC** (`-bgc bgc -nutrient cnp` in the
  build-namelist call). Phenology is not prescribed (not SP mode), so
  `ELAI`/`TLAI`/`TSAI` are states, not forcings. The dashboard's "61%
  interannual variability" for ELAI is mostly the spin-up transient, not
  year-to-year variability.
* **The soil gaps are no-soil cells.** Surface dataset
  `surfdata_0.5x0.5_simyr2010_c230922_with_TOP.nc` (on Perlmutter under
  `/global/cfs/cdirs/e3sm/inputdata/lnd/clm2/surfdata_map/`). Of the 5,883 gap
  cells, matching each output cell to the native 0.5° cells at its center:
  5,750 are `PCT_GLACIER = 100`, 118 are `PCT_WETLAND = 100` along the
  Antarctic ice-shelf edge (no vegetated landunit, so no soil column either),
  7 are glacier/lake/wetland mixes, and 8 have some vegetated land (slivers
  with landfrac ≈ 0, or the Ross Ice Shelf edge). The converse fails: 674
  output cells centred on 100%-glacier native cells do have soil values,
  because the history remap blends in neighbouring vegetated cells. So build
  the soil mask from the output itself, not from `PCT_GLACIER`. An exact check
  needs the remap weights
  `/compyfs/inputdata/fme/map_r05_to_gaussian_180by360_shifted_trintbilin.nc`
  (`hist_horiz_remap_file`), which is on Compy only.
* **Dashboard rebuilt** after a `FORCE=1` rescan of all 720 months (data
  built 2026-10-20 08:38). The fill-value finding now reads ok. The deployed
  `index.html` is the repo version, with the About tab's fill-value history
  removed. Values are unchanged from the 17:14 build.

## 2026-10-19 (later): `elm.h0` fill value fixed upstream

* All 722 `elm.h0` files of both cases were rewritten in place at 17:29–17:30
  PDT, after the first full dashboard build (17:14). They now declare
  `_FillValue = missing_value = 1e20`, which matches the stored sentinel.
  `check_cdf5_fillvalue.py` passes (exit 0) on every file. Sentinel counts are
  unchanged, so only the attributes changed. `ISSUE-elm-fillvalue.md` is marked
  resolved.
* **The dashboard is stale on this point.** Its "72 of 72 … declaring
  `_FillValue` 1e+36" finding comes from `work/scan/*.json`, which recorded
  the attributes before the fix. A plain `build` repeats it; rescan first with
  `prep.py scan --force` (`FORCE=1` in `run_prep.sh`). A one-month test
  rescan took about 20 s per file and recorded `_FillValue` 1e20 with no
  mismatches.
* `build` no longer wipes `data/`. It records the files it writes in
  `data/.build-manifest.json` and removes only files from that list it no
  longer produces, so files added by other stages (inference sources) survive.
  It still rewrites `meta.js` from scratch, so a source added there by hand
  would be lost; inference sources will need to be merged in by `build`.
* New **Land coverage** tab: land mask (or landfrac) next to the coverage of
  the variables selected in the sidebar, one map per variable, a table of
  missing land cells, and PNG export. It needs only a `build` (the per-variable
  gap masks were already in the scans). All 9 gap variables miss the same
  5,883 land cells (7.80% of land area, Greenland and Antarctica).
* Unaffected: the `|x| >= 1e19` missing rule in `audit/prep.py` and the loader,
  and the in-land gaps of the soil fields (a data property, still needs a
  per-variable mask).

## 2026-10-19: decisions, data audit tooling, first verified facts

### Decisions (from the user)

* **6-hour timestep**, the native cadence of the data.
* **PD/PI is an extrapolation test**: train on one case (some of its years held
  out), run inference on the other. Which case trains, and the train/validation
  years, wait for the audit. Normalization statistics come from the training
  case only.

### Audit and dashboard (`audit/`)

* `audit/prep.py scan` reads every `elm.h0` month of both cases plus the
  matching `cpl.fme.x2l` file (about 2.1 TB), one worker per month, resumable.
  `audit/prep.py build` writes the dashboard.
* Code stays in the repo; data goes to
  `/pscratch/sd/i/imanick/land-inference/dashboard/` (`index.html`, `data/`,
  `work/scan/` per-month results, `logs/`). Open `index.html` directly
  (`file://` works, including coastlines). It is modelled on
  `/pscratch/sd/i/imanick/sep26v3-inference/dashboard`, and `lib/` is copied
  from there.
* Run: `sbatch configs/experiments/e3sm_land_f2010/audit/run_prep.sh` (one CPU
  node). First full run: job `60018903`. A two-month test took about 3 minutes
  per file on a login node, all of it I/O.
* Tabs: data audit (generated findings + tables), land-mean time series,
  monthly climatology maps (value or difference from a baseline source), about.
  PD and PI are "sources". **Inference runs are added later as further
  sources** with the same `ts/` and `maps/` file formats, so every view works
  unchanged. Not implemented yet: a stage that converts inference output into
  those files.
* Missing data is `|x| >= 1e19` or non-finite, so the audit works before and
  after the `_FillValue` fix (`ISSUE-elm-fillvalue.md`).
* Land means weight by cell area × `landfrac`. The latitude grid is **not
  uniform**: 180 rows from -89.24° to 89.24°, spacing 0.989–0.997°. Weights
  use midpoint bounds; the maps and coastlines account for the true latitudes.

### Verified so far (two-month test, `0001-01` and `0001-02`, both cases)

To be confirmed or corrected by the full run. The dashboard's findings
regenerate from the data, so trust them over this list.

* **Time stamps differ between streams.** `elm.h0` records carry the *end* of
  their 6-hour interval (`time_bounds[:,1] == time`; file `0001-02` runs
  31.25…59.0). The coupler streams' *files* start one record earlier
  (31.0…58.75). The first month is the exception: x2l has 123 records from
  0.25. **The values match when stamps are equal** ("same" alignment), not
  when shifted by 6 h. Pairing `elm.h0` at t with `x2l` at t gives identical
  `TBOT`, `THBOT`, `QBOT`, `PBOT` and `ZBOT`, and `FLDS`, `FSDS` (with all
  four components), `RAIN` and `SNOW` to about 1e-8 relative. So ELM does
  not repartition rain/snow, and either stream can supply the forcing. Only
  the month-file boundaries differ: a loader merging both streams must not
  assume equal per-file time axes.
* **`WIND` is the exception**: `elm.h0` `WIND` differs from
  `|(Sa_u, Sa_v)|` by a relative RMS of about 0.095. ELM probably adds a
  gustiness term or a floor; not checked. If the forcing comes from x2l, use
  `Sa_u`/`Sa_v`, not `WIND`.
* **In-land gaps:** 5,883 land cells are missing in every record for `TSOI_*`,
  `H2OSOI_*`, `SOILWATER_10CM`, `ZWT` and `QCHARGE` (the "vegetated landunits
  only" fields). On the gap map they sit on **Greenland and Antarctica**, which
  supports the glacier explanation. The set is the same in every record of
  the test months. Those fields need a per-variable mask.
* No valid data over `landfrac == 0` cells, and no NaN/inf/≥1e10 values
  outside the sentinel.
* **Worth a look in the full run** (could be spin-up, early-year or real):
  `ELAI` land mean is about 0.004 m²/m² with a maximum of 0.03 in the first
  months, implausibly small for leaf area. `ZWT` (water table depth) is about
  8.8 m almost everywhere. `QRUNOFF` is negative in many records (min about
  -3.5e-4 mm/s), which ELM allows but which matters for a non-negativity
  corrector.
* `elm.h0.0000-12` is a single record at time 0.0 in both cases (excluded).

### Correction to an earlier claim

`ISSUE-elm-fillvalue.md` first said the coupler files are netCDF4/HDF5. They
are classic netCDF-3 (`NETCDF3_CLASSIC`). Their `_FillValue` is still `1e20`
and matches what is stored, so the conclusion stands.
