# Normalization statistics computed from the historical run

Output: `/pscratch/sd/m/mahf708/2026-08-13-E3SMv3-historical-stats/`

    train-only/atmosphere/  centering.nc  scaling-full-field.nc  scaling-residual.nc
    train-only/ocean/       centering.nc  scaling-full-field.nc
    atmosphere/             centering.nc  scaling-full-field.nc  scaling-residual.nc
    ocean/                  centering.nc  scaling-full-field.nc

**Two sets, from a single pass over the data.**  Both cover **83 atmosphere and
127 ocean fields** -- every variable in the source streams, not just the ones the
current configs use (see "Coverage").

| set | period | atmosphere samples | ocean samples |
|---|---|---|---|
| **`train-only/`** (recommended default) | the training windows only: 1940-01-01 -> 1990-01-01 and 2000-01-01 -> 2040-01-01 | 131 399 | 6569 |
| `atmosphere/`, `ocean/` | the whole record, 1940-01 -> 2065-01 | 182 500 | 9125 |

**Use `train-only/`.**  The full-record set has seen the validation window
(1990-1995) and the held-out test period (2040 onward, where the `12yr_test`
inference block starts), so normalizing with it leaks those years into training.
The full-record set is kept because it is the natural "all the data" reference
and because the difference is small and now quantified (see "Choosing between
the two sets"), not because it is the one to train with.

Everything below applies to both sets unless it says otherwise.

Produced by `compute_hist_stats.py` in this directory:

    P=/pscratch/sd/m/mahf708/2026-08-13-E3SMv3-historical-stats

    uv run python compute_hist_stats.py --realm ocean \
        --out-dir $P/ocean --partials $P/_partials/ocean-partials.pkl --workers 64

    # atmosphere: 3.7 TB of reads, so sharded over three nodes, then aggregated.
    # The partials must go somewhere shared -- a per-node /tmp loses them.
    for i in 0 1 2; do
      srun -N1 -n1 --nodelist=$NODE_i --overlap python compute_hist_stats.py \
        --realm atmosphere --shard $i/3 --partials $P/_partials/atm-shard$i.pkl \
        --partials-only --workers 64 &
    done; wait
    uv run python compute_hist_stats.py --realm atmosphere --out-dir $P/atmosphere \
        --reuse-partials --partials $P/_partials/atm-shard0.pkl,...shard1.pkl,...shard2.pkl

    # the train-only set is the same partials re-aggregated, no re-reading
    uv run python compute_hist_stats.py --realm ocean --out-dir $P/train-only/ocean \
        --reuse-partials --partials $P/_partials/ocean-partials.pkl \
        --years 1940-1989,2000-2039

Wall clock: ocean ~7 min on one node; atmosphere ~23 min on three nodes at
~0.9 GB/s per node (3.7 TB of reads for the atmosphere, 0.3 TB for the ocean).

`$P/_partials/` holds the per-file (count, mean, M2) partials **for every
covered variable**, so any sub-period can be re-aggregated in seconds without
re-reading the run.  Adding a *new* variable is the one thing that still costs a
full re-read, which is why coverage is defined by a rule over the files rather
than by the current config name lists.  Aggregation combines files in sorted-path
order, so it does not depend on the order the pool finished them and reproduces
byte-identical files.

These replace the piControl-derived stats the configs currently point at
(`2026-06-02-E3SMv3-piControl-105yr-coupled-stats/uncoupled_atmosphere` and
`2026-08-12-E3SMv3-hist-ocean-stats-with-FSNS`).  Neither of those was touched.

## What was computed and how

* Source: `v3.LR.historical_0101.aigo` raw run directory, **every file** of every
  stream and **every variable** in those files, no time subsampling within the
  period.
  * atmosphere: `eam.h0`, 6-hourly.  1501 files, 1940-01 .. 2065-01, 182 500
    samples for the full-record set; 1080 files, 1940-01 .. 2039-12, 131 399
    samples for `train-only/`.
  * ocean: `fmeDepthCoarsening5D`, `fmeDerivedFields5D`,
    `fmeSeaiceDerivedFields5D` and the `landfrac5d` aux files, 5-day.  1501
    files per stream and 9125 samples for the full-record set; 1080 files and
    6569 samples for `train-only/`.
* Per file, per field: count / mean / sum of squared deviations over the valid
  points, combined across files with Chan's parallel algorithm.  Exact (to
  floating point) and order-independent, not an approximation.
* **Unweighted** over (time, lat, lon) -- no area or cos(lat) weighting.  This
  matches `scripts/data_process/get_stats.py`, which does
  `ds.mean(dim=["time", "lat", "lon"])`, and was confirmed numerically: the
  piControl `LANDFRAC` centering value is 0.3426250, the unweighted mean of this
  run's `LANDFRAC` is 0.3426344, and the cos(lat)-weighted mean is 0.2934.
  Consequence: the centering values are pole-weighted.  Unweighted global mean
  `TS` is ~279 K, not the ~287 K you get area-weighted.  That is the existing
  convention, not a bug.
* Standard deviations are population (ddof=0), as `xarray.Dataset.std`.
* NaN and `_FillValue` points are excluded, as xarray's default `skipna=True`.
  MPAS files flag land with `_FillValue = 1e20`; that is decoded (the same thing
  `mask_and_scale: true` does in the loader) and those points are dropped.  Any
  value that still exceeds 1e19 after decoding is counted and excluded; the
  count was zero for every stream, including the atmosphere, which the loader
  reads with `mask_and_scale` off.

## Coverage

Each stream contributes two kinds of entry.

**(a) Loader names.**  Every name the three configs ask for, under the name and
with the value the data loader delivers (see the next section).  61 for the
atmosphere and 91 for the ocean, exactly the config names in each case.

Note the atmosphere count was 56 when these statistics were computed, before
the CO2/aerosol channels were added on 2026-08-14.  Five names --
`aerindexall`, `colccn.3`, `lwp`, `lcc`, `cdnc` -- therefore appear in the
tables below under (b), as raw extras, even though the configs now reference
them directly.  The **files are unaffected**: coverage is a deliberate superset
and every one of those names is present with a finite, non-zero scale.  Only
the (a)/(b) bookkeeping below is stale.

**(b) Every other variable in the files, under its raw name.**  A variable
qualifies when it is a floating-point field on (time, lat, lon) or on the time
axis alone, is not one of the auxiliaries `get_stats.py` drops (`mask_*`,
`idepth_*`, `ak_*`, `bk_*`, `P0`, coordinates and bounds) or a calendar
bookkeeping entry (`date`, `datesec`, `nsteph`, `mdt`, ...), and is **not already
read to build an (a) name**.  This is discovered from the files themselves by
`discover_raw_extras()`, not hand-listed.

| stream | (a) | (b) | (b) names |
|---|---|---|---|
| `eam.h0` | 56 | 28 | `AODVISall`, `FLNS`, `FLNT`, `FSDS`, `FSNTOA`, `FSUS`, `QREFHT`, `Q_0..Q_7`, `STWat2m`, `TREFHT`, `aerindexall`, `ccn.3bl`, `cdnc`, `ch4vmr`, `colccn.3`, `f11vmr`, `f12vmr`, `lcc`, `lwp`, `n2ovmr`, `sol_tsi` |
| `fmeDepthCoarsening5D` | 76 | 19 | `layerThicknessCoarsened_0..18` |
| `fmeDerivedFields5D` | 11 | 12 | `evaporationFlux`, `iceRunoffFlux`, `iceRunoffFluxLf`, `icebergHeatFlux`, `riverRunoffFlux`, `riverRunoffFluxTemperature`, `seaIceFreshWaterFlux`, `seaIceHeatFlux`, `seaIceSalinityFlux`, `snowFluxLf`, `sss`, `surfaceHeatFluxTotal` |
| `fmeSeaiceDerivedFields5D` | 2 | 7 | `airStressMeridional`, `airStressZonal`, `iceThicknessMean`, `snowVolumeTotal`, `surfaceTemperatureMean`, `uVelocityGeoCell`, `vVelocityGeoCell` |
| `landfrac5d` | 2 | 0 | |

Every variable carries a `coverage` attribute saying which kind it is.

### Raw names deliberately not added

A name has to mean one thing, so a raw variable that is consumed to build a
loader name does not also appear under its own name.  The value under the loader
name is the one to use:

| raw name | superseded by | why it would collide |
|---|---|---|
| `PRECT` | `surface_precipitation_rate` | m/s vs kg/m2/s (x1000) |
| `PRECST` | `frozen_precipitation_rate` | m/s vs kg/m2/s (x1000) |
| `co2vmr` | `global_mean_co2` | pure rename |
| `windStressZonal` | ocean `TAUX` | sign flipped |
| `windStressMeridional` | ocean `TAUY` | sign flipped |
| `longWaveHeatFluxUp` | ocean `FLUS` | sign flipped |
| `latentHeatFlux` | ocean `LHFLX` | sign flipped |
| `sensibleHeatFlux` | ocean `SHFLX` | sign flipped |
| `shortWaveHeatFlux` | ocean `FSNS` | pure rename |
| `longWaveHeatFluxDown` | ocean `FLDS` | pure rename |
| `snowFlux` | ocean `frozen_precipitation_rate` | pure rename |
| `rainFlux` | consumed by `surface_precipitation_rate` | combine source |
| `sst` (degC) | ocean `sst` (K) | +273.15, same name |
| `iceAreaTotal` | ocean `ocean_sea_ice_fraction` | pure rename |

For the pure renames the two would have been numerically identical, so nothing
is lost; for the rest, keeping both would have put two different quantities in
one file.

### Dropped: constant over the window, zero scale

Three covered variables have a standard deviation of exactly zero, i.e. no
usable normalization, and are omitted from the files rather than shipped as a
divide-by-zero.  The same three are dropped in both windows, so the atmosphere
files hold 84 - 1 = 83 entries and the ocean files 129 - 2 = 127.

| variable | value | note |
|---|---|---|
| `sol_tsi` | -1.0 everywhere, all 1501 files | a sentinel, not a physical TSI; this EAM configuration does not diagnose it |
| `layerThicknessCoarsened_0` | exactly 20.0 m at every valid point and time | the top coarse bin is a fixed 20 m |
| `icebergHeatFlux` | identically 0.0 | already known; this is why the configs exclude it |

## Statistics are of the field the loader delivers, not the field on disk

`compute_hist_stats.py` mirrors `XarrayDataset`: `rename`, then
`overwrite` (`x * multiply_scalar + add_scalar`), then `combine` (weighted sum),
in that order.  So

* `TAUX`/`TAUY`/`FLUS`/`LHFLX`/`SHFLX` are the sign-flipped MPAS fields,
* `sst` is in K (`+273.15`),
* `surface_precipitation_rate` is formed **pointwise** as
  `rainFlux + snowFlux` before any statistic is taken,
* atmosphere `surface_precipitation_rate`/`frozen_precipitation_rate` are
  `PRECT`/`PRECST` times 1000.

Each variable in the output files carries a `loader_transform` attribute
recording what was applied.

## Residual scaling (atmosphere only)

`scaling-residual.nc` is the standard deviation of consecutive-in-time
differences.  Differences are taken **within each file only**, so the 1500
file-boundary pairs (0.8% of all pairs) are skipped; that has no visible effect
at this sample size.

Fields with no variation between consecutive times (`LANDFRAC`, `PHIS`) have a
residual standard deviation of exactly zero, which would be a divide-by-zero.
They are written with their full-field scale instead, which is what the
piControl `scaling-residual.nc` also does for those two fields.  Residual scales
are only ever applied to prognostic names, so these entries are unused.

## `global_mean_co2`

Added to the atmosphere files so that the `co2vmr` -> `global_mean_co2` forcing
documented at the top of `config-train-atm.yaml` no longer needs a hand-patched
normalization entry.  `co2vmr` is one scalar per time step, so its statistics
are over time alone.  In `scaling-residual.nc` it carries its **full-field**
scale, not its true step-to-step scale (~1e-9), which would be a dangerous
divisor; it is never used, since residual scales apply to prognostic names only.

| set | centering | scale |
|---|---|---|
| `train-only/` | 3.658244e-04 | 5.128173e-05 |
| full record | 3.950886e-04 | 7.442359e-05 |

When these statistics were computed, `config-train-atm.yaml` quoted
3.658872e-04 / 5.131649e-05, derived by a completely different route; the
`train-only/` numbers reproduced those to 0.02% and 0.07%.  That was a genuine
independent check both that this pipeline agrees with how those numbers were
originally obtained and that the training window is selected correctly.

**The config has since been updated to quote the values in this table
verbatim, so re-deriving the comparison today is circular.**  The original
figures are kept above as the record of the check that was actually performed.

The full-record scale is 31% larger because CO2 keeps rising to 2065.

## Choosing between the two sets

The training windows are 1940-01-01 -> 1990-01-01 and 2000-01-01 -> 2040-01-01
(`stop_time` exclusive), identical for both realms.  Validation is 1990-1995 and
the held-out `12yr_test` rollouts start 2040-01-03, so the **full-record set has
seen both**; `train-only/` has seen neither.  Prefer `train-only/`.

Window selection is at file granularity, which lands on the config boundaries
almost exactly:

* atmosphere: the 1940-01 file starts at 1940-01-01T06:00, exactly the config's
  `start_time`; the 1989-12 file ends at 1989-12-31T18:00 and the next step,
  1990-01-01T00:00, is the first step of the 1990-01 file, so the first window is
  reproduced exactly.  In the second window the 2000-01 file begins at
  2000-01-01T00:00 while the config starts at 2000-01-01T06:00, so `train-only/`
  contains **one sample the training loader does not use** (1 in 131 399).
* ocean: same story.  1940-01 starts at 1940-01-06, the config's `start_time`;
  1989-12 ends at 1989-12-27 and 1990-01-01 belongs to the 1990-01 file.  The
  2000-01 file begins at 2000-01-01 while the config starts at 2000-01-06, so
  again one extra sample (1 in 6569).

Sample counts are 72.0% of the full record for both realms (131 399 / 182 500
and 6569 / 9125), which is what 90 of 125 years should give.

### What the choice costs

Of the 350 numbers in the five files, 102 move by more than 1% between the two
sets, 48 by more than 2%, and 10 by more than 10%.  The large relative movers are
almost all near-zero means of near-zero-mean fields, where the shift is
irrelevant once divided by the field's own scale (`V_2` moves 124% but 0.0004
sigma; ocean `TAUY` 74% but 0.0014 sigma).  The entries that actually matter:

| realm | stat | field | full record | train-only | change | in sigma |
|---|---|---|---|---|---|---|
| atmosphere | scale | `global_mean_co2` | 7.44236e-05 | 5.12817e-05 | -31.1% | |
| atmosphere | mean | `global_mean_co2` | 3.95089e-04 | 3.65824e-04 | -7.4% | -0.39 |
| atmosphere | scale | `STW_0` | 1.24406e-07 | 9.69112e-08 | -22.1% | |
| atmosphere | scale | `STW_1` | 1.22747e-06 | 1.11445e-06 | -9.2% | |
| atmosphere | scale | `STW_2` | 4.79249e-05 | 4.46480e-05 | -6.8% | |
| atmosphere | mean | `STW_0` | 1.28573e-06 | 1.23946e-06 | -3.6% | -0.37 |
| atmosphere | mean | `ICEFRAC` | 0.102687 | 0.107845 | +5.0% | +0.02 |
| atmosphere | scale | `ICEFRAC` | 0.273979 | 0.280249 | +2.3% | |
| ocean | mean | `iceVolumeTotal` | 0.254539 | 0.281731 | +10.7% | +0.04 |
| ocean | scale | `iceVolumeTotal` | 0.767438 | 0.825839 | +7.6% | |
| ocean | mean | `ocean_sea_ice_fraction` | 0.157653 | 0.165507 | +5.0% | +0.02 |
| ocean | scale | `ocean_sea_ice_fraction` | 0.345390 | 0.352421 | +2.0% | |

The sea-ice and stratospheric-water fields move because the run loses ice and
gains stratospheric water monotonically; dropping 2040-2065 removes the extreme
end of both trends.  Everything else -- all temperatures, winds, fluxes,
salinities, ocean velocities, precipitation, `PS`, `TS`, `LANDFRAC` -- moves by
under 2%.

### `STW_0`, resolved only partly

`STW_0`'s full-record scale is 5.5x the piControl value, because it is dominated
by the secular trend in stratospheric water rather than by variability (within
any single 30-year window the scale is 2.7e-08 to 6.1e-08, close to piControl,
while the mean rises from 1.146e-06 in 1940-1969 to 1.471e-06 in 2040-2065).

Restricting to the training years **does not** resolve this:

| | mean | full-field scale | residual scale |
|---|---|---|---|
| piControl | 1.08516e-06 | 2.27333e-08 | 2.97773e-09 |
| full record | 1.28573e-06 | 1.24406e-07 (5.47x) | 3.11148e-09 |
| `train-only/` | 1.23946e-06 | 9.69112e-08 (4.26x) | 2.97396e-09 |

The scale comes down 22% but is still 4.3x piControl, because the trend spans
the training windows too (1940-2039 is 100 of the 125 years).  So this is a
property of running on a historical/scenario record, not an artifact of
including the test years, and it cannot be fixed by choice of window.  The
**residual** scale, which is what the loss uses, is 2.97396e-09 in `train-only/`
-- within 0.13% of piControl's 2.97773e-09 -- so the loss weighting for `STW_0`
is effectively unchanged.  What changes is the network input scaling: `STW_0` will
be compressed roughly 4x relative to the piControl normalization.  Not obviously
wrong (it keeps a trending field inside a sane normalized range for the whole
rollout) but worth watching if the stratosphere misbehaves.

## Cross-checks performed

Both sets were checked identically.

* Every name in `config-train-atm.yaml`, `config-train-ocn.yaml` and
  `config-train-cpl.yaml` (`in_names`, `out_names`, `next_step_forcing_names`)
  is present, finite and non-zero-scaled -- all 136 coupled-config names across
  the two realm files.
* Loaded through `fme.core.normalizer.NormalizationConfig`: the network
  normalizer for all 55 atmosphere names and all 91 ocean names, and the
  residual normalizer for the 38 prognostic atmosphere names.
* No NaN, no inf, no zero or negative scale in any of the ten files.
* Widening the coverage from the config names to the full rule-(a)+(b) set left
  every pre-existing number untouched.  Verified twice: the per-file moment
  tuples for the 56 + 91 shared names are bit-for-bit equal between the old and
  new partials (217 897 tuples compared), and every one of those names has an
  identical float32 value in all ten published files.
* Compared field by field against the stats they replace.  The atmosphere
  agrees with piControl to within a few percent for essentially every field
  (means and both scales); see "Period" above for `STW_0`, and note `ICEFRAC`
  (-14% in the mean) which simply reflects a warmer run.

  The ocean has four groups of large differences, all of them expected:

  1. **`FLDS`/`FLUS` scales +50-58%, `LHFLX` mean +26%, `frozen_precipitation_rate`
     mean -54%.**  The previous file carried EAM (piControl) values for these;
     they are now the MPAS ocean-side fluxes, which are suppressed under sea ice
     (`shortWaveHeatFlux` is exactly zero at 15% of valid points, the others at
     1.5%).  That bimodality is a real property of the data the loader reads,
     and it inflates the standard deviation relative to an EAM grid-cell mean.
  2. **`ocean_sea_ice_fraction` -52% and `iceVolumeTotal` -64% in the mean.**
     The previous file's sea-ice statistics were taken over a restricted "sea
     ice mask" region covering 40.0% of the grid (`mask_ocean_sea_ice_fraction`
     centering value in that file), whereas the MPAS fields are defined over all
     valid ocean points (69%).  The rest of the gap is the piControl-to-
     historical loss of sea ice: even the 1940-1969 sub-period of this run gives
     0.178 / 0.340, still well below the previous 0.327 / 0.707.
  3. **`TAUX`/`TAUY` and the deepest velocity means change by tens of percent**,
     but they are all tiny compared with their own scale (e.g.
     `velocityMeridionalCoarsened_18`: -3.6e-05 against a scale of 5.6e-03), so
     the normalized difference is negligible.
  4. **`LANDFRAC` scale -1.0%** (0.447356 vs 0.451899) while the mean agrees to
     3e-5 relative.  0.447356 is verified directly against both `eam.h0` and the
     `landfrac5d` aux files, so it is right for this run; the piControl file's
     land mask must differ very slightly (a different remap smooths coastlines,
     which moves the variance much more than the mean).  Not investigated
     further.
