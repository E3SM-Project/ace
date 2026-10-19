# Issue: `elm.h0` files declare `_FillValue = 1e36` but store `1e20`

**Affected:** the regridded ELM land history files (`*.elm.h0.*.nc`) of

```
/global/cfs/cdirs/e3smdata/simulations/v3.LR.F2010.aigo.PD.compy/run/
/global/cfs/cdirs/e3smdata/simulations/v3.LR.F2010.aigo.PI.compy/run/
```

**Found:** 2026-10-19 while planning the land emulator in this directory
(see `README.md`). The files are owned by `mahf708`, read-only to everyone
else.

## The problem

Every lat/lon variable in `elm.h0` except `landfrac` (72 of 73) stores
**`1e20`** at missing cells, but its attributes declare

```
_FillValue    = 1e36f
missing_value = 1e36f
```

Readers that honour CF fill values (xarray's `mask_and_scale=True`,
netCDF4's auto-masking, NCO, CDO, ...) therefore do **not** mask those cells.
They see `1e20` as a real 2 m temperature, soil moisture, runoff and so on.
For ML training that means normalization statistics and losses dominated by
`1e20`. For plots and means it means silently wrong numbers.

Where the `1e20` sits, in record 0 of `elm.h0.0015-07`, PD and PI identical:

| cells holding `1e20` | which variables | what they are |
|---|---|---|
| 37,661 | all 72 | every cell with `landfrac == 0` (64,800 grid cells − 27,139 land cells) |
| +5,883 (43,544 total) | soil fields: `H2OSOI_*`, `TSOI_*`, `ZWT`, ... | land cells with no soil column, presumably glacier/lake |

No cell in any variable equals the declared `1e36`.

## Not affected (checked)

The other regridded streams in the same `run/` directories declare `1e20`, so
their fill values decode correctly:

| stream | declared `_FillValue` |
|---|---|
| `cpl.fme.x2l`, `cpl.fme.l2x` | `1e20` on all lat/lon variables |
| `mosart.h0` | `1e20` (one variable declares none) |

Checked in months `0001-01`, `0015-07` and `0030-12` of both cases.

## How it was verified

To rule out a library artifact, the check does not use netCDF4 or xarray.
[`check_cdf5_fillvalue.py`](check_cdf5_fillvalue.py) (Python stdlib only)
parses the CDF5 header bytes directly from the netCDF format spec, then reads
the stored floats straight from disk:

```bash
python3 check_cdf5_fillvalue.py \
  /global/cfs/cdirs/e3smdata/simulations/v3.LR.F2010.aigo.PD.compy/run/v3.LR.F2010.aigo.PD.compy.elm.h0.0015-07.nc
```

```
variable            _FillValue  missing_value  n(|x|>=1e19)  n(==_FillValue)  status
landfrac                 1e+36          1e+36             0                0  ok
ALBD_nir                 1e+36          1e+36         37661                0  MISMATCH
...
TSA                      1e+36          1e+36         37661                0  MISMATCH
TSOI_0                   1e+36          1e+36         43544                0  MISMATCH
72 variable(s) store sentinels that differ from _FillValue
```

It exits 1 on any mismatch, so it can be looped over all files as a
before/after check. It reads CDF5 only. The coupler streams are netCDF4/HDF5,
so check those with any netCDF tool.

Spot values read the same way (`0001-01`, record 0, PD):

| cell | `landfrac` | `TSA` | `TSOI_0` |
|---|---|---|---|
| 0.5°N, 180.5°E (Pacific) | 0.0 | `1.0000000200408773e+20` | `1e20` |
| 20.4°N, 10.5°E (Sahara) | 1.0 | 275.23 K | 273.34 K |

Coverage: 6 of the 722 `elm.h0` files were checked directly (3 months × 2
cases), with the same result in every one. The full set has not been swept.

## Likely cause (unverified)

ELM's native history output uses `1e36` as its fill value. The step that
regridded it to 180×360, which also produced the `cpl.fme.*` and `mosart.h0`
files, seems to write `1e20` into unmapped/empty cells but copies the source
variable's `1e36` attributes through. The scripts have not been inspected.

## Requested fix

On every `*.elm.h0.*.nc` file in both `run/` directories, set `_FillValue` and
`missing_value` to `1e20` (float32) on each variable that currently declares
`1e36`. This is metadata only, and no data values change. With NCO:

```bash
ncatted -O -a _FillValue,,m,f,1e20 -a missing_value,,m,f,1e20 <file>
```

Notes for whoever applies it:

* Try one file first and rerun `check_cdf5_fillvalue.py` on it. It should
  report `1e+20` and `ok` (non-zero `n(==_FillValue)`) for every variable, and
  exit 0.
* Some netCDF library versions refuse to modify `_FillValue` on an existing
  variable. If `ncatted` fails, rewriting the file (e.g. `ncks` with the
  attribute renamed) is the fallback. Same-size attribute edits should not
  need a full rewrite.
* `landfrac` has no `1e20` cells, so changing its attribute is harmless either
  way.
* If the regrid step is fixed at the source instead, future reprocessing of
  these or other F2010 cases will also come out right.
* `elm.h0.0000-12` is a ~19 MB spin-up stub. Fix it for consistency, though
  the emulator does not use it.

## Workaround if the files cannot be changed

Treat `|x| >= 1e19` as missing in the reader. For the emulator that means a
loader-side rule in `fme` (see `README.md`, "Masking"). With xarray, use
`ds.where(abs(ds) < 1e19)`.
