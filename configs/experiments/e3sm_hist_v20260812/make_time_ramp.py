#!/usr/bin/env python3
"""Generate the `time_ramp` forcing channel for the F09 clock control.

F09 asks whether `global_mean_co2` is carrying a radiative forcing or merely
acting as a clock. `co2vmr` is stored as a `(time,)` scalar -- no spatial
structure at all -- and is monotone over 1940-2040, so a model handed it can
learn "year -> mean state" and extrapolate along a trend line without
representing any forcing response. On the 2040s block CO2 keeps rising, so
clock-following and forcing-response produce the same signature.

This writes a channel with the same shape and the same monotonicity and no
physics: a linear ramp in time, normalized to [0, 1] across the record. F09
appends it to the trunk's `in_names` exactly where F01 appends `global_mean_co2`,
so the two are a matched pair.

Files mirror the h0 monthly chunking and time axis so the result can be pulled
in with a plain `merge:` alongside the h0 stream, the way the ocean config
already merges its derived fields.

    ./make_time_ramp.py --out-dir $SCRATCH/e3sm-hist-aux/time-ramp
    ./make_time_ramp.py --out-dir ... --check      # verify against the h0 axis

Then in the config, replace the bare `dataset:` with:

    dataset:
      merge:
      - data_path: <h0 run dir>          # the existing block, unchanged
        file_pattern: v3.LR.historical_0101.aigo.eam.h0.*.nc
        rename: {...}
        overwrite: {...}
      - data_path: <out-dir>
        file_pattern: time_ramp.*.nc

and append `time_ramp` to `in_names`.

`time_ramp` also needs entries in the normalization statistics, which
`--emit-stats` writes for you: the ramp is analytic, so its mean and standard
deviation are exact and no stats recompute is required.
"""

import argparse
import pathlib

import cftime
import numpy as np
import xarray as xr

# The h0 stream: 6-hourly, noleap, first sample 1940-01-01T06:00, last 2065-01.
RECORD_FIRST = cftime.DatetimeNoLeap(1940, 1, 1, 6)
RECORD_LAST = cftime.DatetimeNoLeap(2065, 1, 1, 0)
STEP_HOURS = 6
UNITS = "days since 1850-01-01 00:00:00"
CALENDAR = "noleap"
RUN_DIR = "/global/cfs/cdirs/e3smdata/simulations/v3.LR.historical_0101.aigo/run"
H0_PATTERN = "v3.LR.historical_0101.aigo.eam.h0.{year:04d}-{month:02d}.nc"

DAYS_IN_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


def month_times(year, month, first=None, last=None):
    """The 6-hourly stamps h0 carries for one month, in a noleap calendar.

    Every day carries 00/06/12/18, so a full month is days x 4. Only the two
    ends of the record are partial: the run starts at 1940-01-01T06:00, so the
    first file is missing that day's 00:00 stamp, and it stops at
    2065-01-01T00:00, so the last file holds a single sample.
    """
    times = [
        cftime.DatetimeNoLeap(year, month, day, hour)
        for day in range(1, DAYS_IN_MONTH[month - 1] + 1)
        for hour in range(0, 24, STEP_HOURS)
    ]
    if first is not None:
        times = [t for t in times if t >= first]
    if last is not None:
        times = [t for t in times if t <= last]
    return times


def all_months(first_year, first_month, last_year, last_month):
    y, m = first_year, first_month
    while (y, m) <= (last_year, last_month):
        yield y, m
        m += 1
        if m > 12:
            m = 1
            y += 1


def ramp_value(times, t0, t1):
    """Linear in time, 0 at the start of the record and 1 at the end."""
    num = cftime.date2num(times, UNITS, CALENDAR)
    a = cftime.date2num([t0], UNITS, CALENDAR)[0]
    b = cftime.date2num([t1], UNITS, CALENDAR)[0]
    return ((np.asarray(num) - a) / (b - a)).astype("float64")


def check_against_h0(out_dir, samples):
    """Confirm the generated axis matches the real h0 axis on sample months."""
    ok = True
    for year, month in samples:
        h0 = pathlib.Path(RUN_DIR) / H0_PATTERN.format(year=year, month=month)
        if not h0.exists():
            print(f"  {year}-{month:02d}  h0 file missing, skipped")
            continue
        ref = xr.open_dataset(h0, decode_timedelta=False).time.values
        gen_path = pathlib.Path(out_dir) / f"time_ramp.{year:04d}-{month:02d}.nc"
        if not gen_path.exists():
            print(f"  {year}-{month:02d}  FAIL generated file missing")
            ok = False
            continue
        gen = xr.open_dataset(gen_path, decode_timedelta=False).time.values
        same = len(ref) == len(gen) and all(r == g for r, g in zip(ref, gen))
        print(f"  {year}-{month:02d}  n={len(gen):<4} vs h0 n={len(ref):<4} "
              f"{'match' if same else 'FAIL'}")
        ok &= same
    return ok


# A linear ramp sampled on a regular axis is uniform on [0, 1], so both moments
# are exact rather than estimated and no stats recompute is needed.
RAMP_MEAN = 0.5
RAMP_STD = 1.0 / (2.0 * np.sqrt(3.0))


def patch_stats(stats_dir):
    """Add a `time_ramp` entry to the three atmosphere statistics files."""
    targets = {
        "centering.nc": RAMP_MEAN,
        "scaling-full-field.nc": RAMP_STD,
        "scaling-residual.nc": RAMP_STD,
    }
    missing = [n for n in targets if not (stats_dir / n).exists()]
    if missing:
        print(f"not a stats directory: {stats_dir} is missing {', '.join(missing)}")
        return 1
    for name, value in targets.items():
        path = stats_dir / name
        with xr.open_dataset(path) as ds:
            ds = ds.load()
        if "time_ramp" in ds:
            print(f"  {name}: time_ramp already present ({float(ds['time_ramp']):.10f})")
            continue
        ds["time_ramp"] = xr.DataArray(np.float32(value))
        ds.to_netcdf(path)
        print(f"  {name}: added time_ramp = {value:.10f}")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--first", default="1940-01")
    p.add_argument("--last", default="2065-01")
    p.add_argument("--check", action="store_true",
                   help="only verify the axis against real h0 files")
    p.add_argument("--emit-stats", action="store_true",
                   help="print the exact mean/std to add to the stats files")
    p.add_argument("--patch-stats", metavar="STATS_DIR", default=None,
                   help="add a time_ramp entry to centering.nc, scaling-full-field.nc "
                        "and scaling-residual.nc in STATS_DIR (writes in place)")
    args = p.parse_args()

    if args.patch_stats:
        raise SystemExit(patch_stats(pathlib.Path(args.patch_stats)))

    fy, fm = (int(x) for x in args.first.split("-"))
    ly, lm = (int(x) for x in args.last.split("-"))
    months = list(all_months(fy, fm, ly, lm))
    out_dir = pathlib.Path(args.out_dir)

    if args.check:
        samples = [months[0], months[len(months) // 2], months[-1],
                   (1990, 1), (2040, 1)]
        print(f"checking {out_dir} against the h0 axis:")
        raise SystemExit(0 if check_against_h0(out_dir, samples) else 1)

    # The ramp spans the whole record, so both endpoints are needed up front.
    t_first, t_last = RECORD_FIRST, RECORD_LAST

    if args.emit_stats:
        # Uniform over [0, 1] on a regular axis: analytic, no recompute needed.
        allv = np.concatenate([
            ramp_value(month_times(y, m, RECORD_FIRST, RECORD_LAST), t_first, t_last)
            for y, m in months])
        print(f"time_ramp  mean={allv.mean():.10f}  std={allv.std():.10f}  "
              f"n={allv.size}")
        print("Add these to centering.nc and BOTH scaling-full-field.nc and")
        print("scaling-residual.nc as a scalar variable named 'time_ramp'.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    for year, month in months:
        times = month_times(year, month, RECORD_FIRST, RECORD_LAST)
        values = ramp_value(times, t_first, t_last)
        ds = xr.Dataset(
            {"time_ramp": ("time", values)},
            coords={"time": times},
        )
        ds["time_ramp"].attrs = {
            "long_name": "linear ramp in time, 0 at record start and 1 at record end",
            "units": "1",
            "comment": "F09 clock control: matches co2vmr's shape and monotonicity, no physics",
        }
        ds.time.encoding.update(units=UNITS, calendar=CALENDAR)
        path = out_dir / f"time_ramp.{year:04d}-{month:02d}.nc"
        ds.to_netcdf(path)
    print(f"wrote {len(months)} files to {out_dir}")
    print("now run with --check to verify the axis against h0, then --emit-stats")


if __name__ == "__main__":
    main()
