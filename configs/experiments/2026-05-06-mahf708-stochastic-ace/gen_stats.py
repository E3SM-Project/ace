"""Generate ACE normalization stats for the E3SM v3.LR.piControl.aigo dataset.

Writes four netCDFs to ``OUT_DIR``:

* ``centering.nc`` - scalar mean of each variable over (time, lat, lon).
* ``scaling-full-field.nc`` - scalar std over (time, lat, lon).
* ``scaling-residual.nc`` - scalar std of the t->t+1 difference over the
  same dims (used for residual prediction normalization).
* ``time-mean.nc`` - (lat, lon) mean over time (the inference aggregator's
  reference field).

Stats are computed over the training subset only (TRAIN_START..TRAIN_STOP)
so they reflect what the model sees during fit, not the validation/inference
periods. The same E3SM-source -> FME-internal rename used by the YAML is
applied here, so the produced files key off FME-internal names.

Implementation: streaming per-file Welford accumulation across a process
pool. Each worker reduces one file to per-variable sufficient statistics,
the main process merges partials in time order (so cross-file diff
bridging is well-defined for scaling-residual). This keeps memory bounded
(~one file per worker) and parallelizes the I/O-bound read, which is the
actual bottleneck on Lustre. A prior version called ``.load()`` on the
full subset; that doesn't scale past a year and the dask Client setup it
fell back to silently fails in the xgns env (``tblib`` is an empty
namespace package there).

Run on a Perlmutter compute node (still touches ~200 GB of disk):

    salloc -N 1 -C cpu -t 02:00:00 -q interactive
    micromamba run -n xgns python gen_stats.py --n-workers 32
"""

from __future__ import annotations

import argparse
import glob
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import xarray as xr

DATA_DIR = "/pscratch/sd/m/mahf708/E3SMv3/v3.LR.piControl.aigo/run"
FILE_PATTERN = "v3.LR.piControl.aigo.eam.h0.*.nc"
OUT_DIR = "/pscratch/sd/m/mahf708/E3SMv3/v3.LR.piControl.aigo/normalization"
TRAIN_START = "0401-01-01"
TRAIN_STOP = "0418-12-31"

# Must mirror XarrayDataConfig.rename in pmgpu_picontrol.yaml, but inverted:
# E3SM-on-disk name -> FME-internal name.
RENAME_SRC_TO_FME = {
    "STW_0": "specific_total_water_0",
    "STW_1": "specific_total_water_1",
    "STW_2": "specific_total_water_2",
    "STW_3": "specific_total_water_3",
    "STW_4": "specific_total_water_4",
    "STW_5": "specific_total_water_5",
    "STW_6": "specific_total_water_6",
    "STW_7": "specific_total_water_7",
    "PRECT": "surface_precipitation_rate",
    "FLUS": "surface_upward_longwave_flux",
    "FSUS": "surface_upward_shortwave_flux",
    "FSUTOA": "top_of_atmos_upward_shortwave_flux",
    "DTENDTTW": "tendency_of_total_water_path_due_to_advection",
}

# Variables we need stats for, by their FME-internal names. Must cover
# in_names U out_names from the YAML stepper config.
FME_NAMES: list[str] = [
    "LANDFRAC", "OCNFRAC", "ICEFRAC", "PHIS", "SOLIN", "PS", "TS",
    *[f"T_{i}" for i in range(8)],
    *[f"specific_total_water_{i}" for i in range(8)],
    *[f"U_{i}" for i in range(8)],
    *[f"V_{i}" for i in range(8)],
    "LHFLX", "SHFLX",
    "surface_precipitation_rate",
    "surface_upward_longwave_flux", "FLUT", "FLDS", "FSDS",
    "surface_upward_shortwave_flux", "top_of_atmos_upward_shortwave_flux",
    "tendency_of_total_water_path_due_to_advection",
]


def _welford_scalar(arr: np.ndarray) -> tuple[int, float, float]:
    """Return (count, mean, M2) reducing over every dim of ``arr``."""
    a = arr.astype(np.float64, copy=False)
    n = a.size
    mean = float(a.mean())
    delta = a - mean
    M2 = float((delta * delta).sum())
    return n, mean, M2


def _welford_merge(
    n_a: int, m_a: float, M2_a: float,
    n_b: int, m_b: float, M2_b: float,
) -> tuple[int, float, float]:
    if n_a == 0:
        return n_b, m_b, M2_b
    if n_b == 0:
        return n_a, m_a, M2_a
    n = n_a + n_b
    delta = m_b - m_a
    mean = m_a + delta * n_b / n
    M2 = M2_a + M2_b + delta * delta * n_a * n_b / n
    return n, mean, M2


def _process_file(args: tuple[str, str, str]) -> dict | None:
    """Worker: reduce one file to per-variable partial stats.

    Returns a dict with the FME variable name -> sub-dict of partials,
    plus a top-level ``nt`` and ``time0`` for sanity logging.
    """
    path, t_lo, t_hi = args
    with xr.open_dataset(path) as ds:
        ren = {k: v for k, v in RENAME_SRC_TO_FME.items() if k in ds.data_vars}
        if ren:
            ds = ds.rename(ren)
        missing = [n for n in FME_NAMES if n not in ds.data_vars]
        if missing:
            print(f"WARN {os.path.basename(path)}: missing {missing}", flush=True)
            return None
        ds = ds[FME_NAMES].sel(time=slice(t_lo, t_hi))
        nt = ds.sizes.get("time", 0)
        if nt == 0:
            return None
        time0 = ds.time.values[0]
        data = {v: ds[v].values for v in FME_NAMES}

    out: dict = {"nt": int(nt), "time0": time0, "vars": {}}
    for v, arr in data.items():
        c, m, M2 = _welford_scalar(arr)
        a64 = arr.astype(np.float64, copy=False)
        ts = a64.sum(axis=0)
        if arr.shape[0] >= 2:
            diff = a64[1:] - a64[:-1]
            dc, dm, dM2 = _welford_scalar(diff)
        else:
            dc, dm, dM2 = 0, 0.0, 0.0
        out["vars"][v] = {
            "count": c, "mean": m, "M2": M2,
            "time_sum": ts, "time_count": int(arr.shape[0]),
            "diff_count": dc, "diff_mean": dm, "diff_M2": dM2,
            "first_step": a64[0].copy(), "last_step": a64[-1].copy(),
        }
    return out


def _write_scalar(stats: dict[str, float], path: str) -> None:
    ds = xr.Dataset({v: xr.DataArray(np.float64(stats[v])) for v in FME_NAMES})
    ds.to_netcdf(path)


def _write_time_mean(
    stats: dict[str, np.ndarray],
    lat: np.ndarray,
    lon: np.ndarray,
    path: str,
) -> None:
    ds = xr.Dataset(
        {v: xr.DataArray(stats[v].astype(np.float32), dims=("lat", "lon")) for v in FME_NAMES},
        coords={"lat": lat, "lon": lon},
    )
    ds.to_netcdf(path)


def main(
    file_pattern: str = FILE_PATTERN,
    start: str = TRAIN_START,
    stop: str = TRAIN_STOP,
    out_dir: str = OUT_DIR,
    n_workers: int = 8,
) -> None:
    files = sorted(glob.glob(os.path.join(DATA_DIR, file_pattern)))
    if not files:
        raise SystemExit(f"no files at {DATA_DIR}/{file_pattern}")
    os.makedirs(out_dir, exist_ok=True)

    # E3SM eam.h0 filenames are ``...h0.YYYY-MM.nc``; lexicographic sort
    # matches time order, so ex.map results arrive in time order and the
    # cross-file diff bridge is well-defined.
    print(f"dispatching {len(files)} files across {n_workers} workers", flush=True)
    args = [(f, start, stop) for f in files]

    # Running accumulators per variable.
    acc: dict[str, dict] = {
        v: {
            "n": 0, "m": 0.0, "M2": 0.0,
            "rn": 0, "rm": 0.0, "rM2": 0.0,
            "ts": None, "tc": 0,
            "prev_last": None,
        }
        for v in FME_NAMES
    }
    total_nt = 0
    used_files = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for i, fs in enumerate(ex.map(_process_file, args)):
            if fs is None:
                continue
            used_files += 1
            total_nt += fs["nt"]
            for v in FME_NAMES:
                p = fs["vars"][v]
                a = acc[v]
                a["n"], a["m"], a["M2"] = _welford_merge(
                    a["n"], a["m"], a["M2"],
                    p["count"], p["mean"], p["M2"],
                )
                a["rn"], a["rm"], a["rM2"] = _welford_merge(
                    a["rn"], a["rm"], a["rM2"],
                    p["diff_count"], p["diff_mean"], p["diff_M2"],
                )
                a["ts"] = p["time_sum"].copy() if a["ts"] is None else a["ts"] + p["time_sum"]
                a["tc"] += p["time_count"]
                if a["prev_last"] is not None:
                    bridge = p["first_step"] - a["prev_last"]
                    bn, bm, bM2 = _welford_scalar(bridge)
                    a["rn"], a["rm"], a["rM2"] = _welford_merge(
                        a["rn"], a["rm"], a["rM2"], bn, bm, bM2,
                    )
                a["prev_last"] = p["last_step"]
            if (i + 1) % max(1, len(files) // 20) == 0 or i + 1 == len(files):
                print(
                    f"  {i+1}/{len(files)} files, {used_files} kept, "
                    f"{total_nt} ts merged, {time.time()-t0:.1f}s",
                    flush=True,
                )

    if used_files == 0:
        raise SystemExit("no timesteps in [start, stop]")

    centering = {v: acc[v]["m"] for v in FME_NAMES}
    full_field = {
        v: float(np.sqrt(acc[v]["M2"] / acc[v]["n"])) if acc[v]["n"] else 0.0
        for v in FME_NAMES
    }
    residual = {
        v: float(np.sqrt(acc[v]["rM2"] / acc[v]["rn"])) if acc[v]["rn"] else 0.0
        for v in FME_NAMES
    }
    time_mean = {v: acc[v]["ts"] / acc[v]["tc"] for v in FME_NAMES}

    with xr.open_dataset(files[0]) as ds0:
        lat = ds0.lat.values
        lon = ds0.lon.values

    writers = [
        ("centering", "centering.nc",
         lambda p: _write_scalar(centering, p)),
        ("scaling-full-field", "scaling-full-field.nc",
         lambda p: _write_scalar(full_field, p)),
        ("scaling-residual", "scaling-residual.nc",
         lambda p: _write_scalar(residual, p)),
        ("time-mean", "time-mean.nc",
         lambda p: _write_time_mean(time_mean, lat, lon, p)),
    ]
    for label, fname, write in writers:
        path = os.path.join(out_dir, fname)
        if os.path.exists(path):
            print(f"{label}: {path} exists, skipping", flush=True)
            continue
        t = time.time()
        write(path)
        print(f"{label}: wrote {path} in {time.time()-t:.1f}s", flush=True)

    print(f"total: {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--pattern", default=FILE_PATTERN, help="glob under DATA_DIR")
    p.add_argument("--start", default=TRAIN_START, help="inclusive start date")
    p.add_argument("--stop", default=TRAIN_STOP, help="inclusive stop date")
    p.add_argument("--out", default=OUT_DIR, help="output directory for the four .nc files")
    p.add_argument("--n-workers", type=int, default=8)
    a = p.parse_args()
    main(
        file_pattern=a.pattern,
        start=a.start,
        stop=a.stop,
        out_dir=a.out,
        n_workers=a.n_workers,
    )
