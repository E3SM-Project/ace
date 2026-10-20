#!/usr/bin/env python
"""Normalization statistics for the HR (ne120 -> 0.25 deg) F2010 EAM run.

Same conventions as configs/experiments/e3sm_hist_v20260812/compute_hist_stats.py
(the LR historical campaign), so the two are comparable:

* unweighted over (time, lat, lon), population std (ddof=0);
* residual scaling = std of consecutive-in-time differences, taken within
  each file only (file-boundary pairs are skipped);
* stats are computed on the RAW file names and then renamed/scaled the way
  the loader does (PRECT/PRECST -> *_precipitation_rate, x1000 to kg/m2/s;
  co2vmr -> global_mean_co2).

Streaming and exact: each (file, variable) gives (n, mean, M2) for the field
and for its time difference; partials are merged with Chan's algorithm.

Usage, one task per node inside an allocation:

    srun --overlap -N 4 --ntasks-per-node 1 -c 128 \
        python compute_stats.py partial --out-dir DIR --stride 3
    python compute_stats.py combine --out-dir DIR
"""

import argparse
import glob
import os
import pickle
import re
import time
from multiprocessing import Pool

import numpy as np
import xarray as xr
from cdf5 import CDF5File  # scripts/cdf5.py: one pread per field

RUN_DIR = "/global/cfs/cdirs/e3smdata/simulations/v3.HR.F2010.aigo.PD.y1/run"
PATTERN = "v3.HR.F2010.aigo.PD.y1.eam.h0.*.nc"
RENAME = {
    "PRECT": "surface_precipitation_rate",
    "PRECST": "frozen_precipitation_rate",
    "co2vmr": "global_mean_co2",
}
MULTIPLY = {"PRECT": 1000.0, "PRECST": 1000.0}
FILL = 1e19
BLOCK = 8  # records per read block
SCRATCH_COPY = "/pscratch/sd/m/mahf708/hr-data/v3.HR.F2010.aigo.PD.y1/eam.h0"
# The ACE2S channel set plus the diagnostics an evaluation is likely to want.
# The pressure-level / aerosol-microphysics fields are skipped: reading every
# variable off CFS ran at ~0.5 GB/s while the scratch copy was also reading it.
VARIABLES = (
    ["LANDFRAC", "OCNFRAC", "ICEFRAC", "PHIS", "SOLIN", "PS", "TS"]
    + [f"{v}_{k}" for v in ("T", "STW", "U", "V", "Q") for k in range(8)]
    + ["Qat2m", "Uat10m", "Vat10m", "Tat2m", "STWat2m", "TREFHT", "QREFHT"]
    + ["LHFLX", "SHFLX", "PRECT", "PRECST", "FLUS", "FLUT", "FLDS", "FSNS"]
    + ["FSUTOA", "FSDS", "FSUS", "FSNTOA", "FLNT", "FLNS", "DTENDTTW"]
    + ["TAUX", "TAUY", "AODVISall", "co2vmr", "lwp"]
)


def file_month(path: str) -> tuple[int, int]:
    m = re.search(r"\.(\d{4})-(\d{2})\.nc$", path)
    assert m is not None, path
    return int(m.group(1)), int(m.group(2))


def select_files(args) -> list[str]:
    files = sorted(glob.glob(os.path.join(args.data_dir, PATTERN)))
    out = []
    for f in files:
        y, mo = file_month(f)
        key = y * 100 + mo
        if args.start <= key < args.stop:
            out.append(f)
    return out[:: args.stride]


def _moments(x: np.ndarray) -> tuple[int, float, float]:
    # Mask only when needed: the boolean-index copy and float64 upcast of
    # every block made this CPU-bound (~8 s per 0.5 GB variable).
    lo, hi = float(np.min(x)), float(np.max(x))
    if not (np.isfinite(lo) and np.isfinite(hi) and max(-lo, hi) < FILL):
        x = x[np.isfinite(x) & (np.abs(x) < FILL)]
    n = x.size
    if n == 0:
        return 0, 0.0, 0.0
    mean = float(x.sum(dtype=np.float64)) / n
    d = x - np.float32(mean)
    m2 = float(np.dot(d.ravel().astype(np.float64, copy=False), d.ravel()))
    return n, mean, m2


def merge(a, b):
    na, ma, m2a = a
    nb, mb, m2b = b
    n = na + nb
    if n == 0:
        return 0, 0.0, 0.0
    d = mb - ma
    return n, ma + d * nb / n, m2a + m2b + d * d * na * nb / n


def _fastest_copy(path: str) -> str:
    """Prefer the complete scratch copy: CFS served ~30 MB/s per reader here
    (per-record strided netCDF reads over DVS)."""
    alt = os.path.join(SCRATCH_COPY, os.path.basename(path))
    if os.path.exists(alt) and os.path.getsize(alt) == os.path.getsize(path):
        return alt
    return path


def one_file(path: str) -> dict:
    t0 = time.time()
    out: dict = {}
    path = _fastest_copy(path)
    with CDF5File(path) as ds:
        names = [
            k
            for k, v in ds.vars.items()
            if v["dims"] in (("time", "lat", "lon"), ("time",))
            and v["dtype"].kind == "f"
            and k in VARIABLES
        ]
        # netCDF-C read these files in 8 KiB read() calls (~20-70 MB/s per
        # reader); scripts/cdf5.py does one pread per field (~0.7-0.9 GB/s).
        nt = ds.numrecs
        last: dict = {}
        for k in names:
            out[k] = [(0, 0.0, 0.0), (0, 0.0, 0.0)]
        for b0 in range(0, nt, BLOCK):
            b1 = min(nt, b0 + BLOCK)
            for k in names:
                data = ds.read_records(k, b0, b1).astype(np.float32, copy=False)
                out[k][0] = merge(out[k][0], _moments(data))
                if k in last:  # pair across the block boundary
                    data_d = np.concatenate([last[k], data], axis=0)
                else:
                    data_d = data
                if data_d.shape[0] > 1:
                    out[k][1] = merge(out[k][1], _moments(np.diff(data_d, axis=0)))
                last[k] = data[-1:]
        out = {k: tuple(v) for k, v in out.items()}
    print(f"{path} {time.time() - t0:.0f}s", flush=True)
    return out


def partial(args):
    files = select_files(args)
    rank = int(os.environ.get("SLURM_PROCID", 0))
    ntask = int(os.environ.get("SLURM_NTASKS", 1))
    mine = files[rank::ntask]
    print(f"rank {rank}: {len(mine)} of {len(files)} files", flush=True)
    acc: dict = {}
    with Pool(args.procs) as pool:
        for res in pool.imap_unordered(one_file, mine):
            for k, (full, diff) in res.items():
                if k not in acc:
                    acc[k] = [(0, 0.0, 0.0), (0, 0.0, 0.0)]
                acc[k][0] = merge(acc[k][0], full)
                acc[k][1] = merge(acc[k][1], diff)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, f"partial-{rank}.pkl"), "wb") as f:
        pickle.dump({"files": mine, "acc": acc}, f)


def combine(args):
    acc: dict = {}
    files: list = []
    for p in sorted(glob.glob(os.path.join(args.out_dir, "partial-*.pkl"))):
        with open(p, "rb") as f:
            d = pickle.load(f)
        files += d["files"]
        for k, (full, diff) in d["acc"].items():
            if k not in acc:
                acc[k] = [(0, 0.0, 0.0), (0, 0.0, 0.0)]
            acc[k][0] = merge(acc[k][0], full)
            acc[k][1] = merge(acc[k][1], diff)
    centering, full_std, resid_std = {}, {}, {}
    for k, (full, diff) in acc.items():
        name = RENAME.get(k, k)
        s = MULTIPLY.get(k, 1.0)
        n, m, m2 = full
        centering[name] = m * s
        full_std[name] = np.sqrt(m2 / n) * s if n else np.nan
        n, _, m2 = diff
        resid_std[name] = np.sqrt(m2 / n) * s if n else np.nan
    attrs = {
        "source": RUN_DIR,
        "files": ",".join(os.path.basename(f) for f in sorted(files)),
        "n_files": len(files),
        "script": "configs/experiments/2026-10-19-e3sm-hr-ace2s/scripts/compute_stats.py",
        "conventions": "unweighted over (time,lat,lon); ddof=0; residual = "
        "within-file consecutive differences",
    }
    for fname, d in [
        ("centering.nc", centering),
        ("scaling-full-field.nc", full_std),
        ("scaling-residual.nc", resid_std),
    ]:
        ds = xr.Dataset({k: ((), np.float64(v)) for k, v in sorted(d.items())})
        ds.attrs.update(attrs)
        ds.to_netcdf(os.path.join(args.out_dir, fname))
    print(f"wrote {len(centering)} variables from {len(files)} files")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["partial", "combine"])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--data-dir", default=RUN_DIR)
    p.add_argument("--start", type=int, default=102, help="YYYYMM, inclusive")
    p.add_argument("--stop", type=int, default=1401, help="YYYYMM, exclusive")
    p.add_argument("--stride", type=int, default=1, help="use every Nth month")
    p.add_argument("--procs", type=int, default=8)
    a = p.parse_args()
    partial(a) if a.mode == "partial" else combine(a)


if __name__ == "__main__":
    main()
