#!/usr/bin/env python3
"""Audit the E3SMv3 F2010 land data and build the land dashboard's data files.

Reads the regridded ELM history (``elm.h0``) of both F2010 cases, plus the
coupler-to-land stream (``cpl.fme.x2l``) for the forcing comparison, and
writes a static dashboard (``index.html`` + ``data/*.js``) that opens from
``file://``. Modelled on /pscratch/sd/i/imanick/sep26v3-inference/dashboard.
The code lives in the repo; the data and work files live under ``--root``.

Stages (each resumable; finished work files are skipped unless --force):

  scan   one worker per (case, month). Reads every lat/lon variable of the
         elm.h0 file once and records: the time axis; per variable, the
         1e20 sentinel coverage (cells that are missing in every / some
         records, inside and outside land), value statistics over valid
         points (min, max, moments, NaN/inf/negative/"huge" counts), the
         land-area-weighted mean and the monthly-mean map. The matching
         x2l file is read for the forcing pairs in FORCING_PAIRS and
         compared under two alignments (see ``_compare_forcing``).
  build  aggregates work/scan into data/:
           meta.js            sources, variables (+ roles/groups/ranges), months
           audit.js           every audit table and the generated findings
           ts/<source>.js     monthly land-mean series, one file per source
           maps/<var>/<source>.js
                              monthly climatology (12 frames, all years),
                              uint16-quantized, gzip + base64
           masks.js           landfrac and the in-land gap count, same encoding,
                              plus a gap bitmask per variable that has gaps
         and copies index.html and lib/ next to data/. Files in data/ that
         build did not write (e.g. inference sources added later) are left
         alone: data/.build-manifest.json lists what the last build wrote,
         and only those files are replaced or, if no longer produced, removed.

Missing data: a value is missing when |x| >= 1e19 (the stored sentinel is
1e20 regardless of the declared _FillValue, see ../ISSUE-elm-fillvalue.md) or
not finite. Everything here works the same before and after the files' fill
value attributes are fixed.

Usage (on a CPU node; see run_prep.sh):
  python prep.py scan --workers 64
  python prep.py scan --force [--months 0001-01 ...]   # redo finished months
  python prep.py build
"""

import argparse
import base64
import concurrent.futures as cf
import gzip
import json
import os
import pathlib
import shutil
import time

import netCDF4
import numpy as np

DATA_ROOT = pathlib.Path("/global/cfs/cdirs/e3smdata/simulations")
CASES = {
    "PD": "v3.LR.F2010.aigo.PD.compy",
    "PI": "v3.LR.F2010.aigo.PI.compy",
}
CASE_LABELS = {
    "PD": "PD (present-day forcing)",
    "PI": "PI (pre-industrial forcing)",
}
YEARS = range(1, 31)
MONTHS = [f"{y:04d}-{m:02d}" for y in YEARS for m in range(1, 13)]
STUB_MONTH = "0000-12"
DT_DAYS = 0.25  # 6-hourly
SENTINEL = 1e19
HUGE = 1e10  # finite, below the sentinel, but implausible for any field here
HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_ROOT = pathlib.Path("/pscratch/sd/i/imanick/land-inference/dashboard")
DEFAULT_LIB = pathlib.Path("/pscratch/sd/i/imanick/sep26v3-inference/dashboard/lib")

# Proposed role of each elm.h0 variable in the emulator (README.md, step 2).
# Not a decision: it only drives grouping and the default selections.
ROLES = {
    "forcing": [
        "TBOT",
        "THBOT",
        "QBOT",
        "PBOT",
        "ZBOT",
        "WIND",
        "RH",
        "FLDS",
        "FSDS",
        "FSDSND",
        "FSDSNI",
        "FSDSVD",
        "FSDSVI",
        "RAIN",
        "SNOW",
        "BCDEP",
        "DSTDEP",
        "OCDEP",
    ],
    "state": [
        "TSOI_0",
        "TSOI_1",
        "TSOI_2",
        "TSOI_10CM",
        "H2OSOI_0",
        "H2OSOI_1",
        "H2OSOI_2",
        "SOILWATER_10CM",
        "H2OSNO",
        "SNOWDP",
        "SNOWICE",
        "SNOWLIQ",
        "FSNO",
        "FH2OSFC",
        "TG",
        "TWS",
        "ZWT",
    ],
    "vegetation": ["ELAI", "TLAI", "TSAI"],
    "static": ["landfrac"],
}  # everything else: "diagnostic"
GROUPS = [
    ("Atmospheric forcing", ROLES["forcing"]),
    (
        "Soil",
        [
            "TSOI_0",
            "TSOI_1",
            "TSOI_2",
            "TSOI_10CM",
            "H2OSOI_0",
            "H2OSOI_1",
            "H2OSOI_2",
            "SOILWATER_10CM",
            "TWS",
            "ZWT",
        ],
    ),
    ("Snow", ["H2OSNO", "SNOWDP", "SNOWICE", "SNOWLIQ", "FSNO"]),
    ("Near-surface state", ["TG", "TSA", "Q2M", "RH2M", "FH2OSFC", "U10"]),
    (
        "Energy & momentum fluxes",
        [
            "FSH",
            "EFLX_LH_TOT",
            "FCEV",
            "FCTR",
            "FGR",
            "FIRA",
            "FIRE",
            "FSA",
            "ALBD_nir",
            "ALBD_vis",
            "ALBI_nir",
            "ALBI_vis",
            "TAUX",
            "TAUY",
        ],
    ),
    (
        "Water fluxes",
        [
            "QRUNOFF",
            "QOVER",
            "QDRAI",
            "QDRAI_PERCH",
            "QINFL",
            "QSNOMELT",
            "QSOIL",
            "QVEGE",
            "QVEGT",
            "QINTR",
            "QCHARGE",
            "QH2OSFC",
            "QRGWL",
            "QSNWCPICE",
        ],
    ),
    ("Vegetation & carbon", ["ELAI", "TLAI", "TSAI", "NEE", "NEM"]),
]
# Physically non-negative fields; a negative value is reported, not masked.
NONNEGATIVE = {
    "ALBD_nir",
    "ALBD_vis",
    "ALBI_nir",
    "ALBI_vis",
    "BCDEP",
    "DSTDEP",
    "OCDEP",
    "ELAI",
    "TLAI",
    "TSAI",
    "FH2OSFC",
    "FSNO",
    "FLDS",
    "FSA",
    "FSDS",
    "FSDSND",
    "FSDSNI",
    "FSDSVD",
    "FSDSVI",
    "FIRE",
    "H2OSNO",
    "H2OSOI_0",
    "H2OSOI_1",
    "H2OSOI_2",
    "PBOT",
    "Q2M",
    "QBOT",
    "RAIN",
    "SNOW",
    "RH",
    "RH2M",
    "SNOWDP",
    "SNOWICE",
    "SNOWLIQ",
    "SOILWATER_10CM",
    "TBOT",
    "TG",
    "THBOT",
    "TSA",
    "TSOI_0",
    "TSOI_1",
    "TSOI_2",
    "TSOI_10CM",
    "U10",
    "WIND",
    "ZBOT",
    "ZWT",
}
# elm.h0 forcing -> the x2l field(s) it should equal; "sum" adds them,
# "hypot" takes the vector magnitude.
FORCING_PAIRS = {
    "TBOT": ("sum", ["Sa_tbot"]),
    "THBOT": ("sum", ["Sa_ptem"]),
    "QBOT": ("sum", ["Sa_shum"]),
    "PBOT": ("sum", ["Sa_pbot"]),
    "ZBOT": ("sum", ["Sa_z"]),
    "WIND": ("hypot", ["Sa_u", "Sa_v"]),
    "FLDS": ("sum", ["Faxa_lwdn"]),
    "FSDS": ("sum", ["Faxa_swvdr", "Faxa_swvdf", "Faxa_swndr", "Faxa_swndf"]),
    "FSDSVD": ("sum", ["Faxa_swvdr"]),
    "FSDSVI": ("sum", ["Faxa_swvdf"]),
    "FSDSND": ("sum", ["Faxa_swndr"]),
    "FSDSNI": ("sum", ["Faxa_swndf"]),
    "RAIN": ("sum", ["Faxa_rainc", "Faxa_rainl"]),
    "SNOW": ("sum", ["Faxa_snowc", "Faxa_snowl"]),
}
# elm.h0 stamps a 6-hour record at the END of its interval, the coupler
# streams at the START; "end" assumes that, "same" assumes equal stamps.
ALIGNMENTS = {"end": DT_DAYS, "same": 0.0}


def _path(case: str, stream: str, month: str) -> pathlib.Path:
    c = CASES[case]
    return DATA_ROOT / c / "run" / f"{c}.{stream}.{month}.nc"


def _role(name: str) -> str:
    for role, names in ROLES.items():
        if name in names:
            return role
    return "diagnostic"


def _missing(a: np.ndarray) -> np.ndarray:
    return ~np.isfinite(a) | (np.abs(a) >= SENTINEL)


def _area_weights(lat: np.ndarray) -> np.ndarray:
    """Relative cell area on the (irregular-in-lat, 1 deg-in-lon) grid."""
    edges = np.concatenate([[-90.0], (lat[1:] + lat[:-1]) / 2, [90.0]])
    w = np.sin(np.deg2rad(edges[1:])) - np.sin(np.deg2rad(edges[:-1]))
    return w[:, None] * np.ones((1, 360))


def _work(root: pathlib.Path) -> pathlib.Path:
    return root / "work" / "scan"


# ---------------------------------------------------------------- scan
def _compare_forcing(elm, elm_t, x2l, land) -> dict:
    """Accumulate elm.h0-minus-x2l statistics over land for each alignment."""
    x2l_t = x2l["time"][:]
    x_index = {int(round(t / DT_DAYS)): i for i, t in enumerate(x2l_t)}
    out: dict[str, dict] = {}
    for name, (how, sources) in FORCING_PAIRS.items():
        e = elm[name][:].astype(np.float64)
        parts = [x2l[s][:].astype(np.float64) for s in sources]
        x_missing = np.zeros(parts[0].shape, dtype=bool)
        for p in parts:
            x_missing |= _missing(p)
        if how == "hypot":
            x = np.hypot(*parts)
        else:
            x = np.sum(parts, axis=0)
        out[name] = {}
        for align, shift in ALIGNMENTS.items():
            pairs = [
                (i, x_index[k])
                for i, t in enumerate(elm_t)
                if (k := int(round((t - shift) / DT_DAYS))) in x_index
            ]
            acc = dict(
                n=0, sd=0.0, sd2=0.0, se=0.0, se2=0.0, maxabs=0.0, n_steps=len(pairs)
            )
            for i, j in pairs:
                ok = land & ~_missing(e[i]) & ~x_missing[j]
                d = e[i][ok] - x[j][ok]
                ev = e[i][ok]
                acc["n"] += int(ok.sum())
                acc["sd"] += float(d.sum())
                acc["sd2"] += float((d * d).sum())
                acc["se"] += float(ev.sum())
                acc["se2"] += float((ev * ev).sum())
                if d.size:
                    acc["maxabs"] = max(acc["maxabs"], float(np.abs(d).max()))
            out[name][align] = acc
    return out


def _scan_one(job) -> str:
    root, case, month, force = job
    out_json = _work(root) / case / f"{month}.json"
    if out_json.exists() and not force:
        return f"{case} {month} skip"
    t0 = time.time()
    path = _path(case, "elm.h0", month)
    rec = {"case": case, "month": month, "path": str(path), "vars": {}}
    maps = {}
    with netCDF4.Dataset(path) as ds:
        ds.set_auto_mask(False)
        rec["format"] = ds.data_model
        t = ds["time"][:].astype(np.float64)
        tb = ds["time_bounds"][:].astype(np.float64)
        rec["time"] = t.tolist()
        rec["time_bounds_ok"] = bool(
            np.allclose(tb[:, 1], t) and np.allclose(tb[:, 1] - tb[:, 0], DT_DAYS)
        )
        lat = ds["lat"][:].astype(np.float64)
        landfrac = ds["landfrac"][:].astype(np.float64)
        land = landfrac > 0
        w_land = _area_weights(lat) * np.where(land, landfrac, 0.0)
        rec["landfrac_sum"] = float(landfrac[land].sum())
        for name, var in ds.variables.items():
            if var.dimensions[-2:] != ("lat", "lon") or name == "landfrac":
                continue
            a = var[:].astype(np.float32)
            sent = np.abs(a) >= SENTINEL
            nonfinite = ~np.isfinite(a)
            valid = ~sent & ~nonfinite
            v = a[valid].astype(np.float64)
            any_s, all_s = sent.any(0), sent.all(0)
            n_cell = valid.sum(0)
            mean_map = np.where(
                n_cell > 0,
                np.where(valid, a, 0).sum(0, dtype=np.float64) / np.maximum(n_cell, 1),
                np.nan,
            )
            ok = np.isfinite(mean_map) & land
            fill = getattr(var, "_FillValue", None)
            rec["vars"][name] = {
                "fill_value": None if fill is None else float(fill),
                "n_eq_fill": 0 if fill is None else int((a == np.float32(fill)).sum()),
                "n_records": int(a.shape[0]),
                "n_valid": int(valid.sum()),
                "n_sentinel": int(sent.sum()),
                "n_nonfinite": int(nonfinite.sum()),
                "n_huge": int((np.abs(v) >= HUGE).sum()),
                "n_negative": int((v < 0).sum()),
                "min": float(v.min()) if v.size else None,
                "max": float(v.max()) if v.size else None,
                "sum": float(v.sum()),
                "sum2": float((v * v).sum()),
                # sentinel coverage, as cell counts on the 180x360 grid
                "ocean_sentinel_all": int((all_s & ~land).sum()),
                "ocean_valid_any": int(((n_cell > 0) & ~land).sum()),
                "land_sentinel_all": int((all_s & land).sum()),
                "land_sentinel_some": int((any_s & ~all_s & land).sum()),
                "land_mean": float((mean_map[ok] * w_land[ok]).sum() / w_land[ok].sum())
                if ok.any()
                else None,
                "gap_bits": base64.b64encode(np.packbits(any_s & land)).decode(),
            }
            maps[name] = mean_map.astype(np.float32)
        try:
            with netCDF4.Dataset(_path(case, "cpl.fme.x2l", month)) as x2l:
                x2l.set_auto_mask(False)
                rec["x2l_time"] = x2l["time"][:].astype(np.float64).tolist()
                rec["forcing"] = _compare_forcing(ds, t, x2l, land)
        except FileNotFoundError:
            rec["forcing"] = None
    (_work(root) / case).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(_work(root) / case / f"{month}.npz", **maps)
    tmp = out_json.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec))
    os.replace(tmp, out_json)  # the json marks the month as done
    return f"{case} {month} {time.time() - t0:.0f}s"


def _stub_info(case: str) -> dict:
    path = _path(case, "elm.h0", STUB_MONTH)
    if not path.exists():
        return {"present": False}
    with netCDF4.Dataset(path) as ds:
        ds.set_auto_mask(False)
        return {
            "present": True,
            "path": str(path),
            "bytes": path.stat().st_size,
            "n_records": len(ds.dimensions["time"]),
            "time": ds["time"][:].astype(float).tolist(),
        }


def scan(root: pathlib.Path, workers: int, force: bool = False) -> None:
    jobs = [(root, c, m, force) for c in CASES for m in MONTHS]
    todo = [
        j for j in jobs if force or not (_work(root) / j[1] / f"{j[2]}.json").exists()
    ]
    print(
        f"scan: {len(jobs) - len(todo)} done, {len(todo)} to do"
        f"{' (--force)' if force else ''}, {workers} workers",
        flush=True,
    )
    with cf.ProcessPoolExecutor(workers) as pool:
        for msg in pool.map(_scan_one, todo, chunksize=1):
            print(msg, flush=True)


# ---------------------------------------------------------------- build
def _js(path: pathlib.Path, kind: str, key, obj, written: set) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(obj, separators=(",", ":"), allow_nan=False)
    path.write_text(f"DASH.loaded({json.dumps(kind)},{json.dumps(key)},{body});\n")
    written.add(path)


MANIFEST = ".build-manifest.json"


def _prune(out: pathlib.Path, written: set) -> None:
    """Remove files the previous build wrote that this one did not; record this
    build's files. Files build never wrote are not touched.
    """
    manifest = out / MANIFEST
    old = set(json.loads(manifest.read_text())) if manifest.exists() else set()
    new = {str(p.relative_to(out)) for p in written}
    for rel in sorted(old - new):
        f = out / rel
        f.unlink(missing_ok=True)
        for d in f.parents:  # drop directories this left empty, up to data/
            if d == out or any(d.iterdir()):
                break
            d.rmdir()
    manifest.write_text(json.dumps(sorted(new), indent=0) + "\n")
    if old - new:
        print(f"build: removed {len(old - new)} stale file(s) from the last build")


def _sig(x, n=6):
    if x is None or not np.isfinite(x):
        return None
    return float(f"{x:.{n}g}")


def _encode(frames: np.ndarray, lo: float, hi: float) -> dict:
    """uint16 quantization over [lo, hi]; 0 marks missing."""
    span = hi - lo if hi > lo else 1.0
    q = np.clip(np.round((frames - lo) / span * 65534) + 1, 1, 65535)
    q = np.where(np.isfinite(frames), q, 0).astype("<u2")
    data = base64.b64encode(gzip.compress(q.tobytes(), compresslevel=6)).decode()
    return {"lo": lo, "hi": hi, "n": int(frames.shape[0]), "data": data}


def _load_case(root: pathlib.Path, case: str):
    recs = []
    for m in MONTHS:
        f = _work(root) / case / f"{m}.json"
        recs.append(json.loads(f.read_text()) if f.exists() else None)
    return recs


def _time_audit(recs: list, case: str) -> dict:
    present = [r for r in recs if r]
    t = np.concatenate([r["time"] for r in present]) if present else np.array([])
    d = np.diff(t)
    days_per_year = 365.0
    return {
        "case": case,
        "files_scanned": len(present),
        "files_expected": len(MONTHS),
        "missing_months": [m for m, r in zip(MONTHS, recs) if not r],
        "n_records": int(t.size),
        "n_expected": int(len(YEARS) * days_per_year / DT_DAYS),
        "first": float(t[0]) if t.size else None,
        "last": float(t[-1]) if t.size else None,
        "n_gaps": int((d > DT_DAYS * 1.001).sum()),
        "n_duplicates_or_backwards": int((d < DT_DAYS * 0.999).sum()),
        "time_bounds_ok": all(r["time_bounds_ok"] for r in present),
        "formats": sorted({r["format"] for r in present}),
        "stub": _stub_info(case),
    }


def _forcing_audit(by_case: dict) -> list:
    rows = []
    for name in FORCING_PAIRS:
        for case, recs in by_case.items():
            for align in ALIGNMENTS:
                acc = dict(n=0, sd=0.0, sd2=0.0, se=0.0, se2=0.0, maxabs=0.0, n_steps=0)
                for r in recs:
                    if not r or not r.get("forcing"):
                        continue
                    a = r["forcing"][name][align]
                    for k in ("n", "sd", "sd2", "se", "se2", "n_steps"):
                        acc[k] += a[k]
                    acc["maxabs"] = max(acc["maxabs"], a["maxabs"])
                if not acc["n"]:
                    continue
                n = acc["n"]
                std_e = np.sqrt(max(acc["se2"] / n - (acc["se"] / n) ** 2, 0.0))
                rms = np.sqrt(acc["sd2"] / n)
                rows.append(
                    {
                        "var": name,
                        "case": case,
                        "alignment": align,
                        "x2l": " + ".join(FORCING_PAIRS[name][1])
                        if FORCING_PAIRS[name][0] == "sum"
                        else "|" + ", ".join(FORCING_PAIRS[name][1]) + "|",
                        "n_steps": acc["n_steps"],
                        "mean_diff": _sig(acc["sd"] / n),
                        "rms_diff": _sig(rms),
                        "max_abs_diff": _sig(acc["maxabs"]),
                        "rel_rms": _sig(rms / std_e if std_e > 0 else None),
                    }
                )
    return rows


def _findings(time_rows, fill_rows, value_rows, pdpi_rows, forcing_rows) -> list:
    """Plain-language findings, generated from the tables so they stay true."""
    out = []
    for r in time_rows:
        msg = (
            f"{r['case']}: {r['files_scanned']}/{r['files_expected']} monthly files, "
            f"{r['n_records']:,} 6-hourly records (expected {r['n_expected']:,}); "
            f"{r['n_gaps']} gaps, {r['n_duplicates_or_backwards']} "
            "duplicate/backward steps."
        )
        level = (
            "ok"
            if (
                r["n_gaps"] == 0
                and r["n_duplicates_or_backwards"] == 0
                and r["n_records"] == r["n_expected"]
            )
            else "warn"
        )
        out.append({"level": level, "text": msg})
        if r["stub"].get("present"):
            out.append(
                {
                    "level": "info",
                    "text": (
                        f"{r['case']}: elm.h0.{STUB_MONTH} is a "
                        f"{r['stub']['n_records']}-record stub "
                        f"(time {r['stub']['time']}); excluded from everything here."
                    ),
                }
            )
    mism = [r for r in fill_rows if r["mismatch"]]
    if mism:
        out.append(
            {
                "level": "bad",
                "text": (
                    f"{len(mism)} of {len(fill_rows)} time-varying variables store "
                    "1e20 at missing cells while "
                    f"declaring _FillValue {mism[0]['fill_value']:.3g} (see "
                    "ISSUE-elm-fillvalue.md)."
                ),
            }
        )
    else:
        out.append(
            {
                "level": "ok",
                "text": "Declared _FillValue matches the stored sentinel in every "
                "variable.",
            }
        )
    leak = [r["var"] for r in fill_rows if r["ocean_valid_any"]]
    out.append(
        {
            "level": "warn" if leak else "ok",
            "text": (
                f"Valid data over non-land cells: {', '.join(leak)}."
                if leak
                else "No variable has valid data over cells with landfrac == 0."
            ),
        }
    )
    gappy = sorted({r["var"] for r in fill_rows if r["land_sentinel_max"]})
    if gappy:
        out.append(
            {
                "level": "warn",
                "text": (
                    f"{len(gappy)} variables are missing at some land cells: "
                    f"{', '.join(gappy)}. "
                    "These need a per-variable mask, not just landfrac."
                ),
            }
        )
    moving = sorted({r["var"] for r in fill_rows if r["land_sentinel_varies"]})
    out.append(
        {
            "level": "warn" if moving else "ok",
            "text": (
                "The set of missing land cells changes over time for: "
                f"{', '.join(moving)}."
                if moving
                else "Where a variable is missing over land, it is missing at the "
                "same cells in every record."
            ),
        }
    )
    bad_vals = sorted({r["var"] for r in value_rows if r["n_nonfinite"] or r["n_huge"]})
    out.append(
        {
            "level": "bad" if bad_vals else "ok",
            "text": (
                "NaN/inf or implausibly large (>=1e10) values in: "
                f"{', '.join(bad_vals)}."
                if bad_vals
                else "No NaN, inf or implausibly large values outside the 1e20 "
                "sentinel."
            ),
        }
    )
    neg = sorted(
        {r["var"] for r in value_rows if r["n_negative"] and r["var"] in NONNEGATIVE}
    )
    if neg:
        out.append(
            {
                "level": "info",
                "text": (
                    "Negative values in nominally non-negative fields: "
                    f"{', '.join(neg)} "
                    "(see the value table for how many and how negative)."
                ),
            }
        )
    veg = [r for r in value_rows if r["var"] in ROLES["vegetation"]]
    for r in veg:
        if r["iav_ratio"] is not None:
            out.append(
                {
                    "level": "info",
                    "text": (
                        f"{r['case']} {r['var']}: interannual variability of monthly "
                        "means is "
                        f"{r['iav_ratio']:.2%} of their magnitude"
                        + (
                            " - consistent with prescribed phenology (treat as "
                            "forcing)."
                            if r["iav_ratio"] < 1e-3
                            else " - varies year to year (prognostic)."
                        )
                    ),
                }
            )
    if forcing_rows:
        best: dict[tuple[str, str], dict] = {}
        for r in forcing_rows:
            k = (r["var"], r["case"])
            if r["rel_rms"] is not None and (
                k not in best or r["rel_rms"] < best[k]["rel_rms"]
            ):
                best[k] = r
        aligns = {r["alignment"] for r in best.values()}
        exact = sorted({r["var"] for r in best.values() if r["rel_rms"] < 1e-4})
        worst: dict[str, float] = {}
        for r in best.values():
            worst[r["var"]] = max(worst.get(r["var"], 0.0), r["rel_rms"])
        loose = [f"{v} ({x:.2g})" for v, x in sorted(worst.items()) if x >= 1e-4]
        out.append(
            {
                "level": "info",
                "text": (
                    "Forcing: the best-matching alignment of x2l to elm.h0 is "
                    f"'{'/'.join(sorted(aligns))}'. "
                    "Identical to 1e-4 of the field's std: "
                    f"{', '.join(exact) or 'none'}. "
                    f"Different (relative RMS): {', '.join(loose) or 'none'}."
                ),
            }
        )
    top = sorted(
        [r for r in pdpi_rows if r["z_land"] is not None],
        key=lambda r: -abs(r["z_land"]),
    )[:8]
    if top:
        out.append(
            {
                "level": "info",
                "text": "Largest PD-PI shifts (land-mean annual difference / "
                "interannual std): "
                + ", ".join(f"{r['var']} {r['z_land']:+.1f}" for r in top)
                + ".",
            }
        )
    return out


def build(root: pathlib.Path, lib_src: pathlib.Path) -> None:
    out = root / "data"
    written: set = set()
    by_case = {c: _load_case(root, c) for c in CASES}
    first = next(r for recs in by_case.values() for r in recs if r)
    names = list(first["vars"])
    with netCDF4.Dataset(first["path"]) as ds:
        ds.set_auto_mask(False)
        attrs = {
            n: (
                getattr(ds[n], "long_name", n),
                getattr(ds[n], "units", ""),
                getattr(ds[n], "cell_methods", ""),
            )
            for n in names
        }
        lat = ds["lat"][:].astype(np.float64)
        lon = ds["lon"][:].astype(np.float64)
        landfrac = ds["landfrac"][:].astype(np.float64)
    land = landfrac > 0
    w_land = _area_weights(lat) * np.where(land, landfrac, 0.0)

    # climatology and annual means per case, streamed from the monthly maps
    clim, clim_std, annual, ts = {}, {}, {}, {}
    gap_union = {n: np.zeros(land.shape, dtype=bool) for n in names}
    gap_inter = {n: np.ones(land.shape, dtype=bool) for n in names}
    for case, recs in by_case.items():
        s = np.zeros((len(names), 12) + land.shape)
        s2 = np.zeros_like(s)
        c = np.zeros_like(s)
        ann = np.full((len(YEARS), len(names)) + land.shape, np.nan, dtype=np.float32)
        ann_s = np.zeros((len(names),) + land.shape)
        ann_c = np.zeros_like(ann_s)
        for i, (m, r) in enumerate(zip(MONTHS, recs)):
            if not r:
                continue
            yi, mi = divmod(i, 12)
            with np.load(_work(root) / case / f"{m}.npz") as z:
                for k, n in enumerate(names):
                    a = z[n].astype(np.float64)
                    ok = np.isfinite(a)
                    s[k, mi][ok] += a[ok]
                    s2[k, mi][ok] += a[ok] ** 2
                    c[k, mi][ok] += 1
                    ann_s[k][ok] += a[ok]
                    ann_c[k][ok] += 1
            for n in names:
                bits = np.unpackbits(
                    np.frombuffer(
                        base64.b64decode(r["vars"][n]["gap_bits"]), dtype=np.uint8
                    )
                )[: land.size].reshape(land.shape)
                g = bits.astype(bool)
                gap_union[n] |= g
                gap_inter[n] &= g
            if mi == 11:
                ann[yi] = np.where(ann_c == 12, ann_s / np.maximum(ann_c, 1), np.nan)
                ann_s[:] = 0
                ann_c[:] = 0
        mean = np.where(c > 0, s / np.maximum(c, 1), np.nan)
        clim[case] = mean.astype(np.float32)
        # spread across years of each calendar month's mean, per cell
        clim_std[case] = np.sqrt(
            np.maximum(np.where(c > 1, s2 / np.maximum(c, 1) - mean**2, np.nan), 0)
        )
        annual[case] = ann
        ts[case] = {
            n: [_sig(r["vars"][n]["land_mean"]) if r else None for r in recs]
            for n in names
        }
        print(f"build: {case} climatology done", flush=True)

    # ------------------------------------------------ audit tables
    time_rows = [_time_audit(recs, case) for case, recs in by_case.items()]
    fill_rows, value_rows = [], []
    for n in names:
        per = [r["vars"][n] for recs in by_case.values() for r in recs if r]
        fv = per[0]["fill_value"]
        fill_rows.append(
            {
                "var": n,
                "fill_value": fv,
                "n_sentinel": sum(p["n_sentinel"] for p in per),
                "n_eq_fill": sum(p["n_eq_fill"] for p in per),
                # stored sentinels that a CF reader would not mask
                "mismatch": sum(p["n_sentinel"] for p in per)
                > sum(p["n_eq_fill"] for p in per),
                "ocean_sentinel_min": min(p["ocean_sentinel_all"] for p in per),
                "ocean_valid_any": max(p["ocean_valid_any"] for p in per),
                "land_sentinel_min": min(p["land_sentinel_all"] for p in per),
                "land_sentinel_max": max(
                    p["land_sentinel_all"] + p["land_sentinel_some"] for p in per
                ),
                "land_sentinel_some": max(p["land_sentinel_some"] for p in per),
                "land_sentinel_varies": bool((gap_union[n] != gap_inter[n]).any())
                or max(p["land_sentinel_some"] for p in per) > 0,
                "land_cells": int(land.sum()),
            }
        )
        for case, recs in by_case.items():
            pc = [r["vars"][n] for r in recs if r]
            nv = sum(p["n_valid"] for p in pc)
            mean = sum(p["sum"] for p in pc) / nv if nv else None
            var = sum(p["sum2"] for p in pc) / nv - mean**2 if nv else None
            # year-to-year spread of each calendar month's mean, relative to its
            # magnitude, land-area weighted: ~0 for a prescribed (repeating) field
            k = names.index(n)
            sd, mu = clim_std[case][k], np.abs(clim[case][k])
            ok = np.isfinite(sd) & np.isfinite(mu) & land[None]
            wk = np.broadcast_to(w_land, sd.shape)[ok]
            denom = float((mu[ok] * wk).sum())
            iav = float((sd[ok] * wk).sum()) / denom if denom > 0 else None
            value_rows.append(
                {
                    "var": n,
                    "case": case,
                    "min": _sig(
                        min(
                            (p["min"] for p in pc if p["min"] is not None), default=None
                        )
                    ),
                    "max": _sig(
                        max(
                            (p["max"] for p in pc if p["max"] is not None), default=None
                        )
                    ),
                    "mean": _sig(mean),
                    "std": _sig(np.sqrt(max(var, 0.0)) if var is not None else None),
                    "n_valid": nv,
                    "n_nonfinite": sum(p["n_nonfinite"] for p in pc),
                    "n_huge": sum(p["n_huge"] for p in pc),
                    "n_negative": sum(p["n_negative"] for p in pc),
                    "frac_negative": _sig(
                        sum(p["n_negative"] for p in pc) / nv if nv else None, 3
                    ),
                    "nonnegative_expected": n in NONNEGATIVE,
                    "iav_ratio": _sig(iav, 4),
                }
            )
    pdpi_rows = []
    if {"PD", "PI"} <= set(annual):
        for k, n in enumerate(names):
            a, b = annual["PD"][:, k], annual["PI"][:, k]
            ok = land & np.isfinite(a).all(0) & np.isfinite(b).all(0)
            if not ok.any():
                continue
            w = w_land[ok] / w_land[ok].sum()
            gm_a = (a[:, ok] * w).sum(1)
            gm_b = (b[:, ok] * w).sum(1)
            sigma = np.sqrt((gm_a.var(ddof=1) + gm_b.var(ddof=1)) / 2)
            cell_sigma = np.sqrt(
                (a[:, ok].var(0, ddof=1) + b[:, ok].var(0, ddof=1)) / 2
            )
            cell_z = np.where(
                cell_sigma > 0,
                (a[:, ok].mean(0) - b[:, ok].mean(0)) / np.maximum(cell_sigma, 1e-30),
                0,
            )
            pdpi_rows.append(
                {
                    "var": n,
                    "pd": _sig(gm_a.mean()),
                    "pi": _sig(gm_b.mean()),
                    "diff": _sig(gm_a.mean() - gm_b.mean()),
                    "rel_diff": _sig(
                        (gm_a.mean() - gm_b.mean()) / abs(gm_b.mean())
                        if gm_b.mean()
                        else None,
                        3,
                    ),
                    "iav_std": _sig(sigma),
                    "z_land": _sig(
                        (gm_a.mean() - gm_b.mean()) / sigma if sigma > 0 else None, 3
                    ),
                    "frac_cells_z2": _sig(
                        float((np.abs(cell_z) > 2).astype(float) @ w), 3
                    ),
                }
            )
    forcing_rows = _forcing_audit(by_case)
    var_meta = []
    for n in names:
        long_name, units, cm = attrs[n]
        allv = np.concatenate([clim[c][names.index(n)][:, land].ravel() for c in clim])
        allv = allv[np.isfinite(allv)]
        diffs = (
            (clim["PD"][names.index(n)] - clim["PI"][names.index(n)])[:, land].ravel()
            if {"PD", "PI"} <= set(clim)
            else np.array([])
        )
        diffs = diffs[np.isfinite(diffs)]
        var_meta.append(
            {
                "name": n,
                "long_name": long_name,
                "units": units,
                "cell_methods": cm,
                "role": _role(n),
                "group": next((g for g, members in GROUPS if n in members), "Other"),
                "p01": _sig(np.percentile(allv, 1)) if allv.size else None,
                "p99": _sig(np.percentile(allv, 99)) if allv.size else None,
                "diff_p99": _sig(np.percentile(np.abs(diffs), 99))
                if diffs.size
                else None,
            }
        )
    findings = _findings(time_rows, fill_rows, value_rows, pdpi_rows, forcing_rows)

    # ------------------------------------------------ write
    sources = [
        {
            "id": c,
            "label": c,
            "long_label": CASE_LABELS[c],
            "kind": "case",
            "path": str(DATA_ROOT / CASES[c] / "run"),
        }
        for c in CASES
    ]
    _js(
        out / "audit.js",
        "audit",
        None,
        {
            "findings": findings,
            "time": time_rows,
            "fill": fill_rows,
            "values": value_rows,
            "pdpi": pdpi_rows,
            "forcing": forcing_rows,
        },
        written,
    )
    for case in CASES:
        _js(out / "ts" / f"{case}.js", "ts", case, ts[case], written)
    for k, n in enumerate(names):
        vm = var_meta[k]
        frames_all = np.concatenate([clim[c][k][:, land] for c in clim])
        frames_all = frames_all[np.isfinite(frames_all)]
        lo = float(frames_all.min()) if frames_all.size else 0.0
        hi = float(frames_all.max()) if frames_all.size else 1.0
        for case in CASES:
            enc = _encode(np.where(land, clim[case][k], np.nan), lo, hi)
            _js(
                out / "maps" / n / f"{case}.js",
                "map",
                f"{n}/{case}",
                {"var": n, "source": case, **enc},
                written,
            )
        vm["map_lo"], vm["map_hi"] = _sig(lo, 8), _sig(hi, 8)
    gap_count = np.where(
        land, sum(g.astype(np.int32) for g in gap_union.values()), np.nan
    )
    _js(
        out / "masks.js",
        "masks",
        None,
        {
            "landfrac": _encode(np.where(land, landfrac, np.nan)[None], 0.0, 1.0),
            "gap_count": _encode(
                gap_count[None].astype(np.float64), 0.0, float(len(names))
            ),
            "gap_vars": {n: int(g.sum()) for n, g in gap_union.items() if g.any()},
            # per variable, land cells missing in any record of either case,
            # np.packbits of the row-major (lat, lon) grid
            "gap_bits": {
                n: base64.b64encode(np.packbits(g & land)).decode()
                for n, g in gap_union.items()
                if g.any()
            },
        },
        written,
    )
    _js(
        out / "meta.js",
        "meta",
        None,
        {
            "title": "E3SMv3 F2010 land (ELM)",
            "built": time.strftime("%Y-%m-%d %H:%M"),
            "sources": sources,
            "vars": var_meta,
            "groups": [g for g, _ in GROUPS] + ["Other"],
            "months": MONTHS,
            "lat": [round(x, 4) for x in lat.tolist()],
            "lon0": float(lon[0]),
            "land_cells": int(land.sum()),
        },
        written,
    )
    _prune(out, written)
    shutil.copy2(HERE / "index.html", root / "index.html")
    if lib_src.is_dir():
        shutil.copytree(lib_src, root / "lib", dirs_exist_ok=True)
    land_json = root / "lib" / "land-110m.json"
    if land_json.exists():  # as a script, so coastlines load from file:// too
        (root / "lib" / "land-110m.js").write_text(
            f"DASH.loaded('land','null',{land_json.read_text()});\n"
        )
    print(f"build: wrote {root}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("stage", choices=["scan", "build"])
    p.add_argument("--root", type=pathlib.Path, default=DEFAULT_ROOT)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument(
        "--lib-src",
        type=pathlib.Path,
        default=DEFAULT_LIB,
        help="directory with plotly.min.js, topojson-client.min.js, land-110m.json",
    )
    p.add_argument("--months", nargs="*", help="limit scan to these YYYY-MM (testing)")
    p.add_argument(
        "--force",
        action="store_true",
        help="scan: redo months that already have work files (all, or --months)",
    )
    a = p.parse_args()
    if a.months:
        global MONTHS
        MONTHS = a.months
    if a.stage == "scan":
        scan(a.root, a.workers, a.force)
    else:
        build(a.root, a.lib_src)


if __name__ == "__main__":
    main()
