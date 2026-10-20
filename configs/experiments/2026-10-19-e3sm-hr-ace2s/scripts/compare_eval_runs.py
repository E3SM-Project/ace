"""Compare two evaluator runs: every netCDF file and every scalar metric.

    python scripts/compare_eval_runs.py REFERENCE_DIR OTHER_DIR [--rtol R] [--atol A]

Both directories are evaluator experiment_dirs run with
``logging.metrics_log_dir=<experiment_dir>/metrics``. Prints, per file and per
metric, the largest absolute and relative difference, and exits non-zero if
any file or metric is missing on one side or differs by more than the
tolerances (defaults suit fp32 runs of the same model on different tilings).
"""

import argparse
import json
import math
import pathlib
import sys

import numpy as np
import xarray as xr


def _netcdf_files(root: pathlib.Path) -> set[str]:
    return {str(p.relative_to(root)) for p in root.rglob("*.nc")}


# GlobalTimer categories, logged as durations next to the summary metrics
TIMERS = {
    "aggregator",
    "compute_derived_variables",
    "data_loading",
    "data_writer",
    "data_writer_io",
    "final_writer_flush",
    "flush_diagnostics",
    "forward_prediction",
    "inference",
    "initialization",
    "validation",
    "wandb_logging",
}


def _is_timing(name: str) -> bool:
    return "per_second" in name or all(part in TIMERS for part in name.split("/"))


def _metrics(root: pathlib.Path) -> dict[tuple[int, str], float]:
    """Scalar metrics by (step, name), from the JSONL files of metrics_log_dir."""
    out: dict[tuple[int, str], float] = {}
    for path in sorted((root / "metrics").rglob("*.jsonl")):
        for line in path.read_text().splitlines():
            record = json.loads(line)
            step = record.pop("step", -1)
            for name, value in record.items():
                if isinstance(value, int | float) and not _is_timing(name):
                    out[(step, name)] = float(value)
    return out


def _diff(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    both = ~(np.isnan(a) & np.isnan(b))
    if not both.any():
        return 0.0, 0.0
    abs_diff = np.abs(a - b)[both]
    scale = np.maximum(np.abs(a), np.abs(b))[both]
    rel = abs_diff / np.where(scale > 0, scale, 1.0)
    return float(np.nanmax(abs_diff)), float(np.nanmax(rel))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=pathlib.Path)
    parser.add_argument("other", type=pathlib.Path)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-5)
    args = parser.parse_args()
    ok = True

    ref_files, other_files = _netcdf_files(args.reference), _netcdf_files(args.other)
    for name in sorted(ref_files ^ other_files):
        print(f"MISSING file {name} in one run")
        ok = False
    for name in sorted(ref_files & other_files):
        with (
            xr.open_dataset(args.reference / name, decode_timedelta=False) as a,
            xr.open_dataset(args.other / name, decode_timedelta=False) as b,
        ):
            if set(a.data_vars) != set(b.data_vars) or dict(a.sizes) != dict(b.sizes):
                print(f"DIFFERENT layout {name}: {dict(a.sizes)} vs {dict(b.sizes)}")
                ok = False
                continue
            worst = (0.0, 0.0, "")
            for var in a.data_vars:
                va, vb = a[var].values, b[var].values
                if not np.issubdtype(va.dtype, np.number):
                    continue
                abs_d, rel_d = _diff(va.astype(np.float64), vb.astype(np.float64))
                if abs_d > args.atol + args.rtol * float(np.nanmax(np.abs(va))):
                    print(f"DIFFERENT {name}:{var} max abs {abs_d:.3g} rel {rel_d:.3g}")
                    ok = False
                worst = max(worst, (abs_d, rel_d, str(var)))
            print(
                f"file {name}: worst {worst[2]} abs {worst[0]:.3g} rel {worst[1]:.3g}"
            )

    ref_metrics, other_metrics = _metrics(args.reference), _metrics(args.other)
    if not ref_metrics:
        print("no metrics found: set logging.metrics_log_dir=<experiment_dir>/metrics")
        ok = False
    for key in sorted(set(ref_metrics) ^ set(other_metrics)):
        print(f"MISSING metric {key} in one run")
        ok = False
    n_bad = 0
    for key in sorted(set(ref_metrics) & set(other_metrics)):
        a, b = ref_metrics[key], other_metrics[key]
        if math.isnan(a) and math.isnan(b):
            continue
        if not math.isclose(a, b, rel_tol=args.rtol, abs_tol=args.atol):
            n_bad += 1
            if n_bad <= 50:
                print(f"DIFFERENT metric {key}: {a!r} vs {b!r}")
    print(f"{len(ref_metrics)} metrics compared, {n_bad} differ")
    ok = ok and n_bad == 0
    print("MATCH" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
