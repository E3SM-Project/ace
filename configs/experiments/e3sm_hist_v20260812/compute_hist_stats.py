#!/usr/bin/env python
"""Compute ACE normalization statistics from the E3SMv3 historical run.

Streaming, exact and embarrassingly parallel: each worker opens one file,
applies exactly the transforms the ``fme`` data loader applies (rename ->
``overwrite`` (``x * multiply_scalar + add_scalar``) -> ``combine`` (weighted
sum of source variables)), and returns (count, mean, M2) per field over the
valid (finite) points.  Partial results are combined with Chan's parallel
variance algorithm, which is exact up to floating point and independent of the
order in which files are processed.

Conventions, chosen to match ``scripts/data_process/get_stats.py`` and the
existing stats files this replaces:

* statistics are **unweighted** over (time, lat, lon) -- no latitude/area
  weighting.  Confirmed numerically: the piControl ``LANDFRAC`` centering value
  (0.3426250) equals the unweighted mean of ``LANDFRAC`` (0.3426344), while the
  cos(lat)-weighted mean is 0.2934.
* standard deviations are population (ddof=0), as ``xarray.Dataset.std``.
* NaN / ``_FillValue`` points are excluded (as ``xarray``'s ``skipna=True``).
* ``scaling-residual.nc`` is the standard deviation of consecutive-in-time
  differences; differences are taken within each file only, so the ~1500
  file-boundary pairs (0.8% of pairs) are skipped.

Usage::

    uv run python compute_hist_stats.py --realm ocean  --out-dir DIR
    uv run python compute_hist_stats.py --realm atmosphere --out-dir DIR
"""

import argparse
import dataclasses
import datetime
import glob
import os
import pickle
import re
import sys
import time
from collections.abc import Mapping, Sequence
from multiprocessing import Pool

import numpy as np
import xarray as xr
from netCDF4 import Dataset  # type: ignore[import-untyped]

RUN_DIR = "/global/cfs/cdirs/e3smdata/simulations/v3.LR.historical_0101.aigo/run"
AUX_DIR = "/pscratch/sd/m/mahf708/e3sm-hist-aux/landfrac5d"
FILL_THRESHOLD = 1e19


@dataclasses.dataclass
class Stream:
    """One on-disk dataset, plus the loader transforms applied to it."""

    key: str
    data_path: str
    file_pattern: str
    loader_targets: Sequence[str]
    rename: Mapping[str, str] = dataclasses.field(default_factory=dict)
    multiply_scalar: Mapping[str, float] = dataclasses.field(default_factory=dict)
    add_scalar: Mapping[str, float] = dataclasses.field(default_factory=dict)
    combine: Mapping[str, Mapping[str, float]] = dataclasses.field(default_factory=dict)
    mask_and_scale: bool = False
    # filled in by discover_raw_extras(): every other variable in the files,
    # recorded under its raw name.  Kept separate so that the loader names and
    # the rule-(b) additions stay distinguishable.
    raw_extras: Sequence[str] = dataclasses.field(default_factory=list)

    @property
    def rename_inverse(self) -> dict[str, str]:
        return {v: k for k, v in self.rename.items()}

    @property
    def targets(self) -> list[str]:
        return list(self.loader_targets) + list(self.raw_extras)

    @property
    def consumed_disk_names(self) -> set[str]:
        """Raw variables read to build the loader names.

        These are excluded from the rule-(b) additions: a name must mean one
        thing, so `windStressZonal` cannot sit alongside its sign-flipped
        `TAUX`, nor degC `sst` alongside the Kelvin `sst`.
        """
        names: set[str] = set()
        for target in self.loader_targets:
            if target in self.combine:
                for source in self.combine[target]:
                    names.add(self.rename_inverse.get(source, source))
            else:
                names.add(self.rename_inverse.get(target, target))
        return names

    @property
    def load_names(self) -> list[str]:
        """Renamed names that must be read, including combine-only sources."""
        names: list[str] = []
        for target in self.targets:
            if target in self.combine:
                for source in self.combine[target]:
                    if source not in names:
                        names.append(source)
            elif target not in names:
                names.append(target)
        return names

    def files(self) -> list[str]:
        return sorted(glob.glob(os.path.join(self.data_path, self.file_pattern)))


_ATMOS_NAMES = [
    "LANDFRAC",
    "OCNFRAC",
    "ICEFRAC",
    "PHIS",
    "SOLIN",
    "PS",
    "TS",
    "LHFLX",
    "SHFLX",
    "FLUS",
    "FLUT",
    "FLDS",
    "FSNS",
    "FSUTOA",
    "DTENDTTW",
    "TAUX",
    "TAUY",
    "Qat2m",
    "Uat10m",
    "Vat10m",
    "Tat2m",
    "surface_precipitation_rate",
    "frozen_precipitation_rate",
    # not currently in in_names, but config-train-atm.yaml documents adding it
    # as a forcing via `rename: {co2vmr: global_mean_co2}`; it is a scalar per
    # time step, so its statistics are over time alone.
    "global_mean_co2",
] + [f"{v}_{i}" for v in ("T", "STW", "U", "V") for i in range(8)]

_DEPTH_NAMES = [
    f"{v}Coarsened_{i}"
    for v in ("salinity", "temperature", "velocityZonal", "velocityMeridional")
    for i in range(19)
]

ATMOSPHERE_STREAMS = [
    Stream(
        key="eam.h0",
        data_path=RUN_DIR,
        file_pattern="v3.LR.historical_0101.aigo.eam.h0.*.nc",
        loader_targets=_ATMOS_NAMES,
        rename={
            "PRECT": "surface_precipitation_rate",
            "PRECST": "frozen_precipitation_rate",
            "co2vmr": "global_mean_co2",
        },
        multiply_scalar={
            "surface_precipitation_rate": 1000.0,
            "frozen_precipitation_rate": 1000.0,
        },
        # the atmosphere loader does not set mask_and_scale, so raw values are
        # used; the run reports how many fill values were seen (expected: 0).
        mask_and_scale=False,
    )
]

OCEAN_STREAMS = [
    Stream(
        key="mpaso.depth5d",
        data_path=RUN_DIR,
        file_pattern="v3.LR.historical_0101.aigo.mpaso.hist.am."
        "fmeDepthCoarsening5D.*.remapped.nc",
        loader_targets=_DEPTH_NAMES,
        mask_and_scale=True,
    ),
    Stream(
        key="mpaso.derived5d",
        data_path=RUN_DIR,
        file_pattern="v3.LR.historical_0101.aigo.mpaso.hist.am."
        "fmeDerivedFields5D.*.remapped.nc",
        loader_targets=[
            "TAUX",
            "TAUY",
            "FSNS",
            "FLDS",
            "FLUS",
            "LHFLX",
            "SHFLX",
            "frozen_precipitation_rate",
            "surface_precipitation_rate",
            "sst",
            "ssh",
        ],
        rename={
            "windStressZonal": "TAUX",
            "windStressMeridional": "TAUY",
            "shortWaveHeatFlux": "FSNS",
            "longWaveHeatFluxDown": "FLDS",
            "longWaveHeatFluxUp": "FLUS",
            "latentHeatFlux": "LHFLX",
            "sensibleHeatFlux": "SHFLX",
            "snowFlux": "frozen_precipitation_rate",
        },
        multiply_scalar={
            "TAUX": -1.0,
            "TAUY": -1.0,
            "FLUS": -1.0,
            "LHFLX": -1.0,
            "SHFLX": -1.0,
        },
        add_scalar={"sst": 273.15},
        combine={
            "surface_precipitation_rate": {
                "rainFlux": 1.0,
                "frozen_precipitation_rate": 1.0,
            }
        },
        mask_and_scale=True,
    ),
    Stream(
        key="mpassi.seaice5d",
        data_path=RUN_DIR,
        file_pattern="v3.LR.historical_0101.aigo.mpassi.hist.am."
        "fmeSeaiceDerivedFields5D.*.remapped.nc",
        loader_targets=["ocean_sea_ice_fraction", "iceVolumeTotal"],
        rename={"iceAreaTotal": "ocean_sea_ice_fraction"},
        mask_and_scale=True,
    ),
    Stream(
        key="landfrac5d",
        data_path=AUX_DIR,
        file_pattern="landfrac5d.*.nc",
        loader_targets=["LANDFRAC", "sea_surface_fraction"],
        mask_and_scale=False,
    ),
]

REALMS = {"atmosphere": ATMOSPHERE_STREAMS, "ocean": OCEAN_STREAMS}
_STREAMS_BY_KEY = {s.key: s for realm in REALMS.values() for s in realm}


def _moments(x: np.ndarray) -> tuple[int, float, float]:
    """(count, mean, sum of squared deviations) over the finite points of x."""
    finite = np.isfinite(x)
    n = int(finite.sum())
    if n == 0:
        return 0, 0.0, 0.0
    if n == x.size:
        mean = float(x.mean())
        m2 = float(((x - mean) ** 2).sum())
    else:
        mean = float(np.where(finite, x, 0.0).sum()) / n
        m2 = float(np.where(finite, (x - mean) ** 2, 0.0).sum())
    return n, mean, m2


def _combine_moments(
    a: tuple[int, float, float], b: tuple[int, float, float]
) -> tuple[int, float, float]:
    """Chan's parallel combination of (count, mean, M2)."""
    na, ma, m2a = a
    nb, mb, m2b = b
    if nb == 0:
        return a
    if na == 0:
        return b
    n = na + nb
    delta = mb - ma
    mean = ma + delta * nb / n
    m2 = m2a + m2b + delta * delta * na * nb / n
    return n, mean, m2


def file_stats(args: tuple[str, str]) -> dict:
    """Per-file (count, mean, M2) of each target field and of its time diffs.

    Fields are handled one at a time so that a worker never holds more than a
    couple of variables in memory; an EAM history file has 55 wanted fields of
    64 MB each in float64.
    """
    stream_key, path = args
    stream = _STREAMS_BY_KEY[stream_key]
    ds = Dataset(path)
    ds.set_auto_maskandscale(stream.mask_and_scale)
    n_fill = 0

    def load(name: str) -> np.ndarray:
        """Read one variable and apply rename/multiply_scalar/add_scalar."""
        nonlocal n_fill
        disk_name = stream.rename_inverse.get(name, name)
        raw = ds.variables[disk_name][:]
        if np.ma.isMaskedArray(raw):
            data = raw.filled(np.nan).astype(np.float64)
        else:
            data = np.asarray(raw, dtype=np.float64)
        # Any surviving _FillValue sentinel (only possible when the loader does
        # not decode it) is counted and excluded.
        sentinel = np.abs(data) > FILL_THRESHOLD
        if sentinel.any():
            n_fill += int(sentinel.sum())
            data = np.where(sentinel, np.nan, data)
        if name in stream.multiply_scalar:
            data = data * stream.multiply_scalar[name]
        if name in stream.add_scalar:
            data = data + stream.add_scalar[name]
        return data

    out = {}
    try:
        n_times = ds.dimensions["time"].size
        for name in stream.targets:
            if name in stream.combine:
                data = None
                for source, coefficient in stream.combine[name].items():
                    term = load(source) * coefficient
                    data = term if data is None else data + term
            else:
                data = load(name)
            assert data is not None
            diffs = data[1:] - data[:-1] if data.shape[0] > 1 else data[:0]
            out[name] = (_moments(data), _moments(diffs))
            del data, diffs
    finally:
        ds.close()
    return {
        "path": path,
        "n_times": n_times,
        "n_fill": n_fill,
        "stats": out,
    }


_DATE_RE = re.compile(r"\.(\d{4})(?:-(\d{2}))?\.(?:remapped\.)?nc$")


def file_date(path: str) -> tuple[int, int]:
    match = _DATE_RE.search(os.path.basename(path))
    if match is None:
        raise ValueError(f"cannot parse a date out of {path}")
    year = int(match.group(1))
    month = int(match.group(2)) if match.group(2) else 1
    return year, month


def run_stream(
    stream: Stream,
    n_workers: int,
    limit: int | None = None,
    shard: tuple[int, int] | None = None,
) -> list[dict]:
    files = stream.files()
    if limit is not None:
        files = files[:: max(1, len(files) // limit)][:limit]
    if shard is not None:
        index, count = shard
        files = files[index::count]
    print(f"[{stream.key}] {len(files)} files, {len(stream.load_names)} variables")
    tasks = [(stream.key, f) for f in files]
    results = []
    t0 = time.time()
    with Pool(n_workers) as pool:
        for i, res in enumerate(pool.imap_unordered(file_stats, tasks, chunksize=1)):
            results.append(res)
            if (i + 1) % 100 == 0 or i + 1 == len(tasks):
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                print(
                    f"[{stream.key}] {i + 1}/{len(tasks)} files "
                    f"{elapsed:.0f}s ({rate:.2f} files/s, "
                    f"eta {(len(tasks) - i - 1) / rate:.0f}s)",
                    flush=True,
                )
    return results


# Auxiliary variables that carry no field to normalize: the ones
# scripts/data_process/get_stats.py drops, plus calendar/bookkeeping entries.
_SKIP_PREFIXES = ("mask_", "idepth_", "ak_", "bk_", "hya", "hyb", "cosp_")
_SKIP_NAMES = frozenset(
    {
        "P0",
        "time",
        "lat",
        "lon",
        "lev",
        "ilev",
        "date",
        "datesec",
        "date_written",
        "time_written",
        "nbdate",
        "nbsec",
        "ndbase",
        "ndcur",
        "nsbase",
        "nscur",
        "nsteph",
        "mdt",
    }
)


def _is_skipped(name: str) -> bool:
    return (
        name in _SKIP_NAMES or name.endswith("_bnds") or name.startswith(_SKIP_PREFIXES)
    )


def discover_raw_extras(stream: Stream) -> list[str]:
    """Rule (b): every remaining variable in the files, under its raw name.

    A variable qualifies when it is a floating-point field on either the
    (time, lat, lon) grid or the time axis alone (a scalar forcing such as
    ``co2vmr``), is not an auxiliary/bookkeeping entry, and is not already
    consumed to build one of the loader names.
    """
    consumed = stream.consumed_disk_names
    ds = Dataset(stream.files()[0])
    try:
        extras = []
        for name, var in ds.variables.items():
            if name in consumed or _is_skipped(name):
                continue
            if var.dtype.kind != "f":
                continue
            if var.dimensions not in (("time", "lat", "lon"), ("time",)):
                continue
            extras.append(name)
    finally:
        ds.close()
    return sorted(extras)


def aggregate(
    results: Sequence[dict],
    year_ranges: Sequence[tuple[int, int]] | None = None,
) -> tuple[dict[str, dict[str, float]], int, int, tuple[int, int], tuple[int, int]]:
    """Combine per-file partials, optionally restricted to given year ranges.

    Files are combined in sorted-path order so that the result does not depend
    on the order in which the pool happened to finish them.
    """
    totals: dict[str, tuple] = {}
    n_times = 0
    n_files = 0
    n_fill = 0
    dates = []
    for res in sorted(results, key=lambda r: r["path"]):
        year, month = file_date(res["path"])
        if year_ranges is not None and not any(
            lo <= year <= hi for lo, hi in year_ranges
        ):
            continue
        n_files += 1
        n_times += res["n_times"]
        n_fill += res["n_fill"]
        dates.append((year, month))
        for name, (moments, dmoments) in res["stats"].items():
            if name not in totals:
                totals[name] = (moments, dmoments)
            else:
                prev, dprev = totals[name]
                totals[name] = (
                    _combine_moments(prev, moments),
                    _combine_moments(dprev, dmoments),
                )
    out = {}
    for name, (moments, dmoments) in totals.items():
        n, mean, m2 = moments
        dn, _, dm2 = dmoments
        out[name] = {
            "mean": mean,
            "std": float(np.sqrt(m2 / n)) if n else np.nan,
            "residual_std": float(np.sqrt(dm2 / dn)) if dn else np.nan,
            "count": n,
            "residual_count": dn,
        }
    if n_fill:
        print(f"WARNING: {n_fill} fill-valued points were excluded")
    return out, n_times, n_files, min(dates), max(dates)


def variable_metadata(streams: Sequence[Stream]) -> dict[str, dict[str, str]]:
    """Attributes for each target, taken from the (first) source variable."""
    meta: dict[str, dict[str, str]] = {}
    for stream in streams:
        path = stream.files()[0]
        ds = Dataset(path)
        try:
            for target in stream.targets:
                if target in stream.combine:
                    source = next(iter(stream.combine[target]))
                    disk = stream.rename_inverse.get(source, source)
                else:
                    disk = stream.rename_inverse.get(target, target)
                var = ds.variables[disk]
                attrs = {
                    k: var.getncattr(k)
                    for k in var.ncattrs()
                    if k in ("units", "long_name", "standard_name", "cell_methods")
                }
                transform = []
                if disk != target:
                    transform.append(f"renamed from {disk}")
                if target in stream.combine:
                    transform.append(
                        "combine "
                        + " + ".join(
                            f"{c} * {s}" for s, c in stream.combine[target].items()
                        )
                    )
                if target in stream.multiply_scalar:
                    transform.append(
                        f"multiply_scalar {stream.multiply_scalar[target]}"
                    )
                if target in stream.add_scalar:
                    transform.append(f"add_scalar {stream.add_scalar[target]}")
                if (
                    stream.add_scalar.get(target) == 273.15
                    and attrs.get("units") == "degC"
                ):
                    attrs["units"] = "K"
                attrs["source_stream"] = stream.key
                attrs["coverage"] = (
                    "loader name"
                    if target in stream.loader_targets
                    else "raw variable (not used by the current configs)"
                )
                if transform:
                    attrs["loader_transform"] = "; ".join(transform)
                meta[target] = {k: str(v) for k, v in attrs.items()}
        finally:
            ds.close()
    return meta


def write_outputs(
    out_dir: str,
    stats: dict[str, dict[str, float]],
    meta: dict[str, dict[str, str]],
    names: Sequence[str],
    n_times: int,
    n_files: int,
    first: tuple[int, int],
    last: tuple[int, int],
    realm: str,
    write_residual: bool,
) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    history = (
        "Created by configs/experiments/e3sm_hist_v20260812/compute_hist_stats.py on "
        f"{datetime.date.today().isoformat()}. SOURCE_RUN: "
        f"v3.LR.historical_0101.aigo ({RUN_DIR}); realm: {realm}; "
        f"DATE_RANGE: {first[0]:04d}-{first[1]:02d} to {last[0]:04d}-{last[1]:02d}; "
        f"{n_files} files. Unweighted statistics over (time, lat, lon) excluding "
        "NaN/_FillValue points, of the fields as the fme data loader delivers "
        "them (rename, multiply_scalar/add_scalar, combine applied)."
    )
    products = [("centering.nc", "mean"), ("scaling-full-field.nc", "std")]
    if write_residual:
        products.append(("scaling-residual.nc", "residual_std"))
    # A field that is constant over the whole window has a full-field scale of
    # exactly zero, i.e. no usable normalization.  Drop it from every product
    # rather than ship a divide-by-zero, and report it to the caller.
    dropped = [
        name
        for name in names
        if stats[name]["std"] == 0.0 or not np.isfinite(stats[name]["std"])
    ]
    names = [name for name in names if name not in set(dropped)]
    for filename, key in products:
        data_vars = {}
        for name in names:
            value = stats[name][key]
            if key == "residual_std" and name == "global_mean_co2":
                # co2vmr is a slowly varying scalar forcing: its true step-to-step
                # standard deviation is ~1e-9, which is a dangerous divisor.  It is
                # never used (residual scales apply to prognostic names only), so
                # the full-field scale is written instead.
                attrs = dict(meta.get(name, {}))
                attrs["note"] = (
                    "full-field scale written in place of the true residual "
                    "scale, which is ~1e-9"
                )
                data_vars[name] = xr.DataArray(
                    np.float32(stats[name]["std"]), attrs=attrs
                )
                continue
            if key == "residual_std" and (value == 0.0 or not np.isfinite(value)):
                # Fields with no in-file time variation (LANDFRAC, PHIS,
                # co2vmr) have a zero residual scale, which would divide by
                # zero.  Fall back to the full-field scale, which is what the
                # piControl stats files do for these fields.
                value = stats[name]["std"]
                attrs = dict(meta.get(name, {}))
                attrs["note"] = (
                    "residual scale is zero (no variation between consecutive "
                    "times); full-field scale used instead"
                )
            else:
                attrs = dict(meta.get(name, {}))
            data_vars[name] = xr.DataArray(np.float32(value), attrs=attrs)
        ds = xr.Dataset(data_vars)
        ds.attrs["input_samples"] = n_times
        ds.attrs["history"] = history
        ds.to_netcdf(os.path.join(out_dir, filename))
        print(f"wrote {os.path.join(out_dir, filename)} ({len(data_vars)} variables)")
    if dropped:
        print(f"DROPPED (constant over the window, zero scale): {dropped}")
    return dropped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--realm", choices=sorted(REALMS), required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--partials",
        required=True,
        help="pickle of per-file partials; a comma-separated list when reusing "
        "the partials of several shards",
    )
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None, help="debug: files/stream")
    parser.add_argument("--reuse-partials", action="store_true")
    parser.add_argument(
        "--shard",
        default=None,
        help="'I/N': process only every Nth file, offset I (for multi-node runs)",
    )
    parser.add_argument(
        "--partials-only",
        action="store_true",
        help="write the partials and stop, without producing stats files",
    )
    parser.add_argument(
        "--years",
        default=None,
        help="restrict to comma-separated year ranges, e.g. 1940-1989,2000-2039",
    )
    args = parser.parse_args()

    streams = REALMS[args.realm]
    # Rule (b): cover every other variable in the files under its raw name.
    # Done before the pool is forked so the workers inherit the full target
    # list.
    for stream in streams:
        stream.raw_extras = discover_raw_extras(stream)
        print(
            f"[{stream.key}] {len(stream.loader_targets)} loader names "
            f"+ {len(stream.raw_extras)} raw variables: {stream.raw_extras}"
        )
    if args.reuse_partials:
        per_stream = {}
        for path in args.partials.split(","):
            with open(path, "rb") as f:
                shard = pickle.load(f)
            for key, results in shard.items():
                per_stream.setdefault(key, []).extend(results)
        seen = {key: len({r["path"] for r in v}) for key, v in per_stream.items()}
        for key, n_unique in seen.items():
            if n_unique != len(per_stream[key]):
                raise ValueError(f"duplicate files in the partials for {key}")
            print(f"[{key}] loaded {n_unique} per-file partials")
    else:
        # Fail now rather than after an hour of reading: when this runs under
        # srun the partials must land on a shared filesystem, not on a
        # per-node /tmp.
        with open(args.partials, "wb"):
            pass
        shard = None
        if args.shard:
            index, count = args.shard.split("/")
            shard = (int(index), int(count))
        per_stream = {}
        for stream in streams:
            per_stream[stream.key] = run_stream(stream, args.workers, args.limit, shard)
        with open(args.partials, "wb") as f:
            pickle.dump(per_stream, f)
        print(f"wrote partials to {args.partials}", flush=True)
        if args.partials_only:
            return 0

    year_ranges = None
    if args.years:
        year_ranges = []
        for chunk in args.years.split(","):
            lo, hi = chunk.split("-")
            year_ranges.append((int(lo), int(hi)))

    stats: dict[str, dict[str, float]] = {}
    n_times_per_stream = {}
    firsts, lasts, n_files_total = [], [], 0
    for stream in streams:
        agg, n_times, n_files, first, last = aggregate(
            per_stream[stream.key], year_ranges
        )
        stats.update(agg)
        n_times_per_stream[stream.key] = n_times
        firsts.append(first)
        lasts.append(last)
        n_files_total += n_files
        print(f"[{stream.key}] {n_files} files, {n_times} time samples, {first}-{last}")

    names = [name for stream in streams for name in stream.targets]
    missing = [name for name in names if name not in stats]
    if missing:
        raise ValueError(
            f"the partials do not cover {missing}; they were produced before "
            "these names were added, so the run has to be re-read"
        )
    meta = variable_metadata(streams)
    # the primary stream defines the sample count (the merged datasets share a
    # time axis; the auxiliary LANDFRAC files simply span a few more months).
    n_times = n_times_per_stream[streams[0].key]
    write_outputs(
        args.out_dir,
        stats,
        meta,
        names,
        n_times,
        n_files_total,
        min(firsts),
        max(lasts),
        args.realm,
        write_residual=(args.realm == "atmosphere"),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
