"""Read throughput of XarrayDataConfig engine "netcdf4" vs "cdf5" on HR files.

Part A: open one file and load isel(time=slice(i, i + 2)) of the training
variables, as XarrayDataset's netcdf read path does per file.
Part B: build a real XarrayDataset over three monthly files and time
dataset[i] (the data loader hot path).

Every trial reads records no earlier trial touched, and the engines alternate,
so neither engine is served from the other's client page cache. The script
cannot see other jobs reading the same files.

    source scripts/env.sh
    python scripts/bench_cdf5_engine.py
"""

import json
import os
import random
import time

import numpy as np
import yaml

from fme.core.dataset.schedule import IntSchedule
from fme.core.dataset.xarray import XarrayDataConfig, XarrayDataset, _open_xr_dataset

EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = "/pscratch/sd/m/mahf708/hr-data/v3.HR.F2010.aigo.PD.y1/eam.h0"
PREFIX = "v3.HR.F2010.aigo.PD.y1.eam.h0."
ENGINES = ("netcdf4", "cdf5")
N_TRIALS = 5


def _training_names() -> tuple[list[str], dict[str, str]]:
    with open(os.path.join(EXP, "config-train-hr.yaml")) as f:
        config = yaml.safe_load(f)
    step = config["stepper"]["step"]["config"]
    names = sorted(set(step["in_names"]) | set(step["out_names"]))
    rename = config["train_loader"]["dataset"]["rename"]
    return names, rename


def _report(label: str, engine: str, nbytes: int, seconds: float):
    print(
        json.dumps(
            {
                "part": label,
                "engine": engine,
                "MB": round(nbytes / 1e6, 1),
                "s": round(seconds, 2),
                "MB_per_s": round(nbytes / 1e6 / seconds, 1),
            }
        ),
        flush=True,
    )


def part_a(names: list[str], rename: dict[str, str], records: list[int]):
    inverse = {v: k for k, v in rename.items()}
    on_disk = [inverse.get(n, n) for n in names]
    path = os.path.join(DATA, PREFIX + "0008-01.nc")
    totals = {e: [0, 0.0] for e in ENGINES}
    for trial in range(N_TRIALS):
        for k, engine in enumerate(ENGINES):
            i = records[2 * trial + k]
            start = time.perf_counter()
            ds = _open_xr_dataset(path, engine=engine, mask_and_scale=False)
            loaded = ds[on_disk].isel(time=slice(i, i + 2)).compute()
            ds.close()
            seconds = time.perf_counter() - start
            nbytes = sum(v.nbytes for v in loaded.data_vars.values())
            _report(f"A trial {trial} rec {i}", engine, nbytes, seconds)
            totals[engine][0] += nbytes
            totals[engine][1] += seconds
    for engine, (nbytes, seconds) in totals.items():
        _report("A total", engine, nbytes, seconds)


def part_b(names: list[str], rename: dict[str, str], samples: list[int]):
    datasets = {}
    for engine in ENGINES:
        config = XarrayDataConfig(
            data_path=DATA,
            file_pattern=PREFIX + "0009-0[1-3].nc",
            engine=engine,
            rename=rename,
            reference_pressure_name="P0",
        )
        start = time.perf_counter()
        datasets[engine] = XarrayDataset(config, names, IntSchedule.from_constant(2))
        print(
            json.dumps(
                {
                    "part": "B build",
                    "engine": engine,
                    "s": round(time.perf_counter() - start, 2),
                    "n_samples": len(datasets[engine]),
                }
            ),
            flush=True,
        )
    totals = {e: [0, 0.0] for e in ENGINES}
    for trial in range(N_TRIALS):
        for k, engine in enumerate(ENGINES):
            idx = samples[2 * trial + k]
            start = time.perf_counter()
            data, *_ = datasets[engine][idx]
            seconds = time.perf_counter() - start
            nbytes = sum(t.numel() * t.element_size() for t in data.values())
            _report(f"B trial {trial} idx {idx}", engine, nbytes, seconds)
            totals[engine][0] += nbytes
            totals[engine][1] += seconds
    for engine, (nbytes, seconds) in totals.items():
        _report("B total", engine, nbytes, seconds)
    # the same sample again, now from the client page cache: the non-I/O floor
    for k, engine in enumerate(ENGINES):
        idx = samples[2 * (N_TRIALS - 1) + k]
        start = time.perf_counter()
        data, *_ = datasets[engine][idx]
        seconds = time.perf_counter() - start
        nbytes = sum(t.numel() * t.element_size() for t in data.values())
        _report(f"B warm repeat idx {idx}", engine, nbytes, seconds)


def main():
    names, rename = _training_names()
    print(f"{len(names)} variables, host {os.uname().nodename}", flush=True)
    rng = random.Random(int(time.time()))
    # sample starts at least 2 apart so no record is read twice
    starts = rng.sample(range(0, 120, 2), 2 * N_TRIALS)
    part_a(names, rename, starts)
    # 3 monthly files of 124/112/124 records; samples span all three
    samples = list(np.array(rng.sample(range(0, 356, 2), 2 * N_TRIALS)))
    part_b(names, rename, [int(s) for s in samples])


if __name__ == "__main__":
    main()
