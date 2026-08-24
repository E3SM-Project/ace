"""Derive a short smoke-test config from a production config-train-*.yaml.

    uv run python configs/experiments/e3sm_hist_v20260812/make_smoke_config.py \
        configs/experiments/e3sm_hist_v20260812/config-train-ocn.yaml \
        $PSCRATCH/smoke-ocn.yaml

Shrinking by hand is error-prone for two reasons this handles automatically:
inference start_indices must land exactly on the ocean's 5-day axis, and the
ocean and atmosphere windows of a coupled config must begin at the same
timestamp. Both are validated at runtime and fail the run otherwise.

Note that `--override` cannot reach anything inside a list (the inference blocks
and the concat/merge members), which is why this is a script rather than a set
of dotlist overrides.
"""

import argparse
import glob
import sys

import xarray as xr
import yaml

RUN = "/global/cfs/cdirs/e3smdata/simulations/v3.LR.historical_0101.aigo/run"
N_INITIAL_CONDITIONS = 16


def _rewrite_patterns(node, years):
    if isinstance(node, dict):
        fp = node.get("file_pattern")
        if isinstance(fp, str):
            fp = fp.replace("5D.*.remapped.nc", f"5D.{years}*.remapped.nc")
            fp = fp.replace("eam.h0.*.nc", f"eam.h0.{years}-*.nc")
            fp = fp.replace("landfrac5d.*.nc", f"landfrac5d.{years}.nc")
            node["file_pattern"] = fp
        for v in node.values():
            _rewrite_patterns(v, years)
    elif isinstance(node, list):
        for v in node:
            _rewrite_patterns(v, years)


def _times(pattern):
    out = []
    for p in sorted(glob.glob(f"{RUN}/{pattern}")):
        ds = xr.open_dataset(p, decode_timedelta=False)
        out += [str(v)[:19].replace(" ", "T") for v in ds.time.values]
        ds.close()
    return sorted(out)


def _set_subsets(node, window):
    """Give every dataset member the same window; None removes it (inference)."""
    if isinstance(node, dict):
        if "file_pattern" in node:
            if window is None:
                node.pop("subset", None)
            else:
                node["subset"] = dict(window)
        for v in node.values():
            _set_subsets(v, window)
    elif isinstance(node, list):
        for v in node:
            _set_subsets(v, window)


def _split_concat(dataset, windows):
    """Assign successive windows to concat members so they do not overlap."""
    if isinstance(dataset, dict) and "concat" in dataset:
        for member, w in zip(dataset["concat"], windows):
            _set_subsets(member, w)
        return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("out")
    ap.add_argument(
        "--years",
        default="194[0-5]",
        help="glob for the years to keep, e.g. '194[0-5]'",
    )
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="global batch size; must be divisible by the rank count you "
        "intend to run on, so the default 4 will not run on 8 ranks",
    )
    ap.add_argument("--experiment-dir", default=None)
    ap.add_argument(
        "--full-data",
        action="store_true",
        help="keep the production globs and train/val windows; shrink "
        "only epochs and the inference block. Use this to "
        "rehearse a production launch (dataset construction over "
        "all ~1500 files per stream is the expensive, and "
        "historically fragile, part).",
    )
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    coupled = "n_coupled_steps" in cfg["inference"][0]
    if not args.full_data:
        _rewrite_patterns(cfg, args.years)

    # --full-data keeps the production windows and initial conditions, so none
    # of the derived windows below are used. Reading the time coordinate of all
    # ~1500 files to compute and then discard them costs minutes.
    ocn: list[str] = []
    train: list[dict] = []
    val: dict = {}
    if not args.full_data:
        ocn = _times(f"*fmeDerivedFields5D.{args.years}*.remapped.nc")
        if len(ocn) < 60:
            sys.exit(f"only {len(ocn)} ocean times matched --years {args.years}")
        # Windows start on the ocean axis so the coupled realms align.
        a, b = len(ocn) // 2, int(len(ocn) * 0.75)
        train = [
            {"start_time": ocn[0], "stop_time": ocn[a // 2]},
            {"start_time": ocn[a // 2], "stop_time": ocn[a]},
        ]
        val = {"start_time": ocn[a], "stop_time": ocn[b]}

    for section, window in (("train_loader", None), ("validation", val)):
        loader = cfg[section] if section == "train_loader" else cfg[section]["loader"]
        if not args.full_data:
            if section == "train_loader":
                if not _split_concat(loader["dataset"], train):
                    _set_subsets(loader["dataset"], train[0])
            else:
                _set_subsets(loader["dataset"], window)
        loader["batch_size"] = args.batch_size
        loader["num_data_workers"] = 2

    cfg["inference"] = [cfg["inference"][0]]
    inf = cfg["inference"][0]
    inf.pop("epochs", None)
    inf["weight"] = 1.0
    _set_subsets(inf["loader"]["dataset"], None)  # inference forbids `subset`
    if coupled:
        inf["n_coupled_steps"] = 4
        inf["coupled_steps_in_memory"] = 2
    else:
        # The default inference aggregator includes StepMeanMetricConfig(step=20),
        # so a rollout shorter than 21 steps fails with MetricNotSupportedError.
        # Keep the metrics rather than overriding log_step_means to [].
        inf["n_forward_steps"] = 24
        inf["forward_steps_in_memory"] = 12
    # The number of inference initial conditions must be divisible by the rank
    # count (train_config.py checks this), so keep 16 as production does -- 4
    # would break every run on 8 or more ranks.
    if not args.full_data:
        # The last initial condition must leave room for the whole rollout:
        # validate_inference_length rejects max_start_index + window_len >
        # len(dataset).
        rollout = inf["n_coupled_steps"] if coupled else inf["n_forward_steps"]
        latest_start = len(ocn) - rollout - 2 - N_INITIAL_CONDITIONS
        if latest_start < 0:
            sys.exit(
                f"--years {args.years} gives only {len(ocn)} ocean times, too few "
                f"for {N_INITIAL_CONDITIONS} initial conditions and a "
                f"{rollout}-step rollout; widen the year glob."
            )
        first_ic = min(b, latest_start)
        inf["loader"]["start_indices"] = {
            "times": ocn[first_ic : first_ic + N_INITIAL_CONDITIONS]
        }

    if args.batch_size % 4 != 0 and args.batch_size not in (1, 2):
        print(
            f"note: batch_size {args.batch_size} only divides rank counts "
            f"{[n for n in (1, 2, 4, 8, 16) if args.batch_size % n == 0]}; "
            f"fme rejects a batch size that the rank count does not divide.",
            file=sys.stderr,
        )

    cfg["max_epochs"] = args.epochs
    cfg["save_checkpoint"] = False
    cfg["logging"]["log_to_wandb"] = False
    if args.experiment_dir:
        cfg["experiment_dir"] = args.experiment_dir

    yaml.safe_dump(
        cfg, open(args.out, "w"), sort_keys=False, default_flow_style=False, width=100
    )
    print(f"wrote {args.out}")
    if args.full_data:
        # Report what the config actually says, not windows that were never
        # applied -- a summary that describes a different config is worse than
        # no summary.
        print(
            "  production globs, train/validation windows and inference "
            "initial conditions kept as-is"
        )
        print(f"  epochs {args.epochs}, batch_size {args.batch_size}")
    else:
        print(f"  train   {train[0]['start_time']} .. {train[-1]['stop_time']}")
        print(f"  val     {val['start_time']} .. {val['stop_time']}")
        print(
            f"  inf ICs {len(inf['loader']['start_indices']['times'])} starting "
            f"{inf['loader']['start_indices']['times'][0]}"
        )


if __name__ == "__main__":
    main()
