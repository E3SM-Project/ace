#!/usr/bin/env python3
"""Assert that every generated run config says what its run id says it says.

`fme.ace.validate_config` proves a config *parses*. It cannot prove that
`E05.aug26.atm.A3_B16_C1_O5_W0_X0.S01` actually has CO2 and both aerosol sets,
batch 16, equal loss weights and no AMP. That is what this checks: the factor
word against the file, for all 35 runs, in about a second on a login node.

    ./check_campaign.py                              # check ../runs
    ./check_campaign.py --dir /some/dir
    ./check_campaign.py --local-batch atm=2          # match the generator

Pass the same `--local-batch` the generator was given. The constant is
duplicated here on purpose -- a check that imports the thing it is checking
cannot catch a mistake in the shared constant -- which means it has to be told
when the generator was run with a non-default value, or it computes the wrong
rank count and reports a divisibility failure that does not exist.

Run it after `generate-campaign.sh` and after any hand edit. A silent
disagreement between a run id and its config is the worst failure mode this
campaign has -- every plot and every conclusion is labelled by the run id.
"""

import argparse
import glob
import os
import pathlib
import sys

import yaml

HERE = pathlib.Path(__file__).resolve().parent

# Kept in step with make_ablation_config.py. Duplicated on purpose: a check that
# imports the thing it is checking cannot catch a mistake in the shared constant.
LOCAL_BATCH = {"atm": 1, "ocn": 2}
GPUS_PER_NODE = 4
AEROSOL_STATE = {
    "A0": (False, False),
    "A1": (True, False),
    "A2": (False, True),
    "A3": (True, True),
}
EMBED = {"embed_dim": 384, "noise_embed_dim": 32}
LOADER = {"num_data_workers": 8, "prefetch_factor": 4}
# One project for both realms, and the team entity rather than the account.
WANDB = {"project": "SamudrACE-E3SMv3", "entity": "e3sm-aig"}

# The page's "disable 2D plots, keep 1D logs". Each of these exists only to make
# images and must be off entirely. `time_mean_denorm`/`time_mean_norm` are NOT
# here: they emit the campaign's headline rmse/bias scalars, so they stay
# enabled and only their plotting is turned off, via the `report_plot` check
# further down.
IMAGE_METRICS_OFF = (
    "zonal_mean", "video", "trend", "seasonal", "near_zero_fraction",
    "enso_coefficient",
)
ONE_STEP_IMAGE_METRICS_OFF = ("snapshot", "mean_map")
# `ipo_index` needs >80 years of rollout and the scored one is 12, so it can
# never build here -- it only ever logs "metric not supported".
UPLOAD_METRICS_OFF = ("ipo_index",)
# Boolean-flag fields belonging to the DEPRECATED Legacy*AggregatorConfig union
# members. dacite picks a union member by shape, so a config using these parses
# fine, warns once, and silently turns the 2D metrics back on.
LEGACY_AGGREGATOR_FIELDS = (
    "log_zonal_mean_images", "log_video", "log_extended_video",
    "log_seasonal_means", "log_histograms", "log_snapshots", "log_mean_maps",
    "log_nino34_index", "log_ipo_index", "log_global_mean_time_series",
)


# Hours per forward step, by realm and ocean cadence. Used to turn an inference
# block's `n_forward_steps` into the physical span of the trajectory it scores.
STEP_HOURS = {("atm", "O5"): 6, ("atm", "O1"): 6, ("ocn", "O5"): 120, ("ocn", "O1"): 24}


def _training_windows(d: dict) -> list[tuple[str, str]]:
    """Every (start_time, stop_time) the train loader actually reads."""
    out: list[tuple[str, str]] = []

    def walk(node):
        if isinstance(node, dict):
            sub = node.get("subset")
            if isinstance(sub, dict) and "start_time" in sub and "stop_time" in sub:
                out.append((str(sub["start_time"]), str(sub["stop_time"])))
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(d["train_loader"]["dataset"])
    return out


def _noleap(stamp: str):
    import cftime

    date, _, clock = str(stamp).partition("T")
    y, m, dd = (int(x) for x in date.split("-"))
    parts = [int(x) for x in clock.split(":")] if clock else []
    parts += [0] * (3 - len(parts))
    return cftime.DatetimeNoLeap(y, m, dd, *parts)


def check_selection_in_sample(d: dict, realm: str, ocean: str) -> list[str]:
    """Checkpoint selection must not see validation or held-out data.

    A weighted inference block picks the checkpoint, so its *whole trajectory* --
    not just its initial condition -- has to lie inside a training subset. A
    12-year rollout started three years before the end of a training window runs
    nine years past it, which is how a "selection is in-sample" claim quietly
    becomes false.
    """
    import datetime

    bad: list[str] = []
    windows = [(_noleap(a), _noleap(b)) for a, b in _training_windows(d)]
    if not windows:
        return ["cannot read the training windows, so selection leakage is unchecked"]
    hours = STEP_HOURS[(realm, ocean)]
    for block in d.get("inference", []):
        if not block.get("weight", 0.0):
            continue  # weight 0 blocks are diagnostics; they are meant to be held out
        span = datetime.timedelta(hours=hours * block["n_forward_steps"])
        for stamp in block["loader"]["start_indices"]["times"]:
            t0 = _noleap(stamp)
            t1 = t0 + span
            if not any(a <= t0 and t1 <= b for a, b in windows):
                bad.append(
                    f"block {block.get('name')!r} is weighted {block['weight']} and "
                    f"selects on {stamp} -> {t1}, which leaves every training "
                    f"window; checkpoint selection would see held-out data"
                )
    return bad


def check(path: pathlib.Path) -> list[str]:
    """Return a list of complaints about one run config, empty if it is sound."""
    bad: list[str] = []
    d = yaml.safe_load(path.read_text())
    step = d["stepper"]["step"]["config"]

    try:
        exp, campaign, realm, word, seed = path.stem.split(".")
        aero, batch_w, co2, lr_w, ocean, weights, amp = word.split("_")
    except ValueError:
        return [f"run id is not <exp>.<date>.<realm>.<tuning_set>.S<seed>: {path.stem}"]

    def want(cond: bool, msg: str) -> None:
        if not cond:
            bad.append(msg)

    batch = d["train_loader"]["batch_size"]
    want(int(batch_w[1:]) == batch, f"{batch_w} but train batch_size is {batch}")
    want(
        d["validation"]["loader"]["batch_size"] == batch,
        "validation batch_size differs from train batch_size",
    )

    ranks, rem = divmod(batch, LOCAL_BATCH[realm])
    want(rem == 0, f"batch {batch} is not a multiple of {realm} local batch")
    if ranks:
        nodes, node_rem = divmod(ranks, GPUS_PER_NODE)
        want(node_rem == 0 and nodes >= 1, f"{ranks} ranks is not whole nodes")
        for block in d.get("inference", []):
            n = len(block["loader"]["start_indices"]["times"])
            want(
                n % ranks == 0,
                f"block {block.get('name')!r} has {n} ICs, not divisible by "
                f"{ranks} ranks",
            )

    want(d["seed"] == int(seed[1:]), f"{seed} but config seed is {d['seed']}")

    if realm == "atm":
        has_co2 = "global_mean_co2" in step["in_names"]
        want((co2 == "C1") == has_co2, f"{co2} but global_mean_co2 present={has_co2}")
        state = ("aerindexall" in step["in_names"], "lwp" in step["out_names"])
        want(state == AEROSOL_STATE[aero], f"{aero} but (inputs, outputs)={state}")
        if state[1]:
            want(
                "lwp" in step["corrector"]["force_positive_names"],
                "aerosol outputs present but not in force_positive_names -- lwp, "
                "lcc and cdnc are non-negative by definition and the corrector is "
                "the only thing enforcing it",
            )
        cfg = step["builder"]["config"]
        for k, v in EMBED.items():
            want(
                cfg[k] == v,
                f"{k} is {cfg[k]}, expected {v} (the page's FOR NASER box)",
            )
        want(
            d["train_loader"].get("time_buffer_pool_size") == 2,
            "time_buffer_pool_size is not 2 -- see EXPERIMENTS.md 'Measurements'",
        )
    else:
        # Cadence has to be consistent across all four merge members: three MPAS
        # streams plus the LANDFRAC aux file, which is materialised per axis
        # because merge members must share sample_start_times. A config that
        # mixes them fails at load with a time-alignment error, or worse, aligns
        # on a subset and silently trains on a fraction of the record.
        want(ocean in ("O1", "O5"), f"unknown ocean cadence {ocean}")
        blob = path.read_text()
        five = [s for s in ("fmeDepthCoarsening5D.", "fmeDerivedFields5D.",
                            "fmeSeaiceDerivedFields5D.", "landfrac5d") if s in blob]
        one = [s for s in ("fmeDepthCoarsening.", "fmeDerivedFields.",
                           "fmeSeaiceDerivedFields.", "landfrac1d") if s in blob]
        if ocean == "O5":
            want(len(five) == 4 and not one,
                 f"O5 but streams are mixed: 5-day={five} 1-day={one}")
        else:
            want(len(one) == 4 and not five,
                 f"O1 but streams are mixed: 1-day={one} 5-day={five}")
            # A 12-year scored rollout is 876 steps at 5-daily and 4380 at 1-daily.
            for block in d.get("inference", []):
                want(block["n_forward_steps"] % 4380 == 0
                     or block["n_forward_steps"] >= 4380,
                     f"O1 block {block.get('name')!r} has "
                     f"{block['n_forward_steps']} forward steps; a 12-year "
                     f"rollout on a daily axis is 4380")

    bad.extend(check_selection_in_sample(d, realm, ocean))

    # L0 holds the base learning rate; L1 scales it by sqrt(batch / 16). Checked
    # numerically rather than by flag, because a wrong lr is invisible in a run id.
    base = 1e-4
    want(lr_w in ("L0", "L1"), f"unknown learning-rate set {lr_w}")
    expected = base if lr_w == "L0" else base * (batch / 16) ** 0.5
    actual = d["optimization"]["lr"]
    want(
        abs(actual - expected) < 1e-12,
        f"{lr_w} at {batch_w} implies lr {expected:.6g}, config has {actual:.6g}",
    )
    want(
        not (lr_w == "L1" and batch == 16),
        "L1 at B16 is identical to L0 -- the scaling is relative to batch 16",
    )

    amp_on = d["optimization"]["enable_automatic_mixed_precision"]
    want((amp == "X1") == amp_on, f"{amp} but AMP={amp_on}")

    w = d["stepper_training"]["loss"].get("weights")
    if weights == "W0":
        want(not w, "W0 is the equal-weight control but a weights block is present")
    else:
        want(bool(w), f"{weights} but no weights block")
        if w:
            # W3 and W4 are the zeroing sets, so all-zero is correct for them.
            want(
                weights in ("W3", "W4") or abs(sum(w.values()) / len(w)) > 0,
                f"{weights} weights are all zero",
            )

    for key in ("checkpoint_save_epochs", "ema_checkpoint_save_epochs"):
        want(d.get(key) == {"step": 1}, f"{key} is {d.get(key)}, expected {{step: 1}}")

    for k, v in WANDB.items():
        want(
            d["logging"].get(k) == v,
            f"logging.{k} is {d['logging'].get(k)!r}, expected {v!r} -- both "
            f"realms share one project so runs are comparable in one workspace",
        )

    for k, v in LOADER.items():
        want(
            d["train_loader"][k] == v,
            f"train_loader.{k} is {d['train_loader'][k]}, expected {v} -- lowering "
            f"it was measured at 3.4x slower",
        )

    # 12yr_test must fire on the final epoch. It runs on
    # range(max_epochs + 1)[start::step], so a start chosen for one run length
    # silently stops scoring the last epoch at another.
    for block in d.get("inference", []):
        sched = block.get("epochs")
        if not sched:
            continue
        fires = range(d["max_epochs"] + 1)[sched.get("start") :: sched.get("step")]
        want(
            bool(fires) and fires[-1] == d["max_epochs"],
            f"block {block.get('name')!r} never fires on the final epoch "
            f"{d['max_epochs']} (start={sched.get('start')}, step={sched.get('step')})",
        )

    # wandb: 2D image metrics off, 1D kept. Checked structurally rather than by
    # building the aggregator, so this stays a one-second login-node check.
    for block in d.get("inference", []):
        agg = block.get("aggregator")
        label = f"inference block {block.get('name')!r}"
        if agg is None:
            bad.append(
                f"{label} has no aggregator block -- 2D image metrics default on"
            )
            continue
        legacy = [k for k in LEGACY_AGGREGATOR_FIELDS if k in agg]
        want(
            not legacy,
            f"{label} uses deprecated legacy flags {legacy}; dacite matches the "
            f"Legacy union member by shape, so this parses and silently re-enables "
            f"the 2D metrics",
        )
        for metric in IMAGE_METRICS_OFF:
            want(
                agg.get(metric, {}).get("enabled") is False,
                f"{label}: {metric} is not explicitly disabled",
            )
        want(
            agg.get("step_diagnostics", {}).get("correction_maps") is False,
            f"{label}: step_diagnostics.correction_maps is not disabled",
        )
        # Upload budget. The `enabled: false` switches above cover the metrics
        # that exist only to make pictures. These three emit one figure per
        # channel while carrying scalars worth keeping, so they are narrowed to
        # a short reference list rather than turned off -- and all three must
        # narrow to the SAME list, or the plots stop being comparable.
        plot_lists = {
            "time_mean_denorm": agg.get("time_mean_denorm", {}).get("plot_variables"),
            "time_mean_norm": agg.get("time_mean_norm", {}).get("plot_variables"),
            "power_spectrum": agg.get("power_spectrum", {}).get("plot_variables"),
            "histogram": agg.get("histogram", {}).get("variables"),
        }
        out = set(step["out_names"])
        for metric, value in plot_lists.items():
            if not (isinstance(value, list) and value):
                bad.append(
                    f"{label}: {metric} has no plot list, so it plots every "
                    f"channel -- one PNG per channel per map metric per block "
                    f"per epoch is what fills the W&B account"
                )
                continue
            want(
                len(value) < len(out),
                f"{label}: {metric} plots all {len(out)} channels; the list is "
                f"meant to drop the interior levels",
            )
            unknown = sorted(set(value) - out)
            want(
                not unknown,
                f"{label}: {metric} plots names that are not outputs {unknown} "
                f"-- a typo here fails silently, it just never plots",
            )
        distinct = {tuple(v) for v in plot_lists.values() if isinstance(v, list)}
        want(
            len(distinct) <= 1,
            f"{label}: the plotted variable lists disagree {plot_lists!r}; the "
            f"point of the list is that one screen shows the same channels as "
            f"maps, spectra and histograms",
        )
        want(
            agg.get("time_mean_norm", {}).get("target") == "norm",
            f"{label}: time_mean_norm.target is not pinned to 'norm'; dacite "
            f"builds this field from the yaml alone, so the 'denorm' default "
            f"wins and the config fails post-init with a union error",
        )
        for metric in UPLOAD_METRICS_OFF:
            want(
                agg.get(metric, {}).get("enabled") is False,
                f"{label}: {metric} is not explicitly disabled",
            )

    vagg = d.get("validation", {}).get("aggregator")
    if vagg is None:
        bad.append(
            "validation has no aggregator block -- snapshot and mean_map default on"
        )
    else:
        legacy = [k for k in LEGACY_AGGREGATOR_FIELDS if k in vagg]
        want(not legacy, f"validation aggregator uses deprecated legacy flags {legacy}")
        for metric in ONE_STEP_IMAGE_METRICS_OFF:
            want(
                vagg.get(metric, {}).get("enabled") is False,
                f"validation: {metric} is not explicitly disabled",
            )
        want(
            vagg.get("ensemble_denorm", {}).get("log_mean_maps") is False,
            "validation: ensemble_denorm.log_mean_maps is not false -- the "
            "ensemble emits crps / ssr_bias / ensemble_mean_rmse mean maps for "
            "every channel, a third of the atmosphere's whole upload",
        )

    # The maps are kept, on disk, as netCDF. Without this the report_plot
    # switches above would be a deletion rather than a relocation.
    want(
        d.get("save_per_epoch_diagnostics") is True,
        "save_per_epoch_diagnostics is not true -- the time-mean fields would "
        "then exist in neither W&B nor experiment_dir",
    )

    # Inputs must be readable by everyone on the reservation, not just the author.
    text = path.read_text()
    for personal in (
        "/pscratch/sd/m/mahf708/2026-08-13",
        "/pscratch/sd/m/mahf708/e3sm-hist-aux",
    ):
        want(personal not in text, f"references personal scratch input {personal}")

    return bad


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dir", default=str(HERE / "runs"))
    p.add_argument("--local-batch", action="append", default=[], metavar="REALM=N",
                   help="samples per rank, matching whatever make_ablation_config.py "
                        "was given (e.g. --local-batch atm=2)")
    args = p.parse_args(argv)

    for spec in args.local_batch:
        realm, _, value = spec.partition("=")
        if realm not in LOCAL_BATCH or not value.isdigit() or int(value) < 1:
            p.error(f"--local-batch expects atm=N or ocn=N, got {spec!r}")
        LOCAL_BATCH[realm] = int(value)

    paths = sorted(pathlib.Path(f) for f in glob.glob(os.path.join(args.dir, "*.yaml")))
    if not paths:
        print(
            f"no configs in {args.dir} -- run generate-campaign.sh first",
            file=sys.stderr,
        )
        return 2

    failed = 0
    for path in paths:
        bad = check(path)
        if bad:
            failed += 1
            print(f"FAIL {path.name}")
            for msg in bad:
                print(f"       {msg}")
    print(f"\nchecked {len(paths)} configs, {failed} with problems")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
