#!/usr/bin/env python3
"""Generate the stage-2 (multi-step fine-tune) configs from stage-1 runs.

Stage 1 is the aug26 campaign: from-scratch training at one forward step
(atmosphere) or four (ocean). Stage 2 takes a finished stage-1 run and keeps
training it on a *stochastic rollout length*, so the model is optimized against
its own multi-step error rather than a single step. The recipe follows
`configs/baselines/era5/ace-train-config-multi-step-finetuning.yaml`.

A stage-2 config is not written by hand and is not derived from
`config-train-<realm>.yaml`. It is derived from the **parent run's own
`config.yaml`**, the one sitting next to its output, so every arm of the
experiment -- aerosol inputs, CO2, loss weights, ocean cadence, batch size,
seed -- is carried across by construction rather than by a second
implementation of `make_ablation_config.py` that could drift from the first.

    ./make_stage2_config.py --list                 # what is eligible, and why not
    ./make_stage2_config.py --all                  # write every eligible child
    ./make_stage2_config.py E12.aug26.ocn.A0_B16_C0_L0_O5_W1_X0.S01

Run ids
-------
The child changes exactly one field of the parent's id:

    E12.aug26.ocn.A0_B16_C0_L0_O5_W1_X0.S01     parent, stage 1
    E12-FT.aug26.ocn.A0_B16_C0_L0_O5_W1_X0.S01  child, stage 2

so the two sort together, the parent is recoverable by `E12-FT` -> `E12`,
and nothing in wandb or on disk can collide with stage 1. The output root is
separately `$PSCRATCH/aug26-ft`, so even a hypothetical id collision cannot
write into `aug26/`.

The parent checkpoint is the **final epoch** (`ckpt_0030.tar` for the
atmosphere, `ckpt_0150.tar` for the ocean), not `best_ckpt.tar`. `best_ckpt` is
selected on one-step validation loss, which is not what stage 2 optimizes, and
`best_inference_ckpt` is selected on a rollout that only fires every 3rd (atm)
or 15th (ocn) epoch and so can be tens of epochs stale. The last epoch is the
most-trained weights and needs no selection metric to justify it. A run that has
not reached its final epoch is not eligible -- that is the point of `--list`.
"""

import argparse
import copy
import pathlib
import sys

import yaml

HERE = pathlib.Path(__file__).resolve().parent
RUNS = HERE / "runs"

# Read and write both go to $PSCRATCH. Measured 2026-09-05 on the filesystem
# matrix: CFS builds the atmosphere dataset index in 1200 s against scratch's
# 100 s, and at the stage-2 read pattern (21-timestep windows, no time_buffer)
# CFS delivers ~0.7 samples/s/node against the ~3 this fine-tune consumes --
# two of four ranks never finished 200 batches in 27 minutes. Scratch delivers
# 4.6. The 2.4% per-step advantage CFS holds does not come close to paying for
# that. See the Perlmutter Filesystem Matrix artifact.
PATH_MAP = {
    "/global/cfs/cdirs/e3smdata/simulations/v3.LR.historical_0101.aigo/run": (
        "/pscratch/sd/m/mahf708/v3.LR.historical_0101.aigo/run"
    ),
    "/global/cfs/cdirs/e3smdata/emulator/SamudrACE-E3SMv3/historical/landfrac5d": (
        "/pscratch/sd/m/mahf708/e3sm-hist-inputs/landfrac5d"
    ),
    "/global/cfs/cdirs/e3smdata/emulator/SamudrACE-E3SMv3/historical/landfrac1d": (
        "/pscratch/sd/m/mahf708/e3sm-hist-inputs/landfrac1d"
    ),
    "/global/cfs/cdirs/e3smdata/emulator/SamudrACE-E3SMv3/historical/stats-2026-08-13": (
        "/pscratch/sd/m/mahf708/e3sm-hist-inputs/stats-2026-08-13"
    ),
}

# Where stage-1 output lives. One run has exactly one owner, and E02/E03/E05 are
# not mine -- reading a parent checkpoint out of someone else's scratch is fine,
# writing anywhere but my own is not.
PARENT_ROOTS = [
    pathlib.Path("/pscratch/sd/m/mahf708/aug26"),
    pathlib.Path("/pscratch/sd/o/olawale/aug26"),
    pathlib.Path("/pscratch/sd/r/rebassoo/aug26"),
]

# The experiments the campaign is taking forward.
#
# E01 is the atmosphere baseline. It was not in the original stage-2 list, but
# every atmosphere arm is scored against it, and a stage-2 arm compared to a
# stage-1 baseline measures the fine-tune rather than the ablation. E11 is
# already here and plays the same role for the ocean.
ELIGIBLE_EXPERIMENTS = {
    "atm": {"E01", "E02", "E03", "E05", "E07", "E08"},
    "ocn": {"E11", "E12", "E13"},
}

# Rollout mixes. The atmosphere's is the ERA5 reference recipe verbatim; at
# 6-hourly, 20 steps is 5 days. The ocean's is the same *shape* -- mass on the
# short draw, a thin tail -- but anchored at 4 rather than 1, because stage 1
# already trains the ocean at four forward steps and a mix starting at 1 would
# be a regression rather than a fine-tune. At 5-daily, 20 steps is 100 days.
#
# Expected rollout length is what sets the cost: 3.0 steps for the atmosphere
# against stage 1's 1, and 7.0 for the ocean against stage 1's 4.
FORWARD_STEP_MIX = {
    "atm": [(1, 0.6), (2, 0.2), (4, 0.1), (12, 0.05), (20, 0.05)],
    "ocn": [(4, 0.6), (8, 0.2), (12, 0.1), (16, 0.05), (20, 0.05)],
}

# Cost tracks the *maximum* of the mix, not its mean. Three things are sized
# from the max: the loader window is `max + n_ic` timesteps for every sample,
# and inside train_on_batch both `get_forward_data` and `forcing_deriver` run
# over the whole window before `n_loss_steps` is drawn. Only the rollout itself
# is short when a short draw comes up, so the 20-step tail carrying 5% of the
# probability costs almost as much as running every batch at 20.
#
# That made the flat ERA5 mix look unaffordable until `time_buffer` was put
# back. Measured 2026-09-06, 4 nodes, atmosphere, 8213 batches/epoch:
#
#     stage-1 reference   window  2, tb 10   0.85 s/step    1.94 h/epoch
#     flat mix max 20     window 21, tb  0   4.97 s/step   11.3  h/epoch
#     flat mix max 20     window 21, tb 10   1.76 s/step    4.0  h/epoch
#
# The stage-1 row reproduces the campaign's documented 2.12 h/epoch, which is
# what makes the other two trustworthy. It is not I/O in the filesystem sense:
# GPU utilisation measured 88.3% during the tb-0 run.
#
# `time_buffer` does NOT lengthen the epoch. It pre-loads windows of
# `n_timesteps + time_buffer` and draws `time_buffer + 1` sub-windows from each,
# and those windows overlap so that no samples are skipped -- both runs above
# report ~8215 batches. What it costs is ordering: independent data is seen only
# every `time_buffer + 1` batches, which `time_buffer_pool_size: 2` partly
# decorrelates. That is the same trade stage 1 already made, and at 2.8x it is
# clearly worth it.
#
# So the atmosphere keeps the flat ERA5 mix rather than a rollout curriculum.
# A curriculum would be cheaper still, but there is no evidence here that it
# converges as well, and stage 2 feeds the coupled fine-tune -- not the place to
# deviate from the validated recipe to save hours that time_buffer already saved.
#
# The ocean gets neither: EXPERIMENTS.md records that time_buffer at a 5-timestep
# window was killed by the host OOM killer there ("Do not add it to the ocean"),
# and at 1.45 -> 3.44 s/step its rollout cost needs no help.
ATM_TIME_BUFFER = 10

STAGE2_EPOCHS = {"atm": 20, "ocn": 40}
PARENT_FINAL_EPOCH = {"atm": 30, "ocn": 150}

# Inline inference. Stage 2 is the last single-realm stage before the coupled
# fine-tune, so its checkpoint is close to a deliverable and the metric that
# picks it has to mean something.
#
# The record is 1940-01 to 2065-01. The atmosphere trains on 1940-1990 and
# 2000-2040, so it has two disjoint out-of-sample regions: the 1990s decade it
# never saw, and everything after 2040.
#
#   heldout_1990s   16 ICs across 1990-1994, 5-year rollouts, every one of them
#                   ending inside the gap. Interpolation -- the forcings sit
#                   inside the training range -- which makes it clean enough to
#                   select on, and it *replaces* the stage-1 selector, which was
#                   16 ICs spread 1940-2027 and therefore largely in-sample.
#   future_2040     extrapolation: CO2 and aerosol levels beyond anything in
#                   training. Ten years rather than five, because a slow drift
#                   is what breaks a coupled run and five years does not show
#                   it. Reported at weight 0 and never selected on -- which is
#                   also what keeps "nothing after 2040 influenced the model"
#                   true, rather than nearly true.
#
# Both realms get both blocks. The ocean's stage-1 subsets are the same two
# windows as the atmosphere's -- they live inside each merge/concat member
# rather than at the top level, which is easy to misread as "no split" -- so
# the two realms are already time-matched and stage 3 can pair their
# checkpoints without one of them having seen the other's held-out decade.
#
# What this buys, stated so it can be claimed: no gradient ever touched
# 1990-2000 or 2040-2065, in either realm; nothing after 2040 entered any
# selection metric; and 2055-2065 is touched by nothing at all -- not training,
# not validation, not inference -- so it is a locked test set. Verified by
# check_stage2_leakage.py against the generated configs rather than asserted.
#
# Cost: inference wall clock is ceil(n_ICs / n_ranks) x n_steps, so only rollout
# length and firing frequency are real levers.
# Initial conditions are sharded across ranks, and
# InlineInferenceConfig.__post_init__ requires n_initial_conditions %
# world_size == 0. That check calls Distributed.get_instance(), so on a login
# node world_size is 1 and *any* count validates -- validate_config cannot
# catch a violation, and dacite swallows the resulting ValueError into an
# unhelpful "can not match type list to union inference". The B32 atmosphere
# arm has 32 ranks and died on exactly this. So the count is derived from the
# arm's rank count rather than fixed, and asserted at generation time.
IC_FLOOR = 16


def ic_count(ranks):
    """Smallest multiple of `ranks` that is at least IC_FLOOR."""
    return max(IC_FLOOR, -(-IC_FLOOR // ranks) * ranks)


def _month_grid(first_year, last_year):
    return [(y, m) for y in range(first_year, last_year + 1) for m in range(1, 13)]


def _atm_ics(n, first_year, last_year):
    """`n` initial conditions spread evenly over a range of whole years.

    Day 3 at 12:00 exists on the atmosphere's 6-hourly axis in every month, so
    these are safe without consulting the record; the ocean's 5-day axis is not
    that forgiving, which is why its lists are literal and verified.
    """
    grid = _month_grid(first_year, last_year)
    if n > len(grid):
        raise ValueError(f"cannot place {n} ICs in {len(grid)} months")
    step = len(grid) / n
    picks = [grid[int(i * step)] for i in range(n)]
    return [f"{y}-{m:02d}-03T12:00:00" for y, m in picks]


# The atmosphere's two windows. Held-out ICs must leave room for a 5-year
# rollout inside the 1990-2000 gap, so they stop in 1994; future ICs must leave
# room for a 10-year rollout before the locked window opens in 2055, so they
# stop in 2043.
# 1990-1993 rather than through 1994: 48 months, so the n=16 case lands on
# exact quarters and reproduces the list the first B16 runs were launched
# with. Changing a running job's evaluation ICs on requeue would make its
# own epochs incomparable on the metric that selects its checkpoint.
ATM_HELDOUT_YEARS = (1990, 1993)
ATM_FUTURE_YEARS = (2040, 2043)

# The ocean axis is 5-daily on a noleap calendar: 1990-01-01 + 5n days, exactly
# 73 stamps a year. These were picked off that grid and checked against the real
# time index -- TimestampList.as_indices raises on a stamp that does not exist,
# minutes into an allocation. Every ocean arm has 4, 8 or 16 ranks, so 16 ICs
# always divide; if one ever needs 32, these lists have to be regenerated
# against the real axis rather than interpolated.
OCN_HELDOUT_ICS = [
    "1990-01-01T00:00:00", "1990-05-01T00:00:00", "1990-08-29T00:00:00",
    "1990-12-27T00:00:00", "1991-05-01T00:00:00", "1991-08-29T00:00:00",
    "1991-12-27T00:00:00", "1992-04-26T00:00:00", "1992-08-29T00:00:00",
    "1992-12-27T00:00:00", "1993-04-26T00:00:00", "1993-08-24T00:00:00",
    "1993-12-27T00:00:00", "1994-04-26T00:00:00", "1994-08-24T00:00:00",
    "1994-12-27T00:00:00",
]
OCN_FUTURE_ICS = [
    "2040-01-01T00:00:00", "2040-04-06T00:00:00", "2040-07-10T00:00:00",
    "2040-10-18T00:00:00", "2041-01-21T00:00:00", "2041-05-01T00:00:00",
    "2041-08-04T00:00:00", "2041-11-07T00:00:00", "2042-02-15T00:00:00",
    "2042-05-21T00:00:00", "2042-08-29T00:00:00", "2042-12-02T00:00:00",
    "2043-03-07T00:00:00", "2043-06-15T00:00:00", "2043-09-18T00:00:00",
    "2043-12-27T00:00:00",
]

# The 32-rank variants, for the coupled stage, which trains at a global batch of
# 32 and therefore shards its inference across 32 ranks. Built the same way and
# subject to the same rule: selected off the real 5-day axis, never interpolated.
#
# Each is a strict superset of the 16-IC list above -- the 16 originals plus the
# midpoints between them, snapped to the real axis -- so a 32-IC score and a
# 16-IC score share half their initial conditions instead of none. The span is
# deliberately unchanged: extending past 1994-12-27 would push a 365-step
# heldout rollout into the 2000+ training window, which the leakage gate
# rejects. Filling to exactly 32 inside a fixed span puts two pairs closer
# together than the rest; coverage matters here, even spacing does not.
OCN_HELDOUT_ICS_32 = [
    "1990-01-01T00:00:00", "1990-03-02T00:00:00", "1990-05-01T00:00:00",
    "1990-06-30T00:00:00", "1990-08-29T00:00:00", "1990-10-28T00:00:00",
    "1990-12-27T00:00:00", "1991-02-25T00:00:00", "1991-03-12T00:00:00",
    "1991-05-01T00:00:00", "1991-06-30T00:00:00", "1991-08-29T00:00:00",
    "1991-10-28T00:00:00", "1991-12-27T00:00:00", "1992-02-25T00:00:00",
    "1992-04-26T00:00:00", "1992-06-25T00:00:00", "1992-08-29T00:00:00",
    "1992-10-28T00:00:00", "1992-12-27T00:00:00", "1993-02-25T00:00:00",
    "1993-04-26T00:00:00", "1993-06-25T00:00:00", "1993-08-24T00:00:00",
    "1993-10-23T00:00:00", "1993-12-27T00:00:00", "1994-02-25T00:00:00",
    "1994-04-26T00:00:00", "1994-06-25T00:00:00", "1994-08-24T00:00:00",
    "1994-10-23T00:00:00", "1994-12-27T00:00:00",
]

OCN_FUTURE_ICS_32 = [
    "2040-01-01T00:00:00", "2040-02-15T00:00:00", "2040-02-25T00:00:00",
    "2040-04-06T00:00:00", "2040-05-21T00:00:00", "2040-07-10T00:00:00",
    "2040-08-29T00:00:00", "2040-10-18T00:00:00", "2040-12-02T00:00:00",
    "2041-01-21T00:00:00", "2041-03-12T00:00:00", "2041-05-01T00:00:00",
    "2041-06-15T00:00:00", "2041-08-04T00:00:00", "2041-09-18T00:00:00",
    "2041-11-07T00:00:00", "2041-12-27T00:00:00", "2042-02-15T00:00:00",
    "2042-04-01T00:00:00", "2042-05-21T00:00:00", "2042-07-10T00:00:00",
    "2042-08-29T00:00:00", "2042-10-13T00:00:00", "2042-12-02T00:00:00",
    "2043-01-16T00:00:00", "2043-03-07T00:00:00", "2043-04-26T00:00:00",
    "2043-06-15T00:00:00", "2043-07-30T00:00:00", "2043-09-18T00:00:00",
    "2043-11-07T00:00:00", "2043-12-27T00:00:00",
]

# Which pair a run gets is set by its rank count, not by hand: the ICs are
# sharded across ranks and InlineInferenceConfig requires the count to divide
# evenly. The 16-rank lists are left exactly as they were so that every ocean
# stage-2 run already scored against them stays comparable to itself.
OCN_ICS = {
    16: (OCN_HELDOUT_ICS, OCN_FUTURE_ICS),
    32: (OCN_HELDOUT_ICS_32, OCN_FUTURE_ICS_32),
}

STEPS_PER_YEAR = {"atm": 1460, "ocn": 73}

# Sixteen initial conditions in every block, both realms. Not a cost choice: the
# ICs are sharded across ranks, so the count has to be a multiple of the rank
# count, and 16 is the only value that covers the 4-, 8- and 16-rank arms at
# once. Cost is ceil(n_ICs / n_ranks) x n_steps, so at B16 the atmosphere's 16
# ranks take them in a single wave and cutting the count would idle ranks for
# identical wall clock; the ocean's 8 ranks take two waves.
INFERENCE_PLAN = {
    "atm": [
        # name, weight, rollout years, epoch schedule
        ("heldout_1990s", 1.0, 5, {"start": 3, "step": 3}),
        ("future_2040", 0.0, 10, {"start": 10, "step": 10}),
    ],
    "ocn": [
        ("heldout_1990s", 1.0, 5, {"start": 5, "step": 5}),
        ("future_2040", 0.0, 10, {"start": 20, "step": 20}),
    ],
}


CAMPAIGN = "aug26"
FT_SUFFIX = "FT"

# Stage-2 output. A separate root from aug26/, so nothing stage 2 does can land
# in a stage-1 run directory even if a run id somehow collided, and firmly in my
# own scratch regardless of who owns the parent.
OUTPUT_ROOT = pathlib.Path("/pscratch/sd/m/mahf708/aug26-ft")


def _mix(outcomes):
    return {"outcomes": [{"steps": k, "probability": pr} for k, pr in outcomes]}


def remap_paths(node):
    """Rewrite every CFS input path to its $PSCRATCH copy, in place."""
    if isinstance(node, dict):
        return {k: remap_paths(v) for k, v in node.items()}
    if isinstance(node, list):
        return [remap_paths(v) for v in node]
    if isinstance(node, str):
        for cfs, scr in PATH_MAP.items():
            if node == cfs or node.startswith(cfs + "/"):
                return scr + node[len(cfs) :]
    return node


def parse_runid(runid):
    """`E12.aug26.ocn.A0_..._X0.S01` -> ('E12', 'aug26', 'ocn', 'A0_..._X0', 'S01')."""
    parts = runid.split(".")
    if len(parts) != 5:
        raise ValueError(f"{runid}: expected 5 dot-separated fields, got {len(parts)}")
    return tuple(parts)


def child_runid(parent_runid):
    """`E12.aug26.ocn...` -> `E12-FT.aug26.ocn...`.

    The stage goes in the experiment field, not the campaign field: these are
    the same data and the same arms as their parents, so `aug26` stays true.
    A two-letter suffix there has precedent -- RF02 already runs alongside E01
    -- and `-` sorts before `.`, so E12-FT lands immediately next to E12 in any
    listing rather than in a separate FT block at the end.
    """
    exp, campaign, realm, word, seed = parse_runid(parent_runid)
    if campaign != CAMPAIGN:
        raise ValueError(f"{parent_runid}: not an {CAMPAIGN} run")
    return ".".join([f"{exp}-{FT_SUFFIX}", campaign, realm, word, seed])


def parent_runid_of(runid):
    exp, campaign, realm, word, seed = parse_runid(runid)
    return ".".join([exp.removesuffix(f"-{FT_SUFFIX}"), campaign, realm, word, seed])


def find_parent(runid):
    for root in PARENT_ROOTS:
        d = root / runid
        if (d / "config.yaml").is_file():
            return d
    return None


def parent_checkpoint(parent_dir, realm):
    """The final-epoch checkpoint, or None if the parent has not got there."""
    epoch = PARENT_FINAL_EPOCH[realm]
    ckpt = parent_dir / "training_checkpoints" / f"ckpt_{epoch:04d}.tar"
    return ckpt if ckpt.is_file() else None


def candidates():
    """Every stage-1 run id in runs/ whose experiment is being taken forward."""
    out = []
    for path in sorted(RUNS.glob("*.yaml")):
        runid = path.stem
        try:
            exp, campaign, realm, _word, _seed = parse_runid(runid)
        except ValueError:
            continue
        if campaign != CAMPAIGN:
            continue
        if exp in ELIGIBLE_EXPERIMENTS.get(realm, ()):
            out.append(runid)
    return out


def build_config(parent_dir, realm, ckpt, runid, ranks):
    cfg = copy.deepcopy(yaml.safe_load((parent_dir / "config.yaml").read_text()))
    cfg = remap_paths(cfg)

    st = cfg["stepper_training"]

    # The one substantive change: a stochastic rollout length instead of a fixed
    # one. `optimize_last_step_only` is inherited, not set here -- it is true for
    # the atmosphere and false for the ocean, and that difference is a property
    # of the arm rather than of the stage.
    st["n_forward_steps"] = _mix(FORWARD_STEP_MIX[realm])

    # Weights only. Optimizer, EMA and scheduler start fresh, which is what the
    # reference recipe does: the parent's Adam moments were accumulated against
    # a one-step objective and are not the right preconditioner for this one.
    st["parameter_init"] = {"weights_path": str(ckpt)}

    cfg["max_epochs"] = STAGE2_EPOCHS[realm]

    # `time_buffer` is dead weight here and a host-memory hazard. It amortized
    # one 12-timestep read over ~11 single-step samples; a 21-timestep window
    # feeds one sample, so the buffer buys nothing and multiplies in-flight
    # host memory by the window growth. The ocean was already killed by the OOM
    # killer with time_buffer at a 5-timestep window (see EXPERIMENTS.md).
    for loader in _all_train_loaders(cfg):
        if realm == "atm":
            # Keep the stage-1 setting: 4.97 -> 1.76 s/step at the 21-timestep
            # window, with the epoch the same length either way. Host memory
            # measured on the compute nodes at the 31-timestep input window:
            # 122 GB used, 128 GB still free of 251.
            loader["time_buffer"] = ATM_TIME_BUFFER
            loader["time_buffer_pool_size"] = 2
        else:
            loader.pop("time_buffer", None)
            loader.pop("time_buffer_pool_size", None)
        # Windows are ~10x (atm) / ~4x (ocn) their stage-1 size, and
        # prefetch_factor multiplies what each of 8 workers holds in flight.
        loader["prefetch_factor"] = 2

    # Validation is sized from the *training* window requirements
    # (`_get_train_window_data_requirements`), so it now reads 21-timestep
    # windows too, and `evaluate_all_steps` defaults to true -- which would
    # score all 20 forward steps of every validation batch and turn a 4-minute
    # validation into an hour. False evaluates only the steps the train stepper
    # drew, which is what the flag exists for.
    for entry in _validation_entries(cfg):
        entry["evaluate_all_steps"] = False

    # Rebuild the inference blocks from the plan. The stage-1 blocks are used as
    # templates -- they carry the aggregator settings, the dataset definition
    # and forward_steps_in_memory, none of which change -- but name, weight,
    # rollout length, schedule and initial conditions are all replaced.
    template = (cfg.get("inference") or [None])[0]
    if template is None:
        raise ValueError("parent config has no inference block to use as a template")
    n_ics = ic_count(ranks)
    if n_ics % ranks:
        raise ValueError(f"{runid}: {n_ics} ICs do not divide {ranks} ranks")
    blocks = []
    for name, weight, years, epochs in INFERENCE_PLAN[realm]:
        if realm == "atm":
            years_range = (
                ATM_HELDOUT_YEARS if name == "heldout_1990s" else ATM_FUTURE_YEARS
            )
            ics = _atm_ics(n_ics, *years_range)
        else:
            ics = OCN_HELDOUT_ICS if name == "heldout_1990s" else OCN_FUTURE_ICS
            if len(ics) % ranks:
                raise ValueError(
                    f"{runid}: the ocean {name} list has {len(ics)} ICs, which does "
                    f"not divide {ranks} ranks -- regenerate it against the real "
                    "5-day axis, do not interpolate"
                )
        block = copy.deepcopy(template)
        block["name"] = name
        block["weight"] = weight
        block["n_forward_steps"] = years * STEPS_PER_YEAR[realm]
        block["epochs"] = dict(epochs)
        block["loader"]["start_indices"] = {"times": list(ics)}
        blocks.append(block)
    cfg["inference"] = blocks

    # Read from the parent wherever it lives, write only into my own scratch.
    # `experiment_dir` is inherited from the parent's config, so for E02/E03/E05
    # it arrives pointing at olawale's directory. The sbatch wrapper does
    # override it via FME_OVERRIDE_ARGS, but a config that names someone else's
    # output directory is one bypassed wrapper away from writing there, and
    # nobody reading the file can tell it is not what runs.
    cfg["experiment_dir"] = str(OUTPUT_ROOT / runid)

    _assert_writes_stay_home(cfg, runid)
    return cfg


def _assert_writes_stay_home(cfg, runid):
    """Every path the run writes to must be under OUTPUT_ROOT.

    The only path that legitimately points outside it is the parent checkpoint,
    which is read once at startup. Everything else -- output directory, and any
    future field that names a destination -- has to be mine.
    """
    weights = cfg["stepper_training"]["parameter_init"]["weights_path"]
    out = cfg["experiment_dir"]
    if not out.startswith(str(OUTPUT_ROOT) + "/"):
        raise ValueError(f"{runid}: experiment_dir {out} is outside {OUTPUT_ROOT}")
    for owner in ("/pscratch/sd/o/olawale", "/pscratch/sd/r/rebassoo"):
        for key, value in _walk_strings(cfg):
            if value.startswith(owner) and value != weights:
                raise ValueError(
                    f"{runid}: {key} points into {owner} but is not the parent "
                    f"checkpoint: {value}"
                )


def _walk_strings(node, prefix=""):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _walk_strings(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _walk_strings(v, f"{prefix}[{i}]")
    elif isinstance(node, str):
        yield prefix, node


def _validation_entries(cfg):
    v = cfg["validation"]
    return v if isinstance(v, list) else [v]


def _all_train_loaders(cfg):
    loaders = [cfg["train_loader"]]
    loaders += [e["loader"] for e in _validation_entries(cfg)]
    return loaders


def build_env(parent_runid, runid, realm, ckpt):
    exp, _campaign, _realm, word, seed = parse_runid(runid)
    parent_env = (RUNS / f"{parent_runid}.env").read_text().splitlines()
    keep = {}
    for line in parent_env:
        if "=" in line and not line.startswith("#"):
            k, _, v = line.partition("=")
            keep[k] = v
    tags = keep.get("WANDB_TAGS", "").split(",")
    tags = [t for t in tags if t and not t.startswith("P")]
    tags = ["ft", "stage2"] + tags
    nodes = keep.get("FME_NODES", "?")
    ranks = keep.get("FME_RANKS", "?")
    return "\n".join(
        [
            "# generated by make_stage2_config.py -- multi-step fine-tune",
            f"# parent {parent_runid}",
            f"# weights {ckpt}",
            f"FME_NODES={nodes}",
            f"FME_RANKS={ranks}",
            "FME_PRIORITY=1",
            f"WANDB_NAME={runid}",
            f"WANDB_RUN_GROUP={CAMPAIGN}.{realm}.{exp}",
            f"WANDB_JOB_TYPE={word}_FT",
            f"WANDB_TAGS={','.join(tags)},{seed}",
            f'WANDB_NOTES="stage-2 multi-step fine-tune of {parent_runid} '
            f'| {nodes} nodes, {ranks} ranks"',
            "",
        ]
    )


def _ranks_from_env(parent_runid):
    """The arm's rank count, from the parent's .env -- the same number
    run-train.sh sizes the job with, so the ICs cannot disagree with the job."""
    for line in (RUNS / f"{parent_runid}.env").read_text().splitlines():
        if line.startswith("FME_RANKS="):
            return int(line.split("=", 1)[1])
    raise ValueError(f"{parent_runid}.env has no FME_RANKS")


def emit(parent_runid, force=False):
    _exp, _campaign, realm, _word, _seed = parse_runid(parent_runid)
    runid = child_runid(parent_runid)
    parent_dir = find_parent(parent_runid)
    if parent_dir is None:
        return runid, "no stage-1 output found in any campaign root"
    ckpt = parent_checkpoint(parent_dir, realm)
    if ckpt is None and not force:
        want = PARENT_FINAL_EPOCH[realm]
        have = sorted((parent_dir / "training_checkpoints").glob("ckpt_[0-9]*.tar"))
        at = have[-1].stem.split("_")[-1].lstrip("0") if have else "0"
        return runid, f"parent is at epoch {at}, needs {want}"
    if ckpt is None:
        ckpt = parent_dir / "training_checkpoints" / "ckpt.tar"

    ranks = _ranks_from_env(parent_runid)
    cfg = build_config(parent_dir, realm, ckpt, runid, ranks)
    (RUNS / f"{runid}.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False, width=100)
    )
    (RUNS / f"{runid}.env").write_text(build_env(parent_runid, runid, realm, ckpt))
    return runid, None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runids", nargs="*", help="stage-1 run ids; default is --all")
    ap.add_argument("--all", action="store_true", help="every eligible stage-1 run")
    ap.add_argument("--list", action="store_true", help="report eligibility only")
    ap.add_argument(
        "--force",
        action="store_true",
        help="use the live ckpt.tar when the parent has not reached its final epoch",
    )
    args = ap.parse_args()

    targets = args.runids or candidates()
    if args.list:
        for parent_runid in targets:
            _exp, _c, realm, _w, _s = parse_runid(parent_runid)
            parent_dir = find_parent(parent_runid)
            if parent_dir is None:
                state = "no output"
            elif parent_checkpoint(parent_dir, realm) is None:
                have = sorted(
                    (parent_dir / "training_checkpoints").glob("ckpt_[0-9]*.tar")
                )
                at = have[-1].stem.split("_")[-1].lstrip("0") if have else "0"
                state = f"epoch {at}/{PARENT_FINAL_EPOCH[realm]}"
            else:
                state = "READY"
            owner = parent_dir.parts[3] if parent_dir else "-"
            print(f"{state:<16} {owner:<9} {parent_runid}")
        return 0

    if not args.runids and not args.all:
        ap.error("give run ids, or --all, or --list")

    written, skipped = [], []
    for parent_runid in targets:
        runid, why = emit(parent_runid, force=args.force)
        (skipped if why else written).append((runid, why))
    for runid, _ in written:
        print(f"wrote runs/{runid}.yaml + .env")
    for runid, why in skipped:
        print(f"skip  {runid}: {why}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
