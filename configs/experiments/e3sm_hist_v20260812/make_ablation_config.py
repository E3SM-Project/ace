#!/usr/bin/env python3
"""Emit every config for the 2026-08-31 hackathon campaign from the baselines.

Source of truth is the hackathon page
https://e3sm.atlassian.net/wiki/spaces/p3ai/pages/6550683662 . The run list,
the factor alphabet and the naming convention below are transcribed from it;
if the page changes, change `RUNLIST` and `FACTOR_DOC` here and regenerate.

What changed on 2026-08-29
--------------------------
The campaign used to be a finetune chain: every run after a trunk started from
its parent's checkpoint, so this script's main job was enforcing that a child's
channel lists were a prefix-superset of its parent's. The page now specifies
independent from-scratch runs -- "ALL EXPERIMENTS start with batch_size=16 as
baseline" -- so there is no parent, no `weights_path`, and no prefix rule. Runs
are siblings, they all start at once, and the only thing that varies between
them is the factor word.

Naming
------
    <exp>.<date>.<realm>.<tuning_set>.S<seed>
    E05.aug26.atm.A3_B16_C1_O5_W0_X0.S01

The experiment prefix is a single incrementing `E` sequence -- E01-E10 are the
atmosphere, E11-E14 the ocean -- rather than the A??/O?? it started as. `A` and
`O` are both factor letters, so `A05...A3_...` and `O01...O5_...` overloaded the
same character twice inside one run id, and a coupled `C??` would have collided
with the CO2 factor as well. The realm is already its own field, so the prefix
did not need to carry it. Old -> new: A01-A10 -> E01-E10, O01-O04 -> E11-E14.

The tuning set is a fixed-order factor word, `A?_B??_C?_O?_W?_X?`:

    A0 no aerosol            A1 aerosol inputs        A2 aerosol outputs
    A3 both
    B08/B16/B32 batch size (global; see "Sizing")
    C0 no CO2                C1 with CO2
    L0 baseline lr           L1 lr x sqrt(batch / 16)
    O1 1-daily ocean step    O5 5-daily ocean step
    W0 equal weights         W1 flux upweight
    W2 away-from-surface dilution
    W3 zero a structurally-zero / near-static channel
    W4 zero a trend-dominated near-zero channel
    X0 baseline              X1 AMP (bf16 autocast)

Sizing
------
`batch_size` is global and is split across ranks by `dist.local_batch_size`.
Both baselines are built for a fixed local batch, so the rank count follows
from the batch size and nothing else has to change:

    atm  local batch 1 -> ranks = batch_size      -> nodes = batch_size / 4
    ocn  local batch 2 -> ranks = batch_size / 2  -> nodes = batch_size / 8

Two divisibility rules then have to hold, and this script enforces both rather
than letting them surface minutes into an allocation as
`UnionMatchError: can not match type "list"`:

  * `validation.loader.batch_size` must be divisible by the rank count. It is
    set equal to `batch_size`.
  * every inference block's initial-condition count must be divisible by the
    rank count. The baseline ships 16 ICs, which covers 4/8/16 ranks; the
    atmosphere's B32 run needs 32, so the list is rewritten (a dotlist override
    cannot index into a yaml list).

It also recomputes `12yr_test`'s `epochs.start`. That block fires on
`range(max_epochs + 1)[start::step]`, so a `start` chosen for one run length
silently stops scoring the final epoch at another.

Usage
-----
    ./make_ablation_config.py --all               # the whole campaign
    ./make_ablation_config.py --exp E05           # one experiment's runs
    ./make_ablation_config.py --list              # print the run list, write nothing
    ./make_ablation_config.py --all --dry-run     # check sizing, write nothing

Regenerate through sbatch-scripts/generate-campaign.sh, which passes the
campaign root and owner consistently.
"""

import argparse
import copy
import dataclasses
import pathlib
import sys

import yaml

HERE = pathlib.Path(__file__).resolve().parent
CAMPAIGN = "aug26"

# ---------------------------------------------------------------- channels --

CO2_IN = ["global_mean_co2"]
# The page writes A1 as "(aerindex, ccn)" and A3 as "(aerindex, aod, lwp, lcc,
# cdnc)". Those two are inconsistent -- A3 drops ccn and adds aod -- so A3 is
# implemented here as A1's inputs plus A2's outputs, which is what makes E03,
# E04 and E05 a clean 2x2 and what "A3: with both aerosol inputs and outputs"
# says one line earlier in the same list. AODVISall does exist in the stats;
# `--aod` adds it if the page is meant literally. See EXPERIMENTS.md "Open
# questions for the page".
AEROSOL_IN = ["aerindexall", "colccn.3"]
AEROSOL_OUT = ["lwp", "lcc", "cdnc"]
AOD_IN = ["AODVISall"]

A_LEVELS = {
    "0": (False, False),
    "1": (True, False),
    "2": (False, True),
    "3": (True, True),
}

# --------------------------------------------------------- loss reweighting --

# Which end of the index range is the surface. These are OPPOSITE between the
# realms and are easy to get backwards, so they are stated once and were
# verified against the centering statistics:
#   atm  T_0 = 220.9 K, T_7 = 277.5 K            -> index 0 is top of atmosphere
#   ocn  temperature_0 = 13.6 C, _18 = 0.7 C     -> index 0 is the surface
SURFACE_AT_LOW_INDEX = {"atm": False, "ocn": True}
MIN_LEVELS_FOR_PROFILE = 4
W2_SPAN = 4.0  # surface weight / far-end weight, mean-normalized to 1.0

# W1, "upweight fluxes in both models, e.g., TOA flux, surface fluxes".
# The atmosphere predicts its fluxes, so this is literal. The ocean predicts no
# fluxes at all -- they are all inputs -- so the ocean's W1 upweights the
# air-sea interface state instead, which is the fluxes' fingerprint and the
# part the coupled model consumes.
W1_UPWEIGHT = 2.0
W1_NAMES = {
    "atm": [
        "LHFLX", "SHFLX", "FLUS", "FLUT", "FLDS", "FSNS", "FSUTOA",
        "TAUX", "TAUY", "DTENDTTW",
        "surface_precipitation_rate", "frozen_precipitation_rate",
    ],
    "ocn": ["sst", "ssh", "ocean_sea_ice_fraction", "iceVolumeTotal"],
}

# W3 and W4 both zero one "poor" channel's loss weight. They are a matched pair
# probing two DIFFERENT reasons a channel can be a poor loss citizen, each picked
# against the training statistics rather than by eye.
#
# W4 -- trend-dominated and near zero. `STW_0` (top-of-atmosphere specific
# humidity) has residual/full-field scale 0.031, the second lowest of any
# atmosphere output after `PS`, and |mean|/std 12.8: almost all of its spread is
# the secular stratospheric-water trend rather than anything predictable
# step-to-step. The ocean analogue is `velocityMeridionalCoarsened_18`, the
# deepest meridional velocity and the most extreme ocean output by |mean|/std
# (0.005). Named for atm by the page; the ocean pick is ours.
#
# W3 -- the neighbouring level, and structural zeros. `STW_1` sits just below
# STW_0 at residual/full 0.70, and the hand-tuned weight set that predates this
# campaign singled out exactly `STW_0` and `STW_1` (both at 0.25), so the pair
# answers how far up the stratospheric-water column zeroing stays harmless. For
# the ocean, `iceVolumeTotal` is structurally zero across most of the domain and
# is already special-cased by the corrector's `zero_where_ice_free_names`, so
# most of its loss is trivially satisfiable and the rest is concentrated at the
# ice edge.
#
# Deliberately NOT used: `FSNS`, `FSUTOA`, `SHFLX` and `DTENDTTW` all have
# residual/full-field scale above 1.0, which looks like "unpredictable" but for
# the first three is just the diurnal cycle, which the model can resolve from
# `SOLIN`. `DTENDTTW` would also confound the moisture-budget corrector, which
# consumes it.
W3_ZERO_NAMES = {
    "atm": ["STW_1"],
    "ocn": ["iceVolumeTotal"],
}
W4_ZERO_NAMES = {
    "atm": ["STW_0"],
    "ocn": ["velocityMeridionalCoarsened_18"],
}
ZERO_NAMES = {"3": W3_ZERO_NAMES, "4": W4_ZERO_NAMES}

# ------------------------------------------------------------ learning rate --

# L0 holds the baseline learning rate at every batch size; L1 scales it by
# sqrt(batch / 16).
#
# Neither is "correct" and that is the point of running both. Linear scaling
# (Goyal et al. 2017) is derived for SGD with momentum; sqrt scaling
# (Krizhevsky 2014, Hoffer et al. 2017) keeps the gradient-noise level fixed,
# since the update variance goes as lr^2 / batch. Both realms use an
# Adam-family optimizer -- FusedAdam for the atmosphere, AdamW for the ocean --
# which normalizes by the gradient's second moment and so is already partly
# invariant to gradient scale; sqrt is the better prior there than linear.
#
# L0 is the campaign default because with one seed per batch arm, changing batch
# and learning rate together gives a result no one can attribute. L1 exists so
# that a null at B32 can be told apart from "B32 needed a bigger step": with
# B32-L0, B32-L1 and B16-L0 in hand the two explanations separate.
LR_REFERENCE_BATCH = 16


def learning_rate(base_lr: float, factors: "Factors") -> float:
    if factors.lr == "0":
        return base_lr
    if factors.lr != "1":
        raise SizingError(f"unknown learning-rate set L{factors.lr}")
    if factors.batch == LR_REFERENCE_BATCH:
        raise SizingError(
            f"L1 at B{LR_REFERENCE_BATCH:02d} is the same as L0 -- the scaling is "
            f"relative to batch {LR_REFERENCE_BATCH}, so it would be a duplicate run"
        )
    return base_lr * (factors.batch / LR_REFERENCE_BATCH) ** 0.5


# ----------------------------------------------------- inference conditions --

# 16 ICs cover 4, 8 and 16 ranks. The 32-rank (B32) atmosphere run needs 32, so
# the mid-year counterpart of each date is interleaved; the 12yr_test window is
# extended to quarterly starts over the same eight held-out years.
IC_TRAIN_WINDOW_32 = [
    f"{y}-{m}-03T12:00:00"
    for y in (1940, 1945, 1950, 1955, 1960, 1965, 1970, 1975,
              1977, 2000, 2005, 2010, 2015, 2020, 2025, 2027)
    for m in ("01", "07")
]
IC_TEST_WINDOW_32 = [
    f"{y}-{m}-03T12:00:00"
    for y in range(2040, 2048)
    for m in ("01", "04", "07", "10")
]

# ------------------------------------------------------------- the run list --

# Samples per rank. This is the single number that sets how many nodes the whole
# campaign needs, so it is a knob rather than a constant: at local batch 1 the
# atmosphere's B16 baseline is 4 nodes and the campaign is 129 nodes; at local
# batch 2 it is 2 nodes and the campaign is 75, which fits the 96-node
# reservation outright.
#
# Both values are MEASURED (2026-08-29): 19.0 GB/GPU at local batch 1 and
# 28.7 GB at 2, on 80 GB cards, so 2 fits with room to spare and is marginally
# better per sample. It is NOT the default anyway, because halving the ranks at
# fixed global batch also DOUBLES the epoch: 4 nodes and 2.11 h/epoch versus
# 2 nodes and 3.79 h/epoch. At local batch 1 the P1-P3 science finishes
# Wednesday night; at 2 nothing finishes before Friday morning. Pass
# `--local-batch atm=2` only if the group prefers all 35 runs concurrent.
# See EXPERIMENTS.md "Measurements".
# O1 vs O5. Both MPAS cadences exist in the run directory: the 5-day streams
# carry a `5D` suffix, the 1-day streams are the un-suffixed ones, 1501 files
# each, 1940-2065. Both are interval MEANS (time_bnds span 5 days and 1 day
# respectively), so switching cadence is a data swap, not a resample.
#
# What has to change together, which is why it lives here and not in a sed:
#   * all three MPAS file patterns lose the `5D`;
#   * LANDFRAC/sea_surface_fraction must be materialised on the matching axis --
#     `make_landfrac_ocn.py --cadence 1d` writes `landfrac1d.<year>.nc`, because
#     merge members have to share sample_start_times;
#   * every inference block's `n_forward_steps` scales x5 to cover the same
#     12-year rollout (876 -> 4380);
#   * an epoch holds 5x the samples, so `max_epochs` comes down to keep the run
#     inside the window (see DEFAULT_EPOCHS).
# The existing inference initial conditions need no change: the 5-day timestamps
# are a subset of the 1-day axis.
OCEAN_STREAMS = ("fmeDepthCoarsening", "fmeDerivedFields", "fmeSeaiceDerivedFields")
LANDFRAC_DIR = {
    "5": "/global/cfs/cdirs/e3smdata/emulator/SamudrACE-E3SMv3/historical/landfrac5d",
    "1": "/global/cfs/cdirs/e3smdata/emulator/SamudrACE-E3SMv3/historical/landfrac1d",
}
OCEAN_STEPS_PER_DAY = {"5": 1 / 5, "1": 1.0}

# ------------------------------------------------------------ inference cost --

# Inline inference is not cheap monitoring, it is a free-running rollout, and at
# the shipped 12-year length it cost 45% of an atmosphere epoch: 2.06 h of
# training against >=1.7 h of inference, measured on job 57775795. Two blocks
# are configured, so an epoch on which both fire spent more wall clock rolling
# out than training, and 30 epochs x 30 runs of that does not fit a 126 h
# reservation.
#
# Length is also what killed every atmosphere run on 2026-08-31. The window loop
# holds no collective, so the ranks drift apart freely and the all-reduce that
# ends the rollout absorbs the whole accumulated skew; at 876 windows that skew
# reached the 30 minute collective timeout. A shorter rollout shortens the drift
# in proportion, so it is the direct fix for the crash as well as for the cost.
# See EXPERIMENTS.md "The inline inference cost".
#
# Stated in physical units, because "5 years" means the same thing to the
# atmosphere at 6-hourly and to the ocean at 5-day and the step counts follow
# from it. Five rather than two or three: the aggregators reduce over whole
# years, and a rollout wants enough of them to say something about drift.
#
# What is NOT lost: validation still runs every epoch at ~4 minutes, so the
# per-epoch loss curve and best-validation checkpoint selection are untouched.
# What drops is the per-epoch ROLLOUT score -- and every epoch's weights are
# saved (checkpoint_save_epochs step 1), so any skipped rollout can be run
# offline afterwards without holding a reservation open for it.
INFERENCE_YEARS = 5
STEPS_PER_YEAR = {"atm": 1460, "ocn": 73}  # 6-hourly; 5-day. O1 is scaled x5.

# Evaluations per run rather than epochs between them. E11 (150 epochs at 5-day)
# and E17 (30 epochs at 1-day) are sample-matched by construction, so a fixed
# epoch stride would score one of them five times as often as the other and make
# the two curves incomparable. Six points, always including the final epoch.
INFERENCE_EVALUATIONS = 6

# Ocean only, halved from 20 on request (2026-08-31). This bounds how many
# forward steps are held between loader reads: it lowers peak memory and RAISES
# the number of reads for a given rollout, so it is a memory knob and not a cost
# one. Watch the ocean rollout wall clock after this change rather than assuming
# it went down.
OCEAN_FORWARD_STEPS_IN_MEMORY = 10


LOCAL_BATCH = {"atm": 1, "ocn": 2}
GPUS_PER_NODE = 4
# Keyed by realm, and for the ocean by cadence: a 1-day epoch holds 5x the
# samples of a 5-day one, so equal wall clock means a fifth of the epochs.
DEFAULT_EPOCHS = {"atm": 30, "ocn": 150, "ocn-O1": 30}


@dataclasses.dataclass(frozen=True)
class Factors:
    """The tuning set, in the page's fixed order A_B_C_O_W_X."""

    aerosol: str = "0"
    batch: int = 16
    co2: str = "0"
    lr: str = "0"
    ocean_step: str = "5"
    weights: str = "0"
    amp: str = "0"

    def word(self) -> str:
        return (
            f"A{self.aerosol}_B{self.batch:02d}_C{self.co2}_L{self.lr}"
            f"_O{self.ocean_step}_W{self.weights}_X{self.amp}"
        )


@dataclasses.dataclass(frozen=True)
class Experiment:
    exp: str
    realm: str
    factors: Factors
    note: str
    seeds: tuple[int, ...] = (1,)
    # Extra single-seed runs at a different batch size, per "add exp with
    # Batch8 / Batch32" on the page.
    batch_variants: tuple[int, ...] = ()
    # The same batch sizes again with the learning rate scaled by sqrt(B/16),
    # so a batch-size result can be told apart from a step-size one.
    lr_scaled_batch_variants: tuple[int, ...] = ()
    priority: int = 2


_B = (8, 32)

RUNLIST: list[Experiment] = [
    # -- atmosphere ---------------------------------------------------------
    Experiment("E01", "atm", Factors(),
               "baseline: no CO2, no aerosol, equal weights",
               seeds=(1, 2, 3), batch_variants=_B,
               # Paired with the L0 arms above: B08/B32 at both the baseline
               # learning rate and the sqrt-scaled one, on the baseline
               # experiment so nothing else varies.
               lr_scaled_batch_variants=_B, priority=1),
    Experiment("E02", "atm", Factors(co2="1"),
               "+ CO2",
               seeds=(1, 2, 3), batch_variants=_B, priority=1),
    Experiment("E03", "atm", Factors(co2="1", aerosol="1"),
               "+ aerosol input (vs E02)"),
    Experiment("E04", "atm", Factors(co2="1", aerosol="2"),
               "- aerosol input + aerosol output (vs E03)"),
    Experiment("E05", "atm", Factors(co2="1", aerosol="3"),
               "+ aerosol input back: both inputs and outputs",
               seeds=(1, 2, 3), batch_variants=_B, priority=1),
    Experiment("E06", "atm", Factors(co2="0", aerosol="3"),
               "- CO2 (vs E05: how aerosols and GHGs interplay)"),
    Experiment("E07", "atm", Factors(co2="1", aerosol="3", weights="1"),
               "+ CO2 + flux-upweighted loss (vs E05, to probe weights)"),
    Experiment("E08", "atm", Factors(co2="1", aerosol="3", weights="2"),
               "away-from-surface weight dilution"),
    Experiment("E09", "atm", Factors(co2="1", aerosol="3", weights="4"),
               "zero STW_0"),
    Experiment("E10", "atm", Factors(co2="1", aerosol="3", amp="1"),
               "+ AMP (bf16 autocast)"),
    # -- ocean --------------------------------------------------------------
    Experiment("E11", "ocn", Factors(),
               "baseline",
               seeds=(1, 2, 3), batch_variants=_B, priority=1),
    Experiment("E12", "ocn", Factors(weights="1"),
               "interface-upweighted loss"),
    Experiment("E13", "ocn", Factors(weights="2"),
               "away-from-surface weight dilution"),
    Experiment("E14", "ocn", Factors(weights="4"),
               "zero the deepest meridional velocity"),
    # -- added 2026-08-29: the campaign uses ~59% of the reservation, so these
    #    fit alongside everything else rather than displacing it.
    Experiment("E15", "atm", Factors(co2="1", aerosol="3", weights="3"),
               "zero STW_1, the level below E09's STW_0"),
    Experiment("E16", "ocn", Factors(weights="3"),
               "zero iceVolumeTotal, structurally zero over most of the domain"),
    Experiment("E17", "ocn", Factors(ocean_step="1"),
               "1-daily ocean stepping (O1) vs E11's 5-daily"),
]


@dataclasses.dataclass(frozen=True)
class Run:
    exp: str
    realm: str
    factors: Factors
    seed: int
    note: str
    priority: int

    @property
    def runid(self) -> str:
        return (
            f"{self.exp}.{CAMPAIGN}.{self.realm}."
            f"{self.factors.word()}.S{self.seed:02d}"
        )

    @property
    def ranks(self) -> int:
        ranks, rem = divmod(self.factors.batch, LOCAL_BATCH[self.realm])
        if rem:
            raise SizingError(
                f"{self.runid}: batch {self.factors.batch} is not a multiple of "
                f"the {self.realm} local batch {LOCAL_BATCH[self.realm]}"
            )
        return ranks

    @property
    def nodes(self) -> int:
        nodes, rem = divmod(self.ranks, GPUS_PER_NODE)
        if rem or nodes < 1:
            raise SizingError(
                f"{self.runid}: {self.ranks} ranks is not a whole number of "
                f"{GPUS_PER_NODE}-GPU nodes"
            )
        return nodes


class SizingError(Exception):
    """Raised when a run's batch size and rank count cannot be reconciled."""


def expand(runlist: list[Experiment]) -> list[Run]:
    """One Run per (experiment, seed) plus one per batch-size variant.

    The batch variants are single-seed by construction: the page counts them as
    "add exp with Batch8 / Batch32" against the bolded experiments only, and
    prices them at one run each.
    """
    runs: list[Run] = []
    for e in runlist:
        for seed in e.seeds:
            runs.append(Run(e.exp, e.realm, e.factors, seed, e.note, e.priority))
        for batch in e.batch_variants:
            runs.append(
                Run(
                    e.exp,
                    e.realm,
                    dataclasses.replace(e.factors, batch=batch),
                    1,
                    f"{e.note} @ batch {batch}",
                    # Batch sweeps are the last thing worth machine time: they
                    # answer an optimizer question, not a science question.
                    priority=4,
                )
            )
        for batch in e.lr_scaled_batch_variants:
            runs.append(
                Run(
                    e.exp,
                    e.realm,
                    dataclasses.replace(e.factors, batch=batch, lr="1"),
                    1,
                    f"{e.note} @ batch {batch}, lr x sqrt({batch}/16)",
                    priority=4,
                )
            )
    # Extra seeds are worth less than the single-seed science ablations, which
    # are the only measurement of their factor at all.
    for i, r in enumerate(runs):
        if r.seed != 1 and r.priority == 1:
            runs[i] = dataclasses.replace(r, priority=3)
    return runs


# ------------------------------------------------------------- config edits --


def level_groups(names: list[str]) -> dict[str, list[int]]:
    """Group `NAME_<int>` channels into vertical families."""
    groups: dict[str, list[int]] = {}
    for name in names:
        head, sep, tail = name.rpartition("_")
        if sep and tail.isdigit():
            groups.setdefault(head, []).append(int(tail))
    return {
        k: sorted(v) for k, v in groups.items() if len(v) >= MIN_LEVELS_FOR_PROFILE
    }


def vertical_weights(out_names: list[str], realm: str) -> dict[str, float]:
    """W2: a monotone surface-heavy profile, mean-normalized to 1.0.

    Mean-normalizing matters. A profile that is not mean 1 changes the total
    loss magnitude as well as its shape, so W2 would be confounded with an
    effective learning-rate change and could not be compared with W0.
    """
    weights: dict[str, float] = {}
    low = W2_SPAN ** -0.5
    high = W2_SPAN ** 0.5
    scale = 2.0 / (low + high)  # make the endpoints average to 1.0
    for head, levels in level_groups(out_names).items():
        n = len(levels)
        for pos, level in enumerate(levels):
            frac = pos / (n - 1)
            if not SURFACE_AT_LOW_INDEX[realm]:
                frac = 1.0 - frac  # index 0 is the far end for the atmosphere
            # frac == 0 at the surface, 1 at the far end
            w = high + (low - high) * frac
            weights[f"{head}_{level}"] = round(w * scale, 4)
    if not weights:
        raise SizingError(f"no vertical families found in {realm} out_names")
    return weights


def loss_weights(kind: str, out_names: list[str], realm: str) -> dict[str, float]:
    if kind == "0":
        return {}
    if kind == "1":
        present = [n for n in W1_NAMES[realm] if n in out_names]
        missing = set(W1_NAMES[realm]) - set(present)
        if missing:
            raise SizingError(
                f"W1 names absent from {realm} out_names: {sorted(missing)}"
            )
        return {n: W1_UPWEIGHT for n in present}
    if kind == "2":
        return vertical_weights(out_names, realm)
    if kind in ZERO_NAMES:
        names = ZERO_NAMES[kind][realm]
        missing = set(names) - set(out_names)
        if missing:
            raise SizingError(
                f"W{kind} names absent from {realm} out_names: {sorted(missing)}"
            )
        return {n: 0.0 for n in names}
    raise SizingError(f"unknown weight set W{kind}")


def apply_channels(config: dict, run: Run, with_aod: bool) -> None:
    """Add the CO2 and aerosol channels this run's factor word asks for.

    The baselines are the A0/C0 corner, so this is add-only. `force_positive_names`
    has to grow with the aerosol outputs: lwp, lcc and cdnc are non-negative by
    definition and the corrector is the only thing enforcing that.
    """
    if run.realm != "atm":
        if run.factors.co2 != "0" or run.factors.aerosol != "0":
            raise SizingError(
                f"{run.runid}: CO2/aerosol factors are atmosphere-only"
            )
        return
    step = config["stepper"]["step"]["config"]
    in_names = list(step["in_names"])
    out_names = list(step["out_names"])

    if run.factors.co2 == "1":
        in_names += CO2_IN
    aero_in, aero_out = A_LEVELS[run.factors.aerosol]
    if aero_in:
        in_names += AEROSOL_IN
        if with_aod:
            in_names += AOD_IN
    if aero_out:
        out_names += AEROSOL_OUT
        corrector = step["corrector"]
        corrector["force_positive_names"] = (
            list(corrector["force_positive_names"]) + AEROSOL_OUT
        )
        # lwp/lcc/cdnc are the signature outputs of these arms, so they belong
        # in the pictures. The baseline's plot list cannot name them -- they do
        # not exist there -- so the arm that adds the outputs adds the plots.
        for names in _plot_lists(config):
            names += AEROSOL_OUT

    dupes = [n for n in set(in_names) if in_names.count(n) > 1]
    if dupes:
        raise SizingError(f"{run.runid}: duplicate in_names {sorted(dupes)}")
    step["in_names"] = in_names
    step["out_names"] = out_names


def _plot_lists(config: dict) -> list[list[str]]:
    """The distinct variable lists that decide which channels get a picture.

    The baselines write one list and alias it into `histogram.variables` and
    three `plot_variables`, so yaml hands back a single object shared by every
    site; this returns it once, by identity, and appending to it updates all of
    them at once. Written as a search rather than a fixed path so that adding an
    aggregator to the baselines does not silently leave it un-narrowed.
    """
    found: dict[int, list[str]] = {}

    def walk(node) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in ("variables", "plot_variables") and isinstance(value, list):
                    found[id(value)] = value
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    for entry in config.get("inference", []):
        walk(entry.get("aggregator", {}))
    return list(found.values())


def apply_sizing(config: dict, run: Run) -> None:
    """Set the batch sizes and make every count divide the rank count."""
    ranks = run.ranks
    config["train_loader"]["batch_size"] = run.factors.batch
    config["validation"]["loader"]["batch_size"] = run.factors.batch

    for block in config.get("inference", []):
        times = block["loader"]["start_indices"]["times"]
        if len(times) % ranks:
            if run.realm != "atm" or ranks != 32:
                raise SizingError(
                    f"{run.runid}: {len(times)} initial conditions in block "
                    f"{block.get('name')!r} do not divide {ranks} ranks, and no "
                    f"replacement list is defined"
                )
            block["loader"]["start_indices"]["times"] = list(
                IC_TEST_WINDOW_32
                if block.get("name") == "12yr_test"
                else IC_TRAIN_WINDOW_32
            )
            times = block["loader"]["start_indices"]["times"]
        if len(times) % ranks:
            raise SizingError(
                f"{run.runid}: replacement IC list ({len(times)}) still does "
                f"not divide {ranks} ranks"
            )
    if run.factors.batch % ranks:
        raise SizingError(
            f"{run.runid}: batch {run.factors.batch} does not divide {ranks} ranks"
        )


def apply_epoch_schedule(config: dict, epochs: int) -> None:
    """Set max_epochs and give every inference block the same fixed number of
    evaluations, ending on the final epoch.

    FME fires a block on ``list(range(1, max_epochs + 1))[start::step]``. The
    range starts at 1 because ``evaluate_before_training`` is off, and that is
    what the previous version of this function got wrong: it solved ``start``
    against ``range(max_epochs + 1)``, one element longer, so the last fire
    landed one stride short and the final epoch was never scored -- the exact
    failure the function exists to prevent. With max_epochs 30 and step 5 it
    fired on 1, 6, ... 26.

    Every block gets the same schedule, so the train-window and held-out scores
    land on the same epochs and can be read against each other.
    """
    config["max_epochs"] = epochs
    step = max(1, epochs // INFERENCE_EVALUATIONS)
    start = (epochs - 1) % step
    fires = list(range(1, epochs + 1))[start::step]
    assert fires and fires[-1] == epochs, (start, step, epochs, fires)
    for block in config.get("inference", []):
        block["epochs"] = {"start": start, "step": step}


def apply_inference_cost(config: dict, run: Run) -> None:
    """Set every inference block's rollout length from INFERENCE_YEARS.

    Called before apply_ocean_cadence, which multiplies the result by 5 to keep
    the O1 runs covering the same span of simulated time as the O5 ones.
    """
    steps = INFERENCE_YEARS * STEPS_PER_YEAR[run.realm]
    for block in config.get("inference", []):
        block["n_forward_steps"] = steps
        if run.realm == "ocn":
            block["forward_steps_in_memory"] = OCEAN_FORWARD_STEPS_IN_MEMORY


def build(baseline: dict, run: Run, epochs: int, with_aod: bool) -> dict:
    config = copy.deepcopy(baseline)
    config["seed"] = run.seed
    apply_channels(config, run, with_aod)
    apply_sizing(config, run)
    apply_epoch_schedule(config, epochs)
    apply_inference_cost(config, run)

    if run.factors.amp == "1":
        config["optimization"]["enable_automatic_mixed_precision"] = True

    base_lr = config["optimization"]["lr"]
    config["optimization"]["lr"] = learning_rate(base_lr, run.factors)

    weights = loss_weights(
        run.factors.weights, config["stepper"]["step"]["config"]["out_names"], run.realm
    )
    loss = config["stepper_training"]["loss"]
    if weights:
        loss["weights"] = weights
    else:
        loss.pop("weights", None)

    if run.realm == "ocn" and run.factors.ocean_step == "1":
        apply_ocean_cadence(config, run)
    return config


# Normalization statistics directories, keyed by ocean cadence. The path
# fragment is unique enough to substitute inside a full path.
OCEAN_STATS = {"5": "/train-only/ocean/", "1": "/train-only/ocean-1d/"}


def apply_ocean_cadence(config: dict, run: Run) -> None:
    """Switch the ocean from the 5-day streams to the 1-day ones."""

    def walk(node):
        if isinstance(node, dict):
            fp = node.get("file_pattern")
            if isinstance(fp, str):
                for stream in OCEAN_STREAMS:
                    node["file_pattern"] = fp = fp.replace(f"{stream}5D.", f"{stream}.")
                if fp.startswith("landfrac5d."):
                    node["file_pattern"] = "landfrac1d.*.nc"
            dp = node.get("data_path")
            if isinstance(dp, str) and dp.rstrip("/").endswith("landfrac5d"):
                node["data_path"] = LANDFRAC_DIR["1"]
            # The 1-day set is a strict superset of the 5-day one (221 variables
            # against 127) but its scales differ where it matters: TAUY 1.48x,
            # precipitation rates 1.40-1.43x, TAUX 1.22x, SHFLX 1.21x. Borrowing
            # the 5-day scales would mis-normalize exactly the channels a
            # cadence experiment is about.
            for key in ("global_means_path", "global_stds_path"):
                sp = node.get(key)
                if isinstance(sp, str):
                    node[key] = sp.replace(OCEAN_STATS["5"], OCEAN_STATS["1"])
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(config)
    for block in config.get("inference", []):
        block["n_forward_steps"] = int(block["n_forward_steps"] * 5)
    if any("5D." in str(v) for v in (config,)):
        raise SizingError(f"{run.runid}: a 5D stream survived the cadence switch")
    if OCEAN_STATS["5"] in str(config):
        raise SizingError(f"{run.runid}: 5-day ocean statistics survived the switch")


def env_file(run: Run) -> str:
    """W&B provenance: read from the environment by wandb, not from the config.

    Deliberately free of identity. Who submitted a run and where its output
    landed are properties of the submission, not of the run list, and
    run-train.sh appends both to WANDB_NOTES at submit time. Baking them in
    here made every file in runs/ different for every teammate, which dirtied
    the worktree the moment they regenerated -- and run-train.sh refuses to
    submit from a dirty worktree, so only the person who generated the
    campaign could launch it.
    """
    tags = ",".join(
        [
            CAMPAIGN,
            run.exp,
            run.realm,
            f"A{run.factors.aerosol}",
            f"B{run.factors.batch:02d}",
            f"C{run.factors.co2}",
            f"L{run.factors.lr}",
            f"O{run.factors.ocean_step}",
            f"W{run.factors.weights}",
            f"X{run.factors.amp}",
            f"S{run.seed:02d}",
            f"P{run.priority}",
        ]
    )
    return "\n".join(
        [
            f"# generated by make_ablation_config.py -- {run.note}",
            # Read by run-train.sh to size the sbatch step. The node count is a
            # property of the config (batch size / local batch / GPUs per node),
            # so it travels with the run rather than living in the sbatch file.
            f"FME_NODES={run.nodes}",
            f"FME_RANKS={run.ranks}",
            f"FME_PRIORITY={run.priority}",
            f"WANDB_NAME={run.runid}",
            f"WANDB_RUN_GROUP={CAMPAIGN}.{run.realm}.{run.exp}",
            f"WANDB_JOB_TYPE={run.factors.word()}",
            f"WANDB_TAGS={tags}",
            f'WANDB_NOTES="{run.note} | {run.nodes} nodes, {run.ranks} ranks"',
            "",
        ]
    )


# -------------------------------------------------------------------- main --


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--all", action="store_true", help="write every run")
    p.add_argument("--exp", action="append", default=[],
                   help="write only these experiments (repeatable), e.g. --exp E05")
    p.add_argument("--realm", choices=["atm", "ocn"], help="restrict to one realm")
    p.add_argument("--list", action="store_true",
                   help="print the run list and the node budget, write nothing")
    p.add_argument("--dry-run", action="store_true",
                   help="build and check every config, write nothing")
    p.add_argument("-o", "--out", default=str(HERE / "runs"))
    p.add_argument("--epochs", type=int, default=None,
                   help=f"override max_epochs (defaults {DEFAULT_EPOCHS})")
    p.add_argument("--local-batch", action="append", default=[], metavar="REALM=N",
                   help="override samples per rank, e.g. --local-batch atm=2. "
                        "Halves the node count per run; only use it once the "
                        "memory has been measured at that value.")
    p.add_argument("--aod", action="store_true",
                   help="read the page's A3 literally and add AODVISall as an "
                        "aerosol input")
    args = p.parse_args(argv)

    for spec in args.local_batch:
        realm, _, value = spec.partition("=")
        if realm not in LOCAL_BATCH or not value.isdigit() or int(value) < 1:
            p.error(f"--local-batch expects atm=N or ocn=N, got {spec!r}")
        LOCAL_BATCH[realm] = int(value)

    all_runs = expand(RUNLIST)
    runs = list(all_runs)
    if args.exp:
        wanted = {e.upper() for e in args.exp}
        runs = [r for r in runs if r.exp in wanted]
    if args.realm:
        runs = [r for r in runs if r.realm == args.realm]
    if not runs:
        print("no runs selected", file=sys.stderr)
        return 2
    if not (args.all or args.exp or args.list or args.dry_run):
        p.error("nothing to do: pass --all, --exp, --list or --dry-run")

    runs.sort(key=lambda r: (r.priority, r.realm, r.exp, r.factors.batch, r.seed))

    if args.list:
        report(runs)
        return 0

    out = pathlib.Path(args.out)
    baselines = {
        realm: yaml.safe_load((HERE / f"config-train-{realm}.yaml").read_text())
        for realm in sorted({r.realm for r in runs})
    }

    if not args.dry_run:
        out.mkdir(parents=True, exist_ok=True)
    written = 0
    for run in runs:
        key = f"{run.realm}-O{run.factors.ocean_step}"
        epochs = args.epochs or DEFAULT_EPOCHS.get(key, DEFAULT_EPOCHS[run.realm])
        config = build(baselines[run.realm], run, epochs, args.aod)
        if args.dry_run:
            written += 1
            continue
        (out / f"{run.runid}.yaml").write_text(
            "# GENERATED by make_ablation_config.py -- do not edit by hand.\n"
            f"# {run.exp}: {run.note}\n"
            f"# {run.nodes} nodes / {run.ranks} ranks, priority P{run.priority}\n"
            + yaml.safe_dump(config, sort_keys=False, default_flow_style=False)
        )
        (out / f"{run.runid}.env").write_text(env_file(run))
        written += 1

    manifest = "\n".join(
        ["\t".join(["priority", "runid", "realm", "nodes", "ranks", "batch",
                    "seed", "note"])]
        + [
            "\t".join(
                [
                    f"P{r.priority}", r.runid, r.realm, str(r.nodes), str(r.ranks),
                    str(r.factors.batch), f"S{r.seed:02d}", r.note,
                ]
            )
            for r in runs
        ]
    ) + "\n"
    # MANIFEST.tsv is what submit-campaign.sh walks, so writing a filtered
    # selection over it turns the whole campaign into whatever subset was
    # generated last -- and the submitter has no way to tell. Only a full
    # generation may write it.
    partial = len(runs) != len(all_runs)
    if not args.dry_run and not partial:
        (out / "MANIFEST.tsv").write_text(manifest)
    elif partial:
        print(
            f"\nselection is {len(runs)} of {len(all_runs)} runs, so "
            f"{out / 'MANIFEST.tsv'} was left alone.\n"
            "Regenerate the whole campaign to refresh it: "
            "./generate-campaign.sh",
            file=sys.stderr,
        )

    report(runs)
    verb = "checked" if args.dry_run else f"wrote {written} configs to {out}"
    print(f"\n{verb}")
    return 0


def report(runs: list[Run]) -> None:
    print(f"{'pri':<4} {'nodes':>5}  runid")
    for r in runs:
        print(f"P{r.priority:<3} {r.nodes:>5}  {r.runid}")
    print()
    by_priority: dict[int, int] = {}
    for r in runs:
        by_priority[r.priority] = by_priority.get(r.priority, 0) + r.nodes
    running = 0
    for pri in sorted(by_priority):
        running += by_priority[pri]
        print(f"  P{pri}: {by_priority[pri]:>3} nodes  (cumulative {running})")
    total = sum(r.nodes for r in runs)
    print(f"  total: {total} nodes across {len(runs)} runs "
          f"({'fits' if total <= 96 else 'EXCEEDS'} the 96-node reservation)")


if __name__ == "__main__":
    raise SystemExit(main())
