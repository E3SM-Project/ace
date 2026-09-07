"""Generate config-train-cpl.yaml by composing the atm and ocn configs.

Building it from the two component configs keeps the three in sync: the coupled
stepper reuses each component's step config verbatim, so a channel change in
either one propagates here.

The production sequence is atm-only and ocn-only pretraining first, coupled
finetune after. Once the component runs have produced checkpoints, regenerate
with them injected (matching Elynn's piControl coupled flow, which set
stepper_training.<realm>.parameter_init.weights_path):

    python make_cpl_config.py
        --atm-ckpt $PSCRATCH/fme-output/hist-train/training_checkpoints/best_ckpt.tar
        --ocn-ckpt $PSCRATCH/fme-output/hist-ocn/training_checkpoints/best_ckpt.tar

weights_path takes a *stepper* checkpoint (best_ckpt.tar or
best_inference_ckpt.tar, not the full training-state ckpt.tar). The component
stepper blocks are copied verbatim into the coupled config, so the pretrained
weights match the coupled realms exactly. With no flags the output is
unchanged (coupled realms train from scratch).
"""

import argparse
import copy
import difflib
import pathlib
import sys

import yaml

# The stage-2 generator owns the held-out initial conditions. Importing them
# rather than restating them means the two stages cannot drift apart silently:
# if stage 2's lists move off the 1990s gap, stage 3's move with them.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import make_stage2_config as stage2  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument(
    "--atm-ckpt",
    default=None,
    help="stepper checkpoint (best_ckpt.tar) to initialize the atmosphere from",
)
ap.add_argument(
    "--ocn-ckpt",
    default=None,
    help="stepper checkpoint (best_ckpt.tar) to initialize the ocean from",
)
ap.add_argument(
    "--out",
    default=None,
    help="write here instead of overwriting the committed config-train-cpl.yaml",
)
ap.add_argument(
    "--check",
    action="store_true",
    help="do not write; exit non-zero with a diff if the committed coupled "
    "config is out of sync with the atm and ocn configs",
)
args = ap.parse_args()

D = pathlib.Path(__file__).resolve().parent
with open(D / "config-train-atm.yaml") as f:
    atm = yaml.safe_load(f)
with open(D / "config-train-ocn.yaml") as f:
    ocn = yaml.safe_load(f)
OUT = args.out or str(D / "config-train-cpl.yaml")

atm_stepper = copy.deepcopy(atm["stepper"])
ocn_stepper = copy.deepcopy(ocn["stepper"])

# The atmosphere's normalization is taken from config-train-atm.yaml verbatim.
# The piControl stats this used to point at came in coupled_atmosphere and
# uncoupled_atmosphere variants, and this script rewrote the path to pick the
# coupled one, on the grounds that a coupled atmosphere takes its TS from the
# ocean. The historical stats draw no such distinction -- they are computed
# from one run, in which the atmosphere is coupled -- so there is nothing to
# rewrite. Reinstate a rewrite here if the configs ever point back at a stats
# set that splits the two.

# ---- data: ocean streams come from the ocn config, atmosphere from the atm one
ocn_val = ocn["validation"]["loader"]["dataset"]


def ocean_window(subset):
    """Ocean merge for one time window.

    The coupled loader takes `ocean` as a merge without concat, so the two
    training windows are concatenated at the coupled level instead (each paired
    with its own atmosphere window) rather than inside the ocean dataset.
    """
    node = copy.deepcopy(ocn_val)
    for member in node["merge"]:
        member["subset"] = dict(subset)
    return node


ocn_inf = ocn["inference"][0]["loader"]["dataset"]
atm_ds = atm["train_loader"]["dataset"]["concat"][0]


def atmos(subset=None, keep_subset=True):
    d = {
        k: copy.deepcopy(atm_ds[k])
        for k in (
            "data_path",
            "file_pattern",
            "rename",
            "reference_pressure_name",
            "overwrite",
        )
        if k in atm_ds
    }
    if subset and keep_subset:
        d["subset"] = dict(subset)
    return d


def strip_subset(node):
    """Inference forbids `subset` on both realms; the glob defines the range."""
    node = copy.deepcopy(node)

    def walk(n):
        if isinstance(n, dict):
            n.pop("subset", None)
            for v in n.values():
                walk(v)
        elif isinstance(n, list):
            for v in n:
                walk(v)

    walk(node)
    return node


# Windows must start on the ocean 5-day axis: CoupledDataset requires the two
# realms to share their first timestamp.
TRAIN_WINDOWS = [
    {"start_time": "1940-01-06", "stop_time": "1990-01-01"},
    {"start_time": "2000-01-06", "stop_time": "2040-01-01"},
]
VAL_A = {"start_time": "1990-01-06", "stop_time": "1995-01-01"}

# The coupled finetune runs on 4 nodes x 4 GPUs. This number sets three things
# that have to agree, so it lives here rather than being written out three
# times: the global train and validation batch (local batch 1), and the number
# of inference initial conditions, which are sharded across ranks and must
# divide evenly -- InlineInferenceConfig.__post_init__ enforces that, but it
# calls Distributed.get_instance(), so on a login node world_size is 1 and the
# check always passes. A violation therefore cannot be caught by validating the
# config; it surfaces minutes into an allocation as a dacite UnionMatchError
# that names none of this. Stage 2 lost two atmosphere runs that way.
NODES = 4
RANKS = NODES * 4

# Length of the coupled finetune, and how often each inference block fires.
#
# 50 rather than the 5 this used to carry. That 5 was a cost budget, not a
# science choice: the campaign README derived "~28 h/epoch, ~6 days for 5
# epochs" from 61.3 s/batch measured 2026-08-24 on 8 ranks, with another job
# competing for the filesystem. Measured again 2026-09-07 on 16 ranks with the
# inputs on scratch, the same production settings run at a median 11.4 s/batch
# (mean 12.0, n=15 intervals, no drift across the epoch), which over this
# config's 411 batches is ~1.3 h/epoch. Five epochs is seven hours, not six
# days, and 50 is about three days -- three segments at the 24 h walltime.
#
# The schedules are periods, not raw indices, because a block fires on
# list(range(1, max_epochs + 1))[start::step] and hand-written start/step
# silently stop meaning what they meant when max_epochs moves. Both periods
# divide max_epochs, so both blocks fire on the final epoch.
STAGE3_EPOCHS = 50
SELECTION_PERIOD = 5  # heldout_1990s: 10 firings, ~17 min each
REPORTING_PERIOD = 10  # future_2040: 5 firings, ~32 min each -- twice the rollout


def _epoch_schedule(period):
    """Fire on every `period`-th epoch, ending on the last one.

    Inference is not free at these rollout lengths. Measured 2026-09-07 from
    the coupled smoke run's window timings: ~2.45 s per coupled step at 16
    initial conditions on 16 ranks. That is ~15 min for the 365-step
    heldout_1990s block and ~30 min for the 730-step future_2040 one, before
    the diagnostics flush. Running the selection block every epoch would add
    ~22% to a 1.3 h epoch, or about 14 hours across 50 epochs; at a period of
    5 it costs under three.
    """
    if STAGE3_EPOCHS % period:
        raise SystemExit(
            f"inference period {period} does not divide max_epochs "
            f"{STAGE3_EPOCHS}, so the block would never fire on the final "
            f"epoch and the last checkpoint would go unscored"
        )
    return {"start": period - 1, "step": period}

# Initial conditions, taken verbatim from the stage-2 generator so stage 3 makes
# the same claim stage 2 makes.
#
# Stage 2 stopped selecting checkpoints on in-sample initial conditions. Its
# weight-1.0 block runs inside the 1990-2000 gap the training windows leave
# open, and the 2040s block is reported at weight 0 and chooses nothing, so
# "the emulator never saw the 1990s or the 2040s" survives the checkpoint
# selection and not just the gradient. A coupled finetune that went back to
# selecting on 1945-2027 would undo that at the last step: the shipped model
# would be the one that scored best on years it trained on.
#
# Both lists are the ocean's, not the atmosphere's. CoupledDataset requires the
# two realms to share their first timestamp, and the ocean's 5-day axis is the
# coarser of the two -- every ocean stamp exists on the atmosphere's 6-hourly
# axis, but not the reverse. These stamps were picked off the real ocean time
# index and verified there; TimestampList.as_indices raises on a stamp that
# does not exist, minutes into an allocation.
#
# Sixteen of each, which is also the rank count: initial conditions are sharded
# across ranks and InlineInferenceConfig.__post_init__ requires the count to
# divide evenly. Change the node count and these lists have to change with it.
IC = list(stage2.OCN_HELDOUT_ICS)
IC_TEST = list(stage2.OCN_FUTURE_ICS)

for _name, _ics in (("heldout_1990s", IC), ("future_2040", IC_TEST)):
    if len(_ics) % RANKS:
        raise SystemExit(
            f"{_name} has {len(_ics)} initial conditions, which does not divide "
            f"across {RANKS} ranks; regenerate the list against the real ocean "
            f"time index rather than padding it."
        )

# Inference aggregator, shared by both blocks.
#
# NOTE: `fme.coupled.aggregator.InferenceEvaluatorAggregatorConfig` is a
# different class from the uncoupled one -- boolean flags only, no per-metric
# typed sub-configs. These four names are its real API, not the deprecated ACE
# union member that looks similar.
#
# CONSEQUENCE: the upload-budget switches the atm and ocn baselines use
# (`time_mean_*.report_plot`, `power_spectrum.plot_variables`,
# `ensemble_denorm.log_mean_maps`) have no coupled equivalent. This config
# builds `TimeMeanMetricConfig(target=...)` internally with plotting on, so a
# coupled run still uploads one map PNG per channel per block per epoch --
# measured at 55 MB/epoch in the atmosphere. No coupled run is in the aug26
# campaign, so this is a known gap rather than a live problem; closing it means
# plumbing report_plot through _build_metrics. See EXPERIMENTS.md "The upload
# budget".
AGGREGATOR = {
    "log_zonal_mean_images": False,
    "log_video": False,
    "log_seasonal_means": False,
    "log_histograms": False,
}

cfg = {
    "experiment_dir": "/pscratch/sd/m/mahf708/fme-output/hist-cpl",
    "save_checkpoint": True,
    "validate_using_ema": True,
    "ema": {"decay": 0.9995, "faster_decay_at_start": False},
    "max_epochs": STAGE3_EPOCHS,
    # The maps go to disk as netCDF rather than to W&B; see AGGREGATOR.
    "save_per_epoch_diagnostics": True,
    "inference": [
        {
            # Selection. Every rollout starts and ends inside the 1990-2000 gap
            # the training windows leave open: the latest IC is 1994-12-27 and
            # five years from there is 1999-12, so the checkpoint that ships was
            # chosen on a decade the coupled model never trained on.
            "name": "heldout_1990s",
            "weight": 1.0,
            "epochs": _epoch_schedule(SELECTION_PERIOD),
            "n_coupled_steps": 365,  # 5 years on the ocean's 5-day axis
            "coupled_steps_in_memory": 2,
            "loader": {
                "num_data_workers": 2,
                "dataset": {
                    "ocean": strip_subset(ocn_inf),
                    "atmosphere": atmos(keep_subset=False),
                },
                "start_indices": {"times": IC},
            },
            "aggregator": AGGREGATOR,
        },
        {
            # Reporting only. weight 0.0 keeps this out of checkpoint selection,
            # which is what lets the 2040s stay a genuine test rather than a
            # validation set under another name.
            #
            # Ten years rather than five, to say something about drift on the
            # timescale the campaign actually cares about. The last IC is
            # 2043-12-27, so the longest rollout ends in 2053 -- still short of
            # the locked window at 2055, which nothing in any stage may touch.
            #
            # A block fires on list(range(1, max_epochs + 1))[start::step], so
            # start 0 / step 4 lands on epochs 1 and 5 -- the first and last.
            # This is the expensive block (730 ocean steps is 14,600 atmosphere
            # steps per initial condition), so it runs twice, not every epoch.
            "name": "future_2040",
            "weight": 0.0,
            "epochs": _epoch_schedule(REPORTING_PERIOD),
            "n_coupled_steps": 730,  # 10 years on the ocean's 5-day axis
            "coupled_steps_in_memory": 2,
            "loader": {
                "num_data_workers": 2,
                "dataset": {
                    "ocean": strip_subset(ocn_inf),
                    "atmosphere": atmos(keep_subset=False),
                },
                "start_indices": {"times": IC_TEST},
            },
            "aggregator": AGGREGATOR,
        },
    ],
    "logging": {
        "log_to_screen": True,
        "log_to_wandb": True,
        "log_to_file": True,
        # One project for the whole campaign, and the team entity rather than an
        # account. A coupled run used to be pointed at ai2cm/samudrace-e3sm-hist,
        # which is a different organisation's workspace.
        "project": "SamudrACE-E3SMv3",
        "entity": "e3sm-aig",
    },
    "train_loader": {
        # Global batch, not per rank: the loader divides it by world size and
        # refuses a remainder ("batch_size must be divisible by the number of
        # parallel workers, got 8 and 16" -- measured 2026-09-07 on 4 nodes).
        # 16 = 4 nodes x 4 GPUs, so local batch stays 1, which is what the
        # coupled memory footprint is sized for. Change the node count and this
        # has to change with it, along with the 16 inference initial conditions.
        "batch_size": RANKS,
        "num_data_workers": 2,
        "prefetch_factor": 1,
        "dataset": {
            "concat": [
                {"ocean": ocean_window(w), "atmosphere": atmos(w)}
                for w in TRAIN_WINDOWS
            ]
        },
    },
    "validation": {
        "loader": {
            "batch_size": RANKS,
            "num_data_workers": 2,
            "prefetch_factor": 1,
            "dataset": {"ocean": copy.deepcopy(ocn_val), "atmosphere": atmos(VAL_A)},
        }
    },
    "optimization": copy.deepcopy(atm["optimization"]),
    "stepper_training": {
        "n_coupled_steps": 4,
        # Coupled training draws an ensemble; the ocean must therefore use an
        # ensemble loss too. Keeping the ocean-only MSE here silently broadcasts a
        # 2-member prediction against a 1-member target.
        "n_ensemble": 2,
        "ocean": {
            "loss_weight": 1.0,
            "optimize_last_step_only": False,
            "n_steps": {
                "outcomes": [
                    {"steps": s, "probability": p}
                    for s, p in [(0, 0.1), (1, 0.3), (2, 0.3), (4, 0.3)]
                ]
            },
            # same ensemble loss as the atmosphere but without its
            # per-variable weights, which name atmosphere fields
            "loss": {
                "type": "EnsembleLoss",
                "kwargs": copy.deepcopy(atm["stepper_training"]["loss"]["kwargs"]),
            },
        },
        "atmosphere": {
            "loss_weight": 1.0,
            "optimize_last_step_only": True,
            "n_steps": {
                "outcomes": [
                    {"steps": s, "probability": p}
                    for s, p in [
                        # Must sum to 1.0: TimeLengthProbabilities renormalizes
                        # silently, so literals that do not sum to 1 mean the
                        # realized distribution is not the one written here.
                        (0, 0.025),
                        (1, 0.275),
                        (2, 0.275),
                        (4, 0.275),
                        (21, 0.1),
                        (41, 0.05),
                    ]
                ]
            },
            "loss": copy.deepcopy(atm["stepper_training"]["loss"]),
        },
    },
    "stepper": {
        "sst_name": "sst",
        "ocean_fraction_prediction": {
            "sea_ice_fraction_name": "ocean_sea_ice_fraction",
            "land_fraction_name": "LANDFRAC",
            "sea_ice_fraction_name_in_atmosphere": "ICEFRAC",
        },
        "ocean": {"timedelta": "5D", "stepper": ocn_stepper},
        "atmosphere": {"timedelta": "6h", "stepper": atm_stepper},
    },
}
if args.ocn_ckpt:
    cfg["stepper_training"]["ocean"]["parameter_init"] = {"weights_path": args.ocn_ckpt}
if args.atm_ckpt:
    cfg["stepper_training"]["atmosphere"]["parameter_init"] = {
        "weights_path": args.atm_ckpt
    }

# Read the inputs from $PSCRATCH, using stage 2's map so all three stages agree
# on where the data lives.
#
# Stage 3 inherited the CFS paths from the atm and ocn configs it composes,
# which stage 2 had already moved. That is not a cosmetic difference: on
# 2026-09-07 a coupled run pointed at CFS sat in "Opening data at
# /global/cfs/..." for over 20 minutes without finishing its dataset index,
# while the same config on scratch cleared the whole init in about 7. CFS
# cannot feed this loader's read pattern under campaign contention.
#
# Applied after the checkpoint paths are set, and the map only matches the CFS
# input roots, so a --atm-ckpt/--ocn-ckpt pointing into another user's scratch
# is left exactly as given -- those are read-only reads we depend on.
cfg = stage2.remap_paths(cfg)

rendered = yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False, width=100)

if args.check:
    existing = pathlib.Path(OUT).read_text() if pathlib.Path(OUT).exists() else ""
    if existing == rendered:
        print(f"{OUT} is up to date")
        sys.exit(0)
    print(f"{OUT} is STALE -- regenerate it by running this script with no flags")
    sys.stdout.writelines(
        difflib.unified_diff(
            existing.splitlines(keepends=True),
            rendered.splitlines(keepends=True),
            fromfile=f"{OUT} (committed)",
            tofile="(generated)",
        )
    )
    sys.exit(1)

with open(OUT, "w") as f:
    f.write(rendered)
print("wrote", OUT)
