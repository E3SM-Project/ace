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

import yaml

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
args = ap.parse_args()

D = "configs/experiments/e3sm_hist_v20260812"
atm = yaml.safe_load(open(f"{D}/config-train-atm.yaml"))
ocn = yaml.safe_load(open(f"{D}/config-train-ocn.yaml"))
OUT = f"{D}/config-train-cpl.yaml"

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
IC = [f"{y}-01-06T00:00:00" for y in [1945, 1955, 1965, 1975, 2005, 2015, 2025, 2035]]

cfg = {
    "experiment_dir": "/pscratch/sd/m/mahf708/fme-output/hist-cpl",
    "save_checkpoint": True,
    "validate_using_ema": True,
    "ema": {"decay": 0.9995, "faster_decay_at_start": False},
    "max_epochs": 5,
    "inference": [
        {
            "name": "inference",
            "weight": 1.0,
            "n_coupled_steps": 876,
            "coupled_steps_in_memory": 2,
            "loader": {
                "num_data_workers": 2,
                "dataset": {
                    "ocean": strip_subset(ocn_inf),
                    "atmosphere": atmos(keep_subset=False),
                },
                "start_indices": {"times": IC},
            },
            "aggregator": {"log_zonal_mean_images": False, "log_histograms": False},
        }
    ],
    "logging": {
        "log_to_screen": True,
        "log_to_wandb": True,
        "log_to_file": True,
        "project": "samudrace-e3sm-hist",
        "entity": "ai2cm",
    },
    "train_loader": {
        "batch_size": 8,
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
            "batch_size": 8,
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
                        (0, 0.025),
                        (1, 0.29),
                        (2, 0.29),
                        (4, 0.29),
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

yaml.safe_dump(
    cfg, open(OUT, "w"), sort_keys=False, default_flow_style=False, width=100
)
print("wrote", OUT)
