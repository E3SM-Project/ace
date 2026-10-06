#!/usr/bin/env python
"""Emit E40, the first E4x coupled fine-tune (2026-10-02): CFT on top of pilot E.

Pilot E2 is the standalone ocean retrained on a single net heat flux with the
ocean heat-content corrector (cft-diag/pilot/ocn-netflux-ohc2.yaml, epoch-5 EMA,
offset -1.62 W/m2 from the training years). Coupled with no fine-tuning it
already reaches an 83-93% SST trend share over 2015-44, against 56-81% for
E24-CFT, but with a worse mean state. E40 asks whether coupled fine-tuning
pulls the mean state back without losing that response.

Naming continues the E3x grammar and adds the E4x factors (campaign design
D16 / physics design section 6):

    A?_B??_C?_L?_O?_V?_W?_X?_F?_H?_E?_D?_G?

  F0  ocean flux form: pilot E2's offline MPAS ocean-side net flux
  H1  ocean heat-content corrector, scaled_temperature
  E1  atmosphere total-energy corrector, trained through (new FSDS/FSUS outputs)
  D0  historical data only
  G0  no cross-realm gradients (gradient accumulation on)

Recipe (campaign design D11 and the S2D addendum):
  * atmosphere E02-FT S03 (A0 C1: no aerosol diagnostics, per D2); ocean pilot E2.
    S03 replaced S01 on 2026-10-03: S01 enters a dark regime (6/7 + 4/8 standalone,
    and once coupled), S02/S03 0/16 each (cft-diag/cpl_e4x/regime_screen).
  * coupler builds hfds from the atmosphere's fluxes (combined_fluxes) and
    zeroes the four flux channels the ocean was trained without
  * atmosphere gains FSDS and FSUS outputs, appended last so the parent's
    weights load into the existing channels, and total_energy_budget_correction
    (constant_temperature) runs in training after one corrector-free epoch, as
    upstream does; unaccounted heating -0.14 W/m2 for historical EAM (Z2)
  * lr 1e-5 with a 3000-iteration LinearLR warm-up (upstream coupled recipe)
  * MSE on the deterministic ocean (CRPS on it is MAE at twice the cost)
  * seeds 1 and 2, model and EMA checkpoints every epoch, at most 10 epochs (extended to 40 on 2026-10-06 by resuming from ckpt.tar)

Submission: both seeds go out as one 8-node bundle (bundles/e4x-cft.txt) at
the 48 h regular-QOS walltime, `sbatch-scripts/bundle.sh bundles/e4x-cft.txt --go`.
"""

import copy
import pathlib
import subprocess
import sys

import yaml

D = pathlib.Path(__file__).resolve().parent
FT = pathlib.Path("/pscratch/sd/m/mahf708/aug26-ft")
PILOT = pathlib.Path("/pscratch/sd/m/mahf708/cft-diag/pilot")
ROOT = pathlib.Path("/pscratch/sd/m/mahf708/aug26-cft")
RUNS = D / "runs"
NODES = 4
MAX_EPOCHS = 40  # 2026-10-06: extended; the 48 h walltime, not this, sets the last epoch
WORD = "A0_B16_C1_L0_O5_V0_W0_X0_F0_H1_E1_D0_G0"
SEEDS = (1, 2)

ATM = FT / "E02-FT.aug26.atm.A0_B16_C1_L0_O5_W0_X0.S03"
OCN_TEMPLATE = FT / "E11-FT.aug26.ocn.A0_B16_C0_L0_O5_W0_X0.S01"  # data layout only
OCN_PILOT_CFG = PILOT / "ocn-netflux-ohc2.yaml"
OCN_PILOT_CKPT = PILOT / "ocn-netflux-ohc2/training_checkpoints/ema_ckpt_0005.tar"

THETAO = {f"temperatureCoarsened_{k}": f"thetao_{k}" for k in range(19)}
NET = {"FSNS": 1.0, "FLDS": 1.0, "FLUS": -1.0, "LHFLX": -1.0, "SHFLX": -1.0}
ZEROED = ("FLUS", "FLDS", "LHFLX", "SHFLX")
ATM_UNACCOUNTED_HEATING = -0.14  # W/m2, historical EAM (cpl_e3x/z2_corrector)


def _rename_depth_levels(node):
    """Rename the coarsened temperature levels wherever a depth file is read."""
    if isinstance(node, dict):
        if "fmeDepthCoarsening5D" in str(node.get("file_pattern", "")):
            node.setdefault("rename", {}).update(THETAO)
        for v in node.values():
            _rename_depth_levels(v)
    elif isinstance(node, list):
        for v in node:
            _rename_depth_levels(v)


def _emit(seed):
    runid = f"E40-CFT.aug26.cpl.{WORD}.S{seed:02d}"
    out = RUNS / f"{runid}.yaml"
    subprocess.run(
        [sys.executable, str(D / "make_cpl_config.py"),
         "--atm-config", str(ATM / "config.yaml"),
         "--ocn-config", str(OCN_TEMPLATE / "config.yaml"),
         "--atm-ckpt", str(ATM / "training_checkpoints/best_inference_ckpt.tar"),
         "--ocn-ckpt", str(OCN_PILOT_CKPT),
         "--nodes", str(NODES), "--out", str(out)],
        check=True, stdout=subprocess.DEVNULL,
    )
    cfg = yaml.safe_load(out.read_text())
    pilot = yaml.safe_load(OCN_PILOT_CFG.read_text())

    cfg["max_epochs"] = MAX_EPOCHS
    cfg["seed"] = seed
    cfg["checkpoint_save_epochs"] = {"step": 1}
    cfg["ema_checkpoint_save_epochs"] = {"step": 1}

    # ocean: pilot E2's stepper (net flux in, heat-content corrector on)
    cfg["stepper"]["ocean"]["stepper"] = copy.deepcopy(pilot["stepper"])
    cfg["stepper_training"]["ocean"]["parameter_init"]["weights_path"] = str(
        OCN_PILOT_CKPT
    )
    cfg["stepper_training"]["ocean"]["loss"] = {"type": "MSE"}
    for section in ("train_loader", "validation", "inference"):
        _rename_depth_levels(cfg.get(section))

    # coupler: hfds from the atmosphere's fluxes, zeroed channels for the ocean
    st = cfg["stepper"]
    st["open_water_flux_scaling"]["names"] = [
        "surface_precipitation_rate", "frozen_precipitation_rate", "hfds"
    ]
    st["combined_fluxes"] = [{"name": "hfds", "terms": dict(NET)}] + [
        {"name": n, "terms": {n: 0.0}} for n in ZEROED
    ]

    # atmosphere: FSDS/FSUS outputs, appended last, and the energy corrector
    acfg = st["atmosphere"]["stepper"]["step"]["config"]
    for name in ("FSDS", "FSUS"):
        if name not in acfg["out_names"]:
            acfg["out_names"].append(name)
    corr = acfg["corrector"]
    corr["total_energy_budget_correction"] = {
        "method": "constant_temperature",
        "constant_unaccounted_heating": ATM_UNACCOUNTED_HEATING,
    }
    corr["corrector_disabled_epochs"] = 1

    # optimizer: upstream coupled fine-tune recipe
    opt = cfg["optimization"]
    opt["lr"] = 1.0e-5
    opt["scheduler"] = {
        "schedulers": [
            {"type": "LinearLR", "step_each_iteration": True,
             "kwargs": {"start_factor": 0.01, "end_factor": 1, "total_iters": 3000}},
            {"type": "ConstantLR", "step_each_iteration": True,
             "kwargs": {"factor": 1.0}},
        ],
        "milestones": [3000],
    }

    text = yaml.safe_dump(cfg, sort_keys=False)
    for a, b in THETAO.items():  # data-writer / aggregator name lists
        text = text.replace(f"- {a}\n", f"- {b}\n")
    out.write_text(text)

    tags = ",".join(["cft", "e4x", "aug26", "E40", "cpl", *WORD.split("_"),
                     f"S{seed:02d}", "pilotE2-ocean", "energy-corrector"])
    (RUNS / f"{runid}.env").write_text(
        "# generated by make_e4x_cft.py -- E4x main arm, CFT on pilot E (2026-10-02)\n"
        f"# atm parent {ATM.name} (+FSDS/FSUS outputs, energy corrector)\n"
        f"# ocn parent pilot E2 {OCN_PILOT_CKPT}\n"
        f"FME_NODES={NODES}\n"
        f"FME_RANKS={NODES * 4}\n"
        "FME_PRIORITY=1\n"
        f"WANDB_NAME={runid}\n"
        "WANDB_RUN_GROUP=aug26.cpl.E40-CFT\n"
        f"WANDB_JOB_TYPE={WORD}_CFT\n"
        f"WANDB_TAGS={tags}\n"
        f'WANDB_NOTES="E4x main arm: CFT on pilot E2 (net flux + heat-content '
        f'corrector) x E02-FT S03 with FSDS/FSUS and the energy corrector | '
        f'{NODES} nodes, {MAX_EPOCHS} epochs, lr 1e-5 + warm-up"\n'
    )
    return runid


def main():
    for p in (ATM / "config.yaml", OCN_PILOT_CFG, OCN_PILOT_CKPT):
        if not p.exists():
            sys.exit(f"missing: {p}")
    runids = [_emit(seed) for seed in SEEDS]
    lines = [
        f"# E4x coupled bundle, 2026-10-02 -- {len(runids)} runs x {NODES} nodes "
        f"= {len(runids) * NODES} nodes, 48 h walltime.",
        "# Generated by make_e4x_cft.py: CFT on pilot E2, energy corrector on.",
        *(f"{r}  {ROOT}" for r in runids),
    ]
    (D / "bundles/e4x-cft.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(runids))


if __name__ == "__main__":
    main()
