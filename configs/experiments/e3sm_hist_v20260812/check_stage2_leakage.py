#!/usr/bin/env python3
"""Prove, from the configs themselves, what the stage-2 emulators ever saw.

The campaign wants to be able to say "the emulators never saw the 1990s or the
2040s" and have it be true rather than nearly true. That claim is not one
property, it is three, and they are not equally strong:

    gradient    a window whose samples produce a loss that is backpropagated
    selection   a window that feeds a metric which chooses a checkpoint --
                inline validation, or an inference block with weight > 0
    reporting   a window scored and logged at weight 0, influencing nothing

A window used only for reporting is genuinely untouched by the model. A window
used for selection is not: the checkpoint that gets shipped was chosen partly
because of it, which is a weak form of exposure but a real one, and it is the
distinction that separates a defensible claim from an overclaim.

This walks every generated stage-2 config, works out the closed time interval
each block actually reads -- for inference that is the initial condition plus
the rollout, which is the part people forget -- and checks the invariants:

    1. no gradient window intersects 1990-01-01..2000-01-01 or 2040-01-01..
    2. no selection window starts on or after 2040-01-01
    3. nothing of any kind intersects the locked window, 2055-01-01..

Exit status is non-zero if any invariant fails, so it can gate submission.

    ./check_stage2_leakage.py                 # every runs/*aug26-ft*.yaml
    ./check_stage2_leakage.py runs/E12.aug26-ft.ocn.....yaml
"""

import datetime
import pathlib
import sys

import yaml

HERE = pathlib.Path(__file__).resolve().parent
RUNS = HERE / "runs"

# A noleap year is 365 days; both realms use that calendar. Rollout length in
# days is n_forward_steps x the realm's step, so the interval an inference block
# reads is [first IC, last IC + rollout].
STEP_DAYS = {"atm": 0.25, "ocn": 5.0}

GAP = ("1990-01-01", "2000-01-01")
FUTURE = "2040-01-01"
LOCKED = "2055-01-01"
RECORD_END = "2065-01-01"


def d(s):
    """Parse a config timestamp to a date, tolerating the T-suffixed form."""
    return datetime.date.fromisoformat(str(s)[:10])


def overlaps(a, b):
    (a0, a1), (b0, b1) = a, b
    return a0 < b1 and b0 < a1


def subsets(node, acc=None):
    """Every `subset` dict anywhere under a dataset definition."""
    acc = [] if acc is None else acc
    if isinstance(node, dict):
        if "subset" in node and isinstance(node["subset"], dict):
            acc.append(node["subset"])
        for key in ("merge", "concat"):
            for member in node.get(key) or []:
                subsets(member, acc)
    elif isinstance(node, list):
        for member in node:
            subsets(member, acc)
    return acc


def windows(node, default_start, default_stop):
    out = []
    found = subsets(node)
    if not found:
        return [(d(default_start), d(default_stop))]
    for sub in found:
        out.append(
            (
                d(sub.get("start_time", default_start)),
                d(sub.get("stop_time", default_stop)),
            )
        )
    return out


def realm_of(runid):
    return runid.split(".")[2]


def audit(path):
    """-> (list of (role, label, start, stop), list of failure strings)."""
    cfg = yaml.safe_load(path.read_text())
    realm = realm_of(path.stem)
    spans, failures = [], []

    for w in windows(cfg["train_loader"]["dataset"], "1940-01-01", RECORD_END):
        spans.append(("gradient", "train_loader", *w))

    validation = cfg["validation"]
    validation = validation if isinstance(validation, list) else [validation]
    for i, entry in enumerate(validation):
        label = entry.get("name") or f"validation[{i}]"
        for w in windows(entry["loader"]["dataset"], "1940-01-01", RECORD_END):
            spans.append(("selection", label, *w))

    for block in cfg.get("inference") or []:
        times = sorted(d(t) for t in block["loader"]["start_indices"]["times"])
        rollout = datetime.timedelta(
            days=block["n_forward_steps"] * STEP_DAYS[realm]
        )
        role = "selection" if block.get("weight", 1.0) > 0 else "reporting"
        spans.append((role, block["name"], times[0], times[-1] + rollout))

    gap = (d(GAP[0]), d(GAP[1]))
    future = (d(FUTURE), d(RECORD_END))
    locked = (d(LOCKED), d(RECORD_END))

    for role, label, start, stop in spans:
        if role == "gradient":
            if overlaps((start, stop), gap):
                failures.append(f"{label}: gradient window {start}..{stop} enters the 1990s gap")
            if overlaps((start, stop), future):
                failures.append(f"{label}: gradient window {start}..{stop} enters 2040+")
        if role == "selection" and stop > d(FUTURE):
            failures.append(f"{label}: selection window {start}..{stop} reaches past {FUTURE}")
        if overlaps((start, stop), locked):
            failures.append(f"{label}: {role} window {start}..{stop} enters the locked window {LOCKED}+")

    return spans, failures


def main():
    args = sys.argv[1:]
    paths = [pathlib.Path(a) for a in args] or sorted(RUNS.glob("*-FT.*.yaml"))
    if not paths:
        print("no stage-2 configs found; generate them first", file=sys.stderr)
        return 1

    bad = 0
    for path in paths:
        spans, failures = audit(path)
        print(f"\n=== {path.stem}")
        for role, label, start, stop in sorted(spans, key=lambda s: (s[0], s[2])):
            print(f"  {role:<10} {label:<16} {start} .. {stop}")
        if failures:
            bad += 1
            for f in failures:
                print(f"  FAIL {f}")
        else:
            print("  ok: no gradient in 1990-2000 or 2040+, no selection past 2040, "
                  f"nothing in {LOCKED}+")
    print()
    if bad:
        print(f"{bad}/{len(paths)} configs violate the split", file=sys.stderr)
        return 1
    print(f"{len(paths)} configs: gradient never sees 1990-2000 or 2040+; "
          f"selection never sees 2040+; {LOCKED}..{RECORD_END} is untouched.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
