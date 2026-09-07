#!/bin/bash
# Entry point for the scrontab timer. Installed with:
#
#     scrontab -e     # see the block at the bottom of this file
#
# Everything the sweeper needs that an interactive shell would have provided:
# the venv (make_stage2_config.py and check_stage2_leakage.py both import
# yaml), a working directory, and a git identity for the auto-commit. A cron
# job on a workflow node starts from almost nothing, so set it all explicitly
# rather than relying on the profile.
set -uo pipefail

EXP_DIR=/pscratch/sd/m/mahf708/ace/configs/experiments/e3sm_hist_v20260812
VENV=/pscratch/sd/m/mahf708/ace/.venv/bin/activate
LOG=/pscratch/sd/m/mahf708/aug26-ft/sweep-cron.log

# NERSC's login profile does not run here, and run-train.sh stages its config
# under ${PSCRATCH} with `set -u` (run-train.sh:105). Without these the
# submission dies with "PSCRATCH: unbound variable" -- which is exactly what
# the 15:53 tick on 2026-09-07 did, silently, because the sweeper piped
# submit-stage2.sh through a grep that kept only "submitted" and "refusing".
export PSCRATCH=/pscratch/sd/m/mahf708
export SCRATCH=$PSCRATCH

# run-train.sh validates the staged config with `uv run` (run-train.sh:152),
# and uv lives in ~/.local/bin, which only the login profile puts on PATH.
# Same class of failure as PSCRATCH above: it does not appear until a tick
# actually has something to submit, so it cannot be caught by a dry run.
#
# Appended, never prepended, and set before the venv is sourced. The system
# python3 on these nodes predates datetime.date.fromisoformat, so any PATH
# entry that shadows the venv's interpreter makes check_stage2_leakage.py die
# with "type object 'datetime.date' has no attribute 'fromisoformat'" and the
# gate then refuses the submission -- a leakage failure that is really a PATH
# failure, which is a genuinely confusing thing to debug at 3am.
export PATH=${PATH:-/usr/bin:/bin}:$HOME/.local/bin

exec >>"$LOG" 2>&1
echo "===== $(date -Is) ====="

# shellcheck disable=SC1090
source "$VENV" || { echo "no venv at $VENV"; exit 1; }
cd "$EXP_DIR" || exit 1

# The sweeper is idempotent -- it skips run ids that are queued, running, or
# already have output -- so a tick that overlaps a slow one does no harm, and
# scrontab will not start a second copy while this one runs anyway.
./sbatch-scripts/sweep-stage2.sh --go
rc=$?
echo "sweep rc=$rc"
exit $rc
