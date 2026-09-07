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
