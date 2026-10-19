#!/bin/bash
# Build the land audit dashboard on one CPU node. Batch:
#   sbatch configs/experiments/e3sm_land_f2010/audit/run_prep.sh
# or inside an interactive allocation:
#   salloc -N 1 -C cpu -q interactive -t 02:00:00 -A e3sm
#   bash configs/experiments/e3sm_land_f2010/audit/run_prep.sh
# Resumable: months already in work/scan are skipped, so a timed-out job can
# simply be resubmitted. Reads ~2.1 TB (elm.h0 + x2l, both cases), I/O bound.
#SBATCH -A e3sm
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH -J land-audit
#SBATCH --nodes=1
#SBATCH --cpus-per-task=256
#SBATCH -t 03:00:00
#SBATCH --output=/pscratch/sd/i/imanick/land-inference/dashboard/logs/%x-%j.out
set -euo pipefail
ROOT=${ROOT:-/pscratch/sd/i/imanick/land-inference/dashboard}
ACE=$(git -C "${SLURM_SUBMIT_DIR:-$(dirname "$0")}" rev-parse --show-toplevel)
PREP="$ACE/configs/experiments/e3sm_land_f2010/audit/prep.py"
export OMP_NUM_THREADS=1
mkdir -p "$ROOT/logs"
cd "$ACE"
echo "$(date) scan";  .venv/bin/python "$PREP" scan  --root "$ROOT" --workers "${WORKERS:-64}"
echo "$(date) build"; .venv/bin/python "$PREP" build --root "$ROOT"
echo "$(date) done: open $ROOT/index.html"
