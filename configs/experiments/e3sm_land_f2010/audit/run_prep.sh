#!/bin/bash
# Build the land audit dashboard on one CPU node. Batch:
#   sbatch configs/experiments/e3sm_land_f2010/audit/run_prep.sh   (from the repo root)
# or inside an interactive allocation:
#   salloc -N 1 -C cpu -q interactive -t 02:00:00 -A e3sm
#   bash configs/experiments/e3sm_land_f2010/audit/run_prep.sh
# FORCE=1 rescans months that already have work files (sbatch --export=ALL,FORCE=1 ...).
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
# No git here: discovery can fail on compute nodes. Run via bash, the script
# sits next to prep.py; via sbatch it is a spooled copy, so use the submit
# directory (the repo root).
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [ -f "$HERE/prep.py" ]; then
  PREP="$HERE/prep.py"
else
  PREP="${SLURM_SUBMIT_DIR:-.}/configs/experiments/e3sm_land_f2010/audit/prep.py"
fi
[ -f "$PREP" ] || { echo "prep.py not found; sbatch from the repo root" >&2; exit 1; }
ACE=$(cd "$(dirname "$PREP")/../../../.." && pwd)
export OMP_NUM_THREADS=1
mkdir -p "$ROOT/logs"
cd "$ACE"
echo "$(date) scan";  .venv/bin/python "$PREP" scan  --root "$ROOT" --workers "${WORKERS:-64}" ${FORCE:+--force}
echo "$(date) build"; .venv/bin/python "$PREP" build --root "$ROOT"
echo "$(date) done: open $ROOT/index.html"
