#!/bin/bash
# Copy the HR EAM history files (atmosphere only) from CFS to $PSCRATCH.
#
# Run inside an allocation, one task per node:
#   srun --overlap -N "$SLURM_NNODES" --ntasks-per-node 1 scripts/stage-data.sh
# Each task takes every Nth file (N = number of tasks) and copies PAR files at a
# time. A file is written as .partial and renamed when complete, so rerunning
# skips what finished and redoes only what was interrupted.
set -euo pipefail
SRC=${SRC:-/global/cfs/cdirs/e3smdata/simulations/v3.HR.F2010.aigo.PD.y1/run}
DST=${DST:-$PSCRATCH/hr-data/v3.HR.F2010.aigo.PD.y1/eam.h0}
PAR=${PAR:-3}
rank=${SLURM_PROCID:-0}; ntask=${SLURM_NTASKS:-1}
mkdir -p "$DST"
mapfile -t files < <(cd "$SRC" && ls v3.HR.F2010.aigo.PD.y1.eam.h0.*.nc | sort)
mine=()
for i in "${!files[@]}"; do (( i % ntask == rank )) && mine+=("${files[$i]}"); done
copy_one() {
  local f=$1
  local s; s=$(stat -c %s "$SRC/$f")
  if [[ -f "$DST/$f" && $(stat -c %s "$DST/$f") == "$s" ]]; then return 0; fi
  local t0=$SECONDS
  cp "$SRC/$f" "$DST/$f.partial" && mv "$DST/$f.partial" "$DST/$f"
  echo "$(date +%T) $(hostname) $f $((s/1000000)) MB $((SECONDS-t0)) s"
}
export -f copy_one; export SRC DST
printf '%s\n' "${mine[@]}" | xargs -P "$PAR" -I{} bash -c 'copy_one {}'
