#!/bin/bash
# Run a list of bench_model.py argument lines across every GPU in the
# allocation. Line i goes to global GPU (i mod NGPU); each GPU runs its share
# serially. Usage (inside the allocation):
#   srun --overlap -N $SLURM_NNODES --ntasks-per-node 1 --gpus-per-node 4 \
#       scripts/sweep.sh sweeps/<name>.txt logs/<name>.jsonl
set -uo pipefail
list=$1; out=$2
source "$(dirname "$0")/env.sh"
node=${SLURM_NODEID:-0}; nnodes=${SLURM_NNODES:-1}
ngpu=$((nnodes * 4))
mapfile -t lines < <(grep -v '^\s*#' "$list" | grep -v '^\s*$')
for g in 0 1 2 3; do
  gid=$((node * 4 + g))
  (
    for i in "${!lines[@]}"; do
      (( i % ngpu == gid )) || continue
      CUDA_VISIBLE_DEVICES=$g python "$HR_EXP/scripts/bench_model.py" ${lines[$i]} 2>&1 \
        | grep '^RESULT ' | sed 's/^RESULT //' >> "$out"
    done
  ) &
done
wait
