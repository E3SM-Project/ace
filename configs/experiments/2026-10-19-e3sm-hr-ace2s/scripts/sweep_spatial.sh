#!/bin/bash
# Node-parallel queue of single-node spatial benchmarks. Each list line is
# "H W <bench_model.py args>"; line i runs on node (i mod NNODES).
#   srun -N K --ntasks-per-node 1 --gpus-per-node 4 sweep_spatial.sh LIST OUT
list=$1; out=$2
node=${SLURM_NODEID:-0}; nnodes=${SLURM_NNODES:-1}
mapfile -t lines < <(grep -v '^\s*#' "$list" | grep -v '^\s*$')
for i in "${!lines[@]}"; do
  (( i % nnodes == node )) || continue
  line=${lines[$i]}; gs=ddp
  if [[ $line == GS=* ]]; then gs=${line%% *}; gs=${gs#GS=}; line=${line#* }; fi
  FME_SPATIAL_GRAD_SYNC=$gs SINGLE=1 timeout 1200 "$(dirname "$0")/bench_spatial.sh" $line \
    | sed -n 's/^RESULT //p' >> "$out"
done
