#!/bin/bash
# Spatial-backend benchmark launcher. Usage:
#   bench_spatial.sh H W <bench_model.py args...>
# SINGLE=1 (default): torchrun --standalone on this node's 4 GPUs.
# SINGLE=0: multi-node; run under `srun --ntasks-per-node 1`, rendezvous on
#           the first node of the step.
source "$(dirname "$0")/env.sh"
# aws-ofi-nccl: without it NCCL uses TCP over hsn (2.2 vs 67.8 GB/s busbw)
[[ ${OFI:-1} == 1 ]] && source "$(dirname "$0")/nccl-ofi.sh"
H=$1; W=$2; shift 2
export FME_DISTRIBUTED_BACKEND=model FME_DISTRIBUTED_H=$H FME_DISTRIBUTED_W=$W
if [[ ${SINGLE:-1} == 1 ]]; then
  launch=(--standalone --nnodes 1)
else
  master=$(scontrol show hostnames "$SLURM_STEP_NODELIST" | head -1)
  launch=(--nnodes "$SLURM_NNODES" --rdzv-backend c10d
          --rdzv-endpoint "$master:${PORT:-29511}" --rdzv-id "bench${PORT:-29511}")
fi
tag=run; prev=; for x in "$@"; do [[ $prev == --tag ]] && tag=$x; prev=$x; done
log=$HR_EXP/logs/runs/$tag.log
torchrun "${launch[@]}" --nproc-per-node 4 "$HR_EXP/scripts/bench_model.py" "$@" > "$log" 2>&1
grep -E '^RESULT' "$log" || { echo "FAILED $tag: see $log"; grep -m3 -E 'Error|error' "$log"; }
