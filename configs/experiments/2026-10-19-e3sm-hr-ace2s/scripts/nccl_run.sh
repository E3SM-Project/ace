#!/bin/bash
source "$(dirname "$0")/env.sh"
[[ ${OFI:-0} == 1 ]] && source "$(dirname "$0")/nccl-ofi.sh"
master=$(scontrol show hostnames "$SLURM_STEP_NODELIST" | head -1)
torchrun --nnodes "$SLURM_NNODES" --nproc-per-node 4 --rdzv-backend c10d \
  --rdzv-endpoint "$master:${PORT:-29700}" --rdzv-id "nccl${PORT:-29700}" \
  "$HR_EXP/scripts/nccl_bench.py" 2>&1 | grep -E "RESULT|Error|NCCL WARN"
