#!/bin/bash
# torchrun wrapper for transpose_bench.py; one task per node.
source "$(dirname "$0")/env.sh"
[[ ${OFI:-1} == 1 ]] && source "$(dirname "$0")/nccl-ofi.sh"
master=$(scontrol show hostnames "$SLURM_STEP_NODELIST" | head -1)
torchrun --nnodes "$SLURM_NNODES" --nproc-per-node "${NPROC:-4}" --rdzv-backend c10d \
  --rdzv-endpoint "$master:${PORT:-29800}" --rdzv-id "tr${PORT:-29800}" \
  "$HR_EXP/scripts/transpose_bench.py" 2>&1 | grep -E "^RESULT|Error"
