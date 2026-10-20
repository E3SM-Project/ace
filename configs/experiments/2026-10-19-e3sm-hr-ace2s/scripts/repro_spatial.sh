#!/bin/bash
# repro_spatial.sh H W <repro_check.py args>  (one node, 4 ranks)
source "$(dirname "$0")/env.sh"
H=$1; W=$2; shift 2
export FME_DISTRIBUTED_BACKEND=model FME_DISTRIBUTED_H=$H FME_DISTRIBUTED_W=$W
torchrun --standalone --nnodes 1 --nproc-per-node $((H*W)) "$HR_EXP/scripts/repro_check.py" "$@" 2>&1 | grep -E "^RESULT|Error"
