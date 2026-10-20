#!/bin/bash
# Spatially parallel fme.ace.evaluator on one node: eval.sh H W CONFIG EXPDIR
here=$(cd "$(dirname "$0")" && pwd)
source "$here/env.sh"
H=$1; W=$2; config=$3; expdir=$4
export FME_DISTRIBUTED_BACKEND=model FME_DISTRIBUTED_H=$H FME_DISTRIBUTED_W=$W
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
exec torchrun --standalone --nnodes 1 --nproc-per-node $((H * W)) \
  -m fme.ace.evaluator "$config" --override experiment_dir="$expdir"
