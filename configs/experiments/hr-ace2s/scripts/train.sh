#!/bin/bash
# Launch fme.ace.train with the spatial backend, one torchrun per node:
#   srun -N <nodes> --ntasks-per-node 1 --gpus-per-node 4 scripts/train.sh \
#       H W CONFIG EXPERIMENT_DIR [k=v ...]
# Extra k=v pairs are appended to the single --override list (argparse keeps
# only the last --override flag, so never pass a second one).
# World = 4 * nodes; the data-parallel size is world / (H * W), and
# train_loader.batch_size must be a multiple of it.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
source "$here/env.sh"
source "$here/nccl-ofi.sh"  # without it inter-node NCCL falls back to TCP
H=$1; W=$2; config=$3; expdir=$4; shift 4
export FME_DISTRIBUTED_BACKEND=model FME_DISTRIBUTED_H=$H FME_DISTRIBUTED_W=$W
# covers the rank skew of inline inference and checkpointing
export FME_DIST_TIMEOUT_MINUTES=${FME_DIST_TIMEOUT_MINUTES:-120}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
master=$(scontrol show hostnames "$SLURM_STEP_NODELIST" | head -1)
mkdir -p "$expdir"
exec torchrun --nnodes "$SLURM_NNODES" --nproc-per-node "${NPROC:-4}" \
  --rdzv-backend c10d --rdzv-endpoint "$master:${PORT:-29900}" \
  --rdzv-id "train${PORT:-29900}" \
  -m fme.ace.train "$config" --override experiment_dir="$expdir" "$@"
