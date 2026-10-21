#!/bin/bash
# One row of sweeps/scaling-matrix.tsv, inside an existing allocation:
#   scaling_run.sh H W BATCH NODELIST RUN [PORT] [k=v ...]
# Production config at full harmonics, training only (no inline inference,
# 1-week validation), step time logged every 5 steps, allocator memory per
# rank in $PSCRATCH/hr-memlog/RUN. srun is cut at 25 min: ~7 min startup
# plus >= 50 steps. Read the result with scripts/step_time.py RUN.
# Rows with fewer than 4 GPUs: NODELIST is one node and the run uses H*W
# of its GPUs. All rows of one study must run in ONE Slurm allocation
# (a run cannot span two jobs: Slingshot VNI_NOT_FOUND).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
H=$1; W=$2; batch=$3; nodes=$4; run=$5; port=${6:-29500}; shift 6 || shift $#
n=$(scontrol show hostnames "$nodes" | wc -l)
ppn=$(( H * W < 4 && n == 1 ? H * W : 4 ))
export FME_MEMTOOLS=1 FME_MEMLOG_DIR=$PSCRATCH/hr-memlog/$run
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$FME_MEMLOG_DIR" "$here/../logs"
srun --overlap --time=25 -N "$n" -w "$nodes" --ntasks-per-node 1 --gpus-per-node 4 \
  --export=ALL,PORT=$port,NPROC=$ppn \
  "$here/train.sh" "$H" "$W" "$here/../runs/scale-full-1ep-noinf.yaml" "$PSCRATCH/hr-runs/$run" \
  train_loader.batch_size="$batch" validation.loader.batch_size="$batch" \
  "train_loader.dataset.subset.stop_time=0013-12-31T18:00:00" \
  "validation.loader.dataset.subset.stop_time=0014-01-08T00:00:00" \
  log_train_every_n_batches=5 checkpoint_every_n_batches=100000 "$@" \
  > "$here/../logs/$run.out" 2>&1 || true
