#!/bin/bash
# Train the HR ACE2S config on N nodes with HxW spatial tiles per sample
# (default 4x1, one replica per node). Submit from this folder:
#
#   cd configs/experiments/hr-ace2s
#   RUN=<name> sbatch -N 8 -C "gpu&hbm40g" scripts/sbatch-train-hr.sh
#   RUN=<name> LAYOUT=8x1 sbatch -N 16 -C "gpu&hbm40g" scripts/sbatch-train-hr.sh
#
# Knobs (environment, all optional except RUN):
#   RUN         run name; the run lives in EXPDIR (default $PSCRATCH/hr-runs/$RUN)
#   LAYOUT      HxW tiles per sample (default 4x1); a tile group wider than a
#               node spans consecutive nodes. Replicas = 4 N / (H W).
#   BATCH       global batch size (default: the number of replicas, one sample
#               per replica)
#   CKPT_EVERY  checkpoint_every_n_batches (default 400)
#   CONFIG      training config (default config-train-hr.yaml)
#   EXTRA       extra space-separated k=v overrides passed to fme
#
# The first launch copies fme/ and the config into EXPDIR, and every requeue
# runs that copy. Under spatial parallelism no restart checkpoint is written on
# SIGTERM, so the job requeues itself at T-5 min and resumes from the last
# checkpoint_every_n_batches save.
#SBATCH -A e3sm_g
#SBATCH -q regular
#SBATCH -C gpu
#SBATCH -J fme-hr-atm
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH -t 48:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
here=$SLURM_SUBMIT_DIR
if [[ ! -x $here/scripts/train.sh ]]; then
  echo "submit from configs/experiments/hr-ace2s" >&2
  exit 1
fi
repo=$(cd "$here/../../.." && pwd)
run=${RUN:?set RUN=<name>}
expdir=${EXPDIR:-$PSCRATCH/hr-runs/$run}
n=$SLURM_NNODES
layout=${LAYOUT:-4x1}
h=${layout%x*}
w=${layout#*x}
if (( 4 * n % (h * w) != 0 )); then
  echo "LAYOUT=$layout does not divide the $((4 * n)) GPUs" >&2
  exit 1
fi
replicas=$((4 * n / (h * w)))
batch=${BATCH:-$replicas}
if (( batch % replicas != 0 )); then
  echo "BATCH=$batch must be a multiple of the $replicas replicas" >&2
  exit 1
fi

# Freeze the code and config once per run, so a requeue trains the same thing.
mkdir -p "$expdir"
if [[ ! -d $expdir/src/fme ]]; then
  rm -rf "$expdir/src.partial"
  mkdir -p "$expdir/src.partial"
  rsync -a --exclude __pycache__ "$repo/fme" "$expdir/src.partial/"
  # the commit, plus any uncommitted changes to fme/
  { git -C "$repo" rev-parse HEAD && git -C "$repo" status --short -- fme; } \
    > "$expdir/src.partial/VERSION" || true
  cp "${CONFIG:-$here/config-train-hr.yaml}" "$expdir/src.partial/config.yaml"
  mv "$expdir/src.partial" "$expdir/src"
fi
export ACE_SRC=$expdir/src

requeue() {
  echo "$(date) walltime signal: requeueing $SLURM_JOB_ID"
  scontrol requeue "$SLURM_JOB_ID"
}
trap requeue USR1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # needed on 40 GB cards
# shellcheck disable=SC2086
srun --ntasks-per-node 1 --gpus-per-node 4 --export=ALL,ACE_SRC="$ACE_SRC" \
  "$here/scripts/train.sh" "$h" "$w" "$ACE_SRC/config.yaml" "$expdir" \
  train_loader.batch_size="$batch" validation.loader.batch_size="$batch" \
  checkpoint_every_n_batches="${CKPT_EVERY:-400}" ${EXTRA:-} &
# in the background so the USR1 trap runs; the job's exit status is srun's
wait "$!"
