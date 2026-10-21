#!/bin/bash
# Production launch for the HR ACE2S config: N nodes, 4x1 spatial tiles per
# node (one data-parallel replica per node), batch_size = N.
#
#   cd configs/experiments/2026-10-19-e3sm-hr-ace2s
#   RUN=hr-e01 sbatch -N 8 scripts/sbatch-train-hr.sh
#
# STATUS: see notes/comm-and-memory.md section 15 for the queued
# walltime/requeue test (2026-10-20). train.sh itself ran interactively on
# 1-4 nodes, on 80 GB and 40 GB cards.
#
# Environment knobs (all optional):
#   CKPT_EVERY  checkpoint_every_n_batches (default 400, ~33 min at 5 s/step)
#   EXTRA       extra space-separated k=v overrides passed to fme
#
# Requeue: the spatial backend does not save a restart checkpoint on SIGTERM
# (fme/core/generics/trainer.py skips it when has_spatial_parallelism), so a
# requeued job resumes from the last `checkpoint_every_n_batches` save. This
# script requeues itself at T-5 min and relies on that interval: 400 batches
# at 5.0 s (full harmonics, 40 GB) is ~33 min lost at most.
#SBATCH -A e3sm_g
#SBATCH -q regular
#SBATCH -C gpu&hbm80g          # 40 GB cards also work with the sharding flags
                               # (full harmonics: 35 GiB allocated at 4x1)
#SBATCH -J fme-hr-atm
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH -t 48:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --output=logs/joblogs/%x-%j.out
set -euo pipefail
here=$SLURM_SUBMIT_DIR
run=${RUN:?set RUN=<name>}
config=${CONFIG:-$here/config-train-hr.yaml}
expdir=${EXPDIR:-$PSCRATCH/hr-runs/$run}
n=$SLURM_NNODES
mkdir -p "$expdir" "$here/logs/joblogs"
# Freeze the code once per run, so a requeue keeps training the same code.
if [[ ! -f $expdir/ACE_SRC ]]; then
  "$here/scripts/snapshot.sh" > "$expdir/ACE_SRC"
fi
export ACE_SRC=$(cat "$expdir/ACE_SRC")

requeue() {
  echo "$(date) walltime signal: requeueing $SLURM_JOB_ID"
  scontrol requeue "$SLURM_JOB_ID"
}
trap requeue USR1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # needed on 40 GB cards
# shellcheck disable=SC2086
srun --ntasks-per-node 1 --gpus-per-node 4 --export=ALL,ACE_SRC="$ACE_SRC" \
  "$here/scripts/train.sh" 4 1 "$config" "$expdir" \
  train_loader.batch_size="$n" validation.loader.batch_size="$n" \
  checkpoint_every_n_batches="${CKPT_EVERY:-400}" ${EXTRA:-} &
wait
