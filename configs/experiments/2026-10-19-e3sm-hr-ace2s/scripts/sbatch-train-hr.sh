#!/bin/bash
# Production launch for the HR ACE2S config: N nodes, 4x1 spatial tiles per
# node (one data-parallel replica per node), batch_size = N.
#
#   cd configs/experiments/2026-10-19-e3sm-hr-ace2s
#   RUN=hr-e01 sbatch -N 8 scripts/sbatch-train-hr.sh
#
# STATUS: written and checked with `sbatch --test-only`, NOT yet run through
# the queue. The same train.sh ran interactively on 1 and 4 nodes on
# 2026-10-19.
#
# Requeue: the spatial backend does not save a restart checkpoint on SIGTERM
# (fme/core/generics/trainer.py skips it when has_spatial_parallelism), so a
# requeued job resumes from the last `checkpoint_every_n_batches` save. This
# script requeues itself at T-5 min and relies on that interval: 400 batches
# at ~2.5 s is ~17 min lost at most.
#SBATCH -A e3sm_g
#SBATCH -q regular
#SBATCH -C gpu&hbm80g          # needs 80 GB cards: ~60 GB/GPU with modes x0.5
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

srun --ntasks-per-node 1 --gpus-per-node 4 --export=ALL,ACE_SRC="$ACE_SRC" \
  "$here/scripts/train.sh" 4 1 "$config" "$expdir" \
  train_loader.batch_size="$n" validation.loader.batch_size="$n" \
  checkpoint_every_n_batches=400 &
wait
