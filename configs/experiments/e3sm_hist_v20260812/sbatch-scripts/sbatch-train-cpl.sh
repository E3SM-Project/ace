#!/bin/bash -l
# coupled finetune (8 ranks, local batch 1)
#
# Submit from this directory with the driver, which stages the config and
# validates it before burning an allocation:
#
#     ./run-train.sh cpl
#
# Submitting this file directly with `sbatch` also works, but then nothing
# validates the config first and $CONFIG_DIR must already be exported.

#SBATCH -A e3sm_g
#SBATCH -q regular
#SBATCH -C gpu&hbm80g          # all three configs require 80 GB cards
#SBATCH -J fme-hist-cpl
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=128
#SBATCH -t 12:00:00
#SBATCH --output=joblogs/%x-%j.out
#SBATCH --signal=USR1@120      # fire the requeue handler 2 min before walltime
#SBATCH --requeue
#SBATCH --open-mode=append

set -x

# Resume by exporting RESUME_JOB_ID=<previous job id>; training then picks up
# from that run's training_checkpoints/ckpt.tar automatically.
if [ -z "${RESUME_JOB_ID}" ]; then
    export FME_OUTPUT_DIR=${PSCRATCH}/fme-output/hist-cpl-${SLURM_JOB_ID}
else
    export FME_OUTPUT_DIR=${PSCRATCH}/fme-output/hist-cpl-${RESUME_JOB_ID}
fi
mkdir -p "$FME_OUTPUT_DIR"

export TRAIN_CONFIG=${CONFIG_DIR}/config-train-cpl.yaml
export FME_TORCHRUN=${FME_TORCHRUN:?set by run-train.sh}
export TRAIN_MODULE=fme.coupled.train
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_PORT=29509       # distinct per realm: two runs on a node collide at 29500
export FME_OVERRIDE_ARGS="experiment_dir=$FME_OUTPUT_DIR"

# Banner. `set -x` makes the log a wall of trace, and every atmosphere run
# used to be named fme-hist-atm, so a log told you almost nothing about which
# of 35 runs it was. Grep the log for "=== run" to get the identity back.
{
  echo "=== run ==========================================================="
  echo "runid        ${RUNID:-<ad-hoc, no run id>}"
  echo "job          ${SLURM_JOB_NAME:-?} / ${SLURM_JOB_ID:-?}"
  echo "restarts     ${SLURM_RESTART_COUNT:-0}"
  echo "nodes        ${SLURM_JOB_NUM_NODES:-?} (${SLURM_JOB_NODELIST:-?})"
  echo "ranks        $(( ${SLURM_JOB_NUM_NODES:-1} * 4 ))"
  echo "config       $TRAIN_CONFIG"
  echo "output       $FME_OUTPUT_DIR"
  echo "commit       $(cat "$CONFIG_DIR/COMMIT" 2>/dev/null || echo unknown)"
  echo "wandb        ${WANDB_NAME:-<unset>} in ${WANDB_RUN_GROUP:-<unset>}"
  echo "started      $(date -Is)"
  echo "==================================================================="
} >&2

# Keep a copy of exactly what ran next to the output.
cp -r "$CONFIG_DIR" "$FME_OUTPUT_DIR/job_config"

srun --nodes=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 --gpus-per-node=4 \
     "$CONFIG_DIR/requeueable-train.sh"
