#!/bin/bash 
#SBATCH --account=e3sm_g
#SBATCH --qos=debug
#SBATCH --constraint=gpu
#SBATCH --time=120:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --job-name=every_5yrs-ace-amip101
#SBATCH --output=slurm_%j.out
#SBATCH --reservation=_CAP_e3smaigs

# replace this with the actual command
# in the form of: 
# srun --gpus-per-node=2 torchrun --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=4 -m fme ...

TRAINING_CONFIG=/pscratch/sd/o/olawale/E3SM_ace/ace/configs/experiments/2026-05-11-olawale-amip101_every_5yrs/config-train-amip101.yaml
ACE_ROOT=/pscratch/sd/o/olawale/E3SM_ace/ace

# wandb config
export WANDB_NAME=amip101_training_evry_5yrs
export WANDB_RUN_GROUP=amip101_1951_2015
export WANDB_API_KEY=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

source "${ACE_ROOT}/.venv/bin/activate" || exit 1
cd "${ACE_ROOT}" || exit 1


HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)

srun torchrun \
    --nnodes="$SLURM_JOB_NUM_NODES" \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${HEAD_NODE}:29500" \
    -m fme.ace.train "${TRAINING_CONFIG}"
    
# Resubmit as a chained job
#sbatch --dependency=afterok:${SLURM_JOB_ID} "$0"