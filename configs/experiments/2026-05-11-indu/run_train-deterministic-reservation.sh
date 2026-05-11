#!/bin/bash 
#SBATCH --account=e3sm_g
#SBATCH --qos=interactive
#SBATCH --constraint=gpu
#SBATCH --time=120:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --job-name=ace-amip151-deterministic
#SBATCH --output=slurm_%j.out
#SBATCH --reservation=_CAP_e3smaigs

TRAINING_CONFIG=/pscratch/sd/i/imanick/e3sm_ace/ace/config-train-151-deterministic.yaml

source ~/.bashrc
load_env || exit 1
cd /pscratch/sd/i/imanick/e3sm_ace/ace || exit 1

HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)

srun torchrun \
    --nnodes="$SLURM_JOB_NUM_NODES" \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${HEAD_NODE}:29500" \
    -m fme.ace.train "${TRAINING_CONFIG}"