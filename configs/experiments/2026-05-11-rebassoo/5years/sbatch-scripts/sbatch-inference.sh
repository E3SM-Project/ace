#!/bin/bash -l

#SBATCH -A m4492_g
#SBATCH -q shared
#SBATCH -C gpu_hbm40g
#SBATCH -J infer-fme
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -t 01:00:00
#SBATCH --output=joblogs/%j.out

set -xe

# directory for saving output from training/inference job
export FME_OUTPUT_DIR=${PSCRATCH}/fme-output/${SLURM_JOB_ID}
mkdir -p $FME_OUTPUT_DIR

# env variables
export WANDB_JOB_TYPE=inference
set +x  # don't print API key to logs
export WANDB_API_KEY=$(cat ~/.config/wandb/api)
set -x

srun -u podman-hpc run --rm --gpu --ipc host \
    -v "${CONFIG_DIR}:/configmount" \
    -v "${FME_VALID_DIR}:/validdata" \
    -v "${FME_CHECKPOINT_PATH}:/ckpt.tar" \
    -v "${FME_OUTPUT_DIR}:/output" \
    --env 'WANDB*' \
    --env 'SLURM*' \
    --env 'OMP_NUM_THREADS' \
    --env 'FME*' \
    $FME_IMAGE \
    python -u -m fme.ace.evaluator /configmount/config-inference.yaml
