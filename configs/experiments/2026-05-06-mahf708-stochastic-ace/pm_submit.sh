#!/bin/bash
#SBATCH -J stochastic-ace-train
#SBATCH -A E3SM
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH -t 2:00:00
#SBATCH -o /lcrc/group/e3sm/ac.ngmahfouz/fme/2026-05-06-mahf708-stochastic-ace/slurm.out

set -e

CONFIG_PATH=configs/experiments/2026-05-06-mahf708-stochastic-ace/pm_config_train.yaml

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)

source $REPO_ROOT/.venv/bin/activate
cd $REPO_ROOT

export FME_DISTRIBUTED_BACKEND=model
export FME_DISTRIBUTED_H=2
export FME_DISTRIBUTED_W=2

srun uv run torchrun \
  --nnodes=$SLURM_NNODES \
  --nproc_per_node=4 \
  --rdzv-backend=c10d \
  --rdzv-endpoint=$MASTER_ADDR:29500 \
  -m fme.ace.train $CONFIG_PATH
