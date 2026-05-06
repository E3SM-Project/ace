#!/bin/bash
#PBS -N stochastic-ace-train
#PBS -A E3SM
#PBS -l select=1:ncpus=128:ngpus=8:mpiprocs=8
#PBS -l walltime=2:00:00
#PBS -j oe
#PBS -o /lcrc/group/e3sm/ac.ngmahfouz/fme/2026-05-06-mahf708-stochastic-ace/pbs.out

set -e

REPO_ROOT=/home/ac.ngmahfouz/ace
CONFIG_PATH=configs/experiments/2026-05-06-mahf708-stochastic-ace/swing_config_train.yaml

NNODES=$(sort -u $PBS_NODEFILE | wc -l)
MASTER_ADDR=$(sort -u $PBS_NODEFILE | head -n1)

pbsdsh -- bash -c "\
  source $REPO_ROOT/.venv/bin/activate && \
  cd $REPO_ROOT && \
  export FME_DISTRIBUTED_BACKEND=model && \
  export FME_DISTRIBUTED_H=2 && \
  export FME_DISTRIBUTED_W=2 && \
  torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=8 \
    --rdzv-backend=c10d \
    --rdzv-endpoint=$MASTER_ADDR:29500 \
    -m fme.ace.train $CONFIG_PATH"
