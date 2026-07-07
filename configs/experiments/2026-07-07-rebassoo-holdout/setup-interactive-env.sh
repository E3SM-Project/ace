#!/bin/bash
# Source this script (do NOT execute it) to set up env vars for interactive runs:
#   source setup-interactive-env.sh

set -x

export FME_IMAGE=e3sm-ace:653607f3

export FME_TRAIN_DIR=/pscratch/sd/o/olawale/E3SM_data/data_processing/output/amip/amip_101
export FME_VALID_DIR=/pscratch/sd/o/olawale/E3SM_data/data_processing/output/amip/amip_101/traindata
export FME_STATS_DIR=/pscratch/sd/o/olawale/E3SM_data/data_processing/output/amip/amip_101/e3sm-amip101-stats/amip101_1951_2015

UUID=$(uuidgen)
export CONFIG_DIR=${PSCRATCH}/fme-config/${UUID}
mkdir -p $CONFIG_DIR

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cp $SCRIPT_DIR/config-train-finn-stochastic.yaml $CONFIG_DIR/train-config.yaml
cp $SCRIPT_DIR/sbatch-scripts/requeueable-train.sh $CONFIG_DIR/requeueable-train.sh

export FME_OUTPUT_DIR=${PSCRATCH}/fme-output/interactive-test-${UUID}
mkdir -p $FME_OUTPUT_DIR

export MASTER_ADDR=$(srun --ntasks=1 hostname)
export MASTER_PORT=29507

export WANDB_JOB_TYPE=training
set +x
export WANDB_API_KEY=$(cat ~/.config/wandb/api)
set -x

echo "Environment ready. Run with:"
echo "  srun -u $CONFIG_DIR/requeueable-train.sh"
