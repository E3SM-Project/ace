
#!/bin/bash

set -x

#export FME_IMAGE=2025081104-hplugin:latest
export FME_IMAGE=e3sm-ace:653607f3

# wandb config
export WANDB_NAME=training1-65year
export WANDB_RUN_GROUP=MayHackathon-e3smACE-stochastic-training-amip101

# directories for input data (training, validation, stats)
export FME_TRAIN_DIR=/pscratch/sd/o/olawale/E3SM_data/data_processing/output/amip/amip_101
export FME_VALID_DIR=/pscratch/sd/o/olawale/E3SM_data/data_processing/output/amip/amip_101/traindata
#export FME_STATS_DIR=/pscratch/sd/r/rebassoo/fme-preprocess/2025-10-16-scream-masked-gfdlnames/2025-10-16-scream
export FME_STATS_DIR=/pscratch/sd/o/olawale/E3SM_data/data_processing/output/amip/amip_101/e3sm-amip101-stats/amip101_1951_2015
#export FME_STATS_DIR=/pscratch/sd/r/rebassoo/fme-preprocess//2025-04-01-e3smv3-1deg/2025-04-01-e3smv3-1deg

# if resuming a failed job, provide its slurm job ID below and uncomment;
# note that information entered above should be consistent with that of
# the failed job
#export RESUME_JOB_ID=52140742

# user should not need to modify below

# copy config to staging area so that local changes between job submission
# and job start will not effect the run
UUID=$(uuidgen)
export CONFIG_DIR=${PSCRATCH}/fme-config/${UUID}
mkdir -p $CONFIG_DIR
if [ -z "${RESUME_JOB_ID}" ]; then
    cp config-train-finn-stochastic.yaml $CONFIG_DIR/train-config.yaml
    #cp config-train-finn-parallel.yaml $CONFIG_DIR/train-config.yaml
    #cp config-train-naser-parallel.yaml $CONFIG_DIR/train-config.yaml
else
    cp ${PSCRATCH}/fme-output/${RESUME_JOB_ID}/job_config/train-config.yaml $CONFIG_DIR/train-config.yaml
fi
cp run-train-perlmutter.sh $CONFIG_DIR/run-train-perlmutter.sh  # copy for reproducibility/tracking
cp sbatch-scripts/requeueable-train.sh $CONFIG_DIR/requeueable-train.sh

#sbatch -t 00:30:00 -q debug sbatch-scripts/sbatch-train.sh  # use this for debugging config/submission
#sbatch -t 05:00:00 sbatch-scripts/sbatch-train.sh
sbatch sbatch-scripts/sbatch-train.sh
