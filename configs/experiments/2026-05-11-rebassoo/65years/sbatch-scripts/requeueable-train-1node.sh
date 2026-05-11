#!/bin/bash

set -x

# this will manually requeue the job and is called if a timeout signal is received
# see https://docs.nersc.gov/jobs/examples/#preemptible-jobs
preempt_handler()
{
    #place here: commands to run when preempt signal (SIGTERM) arrives from slurm
    kill -TERM ${1} #forward SIGTERM signal to the user application
    #if --requeue was used, slurm will automatically do so here
}
timeout_handler()
{
    kill -TERM ${1}
    scontrol requeue ${SLURM_JOB_ID}
}


#podman-hpc run --rm --gpu --ipc host \
podman-hpc run --rm --openmpi-pmi2 --gpu --ipc host \
    -v "${CONFIG_DIR}:/configmount" \
    -v "${FME_OUTPUT_DIR}:/output" \
    -v "${FME_TRAIN_DIR}:/traindata" \
    -v "${FME_VALID_DIR}:/validdata" \
    -v "${FME_STATS_DIR}:/statsdata" \
    --env 'WANDB*' \
    --env 'SLURM*' \
    --env 'FME*' \
    $FME_IMAGE \
    torchrun --nproc_per_node 4\
    --nnodes 1\
#    --rdzv_id $SLURM_JOB_ID\
    --rdzv_backend c10d\
    --rdzv_endpoint $head_node:29500\
    -m fme.ace.train /configmount/train-config.yaml \
    &

pid=$!
trap "preempt_handler '$pid'" SIGTERM #this catches preempt SIGTERM from slurm
trap "timeout_handler '$pid'" USR1 # this catches timeout USR1 from slurm
wait
sleep 120
