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

#export FME_DISTRIBUTED_BACKEND=model 
#export FME_DISTRIBUTED_H=2 
#export FME_DISTRIBUTED_W=2
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=10800
export NCCL_IB_TIMEOUT=7200
#export NCCL_DEBUG=INFO

#podman-hpc run --rm --openmpi-pmi2 --gpu --ipc host \
podman-hpc run --rm --gpu --nccl --ipc host \
    --network=host \
    -v "${CONFIG_DIR}:/configmount" \
    -v "${FME_OUTPUT_DIR}:/output" \
    -v "${FME_TRAIN_DIR}:/traindata" \
    -v "${FME_VALID_DIR}:/validdata" \
    -v "${FME_STATS_DIR}:/statsdata" \
    --env 'WANDB*' \
    --env 'SLURM*' \
    --env 'FME*' \
    --env 'NCCL*' \
    $FME_IMAGE \
    torchrun --nproc_per_node $SLURM_GPUS_PER_NODE \
    --nnodes $SLURM_JOB_NUM_NODES\
    --rdzv-backend=c10d \
    --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
    -m fme.ace.train /configmount/train-config.yaml \
    &


#    --rdzv_id $SLURM_JOB_ID\
#    --rdzv_backend c10d\
#    --rdzv_endpoint $head_node:29500\

    
pid=$!
trap "preempt_handler '$pid'" SIGTERM #this catches preempt SIGTERM from slurm
trap "timeout_handler '$pid'" USR1 # this catches timeout USR1 from slurm
wait
sleep 120
