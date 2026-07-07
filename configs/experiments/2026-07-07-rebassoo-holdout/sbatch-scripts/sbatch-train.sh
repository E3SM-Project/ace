#!/bin/bash -l

#SBATCH -A e3sm_g
#SBATCH -q regular
##SBATCH -q preempt
#SBATCH -C gpu&hbm80g
##SBATCH -C gpu
#SBATCH -J train-fme
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=128
#SBATCH -t 23:00:00
##SBATCH --reservation=_CAP_e3smaigs
#SBATCH --output=joblogs/%j.out
#SBATCH --signal=USR1@60
#SBATCH --requeue
#SBATCH --open-mode=append

# for pre-emptible jobs, update -q to preempt

set -xe

# directory for saving output from training/inference job
if [ -z "${RESUME_JOB_ID}" ]; then
  export FME_OUTPUT_DIR=${PSCRATCH}/fme-output/${SLURM_JOB_ID}
else
  export FME_OUTPUT_DIR=${PSCRATCH}/fme-output/${RESUME_JOB_ID}
fi
mkdir -p $FME_OUTPUT_DIR


export MASTER_ADDR=$(hostname)
export MASTER_PORT=29507

#export head_node=$(hostname)
#export NCCL_DEBUG=INFO
#export MPICH_SMP_SINGLE_COPY_MODE=XPMEM

# TODO: automate

#fme_venv=$($CONFIG_DIR/make-venv.sh $COMMIT | tail -n 1)
#module load python
#conda activate $fme_venv

# env variables
export WANDB_JOB_TYPE=training
set +x  # don't print API key to logs
export WANDB_API_KEY=$(cat ~/.config/wandb/api)
set -x

cp -r $CONFIG_DIR $FME_OUTPUT_DIR/job_config

# run the requeueable job
srun -u $CONFIG_DIR/requeueable-train.sh

sleep 120
