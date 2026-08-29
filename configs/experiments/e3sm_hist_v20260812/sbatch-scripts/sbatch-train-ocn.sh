#!/bin/bash -l
# Samudra ocean (8 ranks / 2 nodes, local batch 2) -- the B16 baseline
#
# Submit from this directory with the driver, which stages the config and
# validates it before burning an allocation:
#
#     ./run-train.sh ocn
#
# The --nodes below is only the default for an ad-hoc run of the committed
# config. A campaign run is sized from its own config: make_ablation_config.py
# writes FME_NODES into ../runs/<runid>.env and run-train.sh passes --nodes to
# sbatch, which overrides this directive. Do not hardcode a campaign size here.
#
# Submitting this file directly with `sbatch` also works, but then nothing
# validates the config first and $CONFIG_DIR must already be exported.

#SBATCH -A e3sm_g
#SBATCH -q regular
#SBATCH -C gpu&hbm80g          # all three configs require 80 GB cards
#SBATCH -J fme-hist-ocn
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=128
#SBATCH -t 12:00:00
#SBATCH --output=joblogs/%x-%j.out
#SBATCH --signal=USR1@120      # fire the requeue handler 2 min before walltime
#SBATCH --requeue
#SBATCH --open-mode=append

set -x

# Resume by exporting RESUME_JOB_ID=<previous job id>; training then picks up
# from that run's training_checkpoints/ckpt.tar automatically.
# A campaign run is addressed by its RUNID, not by a job id: the chain's
# parameter_init.weights_path points at $CAMPAIGN_ROOT/<parent-runid>, so a
# job-id-keyed output directory would leave every finetune unable to find its
# parent. RUNID is set by run-train.sh when a run id is given; without one the
# old job-id behaviour is kept so single ad-hoc runs are unaffected.
if [ -n "${RUNID:-}" ]; then
    export FME_OUTPUT_DIR=${CAMPAIGN_ROOT:-${PSCRATCH}/aug26}/${RUNID}
elif [ -z "${RESUME_JOB_ID}" ]; then
    export FME_OUTPUT_DIR=${PSCRATCH}/fme-output/hist-ocn-${SLURM_JOB_ID}
else
    export FME_OUTPUT_DIR=${PSCRATCH}/fme-output/hist-ocn-${RESUME_JOB_ID}
fi
mkdir -p "$FME_OUTPUT_DIR"

export TRAIN_CONFIG=${CONFIG_DIR}/${CONFIG_NAME:-config-train-ocn.yaml}
export FME_TORCHRUN=${FME_TORCHRUN:?set by run-train.sh}
export TRAIN_MODULE=fme.ace.train
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_PORT=29508       # distinct per realm: two runs on a node collide at 29500
export FME_OVERRIDE_ARGS="experiment_dir=$FME_OUTPUT_DIR"

# Keep a copy of exactly what ran next to the output.
cp -r "$CONFIG_DIR" "$FME_OUTPUT_DIR/job_config"

srun --nodes=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 --gpus-per-node=4 \
     "$CONFIG_DIR/requeueable-train.sh"
