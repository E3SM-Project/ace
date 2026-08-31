#!/bin/bash -l
# ACE2S atmosphere (16 ranks / 4 nodes, local batch 1) -- the B16 baseline
#
# Submit from this directory with the driver, which stages the config and
# validates it before burning an allocation:
#
#     ./run-train.sh atm
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
#SBATCH -J fme-hist-atm
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=128
#SBATCH -t 12:00:00
#SBATCH --output=joblogs/%x-%j.out
#SBATCH --signal=B:USR1@300    # walltime requeue; see the trap at the bottom.
                               # B: = batch shell ONLY. Without it Slurm signals
                               # every process in the step, the 16 python ranks
                               # included, and SIGUSR1 has no handler in FME
                               # (shutdown.py TERMINATION_SIGNALS is SIGTERM,
                               # SIGINT) so they die at default disposition
                               # before writing a restart checkpoint. Measured
                               # 2026-08-30 on job 57758390: REAL_EXIT=138
                               # (128+SIGUSR1) and an empty training_checkpoints/.
                               # 300 s rather than 120: the lead time has to
                               # cover a collective teardown, a ~31 s checkpoint
                               # write and the step's own exit, and a requeue
                               # that loses the race to the walltime does not
                               # happen at all -- TIMEOUT is terminal and
                               # --requeue does not cover it.
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
    export FME_OUTPUT_DIR=${PSCRATCH}/fme-output/hist-atm-${SLURM_JOB_ID}
else
    export FME_OUTPUT_DIR=${PSCRATCH}/fme-output/hist-atm-${RESUME_JOB_ID}
fi
mkdir -p "$FME_OUTPUT_DIR"

export TRAIN_CONFIG=${CONFIG_DIR}/${CONFIG_NAME:-config-train-atm.yaml}
export FME_TORCHRUN=${FME_TORCHRUN:?set by run-train.sh}
export TRAIN_MODULE=fme.ace.train
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_PORT=29507       # distinct per realm: two runs on a node collide at 29500
export FME_OVERRIDE_ARGS="experiment_dir=$FME_OUTPUT_DIR"

# Banner. `set -x` makes the log a wall of trace, and every atmosphere run
# used to be named fme-hist-atm, so a log told you almost nothing about which
# of 35 runs it was. Grep the log for "=== run" to get the identity back.
{
  echo "=== run ==========================================================="
  echo "runid        ${RUNID:-<ad-hoc, no run id>}"
  echo "job          ${SLURM_JOB_NAME:-?} / ${SLURM_JOB_ID:-?}"
  echo "restarts     ${SLURM_RESTART_COUNT:-0}"
  echo "nodes        ${SLURM_JOB_NUM_NODES:-?} (${SLURM_JOB_NODELIST:-?})"
  echo "ranks        $(( ${SLURM_JOB_NUM_NODES:-1} * 4 ))"
  echo "config       $TRAIN_CONFIG"
  echo "output       $FME_OUTPUT_DIR"
  echo "commit       $(cat "$CONFIG_DIR/COMMIT" 2>/dev/null || echo unknown)"
  echo "config sha   $(cat "$CONFIG_DIR/CONFIG_SHA256" 2>/dev/null || echo unknown)"
  echo "wandb        ${WANDB_NAME:-<unset>} in ${WANDB_RUN_GROUP:-<unset>}"
  echo "started      $(date -Is)"
  echo "==================================================================="
} >&2

# Keep a copy of exactly what ran next to the output. `cp -r src dst` nests
# as dst/<uuid> once dst exists, so a requeued or resumed segment used to
# bury its config one level deeper each time -- and run-train.sh now reads
# job_config/<config> back to refuse a mismatched resume.
mkdir -p "$FME_OUTPUT_DIR/job_config"
cp -r "$CONFIG_DIR/." "$FME_OUTPUT_DIR/job_config/"

# Walltime requeue. The trap turns USR1 into the SIGTERM that every layer below
# already handles -- requeueable-train.sh's preempt_handler, torchrun's agent,
# and FME's handler, which tears the collectives down and then writes the
# restart checkpoint. The requeue lives here rather than in
# requeueable-train.sh because with B: the step never sees USR1, and one
# requeue beats one per node.
#
# `scancel --signal`, not `kill -TERM "$srun_pid"`: SIGTERM is one of the few
# signals srun does not forward. It aborts the step instead ("srun: forcing job
# termination") and the tasks come back Killed -- SIGKILL, in the same second,
# so no handler anywhere below ever runs. That is worse than losing the
# checkpoint: a rank SIGKILLed with its NCCL communicators open faults its
# NVLink peers and gets the node cordoned (fme/core/distributed/shutdown.py).
# Measured 2026-08-30 on job 57759729: 16 ranks Killed, no REAL_EXIT line, and
# an empty training_checkpoints/.
#
# scancel without --batch/--full signals the job's steps and not the batch
# shell, which is the target here. Perlmutter runs proctrack/cgroup, so
# slurmstepd delivers the signal to every pid in the step's cgroup --
# requeueable-train.sh, torchrun and the ranks alike.
srun --nodes=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 --gpus-per-node=4 \
     "$CONFIG_DIR/requeueable-train.sh" &
srun_pid=$!
trap 'scancel --signal=TERM --quiet "$SLURM_JOB_ID"; wait "$srun_pid"; scontrol requeue "$SLURM_JOB_ID"' USR1
wait "$srun_pid"
rc=$?
echo "srun exited rc=$rc"
exit $rc
