#!/bin/bash
# Per-node payload run under `srun` by sbatch-train-{atm,ocn,cpl}.sh.
# Same pattern as e3sm_piControl_v20260507/atmosphere/sbatch-scripts/requeueable-train.sh
# (that directory lives on the e3sm/exps/hist branch, not this one):
# torchrun is launched in the background so this shell can catch signals.
#   SIGTERM (preemption, and now also walltime) -> kill torchrun, exit
#   USR1                      -> kept only as a fallback; see below
#
# Walltime is no longer handled here. The batch script carries
# `--signal=B:USR1@120`, so USR1 reaches the batch shell alone and never this
# step -- deliberately, because without B: it also reaches the python ranks,
# which have no SIGUSR1 handler and die before writing a restart checkpoint
# (measured 2026-08-30, job 57758390: REAL_EXIT=138, empty
# training_checkpoints/). The batch script's trap converts it into the SIGTERM
# that every layer here already handles, and owns the `scontrol requeue`.
# Resume is automatic: training relaunches against the same experiment_dir and
# picks up from <experiment_dir>/training_checkpoints/ckpt.tar (verified for
# both realms, see README "Checkpointing and resuming"). Checkpoints are
# written at every epoch boundary, so a requeue always resumes from at most one
# epoch back.
#
# Whether it resumes from *less* than that is doubtful, and the budget is why.
# FME registers save_restart_checkpoints_on_terminate as a post-shutdown
# callback (trainer.py:322), so SIGTERM does start a mid-epoch restart
# checkpoint -- but torchrun's agent gives the ranks one shared 30 s before
# SIGKILL (PContext.close defaults to timeout=30 and LocalElasticAgent._shutdown
# calls it with no argument), the collective teardown spends part of it, and the
# full checkpoint set measured ~31 s to write. Budget a requeue as costing the
# partial epoch in flight -- up to 2.9 h for the atmosphere, ~1.4 h expected --
# and treat a mid-epoch checkpoint that does land as a bonus. Trying costs
# nothing: save_checkpoint writes .<uuid>.tmp and os.replace()s it
# (trainer.py:676-694), so a write cut short cannot damage the checkpoint
# already on disk.
#
# Required environment (exported by the calling sbatch script):
#   TRAIN_CONFIG  absolute path to the config yaml (on shared FS, never /tmp)
#   TRAIN_MODULE  fme.ace.train (atm/ocn) or fme.coupled.train (cpl)
#   FME_TORCHRUN  absolute path to the venv's torchrun (this repo uses uv, so
#                 there is no activated conda env on the compute node)
#   MASTER_ADDR   first node of the allocation
#   MASTER_PORT   distinct per experiment: two runs on one node collide at 29500
# Optional:
#   FME_OVERRIDE_ARGS  space-separated dotlist overrides
#
# Do NOT switch this to the FME_USE_SRUN=1 launcher: on Perlmutter it hardcodes
# cuda device 0 and every rank dies with 'invalid device ordinal' (see README
# "Launching"). torchrun sets the device from LOCAL_RANK, which is correct here.

set -x

preempt_handler()
{
    kill -TERM "${1}"
}

timeout_handler()
{
    kill -TERM "${1}"
    scontrol requeue "${SLURM_JOB_ID}"
}

TRAIN_ARGS=("$TRAIN_CONFIG")

if [[ -n "${FME_OVERRIDE_ARGS:-}" ]]; then
    read -r -a OVERRIDE_ARRAY <<< "$FME_OVERRIDE_ARGS"
    TRAIN_ARGS+=("--override" "${OVERRIDE_ARRAY[@]}")
fi

# Size the rendezvous from the *step*, not the allocation: SLURM_JOB_NUM_NODES
# is allocation-wide, so under an salloc larger than the step (or any srun with
# an explicit --nodes) torchrun would wait forever for nodes that never join.
NNODES="${SLURM_STEP_NUM_NODES:-${SLURM_JOB_NUM_NODES:-1}}"
NPROC="${SLURM_GPUS_PER_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
: "${NPROC:?could not determine GPUs per node; set SLURM_GPUS_PER_NODE}"
echo "rendezvous: nnodes=$NNODES nproc_per_node=$NPROC at $MASTER_ADDR:$MASTER_PORT"

"${FME_TORCHRUN:?must point at the venv torchrun binary}" \
 --nnodes "$NNODES" \
 --nproc_per_node "$NPROC" \
 --rdzv-backend=c10d \
 --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
 -m "$TRAIN_MODULE" "${TRAIN_ARGS[@]}" &

pid=$!
trap "preempt_handler '$pid'" SIGTERM
trap "timeout_handler '$pid'" USR1
wait $pid
rc=$?
# Judge the run by this line and by "DONE ---- rank 0" in the log, not by the
# log tail: time_buffer teardown prints scary-but-harmless tracebacks on
# successful runs (README "Known issues").
echo "REAL_EXIT=$rc"
# No trailing sleep. It used to hold the task open while this script's own
# `scontrol requeue` took effect; the batch script owns the requeue now and
# blocks on srun to issue it, so sleeping here would only spend the walltime
# lead time that the teardown and checkpoint write need.
exit $rc
