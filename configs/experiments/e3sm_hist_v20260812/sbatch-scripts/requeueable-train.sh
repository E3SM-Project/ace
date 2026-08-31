#!/bin/bash
# Per-node payload run under `srun` by sbatch-train-{atm,ocn,cpl}.sh.
# Same pattern as e3sm_piControl_v20260507/atmosphere/sbatch-scripts/requeueable-train.sh
# (that directory lives on the e3sm/exps/hist branch, not this one):
# torchrun is launched in the background so this shell can catch signals.
#   SIGTERM (preemption, and now also walltime) -> kill torchrun, exit
#   USR1                      -> kept only as a fallback; see below
#
# Walltime is no longer handled here. The batch script carries
# `--signal=B:USR1@300`, so USR1 reaches the batch shell alone and never this
# step -- deliberately, because without B: it also reaches the python ranks,
# which have no SIGUSR1 handler and die before writing a restart checkpoint
# (measured 2026-08-30, job 57758390: REAL_EXIT=138, empty
# training_checkpoints/). The batch script's trap converts it into the SIGTERM
# that every layer here already handles, and owns the `scontrol requeue`.
#
# That is a deliberate departure from NERSC's own template
# (docs.nersc.gov/jobs/examples, "Preemptible Jobs"), which uses a bare
# `--signal=USR1@60` and puts both traps in this payload script. Their model
# assumes the application checkpoints on its own schedule and can be killed
# outright by the USR1 broadcast; FME checkpoints on SIGTERM and on nothing
# else, so the broadcast has to be kept off the ranks. The preemption half of
# their template is kept as-is: preemption signals the step, not the batch
# shell -- "there is no way for job preemption to warn the batch script" --
# which is what preempt_handler below is for. NERSC's checkpoint/restart tools
# are not an alternative for either half: DMTCP and MANA are documented as
# unavailable for GPU applications.
# Resume is automatic: training relaunches against the same experiment_dir and
# picks up from <experiment_dir>/training_checkpoints/ckpt.tar (verified for
# both realms, see README "Checkpointing and resuming"). Checkpoints are
# written at every epoch boundary, so a requeue always resumes from at most one
# epoch back.
#
# It resumes from *less* than that too, measured end to end on job 57761772
# (2026-08-30). SIGTERM reached the ranks at 20:30:40.047, the collective
# teardown finished 587 ms later, and save_restart_checkpoints_on_terminate
# (trainer.py:322) had written 6.8 GiB of ckpt.tar by 20:30:51 -- a 10.4 s
# write, well inside the 30 s torchrun's agent allows before SIGKILL
# (PContext.close defaults to timeout=30, and LocalElasticAgent._shutdown calls
# it with no argument). The next segment logged "skip first 148 batches since
# these were already processed for this epoch" and ran the remaining 8,069 of
# the epoch's 8,217.
#
# So budget a requeue at the dataset setup plus the queue wait -- about 21 min
# on CFS for the atmosphere -- not at the partial epoch. The ~31 s figure in
# EXPERIMENTS.md is the whole per-epoch write of ~20 GB, EMA and epoch-numbered
# copies included; this one file is a third of it. A write cut short cannot
# damage the checkpoint already on disk either: save_checkpoint writes
# .<uuid>.tmp and os.replace()s it (trainer.py:676-694).
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
# Keep a process alive for Slurm to SIGKILL. This is NERSC's own instruction for
# preemptible jobs ("ensures a process is still running for slurm to send
# SIGKILL to", docs.nersc.gov/jobs/examples #preemptible-jobs) and it is what
# makes `--requeue` fire: on preemption Slurm requeues a job it killed, so a
# step that tidily exits during the grace period is recorded as finished
# instead. It costs the walltime path 120 s between the ranks exiting and the
# batch script's `wait` returning to issue its requeue, which is part of why
# that lead time is 300 s and not NERSC's 60.
sleep 120
exit $rc
