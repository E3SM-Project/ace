#!/bin/bash
# Batch script for a bundle of runs (see bundle.sh). One srun step per run, on
# disjoint node sets, all inside one allocation; nodes of finished runs idle.
#SBATCH -A e3sm_g
#SBATCH -q regular
#SBATCH -C gpu&a100              # atm/ocn only. bundle.sh overrides this on the
                                 # sbatch command line: a coupled run OOMs on a
                                 # 40 GB card during inline-inference validation,
                                 # so any bundle containing a cpl run goes to
                                 # gpu&hbm80g. Set FME_CONSTRAINT to force it.
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH -t 48:00:00
#SBATCH --output=joblogs/%x-%j.out
#SBATCH --signal=B:USR1@600      # batch shell only, 10 min before the limit:
                                 # more steps have to write restart checkpoints
                                 # than in a single-run job
#SBATCH --requeue
#SBATCH --open-mode=append

set -u
MANIFEST=${BUNDLE_MANIFEST:?set by bundle.sh}
MAX_RESTARTS=${BUNDLE_MAX_RESTARTS:-10}
FME_TORCHRUN=${FME_TORCHRUN:?set by bundle.sh}
export FME_TORCHRUN
RESTARTS=${SLURM_RESTART_COUNT:-0}
LOGDIR=$(dirname "$MANIFEST")/steps
mkdir -p "$LOGDIR"

echo "=== bundle ======================================================"
echo "job        ${SLURM_JOB_NAME} / ${SLURM_JOB_ID}   restarts ${RESTARTS}/${MAX_RESTARTS}"
echo "nodes      ${SLURM_JOB_NUM_NODES} (${SLURM_JOB_NODELIST})"
echo "manifest   $MANIFEST"
echo "started    $(date -Is)"
echo "================================================================="

mapfile -t HOSTS < <(scontrol show hostnames "$SLURM_JOB_NODELIST")

# A run is complete when its log records the final epoch's checkpoint.
run_complete() {  # <output dir> <config yaml>
    local out=$1 cfg=$2 max
    max=$(sed -n 's/^max_epochs: *//p' "$cfg" | head -1)
    [ -n "$max" ] && [ -f "$out/out.log" ] && \
        grep -q "trained for ${max} complete epochs and 0 additional" "$out/out.log"
}

declare -A PID RUNOF
IDX=0
PORT=29600
while IFS=$'\t' read -r runid realm nodes cfgdir root; do
    [ -z "$runid" ] && continue
    out=$root/$runid
    cfg=$cfgdir/$runid.yaml
    hosts=("${HOSTS[@]:IDX:nodes}")
    IDX=$((IDX + nodes))
    PORT=$((PORT + 1))
    if [ "${#hosts[@]}" -ne "$nodes" ]; then
        echo "SKIP $runid: allocation ran out of nodes (needs $nodes)"; continue
    fi
    if run_complete "$out" "$cfg"; then
        echo "DONE $runid already complete; its ${nodes} nodes idle"; continue
    fi
    case "$realm" in
        cpl) module=fme.coupled.train ;;
        *)   module=fme.ace.train ;;
    esac
    nodelist=$(IFS=,; echo "${hosts[*]}")
    mkdir -p "$out/job_config"
    cp -r "$cfgdir/." "$out/job_config/"
    echo "START $runid on $nodelist (port $PORT) -> $out"
    (
        set -a
        # shellcheck disable=SC1090
        . "$cfgdir/$runid.env"
        set +a
        export WANDB_NOTES="${WANDB_NOTES:-} | bundle ${SLURM_JOB_NAME}/${SLURM_JOB_ID}"
        export CONFIG_DIR=$cfgdir TRAIN_CONFIG=$cfg TRAIN_MODULE=$module \
               MASTER_ADDR=${hosts[0]} MASTER_PORT=$PORT \
               FME_DIST_TIMEOUT_MINUTES=${FME_DIST_TIMEOUT_MINUTES:-180} \
               FME_OVERRIDE_ARGS="experiment_dir=$out"
        exec srun --nodes="$nodes" --nodelist="$nodelist" --ntasks-per-node=1 \
             --gpus-per-node=4 --cpus-per-task="${SLURM_CPUS_PER_TASK:-128}" \
             --job-name="$runid" --output="$LOGDIR/${runid}.out" --open-mode=append \
             "$cfgdir/requeueable-train.sh"
    ) &
    PID[$runid]=$!
    RUNOF[$!]=$runid
done < "$MANIFEST"

[ "${#PID[@]}" -gt 0 ] || { echo "nothing to run: every run is complete"; exit 0; }

# Walltime: USR1 reaches this shell only (B:). Signal every step with SIGTERM
# -- the path requeueable-train.sh, torchrun and the trainer already handle by
# writing a restart checkpoint -- then requeue once they have all exited.
WALLTIME=0
trap 'echo "USR1 at $(date -Is): stopping all steps"; WALLTIME=1; scancel --signal=TERM --quiet "$SLURM_JOB_ID"' USR1

while [ -n "$(jobs -rp)" ]; do
    wait -n 2>/dev/null || true
done

echo "=== step results ($(date -Is)) ==="
INCOMPLETE=0
while IFS=$'\t' read -r runid realm nodes cfgdir root; do
    [ -n "${PID[$runid]:-}" ] || continue
    wait "${PID[$runid]}"; rc=$?
    if run_complete "$root/$runid" "$cfgdir/$runid.yaml"; then state=complete; else state=incomplete; INCOMPLETE=$((INCOMPLETE + 1)); fi
    echo "  $runid rc=$rc $state"
done < "$MANIFEST"

if [ "$INCOMPLETE" -gt 0 ] && [ "$WALLTIME" = 1 ]; then
    if [ "$RESTARTS" -lt "$MAX_RESTARTS" ]; then
        echo "requeueing: $INCOMPLETE runs incomplete (restart $((RESTARTS + 1))/$MAX_RESTARTS)"
        scontrol requeue "$SLURM_JOB_ID"
    else
        echo "not requeueing: restart limit $MAX_RESTARTS reached with $INCOMPLETE runs incomplete"
    fi
elif [ "$INCOMPLETE" -gt 0 ]; then
    # Steps ended before the walltime signal and did not finish: a crash, not a
    # timeout. Requeueing would loop on it; leave it for a human.
    echo "not requeueing: $INCOMPLETE runs exited early without finishing; see $LOGDIR"
    exit 1
fi
exit 0
