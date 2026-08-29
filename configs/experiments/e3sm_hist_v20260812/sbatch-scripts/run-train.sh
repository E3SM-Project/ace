#!/bin/bash
# Login-node driver: stage the config, validate it, then submit.
#
#     ./run-train.sh atm|ocn|cpl                        # the committed config
#     ./run-train.sh atm <runid> [--after <jobid>]      # a campaign run
#     ./run-train.sh atm <runid> --no-submit            # stage + validate only
#
# With a run id, the config is taken from ../runs/<runid>.yaml, the matching
# ../runs/<runid>.env is sourced so the run is named in wandb and sized
# correctly (FME_NODES), and the output lands in $CAMPAIGN_ROOT/<runid> rather
# than a job-id directory. The job id is printed on stdout.
#
# The aug26 campaign has no finetune chain -- every run trains from scratch --
# so --after is not needed for it. It is kept for ad-hoc use (e.g. queueing a
# continuation behind a run that is still training).
#
# Staging matters: the job reads the config at *start* time, so editing the
# tree between submit and start would otherwise change what runs. The staged
# copy under $PSCRATCH/fme-config/<uuid> is frozen and is also copied next to
# the run output as job_config/.
#
# Resume a preempted or requeued run:
#     RESUME_JOB_ID=<job id> ./run-train.sh atm

set -euo pipefail

REALM="${1:-}"
case "$REALM" in
    atm|ocn) VALIDATOR=fme.ace.validate_config ;;
    cpl)     VALIDATOR=fme.coupled.validate_config ;;
    *) echo "usage: $0 atm|ocn|cpl [runid] [--after <jobid>] [--no-submit]" >&2; exit 2 ;;
esac
shift

RUNID=""
AFTER=""
NOSUBMIT=0
SIZE=()
while [ $# -gt 0 ]; do
    case "$1" in
        --after) AFTER="${2:?--after needs a job id}"; shift 2 ;;
        # Everything except the sbatch call. This is the pre-flight check to run
        # on Sunday: it exercises staging, the .env, the sizing and the config
        # validator without queueing anything.
        --no-submit) NOSUBMIT=1; shift ;;
        *)       RUNID="$1"; shift ;;
    esac
done

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXP_DIR=$(dirname "$HERE")
REPO_ROOT=$(cd "$EXP_DIR/../../.." && pwd)

# --output=joblogs/%x-%j.out fails to start the job if this does not exist.
mkdir -p "$HERE/joblogs"

UUID=$(uuidgen)
export CONFIG_DIR=${PSCRATCH}/fme-config/${UUID}
mkdir -p "$CONFIG_DIR"

if [ -n "$RUNID" ]; then
    SRC="$EXP_DIR/runs/${RUNID}.yaml"
    [ -f "$SRC" ] || { echo "no config at $SRC -- generate it with make_ablation_config.py" >&2; exit 1; }
    export CONFIG_NAME="${RUNID}.yaml"
    export RUNID
    export CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-${PSCRATCH}/aug26}"
    cp "$SRC" "$CONFIG_DIR/"
    # Provenance: wandb reads these from the environment, not from the config.
    if [ -f "$EXP_DIR/runs/${RUNID}.env" ]; then
        # shellcheck disable=SC1090
        . "$EXP_DIR/runs/${RUNID}.env"
        export WANDB_NAME WANDB_RUN_GROUP WANDB_JOB_TYPE WANDB_TAGS WANDB_NOTES
        cp "$EXP_DIR/runs/${RUNID}.env" "$CONFIG_DIR/"
        # The node count is a property of the config -- batch_size / local batch
        # / 4 GPUs -- so it comes from the .env rather than from the #SBATCH
        # directive, which is only a default for ad-hoc runs. B08 and B32 arms
        # would otherwise silently run at the baseline's node count and either
        # waste half the allocation or fail the batch/rank divisibility check.
        if [ -n "${FME_NODES:-}" ]; then
            SIZE=(--nodes="$FME_NODES")
            echo "sizing: ${FME_NODES} nodes / ${FME_RANKS:-?} ranks" >&2
        fi
    else
        echo "WARNING: no ${RUNID}.env -- the run will be unnamed in wandb" >&2
    fi
else
    export CONFIG_NAME="config-train-${REALM}.yaml"
    cp "$EXP_DIR/config-train-${REALM}.yaml" "$CONFIG_DIR/"
fi
cp "$HERE/requeueable-train.sh" "$CONFIG_DIR/"
cp "$HERE/sbatch-train-${REALM}.sh" "$CONFIG_DIR/"
chmod +x "$CONFIG_DIR/requeueable-train.sh"

# Record what the code was, so the output is reproducible.
git -C "$REPO_ROOT" rev-parse HEAD > "$CONFIG_DIR/COMMIT" 2>/dev/null || true

# Validate before queueing: this catches the batch_size / initial-condition
# divisibility errors, which otherwise surface minutes into an allocation as an
# unhelpful `UnionMatchError: can not match type "list"`.
( cd "$REPO_ROOT" && uv run python -m "$VALIDATOR" \
    --config_type train "$CONFIG_DIR/${CONFIG_NAME}" )

export FME_TORCHRUN="$REPO_ROOT/.venv/bin/torchrun"
[ -x "$FME_TORCHRUN" ] || { echo "no torchrun at $FME_TORCHRUN; run 'uv sync' first" >&2; exit 1; }

echo "staged config: $CONFIG_DIR/$CONFIG_NAME" >&2
[ -n "$RUNID" ] && echo "runid: $RUNID -> ${CAMPAIGN_ROOT}/${RUNID}" >&2

if [ "$NOSUBMIT" = 1 ]; then
    echo "--no-submit: staged and validated, nothing queued" >&2
    exit 0
fi

DEP=()
[ -n "$AFTER" ] && DEP=(--dependency="afterok:${AFTER}")
# During the hackathon window, export RESERVATION=_CAP_aigs_hist or jobs sit in
# the regular queue while the 96 reserved nodes idle. Leave it unset afterwards
# so post-window requeues run on the normal allocation.
[ -n "${RESERVATION:-}" ] && DEP+=(--reservation="${RESERVATION}")

# --parsable prints only the job id, so a driver script can chain on it.
JOBID=$(sbatch --parsable "${SIZE[@]}" "${DEP[@]}" "$HERE/sbatch-train-${REALM}.sh")
echo "submitted ${JOBID}${AFTER:+ (after ${AFTER})}" >&2
echo "$JOBID"
