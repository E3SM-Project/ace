#!/bin/bash
# Login-node driver: stage the config, validate it, then submit.
#
#     ./run-train.sh atm|ocn|cpl                        # the committed config
#     ./run-train.sh atm <runid> [--after <jobid>]      # a campaign run
#     ./run-train.sh atm <runid> --no-submit            # stage + validate only
#     ./run-train.sh atm <runid> --resume               # continue an existing run
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
#
# Email is on by default to $USER@nersc.gov on BEGIN, END, FAIL, REQUEUE and
# TIME_LIMIT_90. Override with FME_MAIL_USER / FME_MAIL_TYPE, or FME_MAIL_TYPE=NONE.
# Already-queued jobs can be retrofitted without resubmitting:
#     scontrol update JobId=<id> MailUser=$USER@nersc.gov \
#         MailType=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT_90

set -euo pipefail

REALM="${1:-}"
case "$REALM" in
    atm|ocn) VALIDATOR=fme.ace.validate_config ;;
    cpl)     VALIDATOR=fme.coupled.validate_config ;;
    *) echo "usage: $0 atm|ocn|cpl [runid] [--after <jobid>] [--no-submit] [--resume] [--force]" >&2; exit 2 ;;
esac
shift

RUNID=""
AFTER=""
NOSUBMIT=0
RESUME=0
FORCE=0
SIZE=()
while [ $# -gt 0 ]; do
    case "$1" in
        --after) AFTER="${2:?--after needs a job id}"; shift 2 ;;
        # Everything except the sbatch call. This is the pre-flight check to run
        # on Sunday: it exercises staging, the .env, the sizing and the config
        # validator without queueing anything.
        --no-submit) NOSUBMIT=1; shift ;;
        # Continue an existing run in place. Required whenever the output
        # directory already holds a checkpoint, because the trainer resumes
        # from it unconditionally (fme/core/generics/trainer.py, `resuming`).
        --resume) RESUME=1; shift ;;
        # Resume even though the config or the commit has changed. This makes
        # one logical run two different experiments; only for a deliberate
        # restart after a code fix.
        --force) FORCE=1; shift ;;
        *)       RUNID="$1"; shift ;;
    esac
done

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXP_DIR=$(dirname "$HERE")
REPO_ROOT=$(cd "$EXP_DIR/../../.." && pwd)

# --output=joblogs/%x-%j.out is relative to the job's working directory, which
# defaults to wherever sbatch was invoked -- so the log landed in a different
# joblogs/ depending on the caller's cwd, and a job whose output directory does
# not exist fails to start with no log at all. Pin both: --chdir below fixes the
# working directory to the experiment dir, and this creates the matching dir.
mkdir -p "$EXP_DIR/joblogs"

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

# Run identity. A run id names the *arms* of the experiment, not the config:
# make_ablation_config.py's --aod, --epochs and --local-batch all change what is
# generated while emitting the same id. So hash the staged config and record it
# beside the output. Two submissions of one id that disagree are then loud
# instead of silent.
sha256sum "$CONFIG_DIR/${CONFIG_NAME}" | cut -d' ' -f1 > "$CONFIG_DIR/CONFIG_SHA256"

# The worktree is what runs, not the recorded commit: .venv is an editable
# install pointing straight at this tree, so an edit between submit and start --
# or before a requeued segment -- silently changes the code while COMMIT still
# says otherwise. Refuse a dirty tree rather than record a lie.
if [ -n "$(git -C "$REPO_ROOT" status --porcelain 2>/dev/null)" ]; then
    if [ "${FME_ALLOW_DIRTY:-0}" = 1 ]; then
        echo "WARNING: worktree is dirty; COMMIT records $(cut -c1-8 < "$CONFIG_DIR/COMMIT") but the job will run whatever is checked out at start time" >&2
    else
        echo "refusing to submit from a dirty worktree -- .venv is an editable install, so the job runs the tree, not the commit. Commit, stash, or set FME_ALLOW_DIRTY=1." >&2
        exit 1
    fi
fi

# Resume guard. The trainer restores <output>/training_checkpoints/ckpt.tar if
# it exists, whatever the newly staged config says, so an accidental second
# submission of a run id continues someone else's run under a different config.
if [ -n "$RUNID" ]; then
    OUT="${CAMPAIGN_ROOT}/${RUNID}"
    PREV_CFG="$OUT/job_config/${CONFIG_NAME}"

    # Queued-duplicate guard. The checks below catch "this run has already run"
    # by looking at the output directory; they cannot catch "this run is
    # already waiting to start", because nothing has been written yet. Two
    # submissions of one id before the first starts is not caught by anything
    # else, and it ends with two jobs writing ckpt.tar in the same directory --
    # which does not fail, it just silently produces one corrupted run.
    #
    # Scope: this sees only $USER's queue, which is the same scope as
    # $CAMPAIGN_ROOT ($PSCRATCH is per person). Two people submitting the same
    # id land in two different directories and two different queues, and no
    # guard here can see that -- one owner per run is what prevents it.
    if [ "$NOSUBMIT" != 1 ]; then
        PENDING=$(squeue -h -u "$USER" --name="$RUNID" -o '%i %T' 2>/dev/null || true)
        if [ -n "$PENDING" ]; then
            echo "$RUNID is already in your queue:" >&2
            echo "$PENDING" | sed 's/^/  /' >&2
            echo "  Submitting again would run two jobs into $OUT and interleave their checkpoint writes." >&2
            echo "  Cancel the existing job first, or pass --force if you know what you are doing." >&2
            [ "$FORCE" != 1 ] && exit 1
        fi
    fi
    if [ -f "$OUT/training_checkpoints/ckpt.tar" ] && [ "$RESUME" != 1 ]; then
        echo "$OUT already holds a checkpoint. Training would resume from it, not start over." >&2
        echo "  continue it:  ./run-train.sh $REALM $RUNID --resume" >&2
        echo "  start over:   move or delete $OUT first" >&2
        exit 1
    fi
    if [ -f "$PREV_CFG" ]; then
        PREV_SHA=$(sha256sum "$PREV_CFG" | cut -d' ' -f1)
        THIS_SHA=$(cat "$CONFIG_DIR/CONFIG_SHA256")
        PREV_COMMIT=$(cat "$OUT/job_config/COMMIT" 2>/dev/null || echo unknown)
        THIS_COMMIT=$(cat "$CONFIG_DIR/COMMIT" 2>/dev/null || echo unknown)
        if [ "$PREV_SHA" != "$THIS_SHA" ] || [ "$PREV_COMMIT" != "$THIS_COMMIT" ]; then
            echo "$RUNID has already run, and this submission does not match it:" >&2
            [ "$PREV_SHA" != "$THIS_SHA" ] && echo "  config  $PREV_SHA -> $THIS_SHA" >&2
            [ "$PREV_COMMIT" != "$THIS_COMMIT" ] && echo "  commit  $PREV_COMMIT -> $THIS_COMMIT" >&2
            echo "  Resuming would make one run id two experiments. Pass --force if that is really what you want." >&2
            [ "$FORCE" != 1 ] && exit 1
        fi
    fi
fi

# Email. NERSC delivers to <user>@nersc.gov. On by default: these jobs are
# --requeue with a 12 h walltime, so a run silently dies or silently restarts
# and nothing on the terminal says so. REQUEUE is the one that matters -- it is
# the difference between "still training" and "has been requeueing all night".
#
#   FME_MAIL_USER=you@lbl.gov     send somewhere else
#   FME_MAIL_TYPE=NONE            turn it off
#   FME_MAIL_TYPE=ALL             every state change, including STAGE_OUT
#
# Volume: roughly (1 + segments) messages per run, where a 63 h atmosphere run
# at a 12 h walltime is 6 segments. The full 35-run campaign is order 250
# messages; filter on the subject, which carries the job name and id.
MAIL_USER="${FME_MAIL_USER:-${USER}@nersc.gov}"
MAIL_TYPE="${FME_MAIL_TYPE:-BEGIN,END,FAIL,REQUEUE,TIME_LIMIT_90}"
MAIL=()
if [ "$MAIL_TYPE" = "NONE" ]; then
    echo "mail: disabled (FME_MAIL_TYPE=NONE)" >&2
else
    MAIL=(--mail-user="$MAIL_USER" --mail-type="$MAIL_TYPE")
    echo "mail: $MAIL_TYPE -> $MAIL_USER" >&2
fi

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
# --job-name defaults to fme-hist-<realm> in the #SBATCH block, which makes all
# 25 atmosphere runs identical in squeue and gives every one of them the same
# log filename. Name the job after the run instead: --output is joblogs/%x-%j.out
# and %x is the job name, so the log becomes joblogs/<runid>-<jobid>.out with no
# further change. Ad-hoc runs keep the generic name.
NAME=()
[ -n "$RUNID" ] && NAME=(--job-name="$RUNID")

JOBID=$(sbatch --parsable --chdir="$EXP_DIR" \
    "${SIZE[@]}" "${DEP[@]}" "${MAIL[@]}" "${NAME[@]}" \
    "$HERE/sbatch-train-${REALM}.sh")
echo "submitted ${JOBID}${AFTER:+ (after ${AFTER})}" >&2
echo "$JOBID"
