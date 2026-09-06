#!/bin/bash
# Queue the stage-2 (multi-step fine-tune) runs.
#
#     ./submit-stage2.sh            # what would be submitted, and nothing else
#     ./submit-stage2.sh --go       # actually submit
#     ./submit-stage2.sh --go E12   # only run ids matching a pattern
#
# This is a thin wrapper over run-train.sh, not a second submission path. All
# it adds is the two things stage 2 needs and stage 1 does not:
#
#   CAMPAIGN_ROOT   $PSCRATCH/aug26-ft rather than $PSCRATCH/aug26, so a stage-2
#                   run cannot land in its parent's output directory
#   a leakage gate  check_stage2_leakage.py must pass before anything queues,
#                   because the whole point of the stage-2 split is a claim
#                   about what the model never saw, and an unverified claim is
#                   worse than no claim
#
# Everything else -- config staging, validation, the dirty-worktree refusal, the
# queued-duplicate guard, node sizing from the .env, wandb identity, mail -- is
# run-train.sh's, unchanged.
#
# Reservation: exported below while the hackathon window is open. After
# 2026-09-09T15:00 the reservation is gone and jobs should just go to the
# regular queue, so unset it rather than pointing at a reservation that has
# expired -- sbatch fails outright on an unknown reservation.

set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXP_DIR=$(dirname "$HERE")

export CAMPAIGN_ROOT=/pscratch/sd/m/mahf708/aug26-ft
RESERVATION_NAME=${RESERVATION:-_CAP_aigs_hist}

GO=0
PATTERN=""
for arg in "$@"; do
    case "$arg" in
        --go) GO=1 ;;
        -*) echo "unknown flag $arg" >&2; exit 2 ;;
        *) PATTERN="$arg" ;;
    esac
done

# The gate. Run it over every stage-2 config, not just the ones being submitted:
# a violation in a sibling means the generator is wrong, and the next run
# generated from it would inherit the same fault.
echo "== verifying the train/validate/test split ==" >&2
if ! "$EXP_DIR/check_stage2_leakage.py" > /tmp/stage2-leakage.$$ 2>&1; then
    cat /tmp/stage2-leakage.$$ >&2
    rm -f /tmp/stage2-leakage.$$
    echo "refusing to submit: the split does not hold" >&2
    exit 1
fi
tail -1 /tmp/stage2-leakage.$$ >&2
rm -f /tmp/stage2-leakage.$$

shopt -s nullglob
CONFIGS=("$EXP_DIR"/runs/*-FT.*.yaml)
[ ${#CONFIGS[@]} -gt 0 ] || { echo "no stage-2 configs; run make_stage2_config.py --all" >&2; exit 1; }

# Reservation capacity, and why this is not just informational.
#
# Stage 1 is still running under a 12 h walltime with --requeue. A requeued job
# *releases its nodes* and goes back into the queue, so if stage 2 has taken
# the free pool in the meantime, a stage-1 run that was 140 epochs into a
# 150-epoch fit sits PENDING behind stage-2 jobs that are 20 h long. Nothing
# fails; the campaign just stops moving.
#
# Stage 2 has no deadline of its own -- it continues in the regular queue after
# the reservation ends on 2026-09-09 -- so there is no reason to race. Leave
# RESERVE_HEADROOM nodes for stage 1 to land on and queue the rest behind.
RESERVE_HEADROOM=${RESERVE_HEADROOM:-16}
CAP=$(scontrol show res "$RESERVATION_NAME" 2>/dev/null | sed -n 's/.*NodeCnt=\([0-9]*\).*/\1/p' | head -1)
USED=$(squeue -R "$RESERVATION_NAME" -h -o '%D' 2>/dev/null | paste -sd+ | bc)
FREE=$(( ${CAP:-0} - ${USED:-0} ))
BUDGET=$(( FREE - RESERVE_HEADROOM ))
echo "== reservation $RESERVATION_NAME: $USED/$CAP used, $FREE free," \
     "$RESERVE_HEADROOM held for stage-1 requeues -> $BUDGET for stage 2 ==" >&2

TOTAL=0
for cfg in "${CONFIGS[@]}"; do
    runid=$(basename "$cfg" .yaml)
    [ -z "$PATTERN" ] || [[ "$runid" == *"$PATTERN"* ]] || continue
    realm=$(echo "$runid" | cut -d. -f3)
    env_file="$EXP_DIR/runs/${runid}.env"
    nodes=$(sed -n 's/^FME_NODES=//p' "$env_file")
    parent=$(sed -n 's/^# parent //p' "$env_file")
    TOTAL=$((TOTAL + nodes))

    fits=""
    [ "$TOTAL" -gt "$BUDGET" ] && fits="  (over budget -- will queue)"

    if [ "$GO" != 1 ]; then
        printf "would submit  %-46s %s nodes  <- %s%s\n" \
            "$runid" "$nodes" "$parent" "$fits"
        continue
    fi
    echo "== $runid ($nodes nodes)$fits ==" >&2
    RESERVATION="$RESERVATION_NAME" "$HERE/run-train.sh" "$realm" "$runid"
done

echo "== $TOTAL nodes requested, $BUDGET available now ==" >&2
[ "$GO" = 1 ] || echo "(dry run -- pass --go to submit)" >&2
