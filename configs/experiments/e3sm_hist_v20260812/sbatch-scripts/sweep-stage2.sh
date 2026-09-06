#!/bin/bash
# Pick up stage-2 runs as their stage-1 parents finish.
#
#     ./sweep-stage2.sh            # report what would happen
#     ./sweep-stage2.sh --go       # generate and submit whatever is newly ready
#
# A stage-2 run is eligible only once its parent reaches its final epoch
# (ckpt_0030 / ckpt_0150), so most of the campaign is not submittable at any
# one moment -- parents finish over hours or days. Rather than watch for that
# by hand, run this: it regenerates configs for every eligible parent and
# submits the ones that are not already queued or running.
#
# Idempotent by construction. make_stage2_config.py skips parents that have not
# finished, run-train.sh refuses a run id already in the queue, and the check
# below skips run ids that already have output. So running it repeatedly is
# safe, and running it on a cron is safe.
#
# It does NOT force anything: a parent still training is left alone rather than
# started from its live ckpt.tar, because the campaign's rule is that stage 2
# begins from the final epoch.

set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXP_DIR=$(dirname "$HERE")
OUT_ROOT=/pscratch/sd/m/mahf708/aug26-ft

GO=0
[ "${1:-}" = "--go" ] && GO=1

cd "$EXP_DIR"

# Regenerate. Parents that have not finished are skipped with a reason; that
# output is the useful part of a dry run, so keep it.
echo "== regenerating stage-2 configs ==" >&2
./make_stage2_config.py --all 2>&1 | sed 's/^/  /' >&2

# A run id is "outstanding" if it has a config but no queue entry and no output
# directory. Anything already running, queued, or previously started is left
# alone -- resuming a stage-2 run is run-train.sh's job, not this script's.
outstanding=()
shopt -s nullglob
for cfg in runs/*-FT.*.yaml; do
    runid=$(basename "$cfg" .yaml)
    if squeue -h -u "$USER" --name="$runid" 2>/dev/null | grep -q .; then
        continue
    fi
    if [ -d "$OUT_ROOT/$runid" ]; then
        continue
    fi
    outstanding+=("$runid")
done

if [ ${#outstanding[@]} -eq 0 ]; then
    echo "== nothing new to submit ==" >&2
    exit 0
fi

echo "== ${#outstanding[@]} newly submittable ==" >&2
for runid in "${outstanding[@]}"; do
    nodes=$(sed -n 's/^FME_NODES=//p' "runs/${runid}.env")
    if [ "$GO" != 1 ]; then
        printf "  would submit  %-46s %s nodes\n" "$runid" "$nodes" >&2
        continue
    fi
    ./sbatch-scripts/submit-stage2.sh --go "$runid" 2>&1 | grep -E "submitted|refusing" >&2 || true
done

[ "$GO" = 1 ] || echo "(dry run -- pass --go to submit)" >&2
