#!/bin/bash
# Run several campaign runs inside ONE Slurm job (prototype, 2026-09-15).
#
#     ./bundle.sh <bundle-file>                 # stage + validate + print, submit nothing
#     ./bundle.sh <bundle-file> --go            # stage, validate and submit
#
# The bundle file has one run per line, `#` comments allowed:
#
#     <runid>  <campaign root>
#     E03-FT.aug26.atm.A1_B16_C1_L0_O5_W0_X0.S01  /pscratch/sd/m/mahf708/aug26-ft
#
# The realm comes from the run id and the node count from runs/<runid>.env
# (FME_NODES), exactly as for a single submission. Every run is staged through
# `run-train.sh --no-submit`, so config validation, the dirty-worktree refusal,
# wandb identity and the resume-from-ckpt.tar behaviour are the single-run
# path's, unchanged. The job's node count is the sum; sbatch-bundle.sh carves
# the allocation into disjoint node sets and starts one srun step per run.
#
# Environment knobs (same names as run-train.sh where they overlap):
#   FME_TIME        walltime (default 48:00:00, the regular-QOS maximum)
#   FME_TIME_MIN    --time-min for backfill (default unset)
#   FME_QOS         QOS (default regular)
#   FME_MAIL_TYPE   default FAIL,REQUEUE,TIME_LIMIT_90; NONE disables
#   BUNDLE_NAME     job name (default the bundle file's basename)
#   BUNDLE_MAX_RESTARTS  walltime requeues before the job stops requeueing
#                   itself (default 10)
#
# Why a bundle at all: nodes of a run that finishes early sit idle until the
# last run ends, and one job means one place in the queue. That trade is
# deliberate here (2026-09-15: plenty of GPU hours, the queue is the
# bottleneck); see AGENTS.md for when it is and is not worth it.

set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXP_DIR=$(dirname "$HERE")

BUNDLE_FILE="${1:?usage: $0 <bundle-file> [--go]}"
GO=0
[ "${2:-}" = "--go" ] && GO=1
[ -f "$BUNDLE_FILE" ] || { echo "no bundle file $BUNDLE_FILE" >&2; exit 1; }

NAME=${BUNDLE_NAME:-$(basename "$BUNDLE_FILE" .txt)}
STAMP=$(date +%Y%m%dT%H%M%S)
BUNDLE_DIR=${PSCRATCH}/fme-bundles/${NAME}-${STAMP}
mkdir -p "$BUNDLE_DIR"
MANIFEST=$BUNDLE_DIR/manifest.tsv
: > "$MANIFEST"

TOTAL=0
while read -r runid root _rest; do
    [ -z "${runid:-}" ] && continue
    case "$runid" in \#*) continue ;; esac
    [ -n "${root:-}" ] || { echo "no campaign root for $runid" >&2; exit 1; }
    realm=$(echo "$runid" | cut -d. -f3)
    env_file="$EXP_DIR/runs/${runid}.env"
    [ -f "$env_file" ] || { echo "no $env_file" >&2; exit 1; }
    nodes=$(sed -n 's/^FME_NODES=//p' "$env_file")
    [ -n "$nodes" ] || { echo "no FME_NODES in $env_file" >&2; exit 1; }

    # One owner per output directory: a run already queued or running on its
    # own would interleave checkpoint writes with its bundled twin.
    queued=$(squeue -h -u "$USER" --name="$runid" -o '%i %T' 2>/dev/null || true)
    if [ -n "$queued" ]; then
        echo "$runid is already in the queue on its own ($queued); take it out of the bundle or cancel it" >&2
        exit 1
    fi

    echo "== staging $runid ($realm, $nodes nodes) ==" >&2
    log=$BUNDLE_DIR/stage-${runid}.log
    if ! CAMPAIGN_ROOT="$root" "$HERE/run-train.sh" "$realm" "$runid" --no-submit > "$log" 2>&1; then
        cat "$log" >&2
        echo "staging $runid failed; nothing submitted" >&2
        exit 1
    fi
    cfg=$(sed -n 's/^staged config: //p' "$log" | tail -1)
    [ -f "$cfg" ] || { cat "$log" >&2; echo "could not find the staged config for $runid" >&2; exit 1; }
    grep -h "^resuming\|^NOTE" "$log" >&2 || true
    printf '%s\t%s\t%s\t%s\t%s\n' "$runid" "$realm" "$nodes" "$(dirname "$cfg")" "$root" >> "$MANIFEST"
    TOTAL=$((TOTAL + nodes))
done < "$BUNDLE_FILE"

[ "$TOTAL" -gt 0 ] || { echo "empty bundle" >&2; exit 1; }
cp "$BUNDLE_FILE" "$BUNDLE_DIR/"
cp "$HERE/sbatch-bundle.sh" "$BUNDLE_DIR/"

echo >&2
echo "bundle $NAME: $(wc -l < "$MANIFEST") runs, $TOTAL nodes" >&2
column -t -s $'\t' "$MANIFEST" | awk '{print "  "$1"  "$2"  "$3" nodes  -> "$5}' >&2
echo "manifest: $MANIFEST" >&2

MAIL_TYPE=${FME_MAIL_TYPE:-FAIL,REQUEUE,TIME_LIMIT_90}
MAIL=()
[ "$MAIL_TYPE" != NONE ] && MAIL=(--mail-user="${FME_MAIL_USER:-${USER}@nersc.gov}" --mail-type="$MAIL_TYPE")
EXTRA=(--qos="${FME_QOS:-regular}" --time="${FME_TIME:-48:00:00}")
[ -n "${FME_TIME_MIN:-}" ] && EXTRA+=(--time-min="$FME_TIME_MIN")

if [ "$GO" != 1 ]; then
    echo "(dry run: staged and validated; pass --go to submit)" >&2
    exit 0
fi

# Same reason as run-train.sh: a SLURM_* variable inherited from an salloc
# shell would override the batch script's own directives.
for _v in $(compgen -v SLURM_ 2>/dev/null || true); do
    [ "$_v" != SLURM_CONF ] && unset "$_v"
done
export BUNDLE_MANIFEST="$MANIFEST" BUNDLE_MAX_RESTARTS="${BUNDLE_MAX_RESTARTS:-10}"
export FME_TORCHRUN="$(cd "$EXP_DIR/../../.." && pwd)/.venv/bin/torchrun"
JOBID=$(sbatch --parsable --chdir="$EXP_DIR" --nodes="$TOTAL" --job-name="$NAME" \
    "${MAIL[@]}" "${EXTRA[@]}" "$BUNDLE_DIR/sbatch-bundle.sh")
echo "$JOBID" > "$BUNDLE_DIR/JOBID"
echo "submitted $JOBID ($TOTAL nodes)" >&2
echo "$JOBID"
