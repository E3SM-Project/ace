#!/bin/bash
# Login-node driver: stage the config, validate it, then submit.
#
#     ./run-train.sh atm|ocn|cpl
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
    *) echo "usage: $0 atm|ocn|cpl" >&2; exit 2 ;;
esac

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXP_DIR=$(dirname "$HERE")
REPO_ROOT=$(cd "$EXP_DIR/../../.." && pwd)

# --output=joblogs/%x-%j.out fails to start the job if this does not exist.
mkdir -p "$HERE/joblogs"

UUID=$(uuidgen)
export CONFIG_DIR=${PSCRATCH}/fme-config/${UUID}
mkdir -p "$CONFIG_DIR"
cp "$EXP_DIR/config-train-${REALM}.yaml" "$CONFIG_DIR/"
cp "$HERE/requeueable-train.sh" "$CONFIG_DIR/"
cp "$HERE/sbatch-train-${REALM}.sh" "$CONFIG_DIR/"
chmod +x "$CONFIG_DIR/requeueable-train.sh"

# Record what the code was, so the output is reproducible.
git -C "$REPO_ROOT" rev-parse HEAD > "$CONFIG_DIR/COMMIT" 2>/dev/null || true

# Validate before queueing: this catches the batch_size / initial-condition
# divisibility errors, which otherwise surface minutes into an allocation as an
# unhelpful `UnionMatchError: can not match type "list"`.
( cd "$REPO_ROOT" && uv run python -m "$VALIDATOR" \
    --config_type train "$CONFIG_DIR/config-train-${REALM}.yaml" )

export FME_TORCHRUN="$REPO_ROOT/.venv/bin/torchrun"
[ -x "$FME_TORCHRUN" ] || { echo "no torchrun at $FME_TORCHRUN; run 'uv sync' first" >&2; exit 1; }

echo "staged config: $CONFIG_DIR"
sbatch "$HERE/sbatch-train-${REALM}.sh"
