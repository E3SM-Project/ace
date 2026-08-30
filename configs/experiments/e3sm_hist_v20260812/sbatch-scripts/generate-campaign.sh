#!/bin/bash
# Regenerate every config for the aug26 campaign.
#
#     ./generate-campaign.sh              # write ../runs
#     CAMPAIGN_LOCAL_BATCH=atm=2 ./generate-campaign.sh   # 2 samples per rank
#     ./generate-campaign.sh --list       # print the run list and node budget
#     ./generate-campaign.sh /some/dir    # write elsewhere
#
# The run list lives in make_ablation_config.py's RUNLIST, transcribed from the
# hackathon page. There is no chain and no ordering constraint any more: every
# run trains from scratch, so this is a single call.
#
# Sizing is per run and comes out of the config, not out of the sbatch file:
# atm nodes = batch_size / 4, ocn nodes = batch_size / 8. run-train.sh reads
# FME_NODES from the generated .env and passes --nodes to sbatch.

set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXP=$(dirname "$HERE")
GEN="$EXP/make_ablation_config.py"
OWNER="${CAMPAIGN_OWNER:-${USER}}"
ROOT="${CAMPAIGN_ROOT:-${PSCRATCH}/aug26}"

if [ "${1:-}" = "--list" ]; then
    exec python3 "$GEN" --list
fi

OUT="${1:-$EXP/runs}"

# Samples per rank. CAMPAIGN_LOCAL_BATCH=atm=2 halves every atmosphere run's node
# count and takes the campaign from 111 nodes to 64 -- see EXPERIMENTS.md
# "Measurements". It is passed to the checker too, which duplicates the constant
# deliberately and would otherwise compute the wrong rank count.
LB=()
[ -n "${CAMPAIGN_LOCAL_BATCH:-}" ] && LB=(--local-batch "$CAMPAIGN_LOCAL_BATCH")

# Clear stale output first. `runs/` is entirely generated, and a factor-word
# change renames every file in it -- without this the old ids linger beside the
# new ones and someone can launch an orphan by hand. check_campaign.py would
# flag them, but only if it is pointed at them.
if [ -d "$OUT" ]; then
    find "$OUT" -maxdepth 1 -type f \( -name '*.yaml' -o -name '*.env' \) -delete
    rm -f "$OUT/MANIFEST.tsv"
fi

python3 "$GEN" --all -o "$OUT" --owner "$OWNER" --campaign-root "$ROOT" "${LB[@]}"

# Assert every emitted config says what its run id says it says. validate_config
# proves a config parses; this proves E05...A3_B16_C1... actually has CO2, both
# aerosol sets, batch 16 and equal weights. A silent disagreement between a run
# id and its config is the worst failure this campaign has, because every plot
# is labelled by the run id.
echo
python3 "$EXP/check_campaign.py" --dir "$OUT" "${LB[@]}"

echo
echo "next: ./submit-campaign.sh --dry-run"
