#!/bin/bash -l
# Normalization statistics for the 1-daily ocean streams (the O1 cadence, E17).
#
#     sbatch sbatch-stats-ocn1d.sh
#
# E17 otherwise borrows the 5-day statistics, which is defensible for a cadence
# comparison -- measured 1-day/5-day standard-deviation ratios are 1.000-1.005
# for temperature, sst, ssh and ice area -- but is a ~12% error on the
# flux-like and deep-velocity channels (latentHeatFlux 1.115,
# velocityMeridionalCoarsened_18 1.130). A production O1 run wants its own.
#
# Sharded because the 1-day streams are 2.42 TiB against the 5-day set's
# 283 GiB: fmeDepthCoarsening alone is 1.4 GiB/file x 1501 files. Each node
# writes partials for its shard; one aggregation pass then produces both the
# full-record and the train-only sets from those partials without re-reading.
#
# Output lands beside the 5-day stats so a config can point at either:
#   <OUT>/ocean-1d/            full record, 1940-2065
#   <OUT>/train-only/ocean-1d/ the training windows only -- USE THIS ONE

# CPU nodes, so the account is `e3sm` and NOT `e3sm_g` -- the _g suffix is the
# GPU allocation, and pairing it with `-C cpu` is rejected as
# "Job request does not match any supported policy".
#SBATCH -A e3sm
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH -J hist-stats-ocn1d
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=256
#SBATCH -t 04:00:00
#SBATCH --output=joblogs/%x-%j.out

set -x
REPO=/pscratch/sd/m/mahf708/ace
EXP=$REPO/configs/experiments/e3sm_hist_v20260812
OUT=${STATS_OUT:-/pscratch/sd/m/mahf708/2026-08-13-E3SMv3-historical-stats}
P=$OUT/_partials
mkdir -p "$P" "$OUT/ocean-1d" "$OUT/train-only/ocean-1d"

cd "$REPO" || exit 1

# One shard per node, partials only. Chan's parallel variance is exact and
# order-independent, so the shard split does not change the answer.
for i in 0 1 2 3; do
    srun -N 1 -n 1 --exclusive uv run python "$EXP/compute_hist_stats.py" \
        --realm ocean-1d --shard "$i/4" \
        --partials "$P/ocn1d-shard$i.pkl" --partials-only \
        --out-dir "$OUT/ocean-1d" --workers 64 &
done
wait

PARTIALS="$P/ocn1d-shard0.pkl,$P/ocn1d-shard1.pkl,$P/ocn1d-shard2.pkl,$P/ocn1d-shard3.pkl"

# Full record, then the training windows only -- both re-aggregate the same
# partials, no second pass over the data.
srun -N 1 -n 1 uv run python "$EXP/compute_hist_stats.py" \
    --realm ocean-1d --out-dir "$OUT/ocean-1d" \
    --reuse-partials --partials "$PARTIALS" --workers 64

srun -N 1 -n 1 uv run python "$EXP/compute_hist_stats.py" \
    --realm ocean-1d --out-dir "$OUT/train-only/ocean-1d" \
    --reuse-partials --partials "$PARTIALS" --workers 64 \
    --years 1940-1989,2000-2039

echo "REAL_EXIT=$?"
ls -l "$OUT/ocean-1d" "$OUT/train-only/ocean-1d"
