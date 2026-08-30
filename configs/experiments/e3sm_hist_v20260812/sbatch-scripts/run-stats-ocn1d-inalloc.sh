#!/bin/bash
# Same work as sbatch-stats-ocn1d.sh, but run from INSIDE an existing salloc.
# CPU-partition sbatch is not available to this account (every `-C cpu`
# combination is rejected as "Job request does not match any supported policy"
# and the GPU queue estimate is days out), so the stats are computed in the
# interactive GPU allocation instead. The work is CPU-bound; the GPUs idle.
set -u
REPO=/pscratch/sd/m/mahf708/ace
EXP=$REPO/configs/experiments/e3sm_hist_v20260812
OUT=${STATS_OUT:-/pscratch/sd/m/mahf708/2026-08-13-E3SMv3-historical-stats}
P=$OUT/_partials
LOG=$EXP/joblogs
mkdir -p "$P" "$OUT/ocean-1d" "$OUT/train-only/ocean-1d" "$LOG"
cd "$REPO" || exit 1

echo "START $(date -Is)  job=${SLURM_JOB_ID:-none}  out=$OUT"

# One shard per node. Chan's parallel variance is exact and order-independent,
# so sharding does not change the answer. -c 128 takes the whole node (the
# allocation defaults to 1 CPU/task, which would leave nproc=2).
for i in 0 1 2 3; do
    srun -N 1 -n 1 -c 128 --exclusive uv run python "$EXP/compute_hist_stats.py" \
        --realm ocean-1d --shard "$i/4" \
        --partials "$P/ocn1d-shard$i.pkl" --partials-only \
        --out-dir "$OUT/ocean-1d" --workers 64 > "$LOG/ocn1d-shard$i.log" 2>&1 &
done
wait
echo "SHARDS_DONE $(date -Is)"
ls -l "$P"/ocn1d-shard*.pkl

PARTIALS="$P/ocn1d-shard0.pkl,$P/ocn1d-shard1.pkl,$P/ocn1d-shard2.pkl,$P/ocn1d-shard3.pkl"

# Both aggregations re-use the same partials; no second pass over 2.42 TiB.
srun -N 1 -n 1 -c 128 --exclusive uv run python "$EXP/compute_hist_stats.py" \
    --realm ocean-1d --out-dir "$OUT/ocean-1d" \
    --reuse-partials --partials "$PARTIALS" --workers 64 > "$LOG/ocn1d-agg-full.log" 2>&1
echo "AGG_FULL_EXIT=$?"

srun -N 1 -n 1 -c 128 --exclusive uv run python "$EXP/compute_hist_stats.py" \
    --realm ocean-1d --out-dir "$OUT/train-only/ocean-1d" \
    --reuse-partials --partials "$PARTIALS" --workers 64 \
    --years 1940-1989,2000-2039 > "$LOG/ocn1d-agg-train.log" 2>&1
echo "AGG_TRAIN_EXIT=$?"

echo "DONE $(date -Is)"
ls -l "$OUT/ocean-1d" "$OUT/train-only/ocean-1d"
