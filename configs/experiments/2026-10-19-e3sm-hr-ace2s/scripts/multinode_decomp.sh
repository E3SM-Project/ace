#!/bin/bash
# Cross-node communication decomposition in the current allocation (4 nodes).
#   ACE_SRC=... multinode_decomp.sh OUT
out=$1; cd "$(dirname "$0")/.."
mapfile -t N < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
run() {  # nodelist H W args...
  local nl=$1; shift
  local n; n=$(echo "$nl" | tr ',' '\n' | wc -l)
  timeout 1500 srun --overlap -N "$n" -w "$nl" --ntasks-per-node 1 --gpus-per-node 4 \
    --export=ALL,ACE_SRC=$ACE_SRC,OFI=1,SINGLE=0,PORT=$((29600 + RANDOM % 300)) \
    scripts/bench_spatial.sh "$@" | sed -n 's/^RESULT //p' >> "$out"
}
A="--embed-dim 384 --htf 0.5 --checkpointing 3 --steps 6 --phases"
P1=${N[0]},${N[1]}; P2=${N[2]},${N[3]}; ALL=$(IFS=,; echo "${N[*]}")
# 2-node layouts, two at a time
run $P1 8 1 $A --batch 2 --tag x-h05-8x1-real & run $P2 4 2 $A --batch 2 --tag x-h05-4x2-real & wait
run $P1 8 1 $A --batch 2 --fake-transpose --no-grad-sync --tag x-h05-8x1-compute & run $P2 4 2 $A --batch 2 --fake-transpose --no-grad-sync --tag x-h05-4x2-compute & wait
run $P1 8 1 $A --batch 2 --no-grad-sync --tag x-h05-8x1-real-nosync & run $P2 4 2 $A --batch 2 --no-grad-sync --tag x-h05-4x2-real-nosync & wait
run $P1 2 4 $A --batch 2 --tag x-h05-2x4-real & run $P2 4 1 $A --batch 4 --tag x-h05-4x1-dp2-real & wait
run $P1 2 4 $A --batch 2 --fake-transpose --no-grad-sync --tag x-h05-2x4-compute & run $P2 4 1 $A --batch 4 --no-grad-sync --tag x-h05-4x1-dp2-real-nosync & wait
# 4-node layouts
run $ALL 16 1 $A --batch 2 --tag x-h05-16x1-real
run $ALL 16 1 $A --batch 2 --fake-transpose --no-grad-sync --tag x-h05-16x1-compute
run $ALL 4 4 $A --batch 2 --tag x-h05-4x4-real
run $ALL 4 4 $A --batch 2 --fake-transpose --no-grad-sync --tag x-h05-4x4-compute
run $ALL 4 1 $A --batch 8 --tag x-h05-4x1-dp4-real
