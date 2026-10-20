#!/bin/bash
# Multi-node scaling runs, sequential, in the current allocation.
#   scaling.sh OUT
out=$1; cd "$(dirname "$0")/.."
run() {  # nodes H W args...
  local n=$1; shift
  OFI=${OFI:-1} PORT=$((29600 + RANDOM % 300)) SINGLE=0 timeout 1200 srun --overlap -N "$n" --ntasks-per-node 1 \
    --gpus-per-node 4 -c 32 scripts/bench_spatial.sh "$@" | sed -n 's/^RESULT //p' >> "$out"
}
A="--embed-dim 384 --checkpointing 3 --batch 2 --steps 5"
run 2 2 2 $A --batch 4 --tag dp2-sp2x2-base
run 4 2 2 $A --batch 8 --tag dp4-sp2x2-base
run 2 2 4 $A --tag dp1-sp2x4-base
run 2 4 2 $A --tag dp1-sp4x2-base
run 4 4 4 $A --tag dp1-sp4x4-base
run 4 2 2 $A --batch 8 --spectral-ratio 0.5 --tag dp4-sp2x2-sr0.5
run 4 2 2 $A --batch 8 --htf 0.5 --tag dp4-sp2x2-htf0.5
