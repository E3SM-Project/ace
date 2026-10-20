#!/bin/bash
# Multi-node scaling on the user's second allocation, from a frozen snapshot.
#   scaling2.sh JOBID NODELIST OUT
job=$1; nodes=$2; out=$3; cd "$(dirname "$0")/.."
S=${ACE_SRC:?export ACE_SRC=\$(scripts/snapshot.sh)}
run() {  # nnodes gs H W args...
  local n=$1 gs=$2; shift 2
  local nl; nl=$(echo "$nodes" | tr ',' '\n' | head -n "$n" | paste -sd,)
  timeout 1500 scripts/srun-other.sh "$job" -N "$n" -w "$nl" --ntasks-per-node 1 \
    --gpus-per-node 4 --export=ALL,ACE_SRC=$S,FME_SPATIAL_GRAD_SYNC=$gs,OFI=1,SINGLE=0,PORT=$((29600 + RANDOM % 300)) \
    scripts/bench_spatial.sh "$@" | sed -n 's/^RESULT //p' >> "$out"
}
A="--embed-dim 384 --checkpointing 3 --steps 5 --phases"
run 2 ddp 4 1 $A --batch 4 --htf 0.5 --tag dp2-sp4x1-htf0.5-ddp
run 2 hooks 4 1 $A --batch 4 --htf 0.5 --tag dp2-sp4x1-htf0.5-hooks
run 3 ddp 4 1 $A --batch 6 --htf 0.5 --tag dp3-sp4x1-htf0.5-ddp
run 2 ddp 4 1 $A --batch 4 --tag dp2-sp4x1-base-ddp
run 2 hooks 4 1 $A --batch 4 --tag dp2-sp4x1-base-hooks
run 3 ddp 4 1 $A --batch 6 --tag dp3-sp4x1-base-ddp
run 2 ddp 4 2 $A --batch 2 --tag dp1-sp4x2-base-ddp
run 3 ddp 4 3 $A --batch 2 --htf 0.5 --tag dp1-sp4x3-htf0.5-ddp
