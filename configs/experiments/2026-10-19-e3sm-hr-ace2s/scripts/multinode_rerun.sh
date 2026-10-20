#!/bin/bash
# Reruns of multinode_decomp.sh runs that lost their rendezvous to a random
# port collision; fixed, distinct ports this time.
out=$1; cd "$(dirname "$0")/.."
mapfile -t N < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
run() {  # port nodelist H W args...
  local port=$1 nl=$2; shift 2
  local n; n=$(echo "$nl" | tr ',' '\n' | wc -l)
  timeout 1500 srun --overlap -N "$n" -w "$nl" --ntasks-per-node 1 --gpus-per-node 4 \
    --export=ALL,ACE_SRC=$ACE_SRC,OFI=1,SINGLE=0,PORT=$port \
    scripts/bench_spatial.sh "$@" | sed -n 's/^RESULT //p' >> "$out"
}
A="--embed-dim 384 --htf 0.5 --checkpointing 3 --steps 6 --phases"
P1=${N[0]},${N[1]}; P2=${N[2]},${N[3]}; ALL=$(IFS=,; echo "${N[*]}")
run 29711 $P1 8 1 $A --batch 2 --no-grad-sync --tag x-h05-8x1-real-nosync & run 29712 $P2 2 4 $A --batch 2 --fake-transpose --no-grad-sync --tag x-h05-2x4-compute & wait
run 29713 $P1 2 4 $A --batch 2 --no-grad-sync --tag x-h05-2x4-real-nosync & run 29714 $P2 4 1 $A --batch 4 --no-grad-sync --tag x-h05-4x1-dp2-real-nosync & wait
run 29715 $ALL 4 4 $A --batch 2 --no-grad-sync --tag x-h05-4x4-real-nosync
run 29716 $ALL 16 1 $A --batch 2 --no-grad-sync --tag x-h05-16x1-real-nosync
echo RERUN-DONE
