#!/bin/bash
# srun into another of the user's allocations from inside this one.
# The inherited SLURM_* variables make srun reject a foreign --jobid
# ("Invalid job id specified"), so drop them first.
#   srun-other.sh JOBID [srun args...] -- cmd...
job=$1; shift
unset $(env | grep -E '^SLURM_' | cut -d= -f1)
exec srun --jobid="$job" --overlap "$@"
