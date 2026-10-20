#!/bin/bash
# Freeze the worktree's fme/ for benchmarking; prints the ACE_SRC to export.
#   export ACE_SRC=$(scripts/snapshot.sh)
src=/pscratch/sd/m/mahf708/ace-hr
id=$(cd $src && git rev-parse --short HEAD)$(cd $src && git diff --quiet -- fme || echo "-dirty$(date +%H%M%S)")
dst=$PSCRATCH/hr-bench-snap/$id
[[ -d $dst/fme ]] || { mkdir -p "$dst" && rsync -a --exclude __pycache__ "$src/fme" "$dst/"; }
echo "$dst"
