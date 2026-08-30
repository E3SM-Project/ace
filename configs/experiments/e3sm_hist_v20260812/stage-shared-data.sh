#!/bin/bash
# One-time: move the two auxiliary INPUT trees off personal scratch onto the
# project's Community File System directory, then repoint the configs at it.
#
#     ./stage-shared-data.sh            # copy + rewrite the configs
#     ./stage-shared-data.sh --check    # report only, change nothing
#
# Why this exists
# ---------------
# The normalization statistics and the 5-day LANDFRAC file are inputs, but they
# were produced into /pscratch/sd/m/mahf708/. That is wrong for a config other
# people run, for two reasons:
#
#   * $PSCRATCH is purged on an inactivity policy and is not backed up, so a
#     colleague cloning this branch in a few months gets FileNotFoundError.
#   * It is readable only because those directories carry world-read bits.
#     NERSC guidance is to share through a project directory with group
#     permissions and to avoid world-readable data
#     (https://docs.nersc.gov/filesystems/sharing/).
#
# CFS is group-readable by default and is the documented way to share within a
# project, so the aux inputs belong beside the run they describe. The whole set
# is ~115 MB.
#
# The raw model output is already on CFS and is not touched by this script.

set -euo pipefail

DEST_ROOT=${FME_HIST_SHARED_ROOT:-/global/cfs/cdirs/e3smdata/emulator/SamudrACE-E3SMv3/historical}
SRC_STATS=/pscratch/sd/m/mahf708/2026-08-13-E3SMv3-historical-stats
SRC_LANDFRAC=/pscratch/sd/m/mahf708/e3sm-hist-aux/landfrac5d

DEST_STATS=$DEST_ROOT/stats-2026-08-13
DEST_LANDFRAC=$DEST_ROOT/landfrac5d

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

echo "source stats    : $SRC_STATS"
echo "source landfrac : $SRC_LANDFRAC"
echo "destination     : $DEST_ROOT"
echo

if [ "$CHECK_ONLY" = 1 ]; then
    for p in "$SRC_STATS" "$SRC_LANDFRAC" "$DEST_STATS" "$DEST_LANDFRAC"; do
        if [ -e "$p" ]; then echo "  present: $p"; else echo "  MISSING: $p"; fi
    done
    echo
    echo "config references still pointing at personal scratch:"
    grep -c '/pscratch/sd/m/mahf708/\(2026-08-13\|e3sm-hist-aux\)' \
        "$HERE"/config-train-*.yaml || true
    exit 0
fi

for p in "$SRC_STATS" "$SRC_LANDFRAC"; do
    [ -d "$p" ] || { echo "source missing: $p" >&2; exit 1; }
done

mkdir -p "$DEST_ROOT"
# -n: never clobber an already-staged copy; re-run is safe. The trailing `/.`
# matters -- `cp -r src dest` with dest already a directory would nest the copy
# at $DEST_STATS/$(basename $SRC_STATS) instead of merging into it.
mkdir -p "$DEST_STATS" "$DEST_LANDFRAC"
cp -rn "$SRC_STATS/."    "$DEST_STATS/"
cp -rn "$SRC_LANDFRAC/." "$DEST_LANDFRAC/"

# Group-readable, not world-readable: the project group is the audience.
chgrp -R e3smdata "$DEST_ROOT" 2>/dev/null || \
    echo "note: could not chgrp; check the destination's group ownership" >&2
chmod -R g+rX,o-rwx "$DEST_ROOT"

# Repoint the two component configs. The coupled config is generated, so it is
# regenerated from them rather than edited.
for f in config-train-atm.yaml config-train-ocn.yaml; do
    sed -i \
        -e "s|$SRC_STATS|$DEST_STATS|g" \
        -e "s|$SRC_LANDFRAC|$DEST_LANDFRAC|g" \
        "$HERE/$f"
done

REPO_ROOT=$(cd "$HERE/../../.." && pwd)
( cd "$REPO_ROOT" && uv run python "$HERE/make_cpl_config.py" )

echo
echo "staged and repointed. Remaining personal-scratch INPUT references:"
grep -c '/pscratch/sd/m/mahf708/\(2026-08-13\|e3sm-hist-aux\)' \
    "$HERE"/config-train-*.yaml || echo "  none"
echo "(experiment_dir values are outputs and are intentionally left alone;"
echo " each user overrides those at launch.)"
