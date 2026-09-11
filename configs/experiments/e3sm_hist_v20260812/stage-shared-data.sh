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
#
# 2026-09-11: the landfrac files gained a third variable,
# atmosphere_flux_fraction (see make_landfrac_ocn.py and AGENTS.md), and the
# coupled stepper refuses to run without it. Re-running this script refreshes
# a stale copy on CFS: files are copied when the source is newer (cp -u), not
# only when the destination is absent, and --check verifies the variable is
# present in what the configs read. landfrac1d (E17's O1 forcing) is staged
# here too now; it used to be copied by hand.

set -euo pipefail

DEST_ROOT=${FME_HIST_SHARED_ROOT:-/global/cfs/cdirs/e3smdata/emulator/SamudrACE-E3SMv3/historical}
SRC_STATS=/pscratch/sd/m/mahf708/2026-08-13-E3SMv3-historical-stats
SRC_LANDFRAC=/pscratch/sd/m/mahf708/e3sm-hist-aux/landfrac5d
SRC_LANDFRAC1D=/pscratch/sd/m/mahf708/e3sm-hist-aux/landfrac1d
# The stage-2/3 generators read the inputs from this scratch mirror rather
# than CFS (make_stage2_config.remap_paths); keep it in step with CFS.
SCRATCH_INPUTS=/pscratch/sd/m/mahf708/e3sm-hist-inputs

DEST_STATS=$DEST_ROOT/stats-2026-08-13
DEST_LANDFRAC=$DEST_ROOT/landfrac5d
# Also read by the configs: landfrac1d is E17's forcing and the simulation
# output is the training data itself. --check reports them because "can I read
# the inputs?" is the question being asked.
DEST_LANDFRAC1D=$DEST_ROOT/landfrac1d
REQUIRED_LANDFRAC_VARS="LANDFRAC sea_surface_fraction atmosphere_flux_fraction"
SIM_ROOT=/global/cfs/cdirs/e3smdata/simulations/v3.LR.historical_0101.aigo/run

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

echo "source stats    : $SRC_STATS"
echo "source landfrac : $SRC_LANDFRAC"
echo "source landfrac1d: $SRC_LANDFRAC1D"
echo "destination     : $DEST_ROOT"
echo

if [ "$CHECK_ONLY" = 1 ]; then
    # Check what the configs actually READ, from the account running this.
    # The two SRC_ trees below are mahf708's staging sources; they are not
    # inputs to anybody's run, so a teammate seeing them missing is fine and a
    # teammate seeing the four destinations present is what matters. The
    # earlier version of this check listed only two of the four destinations
    # and never looked at landfrac1d, which is E17's only input.
    rc=0
    echo "inputs the generated configs read:"
    for p in "$SIM_ROOT" "$DEST_STATS" "$DEST_LANDFRAC" "$DEST_LANDFRAC1D"; do
        if [ -r "$p" ] && ls "$p" > /dev/null 2>&1; then
            echo "  readable: $p"
        else
            echo "  UNREADABLE: $p"; rc=1
        fi
    done
    echo
    echo "landfrac variables the coupled stepper needs ($REQUIRED_LANDFRAC_VARS):"
    REPO_ROOT=$(cd "$HERE/../../.." && pwd)
    for d in "$DEST_LANDFRAC" "$DEST_LANDFRAC1D" "$SCRATCH_INPUTS/landfrac5d" "$SCRATCH_INPUTS/landfrac1d"; do
        f=$(ls "$d"/*.1990.nc 2>/dev/null | head -1)
        if [ -z "$f" ]; then echo "  no 1990 file in $d"; rc=1; continue; fi
        missing=$("$REPO_ROOT/.venv/bin/python" - "$f" $REQUIRED_LANDFRAC_VARS <<'PY'
import sys, xarray as xr
have = set(xr.open_dataset(sys.argv[1]).data_vars)
print(" ".join(v for v in sys.argv[2:] if v not in have))
PY
)
        if [ -z "$missing" ]; then echo "  complete: $d"; else echo "  MISSING $missing: $d"; rc=1; fi
    done
    echo
    echo "staging sources (mahf708 only; not needed to run):"
    for p in "$SRC_STATS" "$SRC_LANDFRAC" "$SRC_LANDFRAC1D"; do
        if [ -e "$p" ]; then echo "  present: $p"; else echo "  absent:  $p"; fi
    done
    echo
    echo "config references still pointing at personal scratch:"
    grep -c '/pscratch/sd/m/mahf708/\(2026-08-13\|e3sm-hist-aux\)' \
        "$HERE"/config-train-*.yaml || true
    [ "$rc" = 0 ] && echo && echo "all inputs readable" \
                  || { echo; echo "MISSING INPUTS -- you are probably not in the e3smdata group" >&2; }
    exit "$rc"
fi

for p in "$SRC_STATS" "$SRC_LANDFRAC" "$SRC_LANDFRAC1D"; do
    [ -d "$p" ] || { echo "source missing: $p" >&2; exit 1; }
done

mkdir -p "$DEST_ROOT"
# -u: copy when the source is newer, so a regenerated file replaces its stale
# copy and a re-run is otherwise a no-op. The trailing `/.` matters --
# `cp -r src dest` with dest already a directory would nest the copy at
# $DEST_STATS/$(basename $SRC_STATS) instead of merging into it.
mkdir -p "$DEST_STATS" "$DEST_LANDFRAC" "$DEST_LANDFRAC1D"
cp -ru "$SRC_STATS/."      "$DEST_STATS/"
cp -ru "$SRC_LANDFRAC/."   "$DEST_LANDFRAC/"
cp -ru "$SRC_LANDFRAC1D/." "$DEST_LANDFRAC1D/"
# The scratch mirror the generated run configs read.
mkdir -p "$SCRATCH_INPUTS/landfrac5d" "$SCRATCH_INPUTS/landfrac1d"
cp -ru "$SRC_LANDFRAC/."   "$SCRATCH_INPUTS/landfrac5d/"
cp -ru "$SRC_LANDFRAC1D/." "$SCRATCH_INPUTS/landfrac1d/"

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

# The venv's python rather than `uv run`: uv locks under $HOME, which fails
# on Perlmutter (flock errno 524), and the venv is already synced.
REPO_ROOT=$(cd "$HERE/../../.." && pwd)
( cd "$REPO_ROOT" && .venv/bin/python "$HERE/make_cpl_config.py" )
"$HERE/regen_cft_configs.sh"

echo
echo "staged and repointed. Remaining personal-scratch INPUT references:"
grep -c '/pscratch/sd/m/mahf708/\(2026-08-13\|e3sm-hist-aux\)' \
    "$HERE"/config-train-*.yaml || echo "  none"
echo "(experiment_dir values are outputs and are intentionally left alone;"
echo " each user overrides those at launch.)"
