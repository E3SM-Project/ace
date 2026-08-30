#!/bin/bash
# Submit the aug26 campaign in priority order.
#
#     ./submit-campaign.sh --dry-run              # print what would be queued
#     ./submit-campaign.sh --preflight            # stage + validate all, queue none
#     ./submit-campaign.sh                        # queue everything (P1..P4)
#     ./submit-campaign.sh --max-priority 3       # queue P1..P3 only
#     ./submit-campaign.sh --only atm             # one realm
#     ./submit-campaign.sh --only E05             # one experiment
#
# There is no dependency graph: the page's redesign makes every run an
# independent from-scratch training, so this just walks ../runs/MANIFEST.TSV in
# priority order and submits. Slurm decides what runs when.
#
# Why priority order matters. The run list adds up to 111 nodes and the
# reservation is 96, so 15 nodes' worth cannot start at once. Submitting in
# priority order means the queue drains in the order the science needs:
#
#   P1  14 nodes  the four bolded baselines at B16 S01 (E01 E02 E05 E11)
#   P2  34 nodes  the single-seed science ablations -- the only measurement of
#                 their factor that exists at all
#   P3  28 nodes  seeds S02/S03 of the bolded four
#   P4  35 nodes  the B08/B32 batch sweeps -- an optimizer question, not a
#                 science question, and the right thing to lose to a queue
#
# P1+P2+P3 = 76 nodes and fits with 20 to spare; P4 lands as capacity frees.
#
# During the hackathon window, export the reservation or every job sits in the
# regular queue while the 96 reserved nodes idle:
#
#     RESERVATION=_CAP_aigs_hist ./submit-campaign.sh
#
# Drop it for anything continuing past the window's end (Sat 2026-09-05 15:00);
# a 12 h segment that cannot finish inside the reservation will not start in it.

# Email is on by default (run-train.sh sets it): $USER@nersc.gov on BEGIN, END,
# FAIL, REQUEUE and TIME_LIMIT_90. The whole campaign is order 250 messages, so
# either filter on the subject or submit with FME_MAIL_TYPE=FAIL,TIME_LIMIT_90
# to hear only about trouble.

set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXP=$(dirname "$HERE")
RUN="$HERE/run-train.sh"
MANIFEST="$EXP/runs/MANIFEST.tsv"

DRY=0
PRE=0
ONLY=""
MAXP=4
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run)      DRY=1; shift ;;
        # The Sunday-night check: exercises staging, the .env, the per-run
        # sizing and the config validator for every run without queueing one.
        --preflight)    PRE=1; shift ;;
        --only)         ONLY="${2:?--only needs a realm or experiment id}"; shift 2 ;;
        --max-priority) MAXP="${2:?--max-priority needs 1..4}"; shift 2 ;;
        *) echo "usage: $0 [--dry-run|--preflight] [--only atm|ocn|E05] [--max-priority N]" >&2
           exit 2 ;;
    esac
done

[ -f "$MANIFEST" ] || {
    echo "no $MANIFEST -- run ./generate-campaign.sh first" >&2; exit 1; }

total=0
count=0
while IFS=$'\t' read -r pri runid realm nodes ranks batch seed note; do
    [ "$pri" = "priority" ] && continue
    [ "${pri#P}" -le "$MAXP" ] || continue
    if [ -n "$ONLY" ] && [ "$realm" != "$ONLY" ] && [ "${runid%%.*}" != "$ONLY" ]; then
        continue
    fi
    total=$((total + nodes))
    count=$((count + 1))
    if [ "$DRY" = 1 ]; then
        printf '%-3s %2s nodes  %-42s %s\n' "$pri" "$nodes" "$runid" "$note"
        continue
    fi
    printf '%-3s %2s nodes  %s\n' "$pri" "$nodes" "$runid"
    if [ "$PRE" = 1 ]; then
        "$RUN" "$realm" "$runid" --no-submit > /dev/null || { echo "PREFLIGHT FAILED: $runid" >&2; exit 1; }
    else
        "$RUN" "$realm" "$runid" > /dev/null
    fi
done < "$MANIFEST"

echo
if [ "$DRY" = 1 ]; then
    echo "$count runs, $total nodes (dry run, nothing submitted)"
elif [ "$PRE" = 1 ]; then
    echo "$count runs, $total nodes -- all staged and validated, nothing queued"
else
    echo "$count runs, $total nodes submitted"
fi
[ "$total" -gt 96 ] && echo "NOTE: $total nodes exceeds the 96-node reservation; \
the tail waits in the queue." >&2
exit 0
