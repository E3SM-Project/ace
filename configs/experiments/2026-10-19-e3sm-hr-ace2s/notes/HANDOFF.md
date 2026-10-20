# Handoff: HR ACE2S bring-up (2026-10-19 evening)

Branch `e3sm/exps/hr-2026.10.0` (worktree `/pscratch/sd/m/mahf708/ace-hr`), not
pushed. Status page: https://claude.ai/artifact/Bc2uTxmhi7Ct1VqXcNnb9u (built
from `report/narrative.html` by `scripts/build_report.py`).

## Done
- Data: all 193 `eam.h0` files copied to `$PSCRATCH/hr-data/v3.HR.F2010.aigo.PD.y1/eam.h0`.
- Code on the branch (on top of v2026.10.0): 7 ported hist commits plus 5 new
  fixes (DDP grad sync, loss/corrector exactness, cdf5 engine, IC divisibility,
  htf builder knob). See `git log 8935f50b6..`.
- Benchmarks: logs/*.jsonl (s1 single GPU, s2/s5 one node, s3 multi-node),
  profiles in logs/prof-*.txt.

## In flight / next
1. Stats: `scripts/compute_stats.py partial` (39 files, stride 4) writes
   `$PSCRATCH/hr-data/stats-2026-10-19/train-stride4/partial-*.pkl`. Then run
   `python scripts/compute_stats.py combine --out-dir <that dir>`
   (from scripts/, after `source env.sh`). If the partials are missing, rerun
   partial: it reads with scripts/cdf5.py and should take ~15 min on 4 idle nodes.
2. The smoke run (`runs/smoke-1n-4x1-base.yaml`, LR stats as a placeholder):
   check `$PSCRATCH/hr-runs/smoke-1n-4x1-base/out.log` for s/step, memory and
   the inline-inference and checkpoint paths.
3. A 4-node run (4x1 tiles x 4 replicas) with `scripts/train.sh` and OFI,
   then the production decision: htf 1.0 or 0.5 (the user's call).
4. Not done: read-only-the-tile loading (4x redundant reads), the restart
   checkpoint under spatial parallelism, a spatial gather for writers.

## Gotchas
- Never edit fme/ while a benchmark imports it live; use `scripts/snapshot.sh`.
- `pkill -f` patterns must not match your own shell command line.
- srun into another allocation: `scripts/srun-other.sh JOBID ...`.
- Multi-node: `source scripts/nccl-ofi.sh` (train.sh and bench_spatial.sh do).
