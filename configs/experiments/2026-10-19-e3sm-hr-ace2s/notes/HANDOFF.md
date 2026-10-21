# Handoff: HR ACE2S bring-up (2026-10-19 evening)

Branch `e3sm/exps/hr-2026.10.0` (worktree `/pscratch/sd/m/mahf708/ace-hr`), pushed to E3SM-Project/ace.
Status page: https://claude.ai/artifact/Bc2uTxmhi7Ct1VqXcNnb9u (built
from `report/narrative.html` by `scripts/build_report.py`).

## Current state (2026-10-21) -- read this first

- Production-ready on 4x A100-40GB per replica, full harmonics, with commits
  through e035d4cbf (prognostic states no longer view their batch; without
  it 1-2 node runs and 2-IC inference OOM on 40 GB). README "Running on 40
  GB A100s" has the requirements, the submit line and the per-epoch cost.
- Throughput 5.00 s/step at 1, 2 and 4 nodes (0.200 samples/s per replica);
  20 epochs ~70 h on 8 nodes. N must be 4 or 8 with the config's 8 ICs.
- Open decisions for the user: hard_thresholding_fraction (config 0.5,
  validated production setting is 1.0 via EXTRA), W&B logging (off), node
  count (sets the global batch).
- Not done: 8+ node measurement (needs one allocation); replica-count
  invariant noise (not needed for training); porting the EMA and get_start
  fixes to the hist branch / upstream (user: not now).

## Done
- Data: all 193 `eam.h0` files copied to `$PSCRATCH/hr-data/v3.HR.F2010.aigo.PD.y1/eam.h0`.
- Code on the branch (on top of v2026.10.0): 7 ported hist commits plus 5 new
  fixes (DDP grad sync, loss/corrector exactness, cdf5 engine, IC divisibility,
  htf builder knob). See `git log 8935f50b6..`.
- Benchmarks: logs/*.jsonl (s1 single GPU, s2/s5 one node, s3 multi-node),
  profiles in logs/prof-*.txt.

## State at ~23:20 PDT
- HR stats: done (`$PSCRATCH/hr-data/stats-2026-10-19/train-stride4/`).
- End-to-end training verified on 1 and 4 nodes (modes x0.5, 4x1 tiles),
  including checkpoint resume. The full baseline OOMs at 4x1.
- Pilot: `$PSCRATCH/hr-runs/pilot-4n-htf05` (4 nodes, scripts/pilot.sh). It dies
  with allocation 60022811 (~00:33). Its checkpoints can be resumed with
  `EXPDIR=$PSCRATCH/hr-runs/pilot-4n-htf05 CONFIG=$PSCRATCH/hr-runs/pilot-4n-htf05/config.yaml
   RUN=pilot sbatch -N 4 scripts/sbatch-train-hr.sh` (the sbatch script is
  untested in the queue; `sbatch --test-only` passes).

## Next
1. The user decides: modes x0.5 (recommended) vs the full baseline (needs 8
   tiles over 2 nodes; poor scaling).
2. Before production: disable the validation snapshot/mean_map images and the
   inference map plots (2+ min per epoch at 0.25 deg); pick the inline
   inference length (360 steps is ~2.5 min at 1 IC per rank).
3. Production: ~8 nodes, 48 h, ~37 h for 20 epochs.
4. Not done: the restart checkpoint on SIGTERM under spatial parallelism, a
   spatial gather for writers/map aggregators, power spectrum under spatial
   parallelism, a faster SHT precompute (~40 s single-threaded per build).

## Gotchas
- Never edit fme/ while a benchmark imports it live; use `scripts/snapshot.sh`.
- `pkill -f` patterns must not match your own shell command line.
- srun into another allocation: `scripts/srun-other.sh JOBID ...`.
- Multi-node: `source scripts/nccl-ofi.sh` (train.sh and bench_spatial.sh do).


## Update 2026-10-20 (allocation 60032495)
- Pushed: E3SM-Project/ace `e3sm/exps/hr-2026.10.0`.
- New fme features, all opt-in and validated on GPU:
  - `optimization.shard_optimizer_state` (ZeRO over each spatial group);
  - `stepper.step.config.builder.config.shard_spatial_parameters` (local
    l-slices of spectral weights and pos_embed tiles);
  - SHT dedup and `gradient_as_bucket_view` (always on).
- With both flags the FULL-harmonics model (hard_thresholding_fraction 1.0)
  trains at 4x1 tiles per node: 45.5 GB peak, 0.255 samples/s per replica.
  The production config still says 0.5; switching back to 1.0 is now a
  pure modeling choice at ~1.65x the step time.
- Serial vs spatial: forward bitwise identical in fp32; fp64 agrees to
  1e-15; sharded vs unsharded drift equals the run-to-run noise floor.
- Comm: 4x1 within a node is 9-14% overhead; tiles across nodes are 35-48%.
  Use data parallelism across nodes.
- Not done: standalone evaluator under spatial parallelism (pre-existing
  failure in flush_diagnostics); use_gradient_accumulation with the ddp
  grad sync (pre-existing, wrong); bf16/overlapped transposes.
