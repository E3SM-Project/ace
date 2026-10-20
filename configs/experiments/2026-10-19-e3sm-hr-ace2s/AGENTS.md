# AGENTS.md: HR ACE2S experiment folder

Guidance and running log for agents working in
`configs/experiments/2026-10-19-e3sm-hr-ace2s/` on branch
`e3sm/exps/hr-2026.10.0`. The repo-wide rules in `/AGENTS.md` still apply.
Read `README.md` for the science and the results. Read `notes/HANDOFF.md`
for exactly where work stopped.

## Ground rules

- **Scope: atmosphere only (ACE2S).** F2010 is an atmosphere + land
  configuration with prescribed SST (docn). There is no ocean model, and
  the user said to ignore the coupler/ELM/MPAS files.
- **Branch base:** upstream tag `v2026.10.0` (8935f50b6, "Release v2026.10.0
  (#1585)"). This is NOT the hist branch. Features that exist only on
  `e3sm/exps/hist-v2026.8.0` (e.g. the XarrayDataset `rename` and
  `reference_pressure_name`) are missing until they are ported. See
  `notes/hist-commit-triage.md`.
- **Worktree:** `/pscratch/sd/m/mahf708/ace-hr`. The user's main checkout
  `/pscratch/sd/m/mahf708/ace` stays on the hist branch. Do not switch it.
- **Python env:** `source scripts/env.sh`. It reuses the main checkout's uv
  venv and puts this worktree first on `PYTHONPATH`. The venv's dependencies
  match v2026.10.0 except `zarrs`, which only the zarr writer tests need.
- **Never push.** The user pushes. Commit only on this branch, and do not add
  `Co-Authored-By: Claude` (repo AGENTS.md).
- **pre-commit cannot run on Perlmutter** (flock errno 524). Run ruff/mypy
  via `uvx` at the versions pinned in `.pre-commit-config.yaml`.
- **Do not pkill with a pattern that matches your own shell command.**
  `pkill -f "compute_stats.py partial"` killed the calling bash (exit 144).
  Use the `[c]ompute_stats` bracket trick.
- **srun into the user's second allocation** requires dropping the inherited
  `SLURM_*` variables. Use `scripts/srun-other.sh JOBID ...`.
- **The session itself runs on an allocation node** (nid008593, job
  60022811). When the allocation ends, the session and every background step
  die with it. Write handoff notes before then.

## Data locations

| what | where |
|---|---|
| raw run dir (source of truth, CFS) | `/global/cfs/cdirs/e3smdata/simulations/v3.HR.F2010.aigo.PD.y1/run` |
| scratch copy of `eam.h0` (training reads this) | `/pscratch/sd/m/mahf708/hr-data/v3.HR.F2010.aigo.PD.y1/eam.h0` (8-OST stripes) |
| normalization stats | `/pscratch/sd/m/mahf708/hr-data/stats-2026-10-19/train-stride4/` |

Scratch is purged on 8-week atime and is not backed up. CFS is the archive.

## Scripts

| script | what |
|---|---|
| `scripts/env.sh` | environment (source it) |
| `scripts/stage-data.sh` | resumable parallel CFS&rarr;scratch copy, one srun task per node |
| `scripts/compute_stats.py` | streaming exact stats (`partial` per node, then `combine`) |
| `scripts/bench_model.py` | isolated fwd+bwd+AdamW step of the real NoiseConditionedSFNO builder on synthetic data; serial or spatial backend; prints one `RESULT {json}` |
| `scripts/sweep.sh` | single-GPU sweep across all GPUs of an allocation |
| `scripts/bench_spatial.sh` | torchrun launcher for the spatial backend (`H W args...`) |
| `scripts/sweep_spatial.sh` | node-parallel queue of single-node spatial runs |
| `scripts/scaling.sh` | sequential multi-node scaling runs |
| `scripts/summarize.py` | JSONL&rarr;table |
| `scripts/build_report.py` | `report/narrative.html` + `logs/*.jsonl` &rarr; `report/index.html` (the artifact) |
| `scripts/srun-other.sh` | srun into another allocation of the user's |

The status artifact is https://claude.ai/artifact/Bc2uTxmhi7Ct1VqXcNnb9u.
To update it, edit `report/narrative.html`, run `python scripts/build_report.py`,
then republish `report/index.html` to the same URL.

## Log

Newest last. Times are PDT on 2026-10-19 unless noted.

- 20:36 Session start on nid008593 inside allocation 60022811 (4 nodes,
  16 A100-80GB, ~4 h). Data: 193 monthly `eam.h0` files of 73.5 GB
  (124 6-hourly snapshots, 143 2D fields, 720x1440 Gaussian, CDF-5),
  0001-01..0017-01. The variable names match the LR hist campaign.
- 20:40 Worktree + branch created at 8935f50b6.
- 20:42 Copy started: ~12 streams, ~5 GB/s aggregate, ~50 min for 14 TB.
- 20:45 **Single GPU OOMs for the baseline**: 2.10 B params (1.70 B
  spectral, 0.40 B pos_embed). The dhconv weight is
  `(modes_lat x C x C)`, so it scales with lmax = 720.
- 20:47 User: also use allocation 60023253 (4 more nodes). User: F2010 =
  atm-lnd only; focus on atm.
- 20:48 **2x2 tiles fit**: 61.1 GB, 5.58 s/step at batch 2.
- 20:50 Single-GPU sweep: everything at 0.25 deg OOMs except
  embed 256 + htf 0.5 (71.5 GB, 5.5 s). 0.5 deg: 35.9 GB / 3.87 s;
  1 deg: 11.5 GB / 0.79 s.
- 20:51 Stats restarted with a 72-variable whitelist and stride 4.
- 20:55 The release lacks XarrayDataConfig `rename` and
  `reference_pressure_name`. Triage of hist-only commits delegated.
- 20:58 2-node DP x (2x2) ran at 18.2 s/step against 5.6 s on one node.
  (nvidia-smi showed 76.5 GB, but that is reserved memory plus NCCL; torch
  peak allocated was 61.1 GB on both.)
- 21:02 **Cause: NCCL over TCP.** torch's pip NCCL has no Slingshot
  plugin. all_reduce busbw on 2 nodes: 2.2 GB/s default, 67.8 GB/s with
  `scripts/nccl-ofi.sh` (NERSC aws-ofi-nccl 2.24.3 plugin under torch's
  NCCL 2.27.5). Every multi-node launch must source it.
- 21:05 Ported 6 hist commits (`notes/hist-commit-triage.md`): 9c264dfae,
  0a511e86f, 91b069b44 (rename/reference_pressure_name), 83d034b34,
  d6307e3b8, a9650919d. Then the fme part of c773f21af, plus the
  pnd_manager timeout (8e41193f0).
- 21:15 The spatial grad sync is now one DDP over the world with a
  sum/data_size comm hook (dabfe58a9). `FME_SPATIAL_GRAD_SYNC=hooks` gives
  the old path. On one node they are identical (4.51 vs 4.55 s); the
  difference is for multi-node runs.
- 21:25 **Live edits of fme/ contaminated running sweeps** (NameError and
  bucket-annotation errors in s4). From here on, benchmarks run from
  `ACE_SRC=$(scripts/snapshot.sh)` and record `fme_src`/`grad_sync` in
  each result.
- 21:35 The 2-node baseline "hang" was the model build:
  torch-harmonics' Legendre precompute runs on one CPU thread per rank
  and took >5 min on nodes busy with the copy and stats jobs (40 s idle).
- 21:50 **netCDF-C reads CDF-5 files with 8 KiB read() calls**
  (strace). That gives 30-105 MB/s per reader against 1.2 GB/s raw. A pread
  reader (`scripts/cdf5.py`) is bit-identical and runs at 650-940 MB/s.
  The stats job was moved onto it.
- 22:05 Loss/corrector spatial fixes committed (88d85d5fb). See
  `notes/spatial-loss-verification.md`; 64/64 parallel tests pass for
  (2,1), (1,2), (2,2), DP2xH2 and serial.
  Run CPU parallel tests with `CUDA_VISIBLE_DEVICES=` or 6 tests fail
  on gloo all_to_all.
- 22:15 `engine: cdf5` for XarrayDataConfig committed (b9cb5a08c,
  `notes/cdf5-engine.md`). The loader sample went from 47-105 to ~750 MB/s
  (median).
- 22:20 s5 frozen-code reruns: **4x1 tiles fastest** (baseline 4.26 s,
  63.3 GB); htf 0.5 is 2.54 s at 44.6 GB; sr0.5+htf0.5 is 2.11 s at 35.2
  GB. bf16 is slower.
- 22:23 Fixed inline inference IC divisibility (world -> data-parallel
  size, 1b1f7391d). Exposed `hard_thresholding_fraction` on
  NoiseConditionedSFNO (3c67da976).
- 22:25 1-node end-to-end smoke run launched (alloc 60023253, nid008532,
  4x1, LR stats as a placeholder):
  `$PSCRATCH/hr-runs/smoke-1n-4x1-base`.
- 22:28 **Source data defect:** `eam.h0.0017-01.nc` on CFS has a corrupt
  header (bytes `c1 9e 9a ba`; netCDF4 cannot open it). Moved out of the
  scratch glob to `$PSCRATCH/hr-data/v3.HR.F2010.aigo.PD.y1/quarantine/`.
  It falls in the test period. **Tell the data owner.**
- 22:31 Host OOM kill during stepper build: torch-harmonics' lru_cache keeps
  ~12 GB of fp64 Legendre tables per rank. Fixed in Distributed.get_sht/
  get_isht (b14fe7a4d): a CPU build goes from 21.3 to 9.4 GB RSS.
- 22:39 The full baseline hits CUDA OOM in real training at 4x1 (70.4 GB
  allocated in the first backward pass). **Production model: modes x0.5.**
- 22:40 HR stats done (`train-stride4/`, 74 vars, 39 files).
- 22:55 Smoke runs, a full epoch each: 1 node 0.42 samples/s, 4 nodes 0.40
  samples/s per replica. Validation, inline inference and EMA/epoch
  checkpoints all work. Checkpoint 20 GB, ~30 s. Validation map aggregators
  add 2.3 min: disable them for production.
- 23:04 4-node pilot started (`$PSCRATCH/hr-runs/pilot-4n-htf05`,
  scripts/pilot.sh, resumable). Resume test: smoke-1n restarted from its
  epoch-1 checkpoint and finished epoch 2 (spatial resume works).
- 23:20 Tile-only loading committed (88a70581d): 4x1 latitude band reads
  2.6x faster, with 4x less host/pinned memory and H2D traffic. An
  end-to-end A/B smoke run (`smoke-1n-4x1-htf05-tile`) is running on
  nid200264.
- Multi-node (s6): 4x1 per node plus data parallelism scales near-perfectly
  (4.26 s on 1 node, 4.25 on 2 and 3); tiles across nodes do not (4x2 over 2
  nodes 3.88 s, 4x3 over 3 nodes for modes x0.5 2.14 s vs 2.54 s at 4x1).
- 23:22 Tile-loading A/B, end to end: losses identical to the full-read path
  at every logged step; 0.42 samples/s on both (GPU-bound); node RAM use
  peaked at ~41 GB above idle against ~150 GB in use before.
- 23:25 The pilot is compute-bound (99% GPU util); 0.34-0.37 samples/s per
  replica, the gap to 0.40 being checkpoint saves every 200 steps.
- 23:30 Hist 1-deg model (0.46 B), 2 nodes x 4, local batch 2: TCP NCCL
  0.873 s/step vs OFI 0.782 (+12%, all in backward). Recommend sourcing
  nccl-ofi.sh in the hist sbatch scripts.

### 2026-10-20 (allocation 60032495, nid200429/200448/200484/200508)
- Branch pushed to E3SM-Project/ace (`e3sm/exps/hr-2026.10.0`). The pilot
  reached step 1700 (loss 0.31); its resumable ckpt is at step 1600.
- Communication decomposition (`--fake-transpose`/`--no-grad-sync` in
  bench_model.py; `notes/comm-and-memory.md`): at 4x1 on one node the comm
  is 9% (h05) / 14% (base); 1x4 16%, 2x2 37%; compute splits ideally.
  Across nodes the comm is 35-48% of the step. Data parallelism is free.
- Transpose microbench: NVLink all-to-all ~200 GB/s aggregate (4 GPUs), 66
  (2 GPUs), Slingshot ~31 per GPU; ~40% of an intra-node transpose is
  torch-harmonics' copies.
- Memory: everything param-sized is replicated per tile. Committed:
  SHT dedup + gradient_as_bucket_view (3fb5c1d7d, -11 GB);
  `optimization.shard_optimizer_state` (45030b9be, ZeRO over the spatial
  group, -12.6 GB). The full-harmonics model now trains end to end at 4x1
  (0.245 samples/s, nvidia-smi 66.7 GB, 33.7 GB checkpoints, ~60 s per save).
- Prototype `--shard-params` (local l-slices of spectral weights + pos_embed
  tile): peak 35 GB, 4% faster. A background agent is implementing it in fme
  (`notes/weight-sharding.md` when done).
- Gotcha (again): a waiter `until ! pgrep -f "<script>"` matches its OWN
  command line and never exits. Wait on log contents or use
  run_in_background on the script itself.
- Random rendezvous ports collided for concurrent multi-node runs; use fixed,
  distinct ports.

### 2026-10-20, 40 GB allocation 60038940
- 40 GB A100s: full harmonics at 4x1 OOM. Modes x0.5 at 4x1 (both
  sharding flags) fit at 39.6 GB, or 38.0 with expandable_segments. Full
  harmonics at 8x1 over 2 nodes fit at 28.6 GB (notes §12b).
- **EMA leak found and fixed (37a2b9a44).** EMATracker.restore() never
  released the parameter copy taken by store(), so after the first epoch-end
  validation a full parameter-sized copy stayed on every GPU, and the next
  store() briefly held two. The production test on 40 GB stepped GPU memory
  up 39.2 -> 40.4 GB at the epoch 1 -> 2 transition and peaked at
  40,439 MiB of 40,960 in epoch 2's validation. This is probably the
  "collapse at the epoch 1 -> 2 transition" the user saw. It is also in the
  hist branch and in upstream ai2cm/ace main.
- Gotcha: `scancel <job>.<step>` on steps launched with `srun --overlap`
  from this session found no matching step (squeue did not list them), and
  the training processes kept running and holding the GPUs. Check with
  nvidia-smi / pgrep and kill with `pkill -f "[f]me.ace.train"` per node.
