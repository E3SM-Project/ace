# Scaling-study plan (for one large allocation, e.g. 80 nodes)

Written 2026-10-21. The matrix is `sweeps/scaling-matrix.tsv` (66 rows); each
row runs with `scripts/scaling_run.sh H W BATCH NODELIST RUN PORT` and is
read with `scripts/step_time.py RUN`. Model: production config, full
harmonics, both sharding flags, expandable_segments.

Definitions: GPUs = H x W x DP; BATCH is `train_loader.batch_size` (global);
DP = GPUs / (H x W); local batch = BATCH / DP. Tile groups larger than 4
GPUs span nodes.

## What is already known (40 GB)

| layout | nodes | s/step | peak GiB | note |
|---|---|---|---|---|
| 4x1 | 1 | 5.00 | 35.6 | production; DP flat to 4 nodes |
| 8x1 | 2 | 3.20 | 20.8 | 78% parallel efficiency vs 4x1 |
| 4x2 | 2 | 3.80 | 22.3 | |
| 2x2, 1x4, 2x1 (x1 or x2 replicas) | 1 | OOM | 36.4+ | spectral-weight sharding is over H only |

Memory columns: "yes" measured or implied by a measured layout with the
same H and fewer GPUs per tile; "try" untested (local batch 2); "yes?"
expected on 80 GB from the h05/base runs, not measured at full harmonics;
"no" measured OOM or follows from one.

## Rules

- One Slurm allocation for the whole study: a run cannot span two jobs
  (Slingshot VNI_NOT_FOUND). Request one card type (`-C "gpu&hbm40g"` or
  `"gpu&hbm80g"`); mixing them paces runs by the slower card.
- Training only: no inline inference, 1-week validation, checkpoints off.
  `scaling_run.sh` cuts each run at 25 min (~7 min startup, then >= 50
  steps). Step times are medians of 5-step intervals after step 15.
- Pack runs: start independent rows at the same time on disjoint nodes,
  each with its own PORT.
- Run the C-4x1 points twice (the 1-4 node points were flat to +-0.2 s).
- H must divide 720 latitudes and W 1440 longitudes; H in 1,2,4,8,16 and W
  in 1,2,4 keep every tile equal.

## A. Cost of one sample vs tiles (one replica, BATCH 1)

Strong scaling of the spatial decomposition: per-sample latency and
communication share. Cross-node tiles start at 8 GPUs.

| H | W | GPUs | nodes | DP | BATCH | local | 40 GB | 80 GB | measured |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 1 | 1 | 1 | 1 | 1 | no | no |  |
| 2 | 1 | 2 | 1 | 1 | 1 | 1 | no | yes? |  |
| 1 | 2 | 2 | 1 | 1 | 1 | 1 | no | yes? |  |
| 4 | 1 | 4 | 1 | 1 | 1 | 1 | yes | yes | 5.00 s (40 GB) |
| 2 | 2 | 4 | 1 | 1 | 1 | 1 | no | yes? |  |
| 1 | 4 | 4 | 1 | 1 | 1 | 1 | no | yes? |  |
| 8 | 1 | 8 | 2 | 1 | 1 | 1 | yes | yes | 3.20 s (40 GB) |
| 4 | 2 | 8 | 2 | 1 | 1 | 1 | yes | yes | 3.80 s (40 GB) |
| 2 | 4 | 8 | 2 | 1 | 1 | 1 | no | yes? |  |
| 16 | 1 | 16 | 4 | 1 | 1 | 1 | yes | yes |  |
| 8 | 2 | 16 | 4 | 1 | 1 | 1 | yes | yes |  |
| 4 | 4 | 16 | 4 | 1 | 1 | 1 | yes | yes |  |
| 16 | 2 | 32 | 8 | 1 | 1 | 1 | yes | yes |  |
| 8 | 4 | 32 | 8 | 1 | 1 | 1 | yes | yes |  |

## B. Fixed GPU budget: spatial vs data parallel (local batch 1)

Throughput (samples/s) at 16, 64 and 256 GPUs. BATCH = DP, so the
optimization differs between rows: this is a throughput comparison.

### 16 GPUs (4 nodes)

| H | W | GPUs | nodes | DP | BATCH | local | 40 GB | 80 GB | measured |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 16 | 4 | 16 | 16 | 1 | no | no |  |
| 2 | 1 | 16 | 4 | 8 | 8 | 1 | no | yes? |  |
| 4 | 1 | 16 | 4 | 4 | 4 | 1 | yes | yes |  |
| 2 | 2 | 16 | 4 | 4 | 4 | 1 | no | yes? |  |
| 1 | 4 | 16 | 4 | 4 | 4 | 1 | no | yes? |  |
| 8 | 1 | 16 | 4 | 2 | 2 | 1 | yes | yes |  |
| 4 | 2 | 16 | 4 | 2 | 2 | 1 | yes | yes |  |
| 16 | 1 | 16 | 4 | 1 | 1 | 1 | yes | yes |  |
| 8 | 2 | 16 | 4 | 1 | 1 | 1 | yes | yes |  |
| 4 | 4 | 16 | 4 | 1 | 1 | 1 | yes | yes |  |

### 64 GPUs (16 nodes)

| H | W | GPUs | nodes | DP | BATCH | local | 40 GB | 80 GB | measured |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 64 | 16 | 64 | 64 | 1 | no | no |  |
| 2 | 1 | 64 | 16 | 32 | 32 | 1 | no | yes? |  |
| 4 | 1 | 64 | 16 | 16 | 16 | 1 | yes | yes |  |
| 2 | 2 | 64 | 16 | 16 | 16 | 1 | no | yes? |  |
| 1 | 4 | 64 | 16 | 16 | 16 | 1 | no | yes? |  |
| 8 | 1 | 64 | 16 | 8 | 8 | 1 | yes | yes |  |
| 4 | 2 | 64 | 16 | 8 | 8 | 1 | yes | yes |  |
| 16 | 1 | 64 | 16 | 4 | 4 | 1 | yes | yes |  |
| 8 | 2 | 64 | 16 | 4 | 4 | 1 | yes | yes |  |
| 4 | 4 | 64 | 16 | 4 | 4 | 1 | yes | yes |  |

### 256 GPUs (64 nodes)

| H | W | GPUs | nodes | DP | BATCH | local | 40 GB | 80 GB | measured |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 256 | 64 | 256 | 256 | 1 | no | no |  |
| 2 | 1 | 256 | 64 | 128 | 128 | 1 | no | yes? |  |
| 4 | 1 | 256 | 64 | 64 | 64 | 1 | yes | yes |  |
| 2 | 2 | 256 | 64 | 64 | 64 | 1 | no | yes? |  |
| 1 | 4 | 256 | 64 | 64 | 64 | 1 | no | yes? |  |
| 8 | 1 | 256 | 64 | 32 | 32 | 1 | yes | yes |  |
| 4 | 2 | 256 | 64 | 32 | 32 | 1 | yes | yes |  |
| 16 | 1 | 256 | 64 | 16 | 16 | 1 | yes | yes |  |
| 8 | 2 | 256 | 64 | 16 | 16 | 1 | yes | yes |  |
| 4 | 4 | 256 | 64 | 16 | 16 | 1 | yes | yes |  |

## C. Data-parallel weak scaling at the production layouts (local batch 1)

Where does the gradient all-reduce stop being free? BATCH grows with the
node count.

| H | W | GPUs | nodes | DP | BATCH | local | 40 GB | 80 GB | measured |
|---|---|---|---|---|---|---|---|---|---|
| 4 | 1 | 4 | 1 | 1 | 1 | 1 | yes | yes |  |
| 4 | 1 | 8 | 2 | 2 | 2 | 1 | yes | yes |  |
| 4 | 1 | 16 | 4 | 4 | 4 | 1 | yes | yes |  |
| 4 | 1 | 32 | 8 | 8 | 8 | 1 | yes | yes |  |
| 4 | 1 | 64 | 16 | 16 | 16 | 1 | yes | yes |  |
| 4 | 1 | 128 | 32 | 32 | 32 | 1 | yes | yes |  |
| 4 | 1 | 256 | 64 | 64 | 64 | 1 | yes | yes |  |
| 4 | 1 | 320 | 80 | 80 | 80 | 1 | yes | yes |  |

| H | W | GPUs | nodes | DP | BATCH | local | 40 GB | 80 GB | measured |
|---|---|---|---|---|---|---|---|---|---|
| 8 | 1 | 8 | 2 | 1 | 1 | 1 | yes | yes |  |
| 8 | 1 | 16 | 4 | 2 | 2 | 1 | yes | yes |  |
| 8 | 1 | 32 | 8 | 4 | 4 | 1 | yes | yes |  |
| 8 | 1 | 64 | 16 | 8 | 8 | 1 | yes | yes |  |
| 8 | 1 | 128 | 32 | 16 | 16 | 1 | yes | yes |  |
| 8 | 1 | 256 | 64 | 32 | 32 | 1 | yes | yes |  |
| 8 | 1 | 320 | 80 | 40 | 40 | 1 | yes | yes |  |

## D. Fixed global batch 16: time to solution at equal optimization

Same BATCH, more GPUs through tiles or local batch. Rows with local batch
2 also test whether 2 samples per replica fit.

| H | W | GPUs | nodes | DP | BATCH | local | 40 GB | 80 GB | measured |
|---|---|---|---|---|---|---|---|---|---|
| 4 | 1 | 64 | 16 | 16 | 16 | 1 | yes | yes |  |
| 8 | 1 | 128 | 32 | 16 | 16 | 1 | yes | yes |  |
| 16 | 1 | 256 | 64 | 16 | 16 | 1 | yes | yes |  |
| 4 | 2 | 128 | 32 | 16 | 16 | 1 | yes | yes |  |
| 4 | 4 | 256 | 64 | 16 | 16 | 1 | yes | yes |  |
| 8 | 1 | 64 | 16 | 8 | 16 | 2 | try | try |  |
| 4 | 1 | 32 | 8 | 8 | 16 | 2 | no | try |  |

## Cost

Sum over rows of nodes x 0.25 h: ~380 node-hours for all 66 rows, ~290
for the 47 rows that fit 40 GB. On 80 nodes, packed, that is ~5-6 h of
wall time. Startup at 64-80 nodes may be slower (model build, data
open); give those rows a 40 min limit if they don't reach step 50.
