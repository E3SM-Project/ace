# Spatial parallelism: where the time and the memory go (2026-10-20)

Allocation 60032495 (4 A100-80GB nodes). Isolated fwd+bwd+AdamW of the real
NoiseConditionedSFNO builder at 720x1440 (`scripts/bench_model.py`), batch 2
(one sample x n_ensemble 2) per spatial group, gradient checkpointing level 3,
code frozen at `hr-bench-snap/3fb5c1d7d`. "h05" = embed 384 with
`hard_thresholding_fraction: 0.5` (1.25 B params). "base" = embed 384 with all
modes (2.10 B). "e256h" = embed 256, 0.5 modes (fits on one GPU: 5.51 s/step).

## 1. Communication pattern

torch-harmonics' `DistributedRealSHT`/`DistributedInverseRealSHT` move data
with all-to-all transposes:

- H split (polar): 2 per transform (c-split/h-local before the Legendre
  contraction, then l-split/c-local after it);
- W split (azimuth): 2 more per transform (before and after the FFT).

Each SFNO block does an SHT and an iSHT. With gradient checkpointing, forward
transposes run again in backward, and every transpose has a transpose in its
backward. So a 4x1 step does ~96 polar transposes, ~0.4 GB each per rank for
h05 (~0.8 GB for base). Each transpose is `split -> .contiguous()` per chunk,
then `all_to_all(list)`, then `torch.cat`: two full copies plus the exchange.

The gradient sync is one DDP all-reduce over the world of the replicated
parameters (sum over tiles, mean over data replicas; commit dabfe58a9).

## 2. Decomposition method

Per layout, four runs:

| run | what it contains |
|---|---|
| `compute` | `--fake-transpose --no-grad-sync`: every transpose replaced by a free reshape to the same local shape (wrong numbers, identical compute), no DDP |
| `real-nosync` | real transposes, no DDP |
| `fake` | fake transposes + DDP |
| `real` | production path |

From these: transposes = real-nosync - compute; grad sync = real - real-nosync.

## 3. One node (4 GPUs, NVLink)

| model | tiles | compute | + transposes | + grad sync | step | comm, % of compute |
|---|---|---|---|---|---|---|
| e256h | 4x1 | 1.35 | +0.12 | +0.03 | 1.50 | 11% |
| e256h | 1x4 | 1.39 | +0.24 | | | 17% (no sync) |
| e256h | 2x2 | 1.37 | +0.58 | | | 42% (no sync) |
| h05 | 4x1 | 2.32 | +0.16 | +0.05 | 2.54 | 9% |
| h05 | 1x4 | 2.39 | +0.35 | +0.03 | 2.76 | 16% |
| h05 | 2x2 | 2.39 | +0.85 | +0.04 | 3.28 | 37% |
| base | 4x1 | 3.74 | (0.51 combined) | | 4.25 | 14% |

- The split compute is ideal: e256h computes in 1.35 s at 4x1 against
  5.51/4 = 1.38 s on one GPU, and every layout computes within 3% of the
  others.
- All of the layout difference is communication. 2x2 pays both transpose
  families at smaller message sizes: 5x the 4x1 transpose cost.
- **4x1 (latitude split) is the right layout**, at 9-14% overhead.

## 4. Memory, base model at 4x1 (GB per GPU, cumulative)

| stage | default | + `gradient_as_bucket_view` | + ZeRO (Adam over 4) | both |
|---|---|---|---|---|
| params + SHT tables | 14.4 | 14.4 | 14.4 | 14.4 |
| + DDP buckets | 22.8 | 22.8 | 22.8 | 22.8 |
| + EMA | 31.2 | 31.2 | 31.2 | 31.2 |
| after first backward | 40.1 | 31.6 | 40.1 | 31.6 |
| after first optimizer step | 48.5 | 48.5 | 36.5 | 36.5 |
| steady-state peak | **71.7** | 71.7 | **60.0** | 60.0 |

Breakdown:

- parameters 8.4 GB (2.1 B fp32);
- SHT Legendre tables 6.0 GB (4 x 1.5 GB, two of them duplicates; commit
  3fb5c1d7d shares them, -3 GB);
- DDP buckets 8.4, grads 8.4 (aliased into the buckets with bucket view;
  3fb5c1d7d), EMA 8.4, Adam 16.8;
- activations about 15 GB.

Real training adds the stepper, loss SHT and data (+~10 GB), which is how the
70.4 GB OOM in the first backward happened.

**Everything parameter-sized is replicated on all four tiles.** The spectral
weights (1.70 B, shape (C, L, C) per block) are sliced to the local l-range in
forward, and pos_embed (0.40 B) to the local tile, so 3/4 of each copy is
never used on a given rank.

## 5. Prototype: shard the l-dependent weights and pos_embed

`--shard-params` keeps only the local l-slice of every spectral weight and the
local tile of pos_embed as parameters. Those gradients are already complete
on their rank, so only the replicated rest (7 M params) is all-reduced.

| base model, 4x1, with EMA | params/GPU | peak GB | s/step |
|---|---|---|---|
| replicated (today) | 2.10 B | 71.7 | 4.28 |
| bucket view + ZeRO | 2.10 B | 57.0 | 4.23 |
| **sharded** | 0.53 B | **35.2** | **4.10** |
| sharded + ZeRO | 0.53 B | 32.2 | 4.09 |
| sharded + ZeRO, embed 512 | 0.90 B | 43.8 | 5.97 |

Sharding halves peak memory, is 4% faster (it removes 99.7% of the
gradient-sync volume), and makes the full-harmonics model and even a wider one
fit. **Reducing harmonics is not necessary; it was compensating for
replication.**

What a real implementation in fme needs:
- sharded parameter creation;
- a gather of the sharded parameters on every checkpoint save (a collective on
  all ranks before the root writes) and a scatter on load, so checkpoints stay
  identical to serial;
- EMA on shards;
- DDP over the data group only for sharded parameters, plus the world sum for
  the replicated ones;
- uneven l-splits handled.

## 6. Cross node (h05, OFI plugin)

| layout | GPUs | compute | step | comm, % of step |
|---|---|---|---|---|
| 4x1 x DP2 | 8 | | 2.53 | data parallelism is free |
| 4x1 x DP4 | 16 | | 2.54 | data parallelism is free |
| 8x1 | 8 | 1.20 | 1.84 | 35% |
| 4x2 | 8 | 1.19 | 2.08 | 43% |
| 2x4 | 8 | | 2.00 | |
| 16x1 | 16 | | 1.34 | |
| 4x4 | 16 | 0.62 | 1.18 | 48% |

Splitting one sample across nodes scales poorly. 16 GPUs process a sample
2.2x faster than 4, but per-GPU throughput is 46% lower than 4x1 with data
parallelism.

## 7. Transpose microbenchmark (`scripts/transpose_bench.py`)

One polar transpose of a (2, 384, 720/P, M) complex64 tensor:

| group | MB/rank | torch-harmonics | all_to_all_single | NCCL floor | floor GB/s |
|---|---|---|---|---|---|
| 4 GPUs, 1 node | 398 | 2.49 ms | 2.44 | 1.51 | 198 |
| 4 GPUs, 1 node, all modes | 797 | 4.85 | 4.83 | 2.95 | 203 |
| 2 GPUs, 1 node | 796 | 7.90 | 7.88 | 6.03 | 66 |
| 8 GPUs, 2 nodes | 199 | 6.53 | 6.14 | 5.54 | 31 |

- Within a node ~40% of a transpose is copies (split -> contiguous, cat).
  `all_to_all_single` needs the same two layout copies, so it gains nothing.
- One NVLink peer gives 66 GB/s; three peers give ~200 in aggregate, which is
  why 2-GPU tile groups are worse.
- The cross-node floor is the network (~31 GB/s per GPU).

What is left within a node: ~2.5% (copies) plus ~2% (grad sync, which
weight sharding removes). The 4x1 path is close to its floor.

## 8. Tile-group size per node

| config | GPUs/replica | s/step | peak GB | samples/s per node |
|---|---|---|---|---|
| h05 4x1 | 4 | 2.54 | 44 | 0.394 |
| h05 2x1 x 2 | 2 | 5.31 | 72 | 0.377 |
| base sharded+ZeRO 4x1 | 4 | 4.09 | 32 | 0.244 |
| base sharded+ZeRO 2x1 x 2 | 2 | 9.35 | 61 | 0.214 |
| base sharded+ZeRO 1x1 x 4 | 1 | OOM | | |

4x1 wins.

## 9. Real training, full harmonics (commits 3fb5c1d7d + 45030b9be)

`runs/smoke-1n-4x1-full-zero.yaml`: hard_thresholding_fraction 1.0,
`shard_optimizer_state: true`, 4x1 tiles on 1 node. It trains at 0.245
samples/s (~4.1 s/step); the loss fell from 4.19 to 1.08 by step 45.
nvidia-smi peaks at 66.7 GB. The checkpoint is 33.7 GB (weights + Adam + EMA,
saved in the plain format) and takes ~60 s with consolidation.

## 10. Serial vs spatial reproducibility (scripts/repro_check.py)

The same NoiseConditionedSFNO (embed 256, 8 blocks, modes x0.5; it fits on
one GPU) gets the same seed, weights, global input and isotropic noise, run
serially and under 4x1, 1x4 and 2x2 tiles. Loss = spatially reduced
sum(pred * w) / numel; SGD for 3 steps. Max relative error vs serial, worst
rank:

| run | fwd step 0 | outputs steps 1-3 | grads | params |
|---|---|---|---|---|
| fp64 360x720, all 3 layouts | 0 | 1.5e-15 | 4-5e-14 | 3e-14 |
| fp32 720x1440, all 3 layouts | **0 (bitwise)** | ~9e-7 | ~3.5e-6 | ~2.7e-6 |
| fp32, deterministic algorithms, 4x1 | 0 | ~9e-7 | 2.0e-6 | 2.7e-6 |
| serial vs serial, fp32, default kernels | 0 | ~4.5e-6 (loss) | | |

- **The decomposition is exact in exact arithmetic.** In fp64 serial and
  spatial agree to machine precision.
- **In fp32 the forward pass is bitwise identical.** Every op is either
  pointwise per grid cell, or (the SHT) sees the full lat/lon extent after
  the transposes, so it reduces the same operands in the same order.
- **Differences start at the first reduction over the grid.** That is the
  loss scalar, and the gradient of every replicated weight (a sum over all
  grid cells: one GEMM reduction serially, per-tile partial sums plus an
  all-reduce spatially). A different summation order rounds differently by
  ~1e-6 relative in fp32, the same size as serial's own run-to-run GPU
  nondeterminism.
- **Bitwise identity would need serial to sum in the same tile-partitioned
  order** (or fp64 accumulation). Deterministic mode does not help, because
  the orders really differ. fp64 agreement is the right correctness check.

The fp64 run needed a test-only copy of the code
(`hr-bench-snap/45030b9be-f64`): `SpectralConvS2`, the conditional layer
and `fme/sht_fix.py` hard-cast the spectral path to float32, so the shipped
model cannot run in float64.

## 11. Parameter sharding, GPU validation (commit 02d312cfa)

`stepper.step.config.builder.config.shard_spatial_parameters=true`, full
harmonics, 4x1 tiles, one node, real `fme.ace.train`, same smoke config as
the unsharded run (`runs/smoke-1n-4x1-full-zero.yaml`, which also sets
`shard_optimizer_state: true`):

| | unsharded | sharded |
|---|---|---|
| params per GPU | 2.10 B | 0.53 B |
| nvidia-smi peak | 66.7 GB | 45.5 GB |
| samples/s | 0.245 | 0.255 |
| loss at steps 5/10/15/20/25 | 4.1949/2.6561/2.0228/1.6589/1.4734 | identical |
| loss at step 115 | 0.6954 | 0.6957 |
| epoch inference error | 0.51337 | 0.51330 |

Checkpoints:
- The epoch checkpoints have the same 681 tensors and shapes.
- Weights differ by at most 2.7% relative (in a conditioning-norm weight
  initialized at zero) after 115 Adam steps; see the noise-floor comparison
  below.
- The sharded run resumed into epoch 2 (step 200, loss 0.5433).
- An **unsharded** run resumed from the **sharded** checkpoint and got the
  same loss at step 200 (0.5433), so the checkpoints are interchangeable.

The CPU parallel suite (102 tests, incl. 20 for sharding) passes at (2,1),
(1,2), (2,2), DP2x(2,1) and serial.

Noise floor: a second, identical unsharded run (`smoke-1n-4x1-full-zero-2`).
Epoch checkpoints after 115 steps:

| pair | global rel. L2 diff of weights | max per-tensor rel. diff |
|---|---|---|
| unsharded 1 vs unsharded 2 | 2.47e-5 | 1.2% (norm0.W_scale_2d) |
| unsharded 1 vs sharded | 2.69e-5 | 2.7% (norm0.W_bias_2d) |
| unsharded 2 vs sharded | 2.70e-5 | 2.8% (norm0.W_bias_2d) |

Sharding is indistinguishable from run-to-run GPU nondeterminism. The
per-tensor maxima are in conditioning-norm weights initialized at zero,
where a tiny absolute difference is a large relative one.

## 12. Do 40 GB A100s suffice? (2026-10-20, last minutes of 60032495)

An attempted 37 GB allocator cap (`scripts/memcap/sitecustomize.py`) did NOT
apply: the ranks' PYTHONPATH no longer had the memcap dir. Uncapped
nvidia-smi usage with both sharding flags:
- full harmonics, 4x1: 45.4 GB per GPU;
- modes x0.5, 4x1: 38.4-39.4 GB per GPU (0.44 samples/s);
- full harmonics, 8x1 over 2 nodes: runs (0.32 samples/s per replica),
  memory not read.

nvidia-smi includes allocator cache, so the true need may be lower, but this
is untested. To settle it, run the smoke config on real 40 GB nodes
(`-C gpu&a100`), or fix the cap by adding the memcap dir to PYTHONPATH inside
scripts/env.sh.

### 12b. Measured on real 40 GB A100s (allocation 60038940, 2026-10-20)

Smoke config (`runs/smoke-1n-4x1-full-zero.yaml`), `shard_spatial_parameters`
+ `shard_optimizer_state`, one full epoch each (train, validation, inline
inference, best/EMA/epoch checkpoints). nvidia-smi sampled every 3 s, of
40,960 MiB:

| setup | outcome | max / p99 MiB | samples/s per replica | samples/s per GPU |
|---|---|---|---|---|
| full harmonics, 4x1, 1 node | CUDA OOM in the first steps (tried to allocate 1.48 GiB) | 38,989 | - | - |
| modes x0.5, 4x1, 1 node | full epoch OK | 39,559 / 39,325 | 0.387 | 0.097 |
| modes x0.5, 4x1, + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | full epoch OK | 38,031 / 38,023 | 0.387 | 0.097 |
| full harmonics, 8x1 over 2 nodes | full epoch OK | 28,591 / 28,449 | 0.310 | 0.039 |

For comparison on 80 GB: full harmonics 4x1 at 0.255 per replica (0.064 per
GPU); modes x0.5 4x1 at 0.42-0.44 per replica (~0.105 per GPU).

- **40 GB, modes x0.5:** fits on one node per replica, with
  expandable_segments for ~2.9 GB of headroom (1.4 GB without).
- **40 GB, full harmonics:** needs 8 GPUs (2 nodes) per replica. It fits
  comfortably (28.6 GB) but costs ~39% per GPU against 80 GB nodes, because
  the tiles are split across nodes.
- The smoke config uses short validation/inference windows. Production adds
  a 360-step inline inference (forward only, lower memory than training) and
  a 1-year validation, which doesn't change the per-step peak.
