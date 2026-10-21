# ACE2S at 0.25 deg on E3SMv3 HR F2010

Trains the ACE2S atmosphere emulator (stochastic `NoiseConditionedSFNO`,
`EnsembleLoss`) on `v3.HR.F2010.aigo.PD.y1`: E3SMv3 at ne120 with EAM output
on a 720x1440 Gaussian grid, perpetual-2010 forcing, prescribed SST. With full
harmonics the network has 2.1 B parameters and does not fit on one GPU, so it
is trained with spatial parallelism.

## Data

- Source: `/global/cfs/cdirs/e3smdata/simulations/v3.HR.F2010.aigo.PD.y1/run/*.eam.h0.*.nc`,
  one netCDF CDF-5 file per month, 124 6-hourly records each, read with
  `engine: cdf5`. Training reads a copy on `$PSCRATCH` (the `data_path` in the
  config); copy the files back from CFS if scratch was purged, leaving out
  `eam.h0.0017-01.nc`, whose header is corrupt.
- Splits: train 0001-02 .. 0013-12, validation 0014, test 0015-01 .. 0016-12.
- Normalization stats: every 4th training month, at the `global_*_path`
  entries of the config.

## Layout

By default each node runs one 4x1 spatial group (one sample split by latitude
band over its 4 GPUs) and nodes are data-parallel replicas
(`FME_DISTRIBUTED_BACKEND=model`, `H=4`, `W=1`, set by `scripts/train.sh`).
`LAYOUT=HxW` gives each sample H*W GPUs; a group wider than a node spans
consecutive nodes. Parameters are sharded over the tiles
(`shard_spatial_parameters: true`); checkpoints keep the unsharded format and
load in serial and unsharded runs.

- Global batch = number of replicas = 4 N / (H W) (one sample per replica).
  `BATCH` must be a multiple of the replica count. It sets the optimization, so keep the node
  count fixed for a whole run.
- The inline-inference initial conditions are split over the replicas: their
  number must be divisible by the replica count, at most 2 per replica on
  40 GB cards. The config's 8 ICs allow 4 or 8 replicas; 16 need 16 ICs.
- One run cannot span two Slurm jobs, so all nodes come from one allocation.

## Requirements

- A venv with fme's dependencies: `<repo>/.venv`, or set `FME_VENV`.
- NERSC's aws-ofi-nccl plugin (`scripts/nccl-ofi.sh`, sourced by `train.sh`).
  Without it inter-node NCCL falls back to TCP sockets and is ~30x slower.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` on 40 GB cards (the sbatch
  script sets it). Without it the first steps run out of memory.
- `FME_DIST_TIMEOUT_MINUTES=120` (set by `train.sh`) so collectives survive the
  rank skew of inline inference and checkpointing.
- 40 GB A100s: training peaks at ~34.0 GiB allocated per GPU; CUDA and NCCL
  take ~2.5 GB more, so the margin is ~3 GiB. On 80 GB A100s set
  `uncheckpointed_filter_blocks: 8` (EXTRA, see below) for a faster step.

## Submit

```bash
cd configs/experiments/hr-ace2s
RUN=hr-e01 sbatch -N 8 -C "gpu&hbm40g" scripts/sbatch-train-hr.sh   # or gpu&hbm80g
```

For 80 GB cards, or any layout with 8 or more GPUs per sample, add
`EXTRA="stepper.step.config.builder.config.uncheckpointed_filter_blocks=8"`.

To strong-scale at a fixed global batch, keep the replicas fixed and grow the
tile group, e.g. batch 8 as `-N 8` (4x1), `-N 16 LAYOUT=8x1` or
`-N 32 LAYOUT=16x1`. H must divide 720 (8, 16, 24, ...) and be at most 32.
`FME_SHT_POLAR_CHUNKS=4` pipelines the SHT transposes with the Legendre
contraction (bitwise identical; 4-6% faster per step on A100 at 4x1 and 8x1);
`FME_SHT_POLAR_MIN_CHUNK_MB` (default 4) is the smallest per-peer message a
chunk may send.

Knobs: `RUN` (required), `EXPDIR` (default `$PSCRATCH/hr-runs/$RUN`), `LAYOUT`
(default 4x1), `BATCH` (default: the replica count), `CKPT_EVERY` (`checkpoint_every_n_batches`, default 400),
`CONFIG`, and `EXTRA` (space-separated `k=v` overrides). The first launch
copies `fme/` and the config to `$EXPDIR/src`, and every requeue runs that
copy. The job requeues itself 5 min before its walltime and resumes from the
last periodic checkpoint. Slurm logs go to `logs/`.

## Measured throughput (full harmonics, one sample per replica)

| layout per replica | cards | uncheckpointed_filter_blocks | s/step | peak GiB |
|---|---|---|---|---|
| 4x1, 1 node | 40 GB | 0 | 3.80 | 34.0 |
| 4x1, 1 node | 80 GB | 0 | 3.40 | 34.0 |
| 4x1, 1 node | 80 GB | 8 | 2.60 | 47.3 |
| 8x1, 2 nodes | 80 GB | 8 | 2.00 | 24.1 |

4x1 gives the most samples per node-hour. 8x1 is ~1.3x faster per sample
for ~15% more node-hours: use it when nodes are plentiful and the batch is
small (e.g. batch 8 on 16 nodes). Within a node, 1x4 (4.40 s) and 2x2
(5.20 s) are slower than 4x1 (4.00 s, all fp32 before the optimizations),
and 2x1 does not fit even on 80 GB. A larger local batch does not raise
throughput (4x1 on 80 GB: 8.1 s at 2 samples). Data-parallel scaling was
flat to 4 nodes. Per epoch, add validation (~15 min / replicas), inline
inference every second epoch (8-14 min) and ~1 min of epoch-end checkpoints.

## Limitations

- Epoch-end checkpoints with identical content are hard links of one file
  (the trainer always replaces files atomically). Tools that rewrite a
  checkpoint in place must write to a new path.

- Under spatial parallelism no restart checkpoint is written on SIGTERM, so
  a requeue loses up to `CKPT_EVERY` steps (~33 min at 400 steps).
- Gradient accumulation over forward steps is not supported under spatial
  parallelism; keep `n_forward_steps: 1` with `optimize_last_step_only`.
- `hard_thresholding_fraction: 0.5` is faster and smaller but truncates the
  spectral filters to l <= 360; 1.0 is the validated setting.
