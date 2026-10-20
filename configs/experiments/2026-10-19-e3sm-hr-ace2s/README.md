# ACE2S on the E3SMv3 HR F2010 run (0.25 deg)

Training the ACE2S atmosphere emulator (stochastic NoiseConditionedSFNO with
EnsembleLoss) on `v3.HR.F2010.aigo.PD.y1`. That run is E3SMv3 at ne120 with
EAM output remapped online to a 720x1440 Gaussian grid, perpetual-2010
forcing, prescribed SST, atmosphere + land only. This folder holds the
configs, launch scripts, benchmarks and notes. Live status page:
https://claude.ai/artifact/Bc2uTxmhi7Ct1VqXcNnb9u.

- Branch: `e3sm/exps/hr-2026.10.0`, based on upstream tag `v2026.10.0`
  (8935f50b6). See "Code changes" for what sits on top.
- Agents: read `AGENTS.md` (rules and running log) and `notes/HANDOFF.md`.

## Data

| | |
|---|---|
| source | `/global/cfs/cdirs/e3smdata/simulations/v3.HR.F2010.aigo.PD.y1/run/*.eam.h0.YYYY-MM.nc` |
| scratch copy (training reads this) | `/pscratch/sd/m/mahf708/hr-data/v3.HR.F2010.aigo.PD.y1/eam.h0/` (8-OST stripes) |
| files | 193 monthly files, 0001-01 .. 0017-01, 73.5 GB each, 14 TB total |
| layout | netCDF classic CDF-5; 124 6-hourly records/month (`time` from 0001-01-01 00Z, noleap); 143 (time, lat, lon) float32 fields; record = 593 MB |
| grid | 720 Gaussian latitudes x 1440 longitudes (0.25 deg); `data_grid: legendre-gauss` |
| vertical | 8 coarsened layers via dimensionless `ak_k`/`bk_k`, so `reference_pressure_name: P0` |

The channel names are the same as in the LR historical campaign
(`configs/experiments/e3sm_hist_v20260812` on `e3sm/exps/hist-v2026.8.0`).
So is the ACE2S channel set: 43 inputs and 50 outputs, with `PRECT`/`PRECST`
renamed and scaled x1000 to kg/m2/s. F2010 has constant GHGs, so there is no
CO2 forcing.

**Splits** (F2010 has no trend, so the splits only need to be disjoint):
train `0001-02-01 .. 0013-12-31` (January of year 1 is dropped as
initialization shock), validation `0014`, test `0015-01 .. 0017-01`.

**Stats**: `scripts/compute_stats.py` computes centering, full-field scaling
and residual scaling from every 4th training month (39 files, 72 variables),
unweighted with ddof 0. These are the LR campaign's conventions, so the two
stats sets are comparable. Output:
`/pscratch/sd/m/mahf708/hr-data/stats-2026-10-19/train-stride4/`.

## Main results

### 1. It does not fit on one GPU, so spatial parallelism is required

At 720x1440 the LR baseline network (embed 384, 8 blocks, noise 32) has
**2.10 B parameters**:

- 1.70 B in the dhconv spectral weights, which are `(modes_lat x C x C)`
  complex and so scale with lmax = 720;
- 0.40 B in the learned positional embedding (`C x 720 x 1440`).

Isolated fwd+bwd+AdamW (`logs/s1-memory.jsonl`) runs out of memory on an
A100-80GB at batch 1 with gradient checkpointing level 3. Only a smaller model
fits on one GPU: embed 256 with modes x0.5 (71.5 GB, 5.5 s/step), which leaves
no room for EMA. The same network at 1 deg uses 11.5 GB and 0.79 s; at 0.5 deg,
35.9 GB and 3.87 s.

### 2. Spatial layout: split latitude (4x1)

On one node with 4 tiles, baseline network, batch 2 (one sample x n_ensemble 2):

| tiles HxW | peak GB | s/step |
|---|---|---|
| 4x1 | 63.3 | **4.26** |
| 1x4 | 59.6 | 4.51 |
| 2x2 | 61.1 | 5.55 |

Parallel efficiency, on a model that also fits on one GPU (5.51 s there):
4x1 1.50 s (92%), 1x4 1.65 s (83%), 2x2 1.99 s (69%).

Model knobs at 4x1:

| model | params | peak GB | s/step |
|---|---|---|---|
| baseline | 2.10 B | 63.3 | 4.26 |
| modes x0.5 (`hard_thresholding_fraction: 0.5`) | 1.25 B | 44.6 | 2.54 |
| modes x0.5 + `spectral_ratio: 0.5` | 0.62 B | 35.2 | 2.11 |
| `filter_num_groups: 8` | 0.62 B | 39.6 | 3.62 |
| bf16 autocast | 2.10 B | | slower |
| checkpointing 2 (vs 3) | | OOM (2x2) | |

Training memory adds an EMA copy (+4 bytes/param, 8.4 GB for the baseline),
so the baseline at 4x1 is tight. **Recommendation: modes x0.5 at 4x1 tiles,
one data-parallel replica per node.** That trades spectral resolution in
the filters (to l <= 360) for 1.7x speed and comfortable memory. It is a
modeling decision for the user.

### 3. Multi-node scaling needs the NCCL OFI plugin

torch's bundled NCCL in the uv venv has no Slingshot (CXI) network plugin and
falls back to TCP sockets. Fp32 all_reduce of 8.4 GB on 2 nodes measured
2.2 GB/s bus bandwidth (6.9 s) on TCP and 67.8 GB/s (0.23 s) with NERSC's
aws-ofi-nccl plugin. Source `scripts/nccl-ofi.sh` in every multi-node launch;
`scripts/train.sh` and `scripts/bench_spatial.sh` already do. A 2-node
data-parallel baseline step took 18.2 s on TCP against 5.6 s on one node. The
multi-node scaling results are in `logs/s6-scaling.jsonl` and on the
status page.

### 4. Reading the files: netCDF-C does 8 KiB reads

strace shows netCDF-C 4.9.3 reading these CDF-5 files in 8192-byte `read()`
calls: 1018 of them for two 4 MB fields. Per-reader throughput was 30-105 MB/s
against ~1.2 GB/s for a plain sequential read, on both scratch and CFS. The
new `engine: cdf5` (`fme/core/dataset/cdf5.py`) parses the header and reads a
record of a field with one `preadv`. It is bit-identical to netcdf4, and a
loader sample of 55 fields goes from 47-105 MB/s to a median of ~750 MB/s.
`config-train-hr.yaml` uses it. `notes/cdf5-engine.md` has the details.

## Code changes on the branch (from v2026.10.0)

Ported from `e3sm/exps/hist-v2026.8.0` (`notes/hist-commit-triage.md`):
9c264dfae frozen_precipitation_rate prefix, 0a511e86f dataset construction at
production size, 91b069b44 `rename`/`reference_pressure_name`/`combine`,
83d034b34 logging basicConfig, d6307e3b8 time-mean `plot_variables`,
a9650919d cross-rank reductions overwriting aggregator state, and c773f21af
configurable collective timeout (plus the same for the spatial backend).

New, with tests:

1. **Spatial gradient sync** (`dabfe58a9`). One DDP over the world with a
   sum/data-size comm hook replaces the per-parameter blocking hooks.
   `FME_SPATIAL_GRAD_SYNC=hooks` restores the old path.
2. **EnsembleLoss and corrector exact under spatial parallelism**
   (`88d85d5fb`). Before this, CRPS gradients came out H*W x serial; the
   energy score was mis-scaled by ~sqrt(H*W), so the 0.9/0.1 mix effectively
   became 0.9/0.05 at 2x2; the logged loss and best-checkpoint selection used
   rank 0's tile only; and the corrector's global means dropped cross-tile
   gradient terms. `notes/spatial-loss-verification.md` has the measurements.
3. **`engine: cdf5`** for XarrayDataConfig (`b9cb5a08c`).
4. **Inline inference IC check** (`1b1f7391d`). The IC count must divide the
   data-parallel size, not the world size; before, a single-node 4-tile run
   with 1 IC was rejected.
5. **`hard_thresholding_fraction` on the NoiseConditionedSFNO builder**
   (`3c67da976`).

Known gaps in the spatial path, not fixed:

- every tile rank reads the full sample and then slices it (4x redundant I/O);
- no restart checkpoint on SIGTERM under spatial parallelism, so set
  `checkpoint_every_n_batches` to fit the walltime;
- writers and map aggregators keep only the root tile; the power spectrum is
  skipped;
- the SHT Legendre precompute is ~40 s per build at lmax 720, single-threaded
  per rank, and much slower on a busy node.

## How to run

```bash
cd /pscratch/sd/m/mahf708/ace-hr/configs/experiments/2026-10-19-e3sm-hr-ace2s
source scripts/env.sh
# freeze the code you are about to benchmark/train with
export ACE_SRC=$(scripts/snapshot.sh)

# systems test (short windows, LR stats as a placeholder)
python scripts/make_smoke.py smoke-1n --dp 1 --lr-stats
srun -N 1 --ntasks-per-node 1 --gpus-per-node 4 \
  scripts/train.sh 4 1 $PWD/runs/smoke-1n.yaml $PSCRATCH/hr-runs/smoke-1n

# production-shaped run: N nodes, 4x1 tiles, batch_size = N
srun -N 4 --ntasks-per-node 1 --gpus-per-node 4 \
  scripts/train.sh 4 1 $PWD/config-train-hr.yaml $PSCRATCH/hr-runs/<name> \
  train_loader.batch_size=4 validation.loader.batch_size=4
```

Benchmarks: `scripts/bench_model.py` (one fwd+bwd+opt step, serial or
spatial), `scripts/sweep.sh` (single GPU), `scripts/sweep_spatial.sh` (one node),
`scripts/scaling*.sh` (multi-node), `scripts/nccl_bench.py`,
`scripts/read_bench.py`. Results go to `logs/*.jsonl`
(`python scripts/summarize.py logs/<x>.jsonl`), with per-run logs in `logs/runs/`.
