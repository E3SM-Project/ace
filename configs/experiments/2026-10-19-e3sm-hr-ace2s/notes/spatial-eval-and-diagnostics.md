# Spatial parallelism: evaluator, writers and map diagnostics

2026-10-20. CPU only (`FME_FORCE_CPU=1`, `CUDA_VISIBLE_DEVICES=`, gloo). Developed in
a scratch copy of `fme/` and copied into the worktree at the end. Not committed.

Under `FME_DISTRIBUTED_BACKEND=model` every rank holds an (h, w) tile of every
field. Scalar metrics are area-weighted reductions, which `GriddedOperations`
already sums across tiles. Everything that outputs a *field* (files, maps,
images, profiles) was taken from the rank's own tile and written or plotted as if
it were global. This note lists what was wrong, the fixes, and the tests.

## TL;DR

| output | before | after |
|---|---|---|
| `python -m fme.ace.evaluator` under spatial parallelism | **crashed** in `flush_diagnostics` (tile-shaped maps given global coords). With any reference data (`time_mean_reference_data`, `monthly_reference_data`) it crashed earlier, in `record_batch` | identical to serial (file values <= 1e-6, scalars to rtol 1e-5) |
| `python -m fme.ace.inference` (standalone) under spatial parallelism | **crashed** at the first step: the initial condition read from file was global, the forcing was tiled | identical to serial |
| `initial_condition.nc`, `restart.nc` | root's tile only, labelled with the root tile's coords (lat 0-3 of 0-7) | global |
| prediction / monthly / custom (`files:`) / time-coarsened / step-diagnostics files | **every** rank wrote its own tile into the same path at once: corrupt HDF5 or the last writer's tile | written by the root only, from the assembled global fields |
| map images (time-mean gen/bias, ref bias, zonal-mean Hovmoller, video, trend, seasonal, ENSO coefficient, near-zero maps, correction maps, one-step snapshot / mean_map / ensemble mean maps) | the root tile plotted as the whole map (and its vmin/vmax in the caption) | global maps |
| per-epoch / reduced diagnostics netCDF (`save_per_epoch_diagnostics`, evaluator `*_diagnostics.nc`) | crash (tile vs global coords) | global |
| power spectrum (inference and one-step) | silently **omitted** under spatial parallelism | computed with the distributed SHT, matches serial |
| per-variable scalar metrics from `area_weighted_*_dict` (one-step `mean/*`, inference `mean/*` series, annual global means, correction series) | **variables mixed across tiles** when the dict order differs between ranks (see "Dict order") | correct |
| seasonal / trend / ENSO-coefficient scalars, annual with a monthly reference | hang or crash: area-weighted reductions issued on the root only | correct |
| ENSO index, IPO index, annual global means | gathered from every rank, so every sample appeared H*W times (biased CRPS / spread); ENSO/IPO regional weights were global vs tiled data (crash); IPO's regional mean was tile-local | correct |
| scalar metrics already reduced by area-weighted ops (rmse, bias, means, std, gradient magnitude, time-mean rmse/bias) | right, apart from the dict-order bug | unchanged |

## Root causes

1. **The evaluator `flush_diagnostics` failure** is not one bug but the general
   one: sub-aggregators' `get_dataset()` returned tile-local maps, and
   `get_reduced_diagnostics` assigned the global coordinates
   (`dataset_info.horizontal_coordinates`), so xarray raised
   `conflicting sizes for dimension 'lat'`. The first one hit was the zonal
   mean (lat band only); every map dataset had the same problem. The training
   run never saw it because `save_per_epoch_diagnostics: false` and validation
   map images are off.
2. **Writers** had only a data-parallel gather. Per-timestep writers are
   refused for >1 data-parallel rank, but nothing stopped them under pure
   spatial parallelism, where every rank opened and wrote the same files.
   `GriddedData.coords` / `InferenceGriddedData.coords` were the *localized*
   coordinates, so the root's `restart.nc` and `initial_condition.nc` silently
   carried its tile with matching tile coords.
3. **Standalone inference** builds `InferenceGriddedData` with a global
   `PrognosticState` from the IC file, which was never scattered.
4. **Dict order.** The data loader returns variables in the order of
   `list(stepper.all_names)`, a frozenset, which every process hashes
   differently (`PYTHONHASHSEED` differs per torchrun rank). Under spatial
   parallelism each variable's area-weighted mean is a collective, so
   `area_weighted_rmse_dict(truth=target, ...)` etc. paired rank 0's tile of
   one variable with rank 1's tile of another. Found by the evaluator test (the
   initial-condition step means of two variables were swapped halves:
   0.0589 / 0.0155 serial vs 0.0225 / 0.0519 spatial, same total). This also
   applies to **training**: the one-step validation per-variable metrics
   (`val/mean/weighted_rmse/*`, `weighted_bias`, `weighted_grad_mag_percent_diff`)
   and the inline-inference `mean/*` series of the HR runs so far may mix
   variables. The loss, `val/mean/loss`, the time-mean metrics
   (`inference/time_mean_norm/rmse/channel_mean`, which selects
   `best_inference_ckpt.tar`) iterate sorted names and are not affected.
5. Root-only branches (`if not dist.is_root(): return`) that then called an
   area-weighted reduction (a collective under spatial parallelism): seasonal,
   trend, ENSO coefficient, annual reference processing.
6. World gathers (`gather_irregular`) of per-sample series that every tile
   holds a copy of.

## Fixes (all in `fme/`)

Core (`fme/core`):

- `Distributed.gather_spatial_tile(t)`: global field on every rank (all-reduce of
  a zero-padded buffer, exact). The global shape is inferred from the tile sizes
  (one tiny all-reduce), so callers need no grid plumbing. CPU tensors are moved
  to the device for the collective and back.
- `Distributed.gather_spatial_tile_to_root(t)`: global field on the spatial root
  only (`torch.distributed.gather` of padded tiles; only the root allocates a
  global buffer). Used by the writers.
- `Distributed.gather_spatial_lat(t)`: for zonal-mean profiles (lat band,
  identical across a tile row).
- `Distributed.data_parallel_gather_irregular`: `gather_irregular` over the data
  group, i.e. each sample once.
- `Distributed.is_spatial_root`, `Distributed.spatial_size`; backend
  `spatial_size()` / `gather_spatial_to_root()` (`base.py`,
  `model_torch_distributed.py`).
- `GriddedOperations.*_dict` visit variables in sorted order.

Writers / data (`fme/ace`):

- `BatchData.gather_spatial_to_root()` (inverse of `scatter_spatial`).
  `_gather_and_write` (IC/restart) gathers spatially first, then over data
  parallelism among the spatial roots.
- `PairedDataWriter` / `DataWriter`: sub-writers are built on the root only;
  each batch is assembled on the root one variable at a time (only the
  variables some sub-writer saves), moved to CPU, then appended. Step
  diagnostics likewise.
- `GriddedData.coords` / `InferenceGriddedData.coords` return the global coords
  (they are only used to build writers).
- `InferenceGriddedData` scatters an initial condition passed as data.

Aggregators: the state stays tile-local (scalars are computed from tiles, as
before); maps are assembled with `gather_spatial_tile` only where they are
plotted or put in a dataset:

- `time_mean.py` (maps, bias maps, datasets; global reference maps are sliced
  to the tile), `zonal_mean.py`, `video.py` (16 frames at a time),
  `step_diagnostics.py`, `near_zero_fraction.py`, `trend.py`, `seasonal.py`,
  `enso/enso_coefficient.py` (reductions on every rank, maps gathered, root
  plots), `one_step/map.py`, `one_step/snapshot.py`, `one_step/ensemble.py`
  (mean maps).
- `spectrum.py`: the `require_no_spatial_parallelism` guard is gone; the
  distributed SHT's (l, m) blocks are squared, assembled and summed over m. The
  leading dims are flattened into the SHT's channel dim and padded to a multiple
  of H*W (it transposes channels across ranks).
- `annual.py`: references processed on every rank in `record_batch`; samples
  gathered over data parallelism only. `utils.LatLonRegion` weights are the
  rank's tile; `ipo_index` regional means reduce across tiles; ENSO/IPO gather
  over data parallelism only.
- Collective loops over variables iterate in sorted order (zonal mean,
  near-zero, spectrum).

Memory: aggregator maps are gathered one variable at a time (a 0.25 deg map is
4 MB). Writers gather one variable per window to the root only, then move it to
CPU; the root's peak extra device memory is ~2x one variable's window
(B x T x 4 MB), the other ranks' one padded tile. Video gathers 16 frames at a
time but is still impractical at 0.25 deg (it keeps n_steps x 720 x 1440 doubles
per variable on CPU, before and after this change).

## Tests (all fail on the code before the change, pass after)

New, under `torchrun` with `-m parallel`, each comparing to a serial reference
computed in-process with `Distributed.force_non_distributed()`:

- `fme/core/distributed/parallel_tests/test_spatial_gather.py`: the three gathers
  equal the global field (even and uneven 9x18 splits, CPU input, NaN kept), bad
  tiles are rejected, and `area_weighted_mean_dict` with a rank-dependent dict
  order equals serial.
- `fme/ace/aggregator/test_spatial_aggregators.py`: `InferenceEvaluatorAggregator`
  with every map-type metric on (time_mean with reference, zonal_mean, video
  extended, power spectrum, trend, seasonal, near-zero maps, correction maps,
  annual with monthly reference, ENSO index and coefficient; 6 years of monthly
  steps), `InferenceAggregator` (inline/standalone), and `OneStepAggregator`
  (snapshot, mean_map, ensemble mean maps, spectrum). Data dicts are ordered
  differently on odd ranks. Compares every scalar log, every image caption
  (vmin/vmax), every plotted array (by recording `plot_imshow`) and every
  `flush_diagnostics` netCDF file.
- `fme/ace/inference/test_spatial_evaluator.py`: `run_evaluator_from_config`
  (plus-one stepper with a corrector, so step diagnostics exist; prediction,
  monthly, region-subset and time-coarsened files, step diagnostics, video and
  correction maps) and the standalone `fme.ace.inference.main`, each run serially
  and spatially; compares every netCDF file under the experiment dir (data
  files, IC, restart, step diagnostics, reduced diagnostics) and every logged
  metric and image caption.
- Shared helpers: `fme/ace/testing/spatial.py`.

Also checked by hand: `python -m fme.ace.evaluator` serial vs `torchrun -n 4`
at 2x2 on a CPU config (all 15 output files match; max diff 1e-6).

## Results

- Parallel suite (`fme/core/distributed/parallel_tests`,
  `fme/ace/data_loading/test_spatial_tile_loading.py`, and the two new test
  files) at (2,1), (1,2), (2,2): **117 passed on every rank** (`--no-timeout`;
  the node had load ~490 on 128 cores from other jobs).
- New tests on the pre-change code: all fail (crashes in `flush_diagnostics`,
  in `record_batch` with reference data, at the first standalone step, and the
  dict-order test).
- Serial `fme/core fme/ace fme/coupled` (-n 32): every failure was rerun on
  both the changed and the original tree. All remaining failures are
  pre-existing and identical on the original: zarrs missing (33),
  `test_writer.py` import error, GraphCast deps, earth2grid/HEALPix, and
  `test_module_registry` backwards-compat for NoiseConditionedSFNO (new builder
  keys from earlier commits). The rest were the 90 s timeout on the loaded
  node; rerun with `--no-timeout` they pass.
- ruff 0.8.1 (format + check) clean; mypy 1.15 on all of `fme`: no issues.

## GPU validation recipe (4 GPUs, one node)

Config: `configs/experiments/2026-10-19-e3sm-hr-ace2s/eval-spatial-check.yaml`
(best_inference_ckpt of prodtest-4n-emafix, 1 IC at 0015-01-01, 20 steps,
`seed: 0`, prediction + monthly + step-diagnostics files for 6 variables, video
for PS/TS, correction maps). Inside an allocation:

```bash
cd /pscratch/sd/m/mahf708/ace-hr/configs/experiments/2026-10-19-e3sm-hr-ace2s
export ACE_SRC=$(scripts/snapshot.sh); source scripts/env.sh
CFG=$PWD/eval-spatial-check.yaml; OUT=$PSCRATCH/hr-runs/eval-spatial-check
# serial reference, 1 GPU (inference only: ~5 GB of weights, fits)
srun -N 1 -n 1 --gpus-per-task 1 python -m fme.ace.evaluator $CFG \
  --override experiment_dir=$OUT/serial logging.metrics_log_dir=$OUT/serial/metrics
# 4x1 tiles, 4 GPUs (no data parallelism: one IC and per-timestep writers)
srun -N 1 -n 1 --gpus-per-task 4 env FME_DISTRIBUTED_BACKEND=model \
  FME_DISTRIBUTED_H=4 FME_DISTRIBUTED_W=1 \
  torchrun --standalone --nproc-per-node 4 -m fme.ace.evaluator $CFG \
  --override experiment_dir=$OUT/4x1 logging.metrics_log_dir=$OUT/4x1/metrics
python scripts/compare_eval_runs.py $OUT/serial $OUT/4x1
```

`compare_eval_runs.py` compares every netCDF file under both dirs (data files,
IC, restart, step diagnostics, `*_diagnostics.nc` including the per-step
`mean_diagnostics.nc` series) and the scalar metrics in `metrics/metrics.jsonl`,
and prints `MATCH`/`MISMATCH`. Expect fp32-level differences (the distributed
SHT sums in a different order; HANDOFF.md reports the serial and spatial
forward bitwise identical in fp32, so most differences may be 0). Note `metrics.jsonl` only keeps
plain Python numbers: the per-step series are numpy scalars and are dropped by
`DiskMetricLogger` (pre-existing), so they are compared through
`mean_diagnostics.nc`. The serial run needs the whole model on one GPU; if it
does not fit, compare `H=4 W=1` with `H=2 W=1` (2 GPUs) instead, which still
tests the gathers on a different tiling.

## Remaining caveats

- **Per-timestep writers still refuse data parallelism** (pre-existing): with
  H*W < number of GPUs, either disable them or use one IC per spatial group
  layout with no data parallelism. IC/restart and all diagnostics work with
  data parallelism + tiles.
- **Histogram aggregator** (off by default) never reduces across ranks: it
  reports the root's samples, and under spatial parallelism the root's tile.
  Pre-existing for data parallelism too; not fixed.
- `SmoothFloodFill` (NaN fill before the spectrum) works per tile: for
  variables with NaN masks the fill differs near tile edges. ACE2S HR fields
  have no NaNs.
- Restart/IC `stepper_state` is taken from the spatial root. That is correct
  for the current contents (global dry-air mass, per-data-rank random state),
  not for any future per-cell state.
- The fix for the dict-order bug is in the consumers (`*_dict` helpers and the
  aggregator loops). Any new code issuing one collective per variable must
  iterate in a rank-independent order. Past HR training logs of per-variable
  one-step metrics are suspect (see root cause 4).
- `gather_spatial_tile` cannot detect a *global* field passed by mistake on every
  rank when it happens to look like a set of tiles (e.g. 2x1 tiles of an 8x16
  field look like a 16x16 field's tiles); it only rejects inconsistent tiles.
