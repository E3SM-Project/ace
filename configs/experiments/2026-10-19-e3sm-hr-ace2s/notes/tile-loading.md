# Tile-only data loading under spatial parallelism

2026-10-19. Uncommitted. I developed and tested it in a scratch copy
(`/pscratch/sd/m/mahf708/hr-agent-tile`), then copied it into the worktree on top of
HEAD `b14fe7a4d`. That commit also touches `distributed.py`; the two changes were merged by
hand and the merged result re-tested.

## Problem

Under `FME_DISTRIBUTED_BACKEND=model`, every rank of a spatial group loaded the full global sample in
its DataLoader workers (`XarrayDataset` -> `BatchData`). It then pinned the sample, copied it to the
GPU, and only then sliced its tile (`GriddedData` -> `BatchData.scatter_spatial` ->
`Distributed.scatter_spatial`). With H x W tiles, that is H*W times the disk I/O, host RAM, pinned
memory and H2D traffic of the data the rank keeps.

## Design

Each rank's tile is computed **once, in the main process**, and handed to the dataset as plain
`(slice, slice)` data. The dataset applies it lazily, before any values are loaded. Because the tile
travels with the pickled or forked dataset, the DataLoader workers need no distributed state.
`ModelTorchDistributed`'s worker path knows the rank and world size but not the (h, w) position, and it
is never consulted. Coordinates, masks and `DatasetInfo.img_shape` stay global. As before,
`GriddedData` localizes them with `properties.localize()`.

- `DatasetABC.supports_spatial_tile` / `set_spatial_tile(tile)` (`fme/core/dataset/dataset.py`).
  The default is unsupported, so a dataset that does not implement it keeps returning global fields.
- `XarrayDataset.set_spatial_tile` (`fme/core/dataset/xarray.py`):
  - It validates a non-empty step-1 slice for each of the last two horizontal dims and stores it as
    `_tile_isel` (dim name -> slice).
  - Per sample, `ds.isel(**self.isel, **self._tile_isel)` runs before `load_series_data`, so xarray's
    lazy indexing passes the tile to the backend. netCDF4 reads the hyperslab, and cdf5 reads the
    latitude band (below). The zarr async path gets the tile through `_selection_tuple` (the old
    `_isel_tuple` with the tile on the horizontal dims).
  - `_shape_excluding_time_after_selection` uses the tile, so broadcasts and the NaN fills for
    missing variables come out at local shape.
  - Time-invariant fields are re-read for the tile once, when the tile is set.
  - Static derived `x/y/z` are sliced.
  - `properties` are unchanged (global).
- Wrappers propagate the tile: `XarraySubset`, `XarrayConcat`, `SubsetDataset` (the time_buffer
  path), `MergedXarrayDataset` and `TimePaddedMergedDataset`. A wrapper supports a tile only if all
  of its children do.
- `Distributed.get_spatial_tile(img_shape)` returns this rank's `(h, w)` slices, or `None` without a
  split (`fme/core/distributed/distributed.py`).
- `Distributed.scatter_spatial` is now shape-aware. It slices a tensor whose last two dims are the
  global shape, passes through one that already has the local shape, and **raises** on any other
  shape (the old code sliced anything silently). This keeps `GriddedData`/`InferenceGriddedData`
  unchanged: their `to_device().scatter_spatial()` handles tiled and global loaders alike, and slicing
  never happens twice. When tiles are loaded, the H2D copy carries only the tile.
- Training/validation (`get_gridded_data`, `fme/ace/data_loading/getters.py`):
  `_restrict_to_spatial_tile` sets the tile right after `config.get_dataset`, unless one of these holds:
  - There is no split.
  - The batch modifier is not `spatially_local`.
  - The dataset cannot load a tile.
- Inference (`get_inference_data` -> `InferenceDataset(..., load_spatial_tile=True)`,
  `fme/ace/data_loading/inference.py`): the tile is set before the persistence sample is read. The
  SST-perturbation lat/lon meshgrids are sliced to the tile, which is exact because the perturbations
  are pointwise. The initial condition comes from the same loader, so it is local either way.
- cdf5 (`fme/core/dataset/cdf5.py`): no functional change was needed. `_CDF5Array._getitem` already
  reads only rows [lo, hi) of the leading slab axis with one `preadv` per record, so an H split costs
  a 1/H read. For a W split it reads the band's full rows and cuts the columns in memory. Per-row
  reads of only the needed columns were 4.1x slower (see the measurements), so I added a comment
  saying why and kept it that way.

### Where global data is still loaded (old path kept, by design)

- `AugmentationConfig.rotate_probability > 0`: `RotateModifier` flips the whole sphere, so it is marked
  `spatially_local = False` and the loader stays global. The loader logs this.
- Datasets without tile support (`DummyDataset`, test `MockDataset`) keep the old path.
- The coupled (`fme/coupled`) and downscaling loaders are untouched. They never call
  `set_spatial_tile`.
- Inference initial conditions read from a file (`InitialConditionConfig`) are a separate path and are
  unchanged.

## Files changed (worktree, uncommitted)

| file | change |
|---|---|
| `fme/core/dataset/dataset.py` | `supports_spatial_tile`, `set_spatial_tile` on `DatasetABC` |
| `fme/core/dataset/xarray.py` | tile isel in sample/time-invariant/static-derived/zarr paths; `XarraySubset` propagation |
| `fme/core/dataset/concat.py`, `subset.py`, `merged.py` | propagate the tile |
| `fme/core/dataset/cdf5.py` | comment only |
| `fme/core/dataset/testing.py` | `write_tile_test_files` helper (netCDF4 or CDF-5, two files, odd 9x15 grid, NaNs) |
| `fme/core/distributed/distributed.py` | `get_spatial_tile`; shape-aware `scatter_spatial` |
| `fme/ace/data_loading/getters.py` | `get_spatial_tile`, `_restrict_to_spatial_tile`; inference passes `load_spatial_tile=True` |
| `fme/ace/data_loading/inference.py` | `load_spatial_tile` param; tile + perturbation lat/lon slicing |
| `fme/ace/data_loading/augmentation.py` | `BatchModifierABC.spatially_local` (Null: True, Rotate: False) |
| `fme/ace/data_loading/batch_data.py` | docstring only |
| `fme/core/dataset/test_spatial_tile.py` (new) | serial unit tests |
| `fme/ace/data_loading/test_spatial_tile_loading.py` (new) | parallel tests |

## Tests

- `fme/core/dataset/test_spatial_tile.py` (14 tests, serial):
  - `XarrayDataset` with a tile equals the global item sliced, bit for bit (NaN-equal), for both
    engines and four tiles (lat band, lon band, 2D, uneven). The items include a time-dependent
    field, a time-invariant field, static-derived `x`, a missing variable filled with NaN, and
    `fill_nans`. The samples cover a sample crossing the file boundary and
    `get_sample_by_time_slice`.
  - Invalid tiles are rejected.
  - Concat and merge-no-concat datasets propagate the tile.
  - cdf5 makes exactly one pread of `rows * nlon * 4` bytes per record for a lat band, and also for a
    lat x lon tile.
  - `scatter_spatial` raises on a wrong shape.
- `fme/ace/data_loading/test_spatial_tile_loading.py` (9 tests, `@pytest.mark.parallel`, pass
  serially too). Each test builds the loader twice: as shipped, and with tile loading switched off
  (`Distributed.get_spatial_tile` patched to return None, which is the old read-global-then-slice
  path). The tests assert:
  - Train/validation loader, netcdf4 and cdf5, with (train=False, workers=0, time_buffer=0) and
    (train=True, workers=2, time_buffer=1): every device batch equals the global CPU batch sliced
    with `get_local_slices`, and times match. The raw loader really yields local-shape tensors. With
    cdf5, every field pread covered exactly the rank's latitude rows.
  - Inference loader, both engines, workers 0 and 2, with a greens-function SST perturbation and
    `persistence_names`: the batches and the initial condition are identical to the old path.
  - `rotate_probability=1.0` keeps the global loader, and its results are unchanged.
- Results:
  - Parallel suite, `-m parallel fme` (all parallel tests in the repo, including
    `fme/core/distributed/parallel_tests`), with `--ignore=fme/core/test_writer.py` because that file
    needs `zarrs` to be collected at all. 102 passed and 6 skipped at (2,1)x2, (1,2)x2 and (2,2)x4;
    108 passed at (2,1)x4 (data-parallel 2). The baseline is 93 passed and 6 skipped (99 at (2,1)x4).
    The difference is the 9 new tests.
  - Serial `fme/core/dataset fme/ace/data_loading fme/core/distributed fme/ace/inference`: the same 34
    failures as the unmodified worktree (zarr/zarrs writer and reader tests, a non-local experiment dir
    test). No new failures.
  - ruff check, ruff format --check and mypy (`--check-untyped-defs` on the touched packages) are clean.

## Measurements

The script is `hr-agent-tile/bench/bench_tile_read.py`. It runs on a compute node (nid008593, CPU
only) with one reader and no workers. It reads one timestep of the 55 config fields
(in_names + out_names) through `XarrayDataset` (engine cdf5, rename and reference pressure as in
`config-train-hr.yaml`), and each read uses a record that no earlier trial touched. Results are the
median of 10 trials, with similar numbers on two files (0007-10, 0009-04):

| tile (per rank) | bytes kept | time / timestep | vs global |
|---|---|---|---|
| global (today, per rank) | 228 MB | 0.28 s | 1.00x |
| 1/4 latitude band (4x1) | 57 MB | 0.11 s | **2.55-2.60x** |
| 2x2 tile (1/2 lat, 1/2 lon) | 57 MB | 0.17 s | 1.65-1.69x |
| 1/4 longitude band (1x4) | 57 MB | 0.25 s | 1.09-1.11x |

A training sample is 6 timesteps (`n_forward_steps` 1, plus the IC, plus `time_buffer` 4), so a
4x1 rank goes from ~1.7 s to ~0.65 s of read per sample.

In every tile case the host tensor, the pinned buffer and the H2D copy are 1/4 of before: 57 MB per
timestep instead of 228 MB, or 0.34 GB instead of 1.37 GB per 6-step sample. Disk bytes shrink only
with the H split. The W split still reads full rows. The raw backend without xarray
(`bench/proto_rows.py`) gives:

| raw read | time |
|---|---|
| full fields | 0.16-0.21 s |
| 1/4 lat band | 0.07 s |
| 1/4 lon band from full rows | 0.21 s |
| 1/4 lon band, one pread per row | 0.89 s |
| one pread per row, 16 threads | 0.48 s |

So per-row reads lose badly on Lustre, and the current behavior is right.

The lat-band speedup is below 4x for two reasons:

- About 40 ms per timestep is fixed overhead: xarray indexing over 55 variables, the
  dataset re-open (~14 ms: header parse plus 124 small preads for the `time` record variable), the
  byteswap, and a `torch.cat` copy.
- 260 KB preads run at ~800 MB/s, against ~1 GB/s for 4 MB preads.

## Caveats / follow-ups

- **4x1 (H split) is the layout that benefits from disk.** It was also the fastest model layout in s5.
  2x2 gets 1.65x. 1x4 gets almost nothing from disk, but still gets the 4x memory/H2D cut.
- The zarr path is implemented (`_selection_tuple`) but **untested here**, because the venv lacks
  `zarrs` and every zarr test fails before reaching it.
- `scatter_spatial` now raises on a tensor whose last two dims match neither the global nor the local
  shape. Before, such a tensor was mis-sliced silently. No test or caller hit this.
- Rotation augmentation keeps the old global path. Making it tile-local would need the mirrored tile
  of the partner rank.
- Possible further wins, not done:
  - Cache the per-file open: the cdf5 header and the `time` index cost ~14 ms per sample per file.
  - Drop the single-tensor `torch.cat` copy in `get_sample_by_time_slice`.
  - For W splits, read the lat band per rank and all-to-all on the GPU.
- Running jobs import frozen snapshots and are unaffected until they are relaunched from a new
  snapshot.
