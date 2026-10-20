# `engine: cdf5` for XarrayDataConfig

2026-10-19. Uncommitted. Developed and tested in a scratch copy (`/pscratch/sd/m/mahf708/hr-agent-cdf5`),
then copied into the worktree on top of HEAD `dabfe58a9`.

## Why

The EAM h0 files are netCDF classic CDF-5: 124 records x 143 `(time, lat, lon)` float32 fields at
720x1440, 593 MB per record. netCDF-C 4.9.3 (netCDF4-python, xarray's `netcdf4` engine) reads them with
8 KiB `read()` calls, which gives 30-105 MB/s per reader on Lustre. In a classic file one record of one
variable is a contiguous slab at a known offset, so it can be read with a single `pread`.

## Design

- `fme/core/dataset/cdf5.py` (new):
  - Header parser for CDF-1, CDF-2 and CDF-5. The only differences are the 32- or 64-bit counts and
    offsets. Record padding is recomputed rather than read from `vsize`, which CDF-1/2 clip at 2^32-1.
    The single-record-variable case uses unpadded records.
  - `_CDF5Array(BackendArray)` declares `IndexingSupport.OUTER`. For a record variable it makes one
    `preadv` per selected record, straight into the output buffer when the rest of the key takes the
    whole slab. Otherwise it reads only the rows [lo, hi) of the first non-record axis that the key
    touches and applies the rest of the key in numpy. A non-record variable is one `preadv`. Data is
    byte-swapped in place to native order.
  - `_CDF5File` keeps the parsed header and a descriptor opened lazily per process. A pid check
    reopens the file after a fork, and pickling drops the descriptor. Reads use `preadv`, so threads
    share no file offset. The descriptor is closed by `ds.close()` (via `set_close`) or on GC.
  - `_CDF5DataStore(AbstractDataStore)` and `CDF5BackendEntrypoint` hand the raw variables to
    xarray's `StoreBackendEntrypoint`. CF decoding is xarray's own code: `mask_and_scale`, cftime
    decoding and char-array concatenation run the same path as `netcdf4`. Attribute types copy
    netCDF4-python's `getncattr`: a numpy scalar for one element, an array otherwise, `str` with NULs
    stripped, and `bytes` for a char `_FillValue`.
  - `get_xarray_engine("cdf5")` returns the entrypoint class. Other names pass through unchanged.
- `fme/core/dataset/xarray.py`:
  - `engine: Literal["netcdf4", "zarr", "cdf5"]`, with a docstring entry.
  - `__post_init__` rejects `cdf5` with a remote (`scheme://`) `data_path`.
  - `_open_xr_dataset` maps `engine` through `get_xarray_engine`. That covers `_get_raw_times*`,
    `_open_file` (the per-sample hot path), the time-invariant load and the variable grouping. The
    one direct `xr.open_dataset` in `XarrayDataset.__init__` is mapped explicitly.
- `fme/core/dataset/merged.py`: maps the engine in its `xr.open_dataset` (available-variable
  inference).
- Not changed: the IC configs (`InitialConditionConfig`, `ComponentInitialConditionConfig`) still
  accept only `netcdf4`/`zarr`.

## Tests

`fme/core/dataset/test_cdf5.py` (37 tests). The test files are written by netCDF4-python with odd dim
sizes, so the 1- and 2-byte slabs need padding.
- `test_open_matches_netcdf4` (CDF-5, CDF-2, CDF-1 x mask_and_scale x decode_times): checks
  `assert_identical` against `engine="netcdf4"`. It also checks that each variable's dtype, the
  encoding dtype, every attribute's Python type and `unlimited_dims` match. The fixture has record and
  static variables in f4/f8/i1/i2/i4/S1, plus u1/u2/i8/u8 (CDF-5 only). It also has a scalar, a
  `_FillValue`, a packed i2 with `scale_factor`/`add_offset`/`missing_value`, a noleap time with
  bounds, char `date_written` and global attributes.
- `test_open_matches_netcdf4_with_no_records` (numrecs = 0).
- `test_isel_matches_netcdf4`, 15 patterns: int, negative int, slices with step and negative step,
  empty slices, sorted and unsorted int arrays, and mixed time/lat/lon/level keys. Each is checked per
  variable and on the whole dataset.
- `test_single_record_variable_has_unpadded_records` (i1/i2/f4). A mutation that always pads fails
  the i1 and i2 cases.
- `test_header_larger_than_first_read`: the header read grows when the first read is too small.
- `test_rejects_non_classic_file`, `test_guess_can_open`, `test_get_xarray_engine`.
- `test_lazy_dataset_survives_pickle`.
- `test_forked_child_reads_with_its_own_descriptor`. A mutation that drops the pid check fails it.

`fme/core/dataset/test_xarray.py`:
- `_get_data` and `get_mock_monthly_netcdfs` take an `nc_format` argument (default `NETCDF4`,
  written with `engine="netcdf4"`).
- `test_cdf5_engine_samples_match_netcdf4_engine[mask_and_scale]`: builds an XarrayDataset over
  CDF-5 monthly mock files with each engine. It compares `properties` and samples 0, last, and one
  that spans two files, bit for bit with NaNs. It uses `isel={"sample": 1}`, which exercises the
  row-range read. A spy asserts the cdf5 backend was actually used.
- `test_cdf5_engine_rejects_remote_paths`.

Suite: `fme/core/dataset` + `fme/ace/data_loading` gives 661 passed, 8 failed, 6 skipped. Baseline
(the worktree without this change) gives 621 passed and the same 8 failures. All 8 are zarr tests that
need `zarrs`, which the shared venv lacks (`BadConfigError: Pipeline class 'zarrs.ZarrsCodecPipeline'
not found`).

Lint: ruff 0.8.1 check/format and mypy 1.15.0 (`--ignore-missing-imports --check-untyped-defs`) are
clean on the five files. `uvx` is not installed on this node, so the pinned versions were installed
with pip into a throwaway venv in the session scratchpad.

## Benchmark

`scripts/bench_cdf5_engine.py`, run on nid008593 (allocation node, CPU only). It reads the 55
training variables (on-disk names, `mask_and_scale=False`).
- A: open one file, then `.isel(time=slice(i, i+2)).compute()`. That is 456 MB per trial, from
  `0008-01.nc`.
- B: a real `XarrayDataset` (rename + `reference_pressure_name`, `n_timesteps=2`) over
  `0009-0[1-3].nc`, timing `dataset[idx]`.

Each trial reads records that no earlier trial in that run read, and the engines alternate. Two runs
were made (3 and 5 trials).

| path | netcdf4 MB/s | cdf5 MB/s | speedup (totals) |
|---|---|---|---|
| A: open + isel 2 records, 55 vars | 71-105 (totals 82, 121*) | 575-974 (totals 684, 885) | 7-8x |
| B: `XarrayDataset[idx]` | 47-105 (totals 62, 113*) | 124**-1030 (totals 257, 805) | 4-7x |
| B warm repeat (page cache; CPU floor) | 0.34 s / sample | 0.16 s / sample | |

\* Run 2's totals include cache hits. A trial 3 read records 56-57, which run 1's cdf5 trial had
already read (1339 MB/s), and B trial 4 ran at 187 MB/s. The median netcdf4 rate is about 100 MB/s.
\*\* One cold first sample in run 1 (3.7 s). The other nine cdf5 samples took 0.44-1.0 s. The median
is about 750 MB/s.

The time-axis reads at dataset build are 124 small preads per file, and they are not slower: B build
took 0.09-0.13 s for cdf5 vs 0.13-0.56 s for netcdf4. Other jobs reading the same files cannot be
seen or excluded.

Commands:

```bash
source configs/experiments/2026-10-19-e3sm-hr-ace2s/scripts/env.sh
# tests
FME_FORCE_CPU=1 python -m pytest -q fme/core/dataset/test_cdf5.py \
  fme/core/dataset/test_xarray.py -k cdf5
FME_FORCE_CPU=1 python -m pytest -q -n 8 fme/core/dataset fme/ace/data_loading
# benchmark
python configs/experiments/2026-10-19-e3sm-hr-ace2s/scripts/bench_cdf5_engine.py
```

To use it, set `engine: cdf5` on each `dataset:` block in `config-train-hr.yaml`.

## Caveats

- Local files only. A `gs://` etc. `data_path` is rejected in `__post_init__`.
- netCDF-4/HDF5 files raise `ValueError("Not a netCDF classic ...")` on open.
- Streaming files (numrecs = 0xFFFFFFFF) are rejected.
- A file truncated mid-variable (NOFILL writers) raises `OSError` on that read, where netCDF-C
  might return fill values.
- Indexing only touches the bytes it needs on the time axis and the first non-record axis. A
  selection on lon alone still reads full (lat, lon) slabs, which is the same I/O as reading the
  whole slab.
