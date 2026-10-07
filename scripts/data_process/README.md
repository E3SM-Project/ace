# Data processing for full model emulation training

This directory contains scripts for generating various datasets needed for FME training, including the FV3GFS primary, baseline, and stats datasets.

It also contains scripts for generating E3SM training data.

The first step in the process to create intermediate datasets (e.g. `make fv3gfs_AMIP_dataset`) uses argo, and can be run on your Google VM.
See the vcm-workflow-control repo for instructions on how to install and run argo.

The second step, which produces monthly netCDF files locally (e.g. `make fv3gfs_AMIP_monthly_netcdfs`), can be run on cirrascale in an interactive session.
To create an interactive session, run the following command from the `scripts/data_process` directory:

```
beaker session create --budget ai2/atec-climate --image beaker://jeremym/fme-2bc0033e --gpus 0 --mount hostPath:///net/nfs/climate=/net/nfs/climate --mount hostpath://$(pwd)=/full-model --workdir /full-model/scripts/data_process --shared-memory 120GiB
```

Doing so will require that your current working directory is a mountable path (e.g. something in /data).
If you'd like to write to a different directory than /net/nfs/climate, you can mount that path instead.

Once inside the image, you will need to authorize access to GCS by running `gcloud auth application-default login` and following the instructions, including to run `gcloud config set project vcm-ml` afterwards.

You can then produce the monthly netCDFs in a target directory by modifying the `OUTPUT_DIR` or `OUTPUT_DIR_AMIP` variable in the make command below.

```
make fv3gfs_AMIP_monthly_netcdfs RESOLUTION=4deg OUTPUT_DIR_AMIP=/data/shared/2023-12-20-vertically-resolved-4deg-fme-amip-ensemble-dataset
```

The stats dataset creation step (e.g. `make fv3gfs_AMIP_stats_beaker_dataset`) must be run in the fme conda environment (created by `make create_environment` at the top level of this repo), and additionally requires the beaker client is installed ([install instructions](https://beaker-docs.apps.allenai.org/start/install.html)).


For healpix data (both `healpix_ace` and `healpix_dlwp`), you will need to use the annad/dlwp-datapipe image.

You can either run the target with gantry or use the --bare flag, passing your own beaker secrets to the usual session command.

Update `configs/healpix-1deg-8layer-1940-2022.yaml` to point at the latest era5 data on gcs and the current date, i.e., variable_sources `2024-06-20-era5-1deg-8layer-1940-2022.zarr`; data_output_directory: `/climate-default/[DATE]-healpix-era5-dataset`.

If using gantry, be sure to run `make healpix_ace_dataset_gantry` before running `make healpix_dlwp_dataset_gantry`, or update the config to point at an existing hpx-ace dataset.

The output will be written to the `/climate-default` file directory on weka.

Example bare usage: `cd full-model/scripts/data_process && make healpix_ace_dataset`. You may also want to run in the background using nohup: `nohup make healpix_ace_dataset > compute_hpx.log 2>&1 &`.

Example bare session creation (use your own ssh secrets): `beaker session create --name annad/dlwp-ace-datapipe --image beaker://annad/dlwp-datapipe --remote --cluster ai2/phobos-cirrascale --bare --mount src=weka,ref=climate-default,dst=/climate-default  --mount src=weka,ref=climate-default,subpath=annad,dst=/root --workdir=/root --mount src=secret,ref=ssh-key,dst=/secret-files/.ssh/id_ed25519     --mount src=secret,ref=git-config,dst=/secret-files/.gitconfig --budget ai2/atec-climate --shared-memory 120GiB`
## SamudrACE-E3SMv3 initial conditions from E3SM restarts

`create_e3sm_restart_ic.py` turns E3SM restart output into the initial condition
pair that `fme.coupled.inference` expects, for the
[SamudrACE-E3SMv3](https://huggingface.co/allenai/SamudrACE-E3SMv3) checkpoint.
Each restart directory must contain one each of `*.eam.i.*.nc`,
`*.mpaso.rst.*.nc` and `*.mpassi.rst.*.nc`:

```
python create_e3sm_restart_ic.py --config configs/e3smv3-restart-ic.yaml
```

Only the 38 atmosphere and 80 ocean *prognostic* variables are written; the
forcing and time-invariant fields (`LANDFRAC`, `PHIS`, `SOLIN`, `ak_*`, `bk_*`,
`idepth_*`, `mask_*`, ...) come from the forcing dataset published alongside the
checkpoint. Set `stack: true` to get one file pair with one time per restart,
which is what `initial_condition.start_indices.n_initial_conditions` in the
inference config consumes.

### Required remap weights

The ocean map (`map_IcoswISC30E3r5_to_gaussian_180by360_shifted.nc`) and the
target grid are published under
`https://web.lcrc.anl.gov/public/e3sm/inputdata/fme/`. The atmosphere map is
not, because the EAM restart state lives on the spectral-element GLL grid rather
than the `ne30pg2` physics grid the history files use. Build it once:

```bash
# 1. Extract the ACE target grid (shifted gaussian 180x360) from any published
#    map file, so the destination is bit-identical to the training data's.
python -c "
import numpy as np, xarray as xr
m = xr.open_dataset('map_ne30pg2_to_gaussian_180by360_shifted.nc', decode_cf=False)
xr.Dataset({
    'grid_dims': ('grid_rank', m.dst_grid_dims.values.astype('i4')),
    'grid_center_lat': ('grid_size', m.yc_b.values),
    'grid_center_lon': ('grid_size', m.xc_b.values),
    'grid_corner_lat': (('grid_size', 'grid_corners'), m.yv_b.values),
    'grid_corner_lon': (('grid_size', 'grid_corners'), m.xv_b.values),
    'grid_imask': ('grid_size', np.ones(m.sizes['n_b'], dtype='i4')),
    'grid_area': ('grid_size', m.area_b.values),
}).to_netcdf('dst_gaussian_180by360_shifted.scrip.nc')
"

# 2. Fetch the ne30np4 GLL dual grid (48602 cells, matching eam.i's ncol_d).
curl -O https://web.lcrc.anl.gov/public/e3sm/inputdata/share/meshes/homme/ne30np4_pentagons.091226.nc

# 3. Generate the conservative map (takes seconds).
ncremap -s ne30np4_pentagons.091226.nc \
        -g dst_gaussian_180by360_shifted.scrip.nc \
        -m map_ne30np4_to_gaussian_180by360_shifted.nc
```

### Fields E3SM restarts do not carry

Two prognostics have to be derived rather than read, and one has to be
approximated. All three are documented at their implementation site:

- **Ocean velocity.** MPAS-Ocean checkpoints only `normalVelocity` on edges, so
  `reconstruct_cell_velocity` solves an edge-length-weighted least squares
  problem per cell and level. It reproduces MPAS's own
  `surfaceVelocityZonal`/`surfaceVelocityMeridional` to a correlation of 1.0000
  and an RMSE of 5e-4 m/s.
- **`ssh`.** Diagnosed as `sum(layerThickness) - bottomDepth`, which reproduces
  the coupler's `o2x_ox_So_ssh` to round-off, then corrected for the ice-shelf
  draft and the sea ice load (see the comment in `build_ocean_native`).
- **`Tat2m`, `Qat2m`, `Uat10m`, `Vat10m`, and the land tile of `TS`.** EAM
  computes these inside its surface-layer scheme and never checkpoints them, so
  they are approximated from the lowest model level. They are prognostic in the
  emulator but tightly constrained by the rest of the state.

### Every wetmask point must have a value

Initial conditions are handed to the steppers exactly as they are read:
`ComponentInitialConditionConfig` has no `fill_nans` option, and unlike
`XarrayDataConfig` it does no NaN handling. A single missing value *inside* an
ocean wetmask therefore spreads across the globe within one step, and the run
returns NaN everywhere -- in both components, since the ocean state feeds the
atmosphere through the coupler. Two things in this script exist only to prevent
that, and both are on by default:

- `ocean.exclude_ice_shelf_cavities: false`. The E3SMv3 mesh resolves
  sub-ice-shelf cavities and the training wetmasks cover them, so they must be
  kept. Their raw `ssh` reaches -1700 m, but removing the `landIceDraft` load
  already brings them back into range.
- `masks.fill_masked_gaps: true`. The training wetmasks are marginally wider
  than what a given restart's bathymetry supports near shelf breaks. The
  leftover points (~1300 out of 5.2 million for the 1940 historical restart)
  are filled from the layer above, which the nested masks guarantee is wet.

To check a generated file before spending GPU hours on it:

```python
import numpy as np, xarray as xr
ic = xr.open_dataset("out/..._ocean_ic.nc")
forcing = xr.open_dataset("forcing_data/ocean-forcing-1yr.nc")
for name in ic.data_vars:                      # must print nothing
    mask = forcing.get(f"mask_{name.rsplit('_', 1)[-1]}", forcing["mask_2d"])
    bad = int(((mask.values > 0) & ~np.isfinite(ic[name].isel(time=0).values)).sum())
    if bad:
        print(name, bad)
```

### Lining the time coordinate up with the forcing

Inference selects forcing by timestamp, so every initial condition time must
exist in the forcing dataset. The published forcing is one piControl year
(starting `0425-01-03 12:00:00`, ocean steps every 5 days) played back
`n_repeats` times, so a historical restart date such as `1940-01-01` is simply
not in it. Use `time.source: explicit` with one timestamp per restart drawn
from the forcing calendar; `time.source: restart` (the default) is only right
when the forcing actually spans the restart dates.

### Verifying end to end

The checkpoint needs `ZonallyPeriodicBilinearUpsample`, added in #1316, so it
cannot be loaded by an older `fme`. With a current checkout, two coupled steps
on CPU take a few minutes and are enough to confirm an initial condition is
sound:

```bash
FME_FORCE_CPU=1 python -m fme.coupled.inference inference-config.yaml
```

A healthy ocean run holds its NaN fraction at exactly the land fraction (0.3072
for `sst`) at every step; the atmosphere stays at 0.0000. Any step at 1.0000
means a missing value reached the model.
