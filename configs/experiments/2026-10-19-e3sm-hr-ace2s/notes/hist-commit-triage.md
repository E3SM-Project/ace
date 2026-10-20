# HIST -> REL commit triage (fme/)

- HIST = `4a48f9028` (e3sm/exps/hist-v2026.8.0), REL = `8935f50b6` (v2026.10.0).
- Merge base `78ea64f15` (2026-08-10). `git log --no-merges 8935f50b6..4a48f9028 -- fme`
  gives 22 commits and no merges. `git cherry 8935f50b6 4a48f9028` reports **no**
  patch-equivalent commits, so every "in REL?" below was checked by content
  (`git grep` / `git show 8935f50b6:<path>`).
- Cherry-pick testing ran in a throwaway detached worktree on REL (since removed).
  First each commit was picked alone with `-n`, then the port set was applied in
  sequence. No branches were created and nothing was committed or pushed.
- Target: atmosphere-only ACE2S (NoiseConditionedSFNO), raw EAM netCDF via
  XarrayDataset with `rename`, `overwrite.multiply_scalar` and `reference_pressure_name`,
  spatial-parallel (`model`) backend, multi-node torchrun, and Slurm requeue.

## Table

| sha | subject | in REL? (evidence) | atm-relevant? | cherry-pick clean? | recommendation |
|---|---|---|---|---|---|
| 9c264dfae | Restore bare frozen_precipitation_rate name prefix | **No**. In REL, `atmosphere_data.py:42` lists only `["total_frozen_precipitation_rate"]` | **Yes**. The HIST atm config renames `PRECST: frozen_precipitation_rate`. Without the fix, `AtmosphereData.frozen_precipitation_rate` falls back to ICEsfc+... and then to **zeros**, so the energy-flux derived quantities silently lose frozen precip | Clean | **port** (or else rename PRECST to `total_frozen_precipitation_rate` in the config) |
| 0a511e86f | Make dataset construction survive production width and file count | **No**. REL `xarray.py` still has `multiprocessing.Pool` in `_get_raw_times` (l.154-162, `GET_RAW_TIMES_NUM_FILES_PARALLELIZATION_THRESHOLD`) and globs through fsspec in `get_raw_paths` | **Yes, critical**. A fork-after-CUDA/NCCL Pool deadlock trips the NCCL watchdog during DDP setup, and many raw EAM files make the fsspec glob slow | Clean | **port** |
| 91b069b44 | Add rename-aware, decode-aware and combinable variable loading to XarrayDataset | **No**. REL `XarrayDataConfig` has no `rename`, `reference_pressure_name`, `combine`, `mask_and_scale` or `OverwriteConfig.add_scalar` (`git grep reference_pressure_name 8935f50b6` returns nothing) | **Yes, required**. The planned config uses `rename` and `reference_pressure_name`. It also fixes renamed time-invariant variables raising KeyError | Conflicts in `test_xarray.py` alone; **clean once 0a511e86f is applied first** | **port** (after 0a511e86f) |
| 83d034b34 | Force logging.basicConfig so the configured log level takes effect | **No**. REL `logging_utils.py:97-101` calls `basicConfig` without `force=True` | Yes. An early `logging.warning` in a `__post_init__` silences INFO for the whole run (empty out.log) | Clean | **port** |
| 02caa33e5 | Validate that the atmosphere actually forces the ocean in CoupledStepperConfig | No (its error strings are absent from `fme/coupled`) | No (coupled) | Conflict (`coupled/test_stepper.py`) | skip |
| 3d26128ec | Add DataWriterConfig.prediction_names | No (REL `DataWriterConfig` has only `names`) | Inference only. **REL already has an equivalent**: `files: [FileWriterConfig(label, names, ...)]` with `save_prediction_files: false` (different filenames) | Clean | optional; skip unless old inference yamls using `prediction_names` must run |
| c5d39a0fa | Make gradient checkpointing differentiable in the conditional SFNO | **Yes**. Upstreamed as `22625c008` (#1493). All four `checkpoint(..., use_reentrant=False)` calls are in REL `conditional_sfno/{layers,sfnonet}.py`, which NoiseConditionedSFNO builds on | Yes | Conflict (test file only; the code is already there) | **already-there** |
| fd20fe981 | update with full campaign (fme part: empty-`prediction_names` guard + test import tidy) | No | Only together with 3d26128ec; the rest is campaign configs | The whole commit conflicts (configs); the fme-only diff applies cleanly on top of 3d26128ec+91b069b44 | skip (fold into 3d26128ec if that is ported) |
| d6307e3b8 | Let the time-mean metrics choose which channels get a picture (`report_plot`, `plot_variables` on `TimeMeanMetricConfig`) | **No**. REL `time_mean.py` has no `report_plot`/`plot_variables`. REL's `report_plot` hits are the spectrum and histogram toggles (#1208), not time-mean. This is the W&B-quota fix from memory | Yes. It cuts the gen_map/bias_map/ref_bias_map PNGs per channel per epoch. At 0.25° each PNG is larger, so the saving matters more | Clean (both classes' image sites are gated after the pick; `test_time_mean.py` passes) | **port** |
| 5d2a07825 | Deafen the termination handler to its DataLoader workers dying | **Superseded**. REL rewrote shutdown (#1425/#1435/#1438): the abort and the post-abort callbacks (restart checkpoint) run on a dedicated listener thread, the main thread parks at batch boundaries, the listener ignores non-termination wakeups such as SIGCHLD (`_signal_listener.py:131-137`, `test_other_handled_signals_do_not_trigger_the_abort`), and forked workers are disarmed (`test_shutdown_dataloader.py`) | Relevant problem, but the HIST code targets the old `handle_termination_signals` | Conflict (`shutdown.py`, `test_shutdown.py`). The code it edits no longer exists | skip. Validate with a real walltime requeue instead |
| 25b8e6819 | Mark a requeued run preempting so wandb stops calling it failed | **No** (no `mark_preempting` in REL `wandb.py`) | Yes (cosmetic: the W&B run shows `failed` between requeue segments) | **Conflict** in `generics/trainer.py`. Needs `add_post_shutdown_callback` changed to `add_post_abort_callback`, and REL's `test_trainer.py::test_resume_after_interrupted_training_during_epoch[preempt]` unpacks exactly one callback (`(save_on_terminate,) = registered_callbacks`) and fails | optional, needs adaptation. Caution: REL forbids the logging module and collectives in post-abort callbacks, and the network call eats into the 30 s budget |
| 72ef48865 | Absorb the child death with a callable, not SIG_DFL | Superseded (see 5d2a07825) | as 5d2a07825 | Conflict | skip |
| c773f21af | Let the collective timeout cover the inline inference rank skew (`FME_DIST_TIMEOUT_MINUTES`) | **No**. REL `torch_distributed.py:62,68,86` hardcode `timedelta(minutes=30)` | **Yes, and REL is worse under spatial parallel**: the `model` backend initialises through `external/pnd_manager.py:459-477` `dist.init_process_group(...)` with **no timeout**, which is torch 2.10's NCCL default of **10 min** (measured: `default_pg_nccl_timeout=0:10:00`) | Code and test clean; only the campaign sbatch scripts conflict | **port and extend**: also pass `timeout=_process_group_timeout()` in `pnd_manager.py`, or the HR run gets no benefit |
| 021745334 | Refuse an inline coupled inference that overshoots its rollout | No (only the standalone `coupled/inference/evaluator.py:276` has the check) | No (coupled) | Clean | skip |
| ab0af55f5 | Do not duplicate a sea ice thickness name the migration already has | No (REL `corrector/ocean.py:499` still appends unconditionally) | No (ocean) | Clean | skip |
| 9ba06e105 | Scale atmosphere-to-ocean fluxes by the ocean's open-water fraction | No (`open_water_flux_scaling` 0 hits) | No (coupled) | Clean | skip |
| 8f61e6988 | Add an exponent on the ocean-fraction weight of SST prescription (`interpolate_weight_power`) | No (0 hits) | Marginal. It touches `core/prescriber.py`, but the default of 1.0 is a no-op and the motivation is coupled | Clean | skip (revisit for coupling) |
| db6d2f53e | Scale the atmosphere-to-ocean wind stress by a static ocean-area factor | No (`static_flux_scaling` 0 hits) | No (coupled) | Conflict (depends on 9ba06e105) | skip |
| 57dedf55a | Let the coupler build ocean forcings from atmosphere outputs | No (`combined_fluxes` 0 hits) | No (coupled) | Conflict | skip |
| e87618086 | Let a combined ocean forcing include the ocean's own emission | No (`sst_emission_coefficient` 0 hits) | No (coupled) | Conflict | skip |
| a9650919d | Fix cross-rank reductions overwriting aggregator state (val mean_map N_ranks x bug) | **No**. REL still calls in-place `dist.reduce_mean(self._gen_data[name]) / n` (`one_step/map.py:92,96`), `reduce_mean(_mean_spectrum)` without clone (`inference/spectrum.py:75`) and `reduce_sum(self._sum[k])` (`tensor_dict_accumulator.py:60`). Both REL backends `all_reduce` in place. The same fix as `928d5cada`, which is only on e3sm/e4x-w7 | **Yes**. The val ensemble mean_map images scale with the data-parallel rank count. I audited the other REL reduce call sites (video, reduced, near_zero_fraction, trend, zonal_mean, step_diagnostics) and they all reduce fresh tensors | Clean (its tests pass) | **port** |
| 028b1bed5 | Add an additive-profile method to the ocean heat-content corrector | No (`additive_profile` 0 hits) | No (ocean) | Conflict | skip |

## Verification of the port set

I applied, in order, 9c264dfae, 0a511e86f, 91b069b44, 83d034b34, 3d26128ec, the fme part
of fd20fe981, d6307e3b8, the fme part of c773f21af, a9650919d, and 25b8e6819 with the
conflict resolved by hand. Then I ran test_xarray, test_merged, test_atmosphere_data,
test_data_writer, test_time_mean, test_spectrum, one_step/test_map,
test_tensor_dict_accumulator, test_wandb, test_distributed and test_trainer on CPU
against the same files on plain REL. The only new failures were:

- `test_trainer.py::test_resume_after_interrupted_training_during_epoch[preempt]`, caused
  by 25b8e6819 registering a second post-abort callback. The test needs updating if that
  commit is ported.
- `test_xarray.py::test_rename[mock_monthly_zarr...]`, an environment issue: the shared
  `.venv` lacks `zarrs` (`BadConfigError: zarrs.ZarrsCodecPipeline`). That same error
  fails 8 zarr tests on plain REL too.

The other 11 failures also fail on plain REL (the zarrs env, and 2 in test_data_writer).
I did not run lint or mypy (no mypy in the venv; pre-commit is broken on Perlmutter).

## Notes for the HR run (found while triaging)

- **REL does not write the restart checkpoint on SIGTERM under spatial parallelism.**
  In `generics/trainer.py`, the post-abort callback is registered only
  `if not dist.has_spatial_parallelism`, because the sharded state would need the aborted
  communicators. On a walltime requeue, mid-epoch progress therefore comes only from
  `checkpoint_every_n_batches` (default 1000). Set it so one interval costs well under
  the walltime margin.
- No HIST commit gates the per-channel `val/ensemble/*/mean_map` images
  (one_step aggregator). d6307e3b8 covers inline-inference time-mean only.

## Recommended ordered cherry-pick list (onto REL)

1. `9c264dfae`: frozen_precipitation_rate prefix (PRECST rename resolves).
2. `0a511e86f`: serial and memoized `_get_raw_times` plus stdlib glob (no fork-after-NCCL deadlock).
3. `91b069b44`: `rename` / `reference_pressure_name` / `combine` / `mask_and_scale` / `add_scalar`. Pick it after 0a511e86f and it is clean.
4. `83d034b34`: `basicConfig(force=True)`.
5. `d6307e3b8`: time-mean `report_plot` / `plot_variables`.
6. `a9650919d`: in-place cross-rank reduction fix.
7. `c773f21af` (fme/ only: `git show c773f21af -- fme | git apply -3`), plus a follow-up
   commit that threads `_process_group_timeout()` into `external/pnd_manager.py`
   `init_process_group`.

Optional:

8. `25b8e6819`, adapted to `add_post_abort_callback`, with the `[preempt]` test fixed.
9. `3d26128ec` plus the fme hunk of `fd20fe981`, only if `prediction_names` configs must
   carry over. Otherwise use `DataWriterConfig.files`.

Do not port: c5d39a0fa (already upstream as #1493), 5d2a07825 and 72ef48865 (superseded
by the REL listener-thread shutdown), and all coupled or ocean commits (02caa33e5,
021745334, ab0af55f5, 9ba06e105, 8f61e6988, db6d2f53e, 57dedf55a, e87618086, 028b1bed5).
