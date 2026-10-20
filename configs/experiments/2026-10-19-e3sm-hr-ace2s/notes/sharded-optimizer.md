# Sharded optimizer state (`optimization.shard_optimizer_state`)

2026-10-20. Not committed.

## Why

Under the spatial backend (`FME_DISTRIBUTED_BACKEND=model`), every tile rank
holds a full copy of the parameters, the gradients and the Adam state. For the
2.1 B-parameter 0.25 deg NoiseConditionedSFNO, the Adam state alone is 16.8 GB
per GPU. In an isolated benchmark, ZeroRedundancyOptimizer over the 4 tile
ranks cut the steady-state peak from 71.7 to 60.0 GB at the same step time
(4.24 vs 4.28 s). Real training of that model OOMs at about 70 GB today.

## Config

```yaml
optimization:
  optimizer_type: AdamW        # Adam, AdamW and FusedAdam all work
  kwargs: {fused: true}
  shard_optimizer_state: true  # default false
```

- **With spatial parallelism**, the optimizer is wrapped in
  `torch.distributed.optim.ZeroRedundancyOptimizer` over the **spatial**
  process group. That group is one node in our layouts. Each data-parallel
  replica keeps its own shards. Each rank steps the parameters it owns, then
  broadcasts them to its group.
- **Without spatial parallelism** (serial, purely data-parallel, or the model
  backend at 1x1), the flag does nothing and logs one info line. I chose not to
  shard over the data group for two reasons. The post-abort (preemption)
  restart save runs exactly in that case, and it cannot run a collective
  because its communicators are already aborted. Also, pure-DP runs are not
  short of memory.
- **`lr_tuning` together with the flag is rejected** in `__post_init__` of
  both the ace and coupled `TrainConfig`. The LR-tuning trial copies the
  optimizer state with `get_state()` on every rank, and sharded state is only
  complete on the root, inside a collective.

## Design

New module `fme/core/optimizer_sharding.py` with
`SpatiallyShardedOptimizer(ZeroRedundancyOptimizer)`:

- **Partitioning and the step** come unchanged from ZeRO: a greedy split by
  `numel`, the local optimizer stepping this rank's share, and a per-parameter
  broadcast from each owner. `fused=True` passes through to the local AdamW.
- **Hyperparameters.** The LR scheduler is built on the wrapper, and the
  wrapper's `param_groups` hold all parameters. ZeRO copies the wrapper's
  hyperparameters to the local optimizer at every step and at every gather.
  So scheduler LR changes, `set_learning_rate`, and `learning_rate` (which
  reads `param_groups[0]["lr"]`) all work. `_sync_param_groups` is overridden
  to mirror the groups exactly (it also deletes keys missing from the source).
  Without that, fine-tuning would leave a key it had removed, such as the
  checkpoint's `initial_lr`, in the local optimizer, and that key would come
  back after the next step.
- **Gradient clipping, AMP and accumulation are unchanged.** After the DDP
  all-reduce, every rank holds the full, identical gradient.
  - `_clip_gradients` iterates the wrapper's groups, which contain all
    parameters, so it computes the same norm on every rank.
  - `GradScaler.unscale_/step` likewise see all parameters. They get the same
    `found_inf` on every rank, so every rank makes the same step/skip decision
    and the collective inside `step()` stays matched. The wrapper has no
    `_step_supports_amp_scaling`, so the scaler unscales and then calls
    `step()` as it does for a plain optimizer.
  - Gradient accumulation only changes when `backward` runs.
- **Saving.** `gathered_on(global_rank)` is a context manager and a
  collective over the shard group. Ranks outside the root's group (other
  data-parallel replicas) skip it.
  - Each non-root rank sends a pickled structure with its per-parameter
    state, keyed by the global index. 0-dim tensors such as `step` travel
    inside the pickle, and larger tensors are replaced by placeholders.
    Each such tensor then follows with a point-to-point `send`.
  - The root receives the tensors one at a time into a device buffer and
    copies each to the CPU.
  - The result is `torch.optim.Optimizer.state_dict(self)` (the packing of
    the plain optimizer) with the gathered state filled in. It is freed when
    the context exits.
  - `state_dict()` raises outside the context, so a stale or partial state
    cannot be written by accident.
  - I did not use ZeRO's own `consolidate_state_dict`, for two reasons.
    First, it broadcasts every shard to every rank as a pickled byte tensor
    on the GPU, which costs a few GB of transient device memory per rank.
    Second, it copies GPU to CPU with `non_blocking=True` and never
    synchronizes, which pins memory and races the read.
- **Loading.** `load_state_dict(full)` runs on every rank. It builds a local
  state dict (local indices) for the parameters this rank owns, then calls
  the local optimizer's own `load_state_dict`. This means the local
  optimizer's placement rules apply: fused AdamW puts `step` on the device.
  ZeRO's built-in loader forces `step` onto the CPU, which a fused CUDA step
  would reject. The input dict is not mutated, which ZeRO's loader also does.
  Group-count and group-size mismatches raise the same `ValueError` text as
  the plain optimizer, so the fine-tune mismatch path keeps its message.
- **Interchangeable on disk.** On disk, the format equals the plain
  optimizer's `state_dict()`: the same `state` keys and tensors, and the same
  `param_groups` contents. There are two cosmetic differences. Sharded runs
  save the state tensors on the CPU, while unsharded GPU runs save them on the
  GPU. Every loader uses `map_location="cpu"` or the optimizer's cast, so the
  device does not matter. The key order inside each group also differs, which
  does not affect dict equality.
  - Sharded checkpoints therefore load into unsharded runs and the reverse.
    Inference never reads `optimization`.
- **Plumbing.**
  - `OptimizationABC.consolidated_state()` is a no-op context by default.
    `Optimization` takes a `consolidated_state` callable from `build()`:
    `nullcontext`, or `partial(sharded.gathered_on, 0)`.
  - `Distributed.get_spatial_process_group()` returns None without spatial
    parallelism.
  - `DistributedBackend.spatial_process_group()` is a non-abstract method
    that defaults to None.

### Backend hook (for `model_torch_distributed.py`)

Add this to `ModelTorchDistributed`:

```python
    def spatial_process_group(self) -> torch.distributed.ProcessGroup | None:
        if self._h_size > 1 or self._w_size > 1:
            return self._spatial_group
        return None
```

Until that method exists, `Distributed.get_spatial_process_group()` falls back
to `getattr(backend, "_spatial_group", None)`, so things work either way.
Once the method is in, the fallback is dead code and can be removed.

## Save/load call-site audit

The checkpoint writer is the global root (rank 0). `save_checkpoint` raises
on any other rank.

| site | includes optimization | who reached it before | now |
|---|---|---|---|
| `train_one_epoch`, every `checkpoint_every_n_batches` | yes (`latest`) | root only (`_should_save_checkpoints()` gate) | all ranks gate on `params.save_checkpoint` + batch count, then call `_save_checkpoints_on_root(self._save_restart_checkpoints)` |
| `train_one_epoch`, end-of-epoch restart save | yes (`latest`) | root only | same; `_last_saved_num_batches_seen` is now updated on every rank, so the condition agrees across ranks |
| `train`, end-of-epoch `save_all_checkpoints` | best/best-inference/ema-epoch: no; restart + epoch ckpt: yes | root only | `_save_checkpoints_on_root(save_epoch_checkpoints)`: one gather serves both optimization-carrying files (no step happens in between) |
| post-abort termination callback | yes | root, from a listener thread | unchanged; it is only registered without spatial parallelism, and so never sees sharded state |
| `_maybe_tune_lr` / `run_lr_tuning_trial` (`get_state`/`load_state` on all ranks) | yes | all ranks | forbidden with the flag (config validation) |
| `_restore_checkpoint` (resume) | load | all ranks, full ckpt on CPU | unchanged; every rank keeps its share |
| `resume_optimizer_ckpt_path` -> `load_optimizer_state_for_finetuning` | load | all ranks | unchanged; the configured hyperparameters win on both the wrapper and the local optimizer |
| downscaling `train.py`/`video_train.py` `_save_checkpoint` | yes | root only | **not wired**; downscaling does not run spatially. With the flag under spatial parallelism, the first save would raise the `state_dict()` RuntimeError (loud, no hang) |

`_save_checkpoints_on_root(save)` does nothing unless `params.save_checkpoint`
is set (the same value on every rank). Otherwise it enters
`optimization.consolidated_state()` on every rank and calls `save()` only where
`_should_save_checkpoints()` (root). Every gate before it is a deterministic
function of the batch/epoch counters and config, which are the same on every
rank, so all ranks of the root's group reach the collective together.

## Tests

- **`fme/core/distributed/parallel_tests/test_sharded_optimizer.py`** has 9
  tests. Each one trains a 3-layer MLP through `dist.wrap_module`, using the
  real gradient sync, and gives every rank different inputs, as tiles have.
  The cases are AdamW, fused AdamW, Adam, and StepLR + `max_grad_norm` +
  gradient accumulation.
  - **Sharded vs unsharded:** parameters after 4 steps agree (rtol 1e-6),
    with identical LR. Each rank's local state is smaller than the full
    state. The gathered state dict equals the unsharded `state_dict()`
    (groups, keys and tensors), and the scheduler state matches.
  - **Round trip:** save, broadcast to all ranks, load, then continue 3
    steps, in the directions sharded to sharded, sharded to unsharded and
    unsharded to sharded. Each matches continuing without the round trip.
  - **Fine-tune from file:** sharded and unsharded runs match, and the
    configured `lr`/`weight_decay` hold on both the wrapper and the local
    optimizer, with no leaked `initial_lr`.
  - **Mutation checks:** skipping the load fails 5 tests, dropping a rank's
    state in the gather deadlocks, and the plain ZeRO group sync fails the
    fine-tune test.
- **`fme/core/test_optimization.py`:** the flag is a no-op serially (plain
  AdamW, bitwise-equal training, `get_state` works without the context), and
  the config round-trips.
- **`fme/core/generics/test_trainer.py::test_checkpoint_saves_consolidate_optimizer_state_on_every_rank`**
  runs the trainer as root and as non-root (`is_root` patched), with mid-epoch
  and epoch checkpoints. It checks two things. First, the non-root enters the
  collective as often as the root (12 times). Second, every optimization-state
  read for a checkpoint happens inside it. The test fails on the old trainer.
- **`fme/ace/train/test_train_config.py`, `fme/coupled/train/test_train_config.py`:**
  `lr_tuning` together with the flag raises.

Results (CPU/gloo, `CUDA_VISIBLE_DEVICES=`): see the table below.

| suite | result |
|---|---|
| `test_sharded_optimizer.py`, serial / (2,1) / (1,2) / (2,2) / DP2x(2,1) | 9/9 passed on every rank |
| full `-m parallel fme` (2,1), (1,2), (2,2) | 111 passed, 6 skipped per rank |
| full `-m parallel fme` DP2x(2,1), 4 ranks | 117 passed per rank |
| full `-m parallel fme` serial | 108 passed, 9 skipped |
| `fme/core/test_optimization.py` + `fme/core/generics` + both `test_train_config.py` | 250 passed, 2 skipped |
| `fme/ace/test_train.py` + `fme/coupled/train` (end to end) | 64 passed (HEALPix deselected: `earth2grid` is not installed, and it fails the same way on the unmodified tree) |
| ruff check / ruff format --check / mypy on the 13 touched files | clean |

The parallel runs pass `--ignore=fme/core/test_writer.py`, which imports the
missing `zarrs` package (an existing environment gap).

## Caveats / risks

- **Untested on GPU/NCCL.** The send and recv buffers are on
  `param.device`. The pickled structure goes through `send_object_list`,
  which uses the current CUDA device for NCCL. The fused-AdamW `step`
  placement on load and the AMP GradScaler path are covered by reasoning
  only: GradScaler("cuda") is disabled on CPU. **First GPU run: train a few
  steps with the flag, check the checkpoint loads into an unsharded
  optimizer, and resume.**
- **Gather cost.** At each checkpoint that includes the optimizer state, the
  root's group sends about 3/4 of 16.8 GB point to point to the root. The
  root holds the full state on the CPU (about 16.8 GB of pageable memory)
  until the epoch-end saves finish. The extra device memory is one state
  tensor at a time. The wall time is not measured, but should be seconds
  against the existing ~30 s save.
- **Per-step cost.** ZeRO broadcasts every parameter from its owner every
  step, about 8.4 GB per group per step at 2.1 B parameters, unbucketed.
  The isolated benchmark showed no step-time change on NVLink. Frozen
  parameters are broadcast too. `parameters_as_bucket_view` is not exposed.
- **Parameters must be identical across the spatial group.** In
  `FME_SPATIAL_GRAD_SYNC=ddp` mode, DDP over the world broadcasts rank 0's
  parameters at wrap time, so this holds. In `hooks` mode, it relies on equal
  init seeds. If a rank ever held a rank-local parameter, ZeRO would
  overwrite it with the owner's copy.
- **Full checkpoint on every rank at fine-tune time.**
  `resume_optimizer_ckpt_path` still moves the full optimizer state onto
  every rank's GPU before loading (pre-existing `_tensors_to_device`). For
  this model that is about 16.8 GB of transient device memory at build time.
  Resume (`_restore_checkpoint`) keeps it on the CPU.
- **Downscaling trainers** and **`lr_tuning`** are not supported with the flag
  (see the audit).
