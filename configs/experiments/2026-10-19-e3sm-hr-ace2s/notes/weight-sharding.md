# Spatially sharded parameters (`shard_spatial_parameters`)

2026-10-20. Not committed.

## Why

Under the spatial backend (`FME_DISTRIBUTED_BACKEND=model`) every rank held a
full replica of every parameter. Two kinds of parameter are only used on a
rank-dependent slice:

- `SpectralConvS2.weight` (and the spectral LoRA factors), shape
  `(groups, lmax, C, C, 2)`: forward used `weight[:, l_slice]`, the rank's
  range of spherical-harmonic degrees under the polar (H) split;
- the positional embeddings: the SFNO `pos_embed`, and `NoiseConditionedModel`'s
  `pos_embed` / `label_pos_embed`: forward used `[..., h_slice, w_slice]`.

For the 0.25 deg model (embed 384, 8 layers, all modes) these are 1.70 B + 0.40 B
of the 2.10 B parameters. The benchmark prototype that kept only the slices
cut peak memory from 71.7 to 35.2 GB and step time from 4.28 to 4.10 s
(`comm-and-memory.md` section 5). This is the real implementation.

## Config

```yaml
stepper:
  step:
    config:
      builder:
        type: NoiseConditionedSFNO
        config:
          shard_spatial_parameters: true   # default false
```

Override form: `stepper.step.config.builder.config.shard_spatial_parameters=true`.

**Why the builder.** The model alone knows which parameters are used on
which slice. Everything else (gradient sync, optimizer, EMA, checkpoints) is
generic and keys off metadata the model attaches to its parameters, so no
other config needs to know about the flag. The builder already holds an
execution-only knob of the same kind (`checkpointing`), and the flag travels
with the checkpoint: inference under spatial parallelism shards too, with no
extra configuration. Serially, or under pure data parallelism, it does
nothing, so a checkpoint trained with it loads anywhere. A distributed
setting (an env var such as `FME_SPATIAL_GRAD_SYNC`) would have needed a
separate path into model construction and would not be in the checkpoint.

## Design

### Shard metadata (`fme/core/distributed/parameter_sharding.py`, new)

- `ParameterShard(global_shape, index, split, owner)`: this rank's slice of the
  full tensor (`index`, from `Distributed.get_local_slices`, so uneven splits
  work), which spatial axes split it (`split = (h, w)`), and whether this rank
  is the one that contributes the slice when the full tensor is assembled
  (`owner`: one rank per distinct slice).
- `spatial_shard(global_shape, split_shape, dims)` builds it: `dims` names the
  tensor dims split like the H and W axes of `split_shape`. Spectral weights:
  `split_shape=(lmax, mmax)`, `dims=(1, None)`, so only the polar split shards
  them; with W > 1 the azimuth ranks hold the same l-slice. pos_embed:
  `split_shape=img_shape`, `dims=(-2, -1)`. Returns None when the layout
  leaves the tensor whole (serial, pure DP, H = 1 for spectral weights), so
  the parameter stays an ordinary replicated one.
- `shard_parameter(module, name, shard)` replaces the parameter by
  `full[index].clone()` after the module drew the full initialization, so
  sharded init == slicing the serial init. It registers two module hooks:
  - `load_state_dict` pre-hook: a full tensor is sliced to the local shard; a
    local-shaped tensor is accepted as is (in-memory round trips, e.g.
    `Stepper.replace_multi_call`). It is registered after `SpectralConvS2`'s
    legacy-format hooks, which therefore still see full tensors.
  - `state_dict` post-hook: inside `gathered_parameter_shards(dst)` it
    replaces each local shard by the full tensor; outside it returns local
    shards.
- `gathered_parameter_shards(dst=0 | None)`: a collective context. Each read of
  a sharded tensor inside it does one all-reduce over the spatial group of a
  zero buffer into which the owner wrote its slice. Ranks of every
  data-parallel replica take part (each in its own spatial group). The full
  tensor goes to the CPU of `dst` (the root) or, with `dst=None`, of every
  rank. Other ranks get their local value back. Transient device memory: two
  full copies of one tensor (pos_embed: 2 x 1.6 GB at 0.25 deg).

### Model

- `SpectralConvS2(shard_spatial_parameters=...)`: shards `weight`, `lora_A`,
  `lora_B` along l. Forward indexes them with `_weight_l_index` (the l-slice,
  or `slice(None)` when sharded). Spectral LoRA under the polar split was
  broken before (the full-l LoRA factors met the local-l activations in the
  einsum). It now uses the local slice too, sharded or not.
- `SFNONetConfig.shard_spatial_parameters` reaches every block's filter and
  the SFNO `pos_embed` (`_pos_embed_index`).
- `NoiseConditionedModel(shard_spatial_parameters=...)`: `pos_embed` and
  `label_pos_embed`.
- `NoiseConditionedSFNOBuilder.shard_spatial_parameters` (default False) wires
  it through. **The builder also starts every rank's CPU random state from the
  root's before building** (only with the flag and spatial parallelism). The
  ranks' states really do differ at build time in real training: the root
  does setup work the others skip (measured: different RNG hashes on ranks 0
  and 1 at `build_trainer` start). Replicated parameters never cared, because
  DDP broadcasts rank 0's values. Sharded ones are not broadcast, so without
  this the slices would come from different initializations. A parallel test
  seeds the ranks differently and fails without the sync.

### Gradient sync (`ModelTorchDistributed.wrap_module`)

A sharded parameter's gradient is complete on its rank except for:

1. a sum over the ranks that hold the same slice: the azimuth ranks for a
   polar-split spectral weight, because each one contracts the weight with
   its own m-range;
2. the mean over data-parallel replicas.

| mode | replicated params | sharded, no replicas (pos_embed, spectral at W = 1) | sharded with replicas (spectral at W > 1) |
|---|---|---|---|
| `ddp` (default) | DDP over the world + sum/data_size comm hook, as before. Sharded names are excluded with `DDP._set_params_and_buffers_to_ignore_for_model`, which also skips the shape check and rank-0 broadcast that would break them. | if data_size > 1: async `all_reduce(grad / D)` over the data group from a post-accumulate-grad hook, overlapping backward; the backward waits for all of them in an engine callback, like DDP's finalize | per-pass gradient hook: `all_reduce(grad / D)` over the (data x w) group, created from the device mesh; stream-synchronous |
| `hooks` | DDP over the data group + spatial-sum hooks | DDP over the data group (its ranks hold the same slice) | DDP over the data group + per-pass sum hook over the w group |

With 4x1 on one node (the production layout), sharded parameters need no
communication at all: 99.7% of the gradient-sync volume disappears.
`gradient_as_bucket_view=True` is unaffected (ignored parameters keep
ordinary `.grad`s). DDP spells the root module's own parameters with a
leading dot in one of its two ignore checks, so both spellings are passed.

### Optimizer (`fme/core/optimization.py`, `fme/core/optimizer_sharding.py`)

- **Plain optimizer:** a sharded parameter's state (Adam moments) has the
  local shape. Each rank steps its slice with its complete gradient.
- **`shard_optimizer_state: true` (ZeRO) now works with sharding.** ZeRO cannot
  take rank-local parameters: its partition is computed from shapes (uneven
  splits give different partitions on different ranks), and it broadcasts
  each parameter from its owner. `SplitShardedOptimizer` (new) steps the
  replicated parameters with the existing `SpatiallyShardedOptimizer` and the
  sharded ones with a plain optimizer of the same class. It exposes one param
  group with all parameters, whose hyperparameters (scheduler LR,
  `set_learning_rate`, fine-tune resets) reach both inner optimizers at every
  step. Its `state_dict` is a single unsharded optimizer's. With full sharding
  the replicated rest is only 6.7 M parameters, so ZeRO saves little on top
  of sharding at 4x1. It matters at W > 1, where the spectral weights stay
  replicated.
- **Gradient clipping:** `_sharded_grad_norm` = sqrt(norm^2 of the replicated
  grads + spatial all-reduce of norm^2 of the owners' sharded grads). It
  equals the serial global norm and is the same on every rank. It is then
  applied with `clip_grads_with_norm_`.
- **AMP is rejected** with sharded parameters (`ValueError` in
  `OptimizationConfig.build`): GradScaler's inf check would then differ
  between ranks. HR runs do not use AMP (bf16 was slower).
- **State dict:** `consolidated_state()` (collective, entered on every rank by
  the trainer, as for ZeRO) also gathers every sharded parameter's state
  tensors (the ones of local shape; `step` stays the root's) to the root's
  CPU. `get_state()` merges them in and raises if called outside the
  consolidation. `load_state` and `load_optimizer_state_for_finetuning` slice
  full-shaped entries to the local shard (`_localize_state_dict`) before
  `optimizer.load_state_dict`.

### EMA (`fme/core/ema.py`)

`EMATracker` records the shard of each tracked parameter at construction. The
averages are local shards, updated in place on the local parameters.
`get_state` gathers them inside `gathered_parameter_shards`. `from_state`,
`load_ema_state_for_finetuning` and `copy_params_from_state` (inference with
EMA weights) slice full tensors.

### Checkpoints (`fme/core/generics/trainer.py`)

The format is unchanged: full tensors everywhere. The bitwise e2e check below
shows identical checkpoints, keys and shapes included.

**Saving is collective with sharded parameters.** Reading the stepper's or the
EMA's state assembles full tensors, and a save can read them with the EMA
applied (best and EMA-epoch checkpoints under `validate_using_ema`). So one
up-front gather is not enough. `Trainer` sets `_collective_checkpoints =
has_parameter_shards(stepper.modules)`. When it is set, `_save_checkpoints_on_root`
runs `save()` on **every** rank inside `optimization.consolidated_state()` and
`gathered_parameter_shards(dst=0)`. `save_checkpoint` reads the stepper and
EMA state on every rank and only the root writes.
`save_all_checkpoints` broadcasts the root's choice of best checkpoints
(`scatter_object`), so every rank runs the same sequence of collectives.
`save_checkpoint` outside that context raises. Without sharded parameters the
old root-only path is unchanged. `lr_tuning` with sharding is rejected (it
copies state in memory on every rank).

**Loading needs no new code path.** `stepper.load_state(full)` slices in the
module pre-hook, `optimization.load_state(full)` slices, and
`EMATracker.from_state(full)` slices.

## Call sites audited

| site | path | status |
|---|---|---|
| trainer mid-epoch / end-of-epoch restart saves | `_save_checkpoints_on_root(_save_restart_checkpoints)` | collective; tested |
| trainer epoch saves: best, best-inference, restart, EMA epoch, epoch ckpt | `_save_checkpoints_on_root(save_epoch_checkpoints)` | collective; best/EMA ones read the stepper with EMA applied on every rank; tested e2e |
| post-abort restart save | root-only listener thread | not registered under spatial parallelism (unchanged), so it never sees shards |
| resume `_restore_checkpoint` | full ckpt on CPU, every rank | slices; tested (resume = continued run, bitwise e2e) |
| inference / evaluator `load_stepper` -> `Stepper.from_state` | builds the sharded model (flag from the checkpoint), loads full | slices; parallel test: sharded-run checkpoint, raw and EMA weights, sharded/unsharded/serial forward agree |
| `load_ema_params_if_available` (`use_ema_if_available`) | `copy_params_from_state` | slices; tested |
| fine-tune `parameter_init.weights_path` | `load_weights_and_history` -> `load_stepper` (sharded if that ckpt had the flag) -> `module.state_dict()` | now inside `gathered_parameter_shards(dst=None)` (full on every rank, CPU); `overwrite_weight_initial_slice` slices for sharded targets; tested (`overwrite_weights`) |
| `CopyWeightsConfig.apply` (copy after batch) | `overwrite_weight_initial_slice` | slices (full base weights live on the CPU, so each copy is a host-to-device transfer) |
| L2-SP regularizer (`alpha`/`beta`) | base weights vs params | base weights sliced once; **but L2-SP is not spatially correct with or without sharding** (per-rank norms, replicated params' penalty summed over tiles); pre-existing |
| `resume_optimizer_ckpt_path` | `_load_finetune_optimization_state` | slices; still moves the full optimizer state to every GPU first (pre-existing, ~17 GB transient at 0.25 deg) |
| `resume_ema_ckpt_path` | `load_ema_state_for_finetuning` | slices on the CPU, then moves |
| `Stepper.replace_multi_call/replace_ocean/...`, `wildcard`, `parameter_init` key checks | in-memory `state_dict()` | local shards in, local shards out |
| `lr_tuning` | in-memory copies of stepper/optimizer/EMA on every rank | rejected in `Trainer.__init__` |
| coupled `_load_stepper_weights_and_history_factory`, downscaling trainers | `module.state_dict()` | not wired (no spatial support there; the flag is NoiseConditionedSFNO-only) |
| `count_parameters` log | local | logs this rank's count (0.53 B at 4x1) |

## Tests

| file | what |
|---|---|
| `fme/core/distributed/parallel_tests/test_parameter_sharding.py` (20 tests) | small NoiseConditionedSFNO on a 9x16 grid (lmax 9: uneven polar split), model + area-mean loss + AdamW + EMA driven as the trainer does, each rank on its tile of its replica's samples; cases: SFNO pos_embed; context pos_embed + label_pos_embed + labels; spectral LoRA + ZeRO; clip + StepLR + ZeRO; gradient accumulation + clip under `FME_SPATIAL_GRAD_SYNC=hooks`. Init == sliced serial init (bitwise, incl. which tensors are sharded per layout); init uses the root's random state when ranks differ; sharded vs unsharded training (params, EMA, full optimizer state dict, scheduler) rtol 1e-5; vs a serial in-process run (`force_non_distributed`) rtol/atol 1e-4; checkpoint -> resume into sharded, unsharded and serial == continuing; inference from a sharded checkpoint (raw and EMA) sharded == unsharded (1e-5) == serial (1e-4); local in-memory round trip; gather to every rank -> `overwrite_weights` |
| `fme/core/distributed/test_parameter_sharding.py` (10) | serial: flag is a no-op (bitwise), hook semantics, gather semantics, EMA/optimizer full-format round trip, AMP rejection, `overwrite_weights` |
| `fme/core/generics/test_trainer.py` (+3) | sharded stepper: checkpoints written by root have full shapes (stepper, EMA, Adam state), resume slices back; a non-root rank reads the state inside the gather exactly as often as the root and writes nothing; direct `save_checkpoint` outside the collective raises; `lr_tuning` rejected |
| `scripts/check_weight_sharding_e2e.py` | real `python -m fme.ace.train` under torchrun (CPU, 2x1): 1 epoch then resume, sharded vs unsharded, with `validate_using_ema`, inline inference, best/restart checkpoints. Result: **every checkpoint tensor (247 in ckpt.tar: stepper, EMA, Adam, scheduler) bitwise equal**, after both segments. Not a pytest test: each torchrun segment takes 20-70 s on a shared node and tests are capped at 90 s. It aligns the CPU random state of both runs at build (sitecustomize), see "RNG" below. |

Mutation checks: no azimuth-replica sum -> the (2,2) training tests fail;
no RNG sync -> `test_sharded_init_uses_the_root_random_state` fails.

Results (CPU/gloo, `CUDA_VISIBLE_DEVICES=`):

| suite | result |
|---|---|
| `parallel_tests/test_parameter_sharding.py` at (2,1), (1,2), (2,2), (4,1), DP2x(2,1), DP2x(2,2), and serial | 20/20 on every rank (serial: 20) |
| `test_parameter_sharding.py` (serial) + trainer tests | 10/10 + 3/3 |
| `scripts/check_weight_sharding_e2e.py` | segment 1 and 2 checkpoints bitwise equal (resume check re-added afterwards, rerun pending) |
| serial suites: core/models, core/distributed, core/generics, core/step, test_optimization, test_ema, ace/stepper, ace/train, ace/registry, ace/test_train.py, coupled/train (`-n 16`) | 1063 passed, 58 failed: 57 `TimeoutException` (90 s cap; node loaded by concurrent runs) + 1 pre-existing earth2grid. A rerun of the 57 at `-n 6` still had 19 timeouts and no assertion failure. **A clean rerun on an idle node is still owed.** |
| full `-m parallel fme` suite | **not run** (time) |
| ruff check / format / mypy on all touched files | clean |

## Measured: per-rank parameters at 0.25 deg (CPU build, 4x1)

`NoiseConditionedSFNOBuilder(embed_dim=384, noise_embed_dim=32, isotropic,
num_layers=8, affine_norms, hard_thresholding_fraction=1.0,
shard_spatial_parameters=True)`, 43 in / 50 out channels, 720x1440, H=4:

| | params | fp32 |
|---|---|---|
| full model (and per rank without sharding) | 2103.5 M | 8.41 GB |
| per rank, sharded | **530.9 M** (524.2 M sharded slices + 6.7 M replicated) | **2.12 GB** |

9 sharded tensors (8 spectral weights + pos_embed). The build took 23-26 s
per rank, with 11.0 GB peak RSS. The full init is drawn and then sliced, so
host memory at build is unchanged. Per-GPU memory then scales the same way:
the parameter, gradient, EMA and Adam terms of the 71.7 GB budget
(8.4 + 8.4 + 8.4 + 16.8 GB) become a quarter.

## Pre-existing issues found on the way (not fixed here)

1. **`FME_SPATIAL_GRAD_SYNC=ddp` (the default) is wrong with
   `use_gradient_accumulation: true`.** Each extra backward pass all-reduces
   the *accumulated* gradient with a SUM over tiles, so earlier passes are
   counted H*W times. Measured on the test model: 0.05 max param error vs
   serial after 3 steps (`hooks` mode: 1e-5). This holds with or without
   sharding; sharded parameters themselves are right. HR configs use
   `use_gradient_accumulation: false` (one backward per step), so they are
   unaffected. Fix: `no_sync()` on all but the last backward, or use `hooks`.
2. **The evaluator under spatial parallelism fails in
   `InferenceAggregator.flush_diagnostics`** (`conflicting sizes for dimension
   'lat': 16 vs 8`), with sharded and unsharded checkpoints alike. Stepper
   loading and stepping work. The prediction file is then left unreadable.
3. **CPU random states differ between ranks at model build** in real training.
   This is harmless for replicated parameters (DDP broadcast). On GPU the noise
   uses the CUDA RNG, seeded identically on all ranks. On CPU-only runs,
   isotropic noise drawn from per-rank CPU states would be inconsistent
   across tiles of an unsharded model. The sharded build aligns the states.
4. The per-rank `loss` metric differed between the two ranks of the e2e run
   (1.97 vs 2.06 on step 1, in both sharded and unsharded). Probably a
   rank-local metric, not the training loss. Worth a look.

## Caveats / risks

- **Not run on GPU/NCCL.** Specific points: the `(data, w)` mesh group (only
  for W > 1) is created lazily at wrap time; the async data-group all-reduces
  in post-accumulate hooks use NCCL streams with `work.wait()` in the backward
  callback; the gathers allocate two transient full tensors on the device
  (3.2 GB for pos_embed).
- **Checkpoint gather cost:** one all-reduce per sharded tensor per read. The
  restart and epoch saves read the stepper (2.1 B), the EMA (2.1 B) and the
  Adam state (4.2 B) once each. Within a node that is seconds against the
  existing ~60 s save.
- **The CPU RNG sync changes non-root ranks' CPU random streams after the
  build** (they follow the root's). It is only active with the flag under
  spatial parallelism, and on GPU the noise uses the CUDA RNG.
- W > 1 layouts: the spectral weights stay replicated across azimuth ranks
  (sharded only along l) and need a stream-synchronous sum over the azimuth
  group. That is correct but not overlapped; 4x1 needs none of it.
- `makani-linear` filters and DISCO (`local_blocks`) are not sharded.

## GPU validation (1 node, 4 x A100, ~15 min)

Baseline is the existing unsharded run `$PSCRATCH/hr-runs/smoke-1n-4x1-full-zero`
(log `logs/smoke-1n-4x1-full-zero.out`, nvidia-smi 66.7 GB, 0.245 samples/s).

```bash
cd $HR_EXP   # after: source scripts/env.sh
# 1. sharded + ZeRO, same config otherwise
srun -N 1 --ntasks-per-node 1 --gpus-per-node 4 scripts/train.sh 4 1 \
  runs/smoke-1n-4x1-full-zero.yaml $PSCRATCH/hr-runs/smoke-1n-4x1-full-shard \
  stepper.step.config.builder.config.shard_spatial_parameters=true \
  2>&1 | tee logs/smoke-1n-4x1-full-shard.out
```

Check:
1. The first ~20 step losses track the baseline log. Exact equality is not
   expected (NCCL reduction order).
2. Peak memory: ~35-40 GB in nvidia-smi (was 66.7), with s/step <= baseline.
3. Checkpoints: same keys and shapes as the baseline's:
   ```bash
   python - <<'EOF'
   import os, torch
   r = os.environ["PSCRATCH"] + "/hr-runs/"
   a = torch.load(r + "smoke-1n-4x1-full-shard/training_checkpoints/ckpt.tar", map_location="cpu", weights_only=False)
   b = torch.load(r + "smoke-1n-4x1-full-zero/training_checkpoints/ckpt.tar", map_location="cpu", weights_only=False)
   sa, sb = a["stepper"]["step"]["module"], b["stepper"]["step"]["module"]
   assert {k: getattr(v, "shape", None) for k, v in sa.items()} == {k: getattr(v, "shape", None) for k, v in sb.items()}
   oa, ob = a["optimization"]["optimizer_state_dict"]["state"], b["optimization"]["optimizer_state_dict"]["state"]
   assert {i: {k: v.shape for k, v in s.items()} for i, s in oa.items()} == {i: {k: v.shape for k, v in s.items()} for i, s in ob.items()}
   assert {k: v.shape for k, v in a["ema"]["ema_params"].items()} == {k: v.shape for k, v in b["ema"]["ema_params"].items()}
   print("same format")
   EOF
   ```
4. Resume: rerun command 1 unchanged (it resumes from `ckpt.tar`).
5. Cross-load: copy the sharded run's experiment dir and resume it without the
   flag (`...shard_spatial_parameters=false`); it must start at the saved
   step with the same loss. The reverse (resume the baseline dir with the flag)
   checks unsharded -> sharded.
6. Optional: `optimization.shard_optimizer_state=false` to see sharding alone
   (expect ~35 GB per the prototype, vs 32 with ZeRO).
