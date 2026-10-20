# Spatial-parallel loss verification (ACE2S, EnsembleLoss)

2026-10-19. CPU only (`FME_FORCE_CPU=1`, `CUDA_VISIBLE_DEVICES=`), gloo, torchrun 2 and 4 ranks.
Code: worktree `/pscratch/sd/m/mahf708/ace-hr`, HEAD `8e41193f0`, plus the uncommitted
`model_torch_distributed.py` change that adds `FME_SPATIAL_GRAD_SYNC` (see section 6). The
first run was at session-start HEAD `8935f50b6` (v2026.10.0), before that change. Both gave
the same numbers in every row.

## TL;DR

| concern | verdict | size at 2x2 (720x1440 run) |
|---|---|---|
| (a) CRPS/EnsembleLoss tile-local `.mean` plus spatial SUM of grads | **Confirmed.** CRPS gradient is exactly `H*W` times serial, with the same direction | x4.00 |
| (b) energy score uses local spectral sizes and sets `mode_weights[...,0]` on every w-rank | **Confirmed, but mostly a constant scale.** The ES gradient is about `sqrt(H*W)` times serial. The m=0 bug only bends direction for W>1, and only slightly (cos 0.9997 at 16x32, negligible at 721 m's) | x~2.0 |
| (a)+(b) combined 0.9/0.1 | Not a pure scale: CRPS is scaled by HW and ES by sqrt(HW), so the effective mix is **0.9 : 0.05** (ES share 5.3%, not 10%) | grad x3.72, cos 0.9999 |
| (c) logged train/val loss is the root tile only | **Confirmed.** Spatial ranks never reduce it. ES rank values range from 0 to 1.04x serial; one 2x2 rank holds only m>l modes and logs ES = 0 | n/a |
| (d) isotropic noise is consistent only if the RNG streams match | **Confirmed both ways.** Same seed gives bit-identical noise (0 diff, all 3 layouts). A rank-dependent seed or one extra draw on one rank breaks it (cos 0.28-0.85) | n/a |
| (e) **new**: corrector's mid-graph global means (`area_weighted_mean` -> `_AutogradAllReduce`, identity backward) | **Real bug, small.** Gradients through `conserve_dry_air` + `moisture_budget_correction` are missing the cross-tile terms: cos 0.9987-0.9994, norm 0.997-0.999, up to 4% pointwise | cos 0.9989 |
| bonus: uncommitted `FME_SPATIAL_GRAD_SYNC=ddp` (now the default) | **Crashes** on torch 2.10: `ValueError: Communication hook: bucket annotation should be dist.GradBucket.` | blocks training |

Practical bottom line for the HR run (AdamW fused, wd 0.01, no `max_grad_norm`, no AMP):

- The xHW constant is harmless.
- The CRPS/ES re-weighting is a real but mild change of objective. You can undo it from config: `energy_score_weight: 0.2` at 2x2.
- The logged loss and `best_ckpt.tar` (chosen by validation loss) are root-tile quantities. `best_inference_ckpt.tar` is fine.
- The `ddp` grad-sync path must be fixed (one-line) or run with `FME_SPATIAL_GRAD_SYNC=hooks`.

## Setup

New test: `fme/core/distributed/parallel_tests/test_ensemble_loss_spatial.py`. It is not committed.

Every test computes the same quantity twice in one process:

- **serial**: inside `Distributed.force_non_distributed()`, on the global field, with the non-distributed SHT.
- **spatial**: with the session backend, on this rank's tile, with the distributed SHT and the `wrap_module` gradient sync.

The test then compares the two. The loss is built the way `Stepper.build_loss` builds it: `StepLossConfig(type="EnsembleLoss", kwargs).build(gridded_ops, out_names, normalizer, channel_dim=-3)`, with ensemble-shaped `[B=2, E=2, H, W]` predictions and `[B, 1, H, W]` targets on 4 channels. The grid is 16x32 Legendre-Gauss, so all tiles are even. Spectral `lmax=16`, `mmax=17`, so the m-split is 9/8 for W=2.

Tests:

1. `test_ensemble_loss_prediction_gradient[crps|energy|combined]`
   - Computes loss value and d(loss)/d(predictions).
   - The spatial gradients are gathered from the tiles.
   - Because the distributed SHT transposes route gradients across ranks, the gathered gradient equals the gradient of the **sum over ranks of each rank's local loss**. That sum is what the parameter update follows.
2. `test_ensemble_loss_parameter_gradient[crps|energy|combined|area_weighted_mse_control]`
   - Tiny `NoiseConditionedSFNO` (embed 8, noise 4, isotropic, linear filter, MLP, 2 layers, affine norms, pos_embed).
   - Weights are copied serial -> spatial and the noise seed is the same.
   - Compares `p.grad` after the spatial sync against serial.
   - The control is an `area_weighted_mean` MSE (a correctly all-reduced loss). It proves the model, the noise, and the hooks are correct, so any deviation comes from the loss.
3. `test_isotropic_noise_consistent_across_tiles[same_seed|seed_plus_rank|one_extra_draw_on_last_rank]`
   - Calls `isotropic_noise` with the model's own distributed `itrans_up`, gathers the result, and compares it with serial.
4. `test_corrector_function_gradient[gen_near_target|gen_independent]`
   - Runs the real `AtmosphereCorrectorConfig(conserve_dry_air=True, moisture_budget_correction="advection_and_precipitation")` corrector on fields of roughly realistic magnitude.
   - The dry-air reference comes from the IC.
   - The loss is a correctly reduced area-weighted MSE.
   - Compares the gradient w.r.t. the generated fields.
5. `test_corrector_parameter_gradient[False|True]`
   - Step-level version of test 4 with a tiny SFNO.
   - In this synthetic step setup the cross-tile terms are too small to see (agreement better than 1e-6 even on current code). Test 4 is the informative one.

The root rank writes one JSON line per case to `$FME_SPATIAL_LOSS_REPORT`. Assertions encode "spatial == serial" and run after all collectives, so failures cannot hang the other ranks.

## Commands

```bash
source /pscratch/sd/m/mahf708/ace-hr/configs/experiments/2026-10-19-e3sm-hr-ace2s/scripts/env.sh
cd /pscratch/sd/m/mahf708/ace-hr
T=fme/core/distributed/parallel_tests/test_ensemble_loss_spatial.py
# serial sanity (serial vs serial): 14 passed
CUDA_VISIBLE_DEVICES= FME_FORCE_CPU=1 python -m pytest -q $T
# spatial matrix
for hw in "2 1" "1 2" "2 2"; do set -- $hw
  CUDA_VISIBLE_DEVICES= FME_FORCE_CPU=1 FME_DISTRIBUTED_BACKEND=model \
  FME_DISTRIBUTED_H=$1 FME_DISTRIBUTED_W=$2 FME_SPATIAL_GRAD_SYNC=hooks \
  FME_SPATIAL_LOSS_REPORT=$PWD/results.jsonl \
  torchrun --standalone --nproc-per-node $(( $1 * $2 )) -m pytest -q -p no:cacheprovider $T
done
```

`CUDA_VISIBLE_DEVICES=` is required for the 2-rank layouts on a GPU node. With CUDA visible, the DeviceMesh group that spans the whole world reports backend `'cpu:gloo,cuda:nccl'`. `_gloo_patch` only matches `== "gloo"`, so torch_harmonics' `all_to_all` raises `Backend gloo does not support alltoall`. The existing `test_step.py -k regression` fails the same way at 2 ranks. This is a test-infra issue, not a training issue.

## Results (current code; `hooks` and annotation-fixed `ddp` give identical numbers)

Ratios are spatial / serial. Columns:

- "root": the loss value on global rank 0, which is what gets logged.
- "Σranks": the sum of the per-rank loss values, i.e. the effective objective.
- "|g|" and "cos": norm ratio and cosine of the gradient.

### Gradient w.r.t. predictions (loss built via `StepLossConfig`)

| H,W | term | root | Σranks | \|g\| | cos |
|---|---|---|---|---|---|
| 2,1 | crps | 1.010 | 2.000 | 2.000 | 1.00000 |
| 2,1 | energy | 0.339 | 1.414 | 1.414 | 1.00000 |
| 2,1 | 0.9/0.1 | 0.915 | 1.917 | 1.944 | 0.99890 |
| 1,2 | crps | 0.999 | 2.000 | 2.000 | 1.00000 |
| 1,2 | energy | 1.065 | 1.352 | 1.360 | 0.99324 |
| 1,2 | 0.9/0.1 | 1.009 | 1.908 | 1.938 | 0.99853 |
| 2,2 | crps | 1.025 | 4.000 | 4.000 | 1.00000 |
| 2,2 | energy | 0.466 | 1.912 | 1.924 | 0.99324 |
| 2,2 | 0.9/0.1 | 0.945 | 3.703 | 3.805 | 0.99628 |

### Gradient w.r.t. parameters of a tiny NoiseConditionedSFNO

The forward output matches serial exactly (max rel diff 0) in every layout.

| H,W | term | root | \|g\| | cos |
|---|---|---|---|---|
| 2,1 | crps | 1.016 | 2.000 | 1.00000 |
| 2,1 | energy | 0.353 | 1.414 | 1.00000 |
| 2,1 | 0.9/0.1 | 0.923 | 1.921 | 0.99997 |
| 2,1 | AW-MSE control | 1.000 | 1.000 | 1.00000 |
| 1,2 | crps | 0.992 | 2.000 | 1.00000 |
| 1,2 | energy | 1.045 | 1.351 | 0.99969 |
| 1,2 | 0.9/0.1 | 1.000 | 1.912 | 0.99996 |
| 1,2 | AW-MSE control | 1.000 | 1.000 | 1.00000 |
| 2,2 | crps | 1.040 | 4.000 | 1.00000 |
| 2,2 | energy | 0.485 | 1.910 | 0.99969 |
| 2,2 | 0.9/0.1 | 0.963 | 3.718 | 0.99989 |
| 2,2 | AW-MSE control | 1.000 | 1.000 | 1.00000 |

Logged (root) value per rank at 2,2, as a ratio to serial:

- crps: 1.025, 0.996, 0.974, 1.006
- energy: 0.466, **0.000**, 1.039, 0.407

Rank (h=0, w=1) holds l in [0,8) and m in [9,17). Every mode in that block has m > l, so the block is identically zero. At 720x1440 with 2x2 the same holds for l<360, m>=361: one GPU does ES work on an all-zero block.

### Isotropic noise (gathered tiles vs serial, same seed)

| H,W | same seed | seed+rank (cos) | one extra randn on last rank (cos) |
|---|---|---|---|
| 2,1 | identical (0) | 0.275 | 0.305 |
| 1,2 | identical (0) | 0.853 | 0.851 |
| 2,2 | identical (0) | 0.316 | 0.851 |

### Corrector (conserve_dry_air + advection_and_precipitation), gradient w.r.t. generated fields

| H,W | regime | \|g\| | cos | worst field cos (PRESsfc) | max pointwise rel err |
|---|---|---|---|---|---|
| 2,1 | gen near target | 0.9985 | 0.99917 | 0.99839 | 2.8% |
| 2,1 | gen independent | 0.9977 | 0.99887 | 0.99767 | 3.6% |
| 1,2 | gen near target | 0.9989 | 0.99935 | 0.99874 | 2.5% |
| 1,2 | gen independent | 0.9973 | 0.99873 | 0.99723 | 3.6% |
| 2,2 | gen near target | 0.9982 | 0.99891 | 0.99741 | 4.4% |
| 2,2 | gen independent | 0.9969 | 0.99874 | 0.99626 | 3.9% |

A contrived mid-graph probe gives cos 0.9996 at 2x2. The probe is `x - area_weighted_mean(x)`, then a weighted MSE.

### With the proposed fixes (section 7)

Every row above, in all three layouts and both grad-sync modes, gives 1.0000 for root, Σ/P, |g| and cos, with max rel diff below 6e-7. The logged loss is identical on every rank. In the fixed copy, `fme/core/test_loss.py`, `fme/core/corrector`, `test_gridded_ops.py` (234 passed) and the existing parallel tests at 2x2 (50 passed) still pass.

## Conclusions per concern

**(a) Confirmed, exact.**

- `CRPSLoss` returns the full `(B, C, H_loc, W_loc)` field, and `StandardLoss.reduce_to_channel` takes a tile-local `.mean`.
- Each rank's graph is therefore its tile's *mean*, not its *share* of the global mean.
- The spatial grad sync (`hooks` or `ddp`) SUMs over tiles, so the gradient is `Σ_tiles mean_tile = HW x mean_global` for equal tiles. 720/2 and 1440/2 are even, so this holds exactly.
- The direction is exact (cos 1.0000).
- The same applies to plain `MSE` and `L1` (`_wrap_elementwise` uses `.mean`) and to `FiniteDifferenceCRPS`.
- `AreaWeightedMSE` is correct because it all-reduces local sums.

**(b) Confirmed, and mostly a constant.** `EnergyScoreLoss` computes each rank's term as `Σ_local w·es / (2·sqrt(n_l,loc·n_m,loc))`. The Σ over ranks is then `sqrt(HW)` times the serial `Σ_all / (2·sqrt(L·M))` (1.414 at 2x1, 2.0 at 2x2). It is not exactly constant for two reasons:

1. **Uneven m-split.** mmax=721 splits 361/360 at W=2, so the per-rank factors differ by 0.14%.
2. **m=0 weight on every w-rank.** `mode_weights[...,0]=1` is applied on every w-rank, so global m=361 gets weight 1 instead of 2. That is one column of 360 modes out of about 260k valid modes.

Both are negligible at 720x1440. In the 16x32 test they bend the ES direction to cos 0.9932 (prediction gradient) and 0.9997 (parameter gradient). The *material* effect is the relative weight: the CRPS gradient is multiplied by HW and the ES gradient by sqrt(HW), so at 2x2 the trained objective is ∝ 0.9·CRPS + 0.05·ES instead of 0.9·CRPS + 0.1·ES. `SpectralWhitening.factor` also takes a local mean over m, which would be wrong for W>1, but whitening is off in this config.

**(c) Confirmed.**

- `TrainAggregator` does `dist.reduce_mean(loss)`, which reduces over the **data** group only. `loss_step_*` and the per-channel losses work the same way.
- The logged train loss, the validation loss and `best_ckpt.tar` selection are rank 0's tile: its quarter of the grid for CRPS, and its low-(l,m) spectral block scaled by `1/(2·sqrt(N/4))` for ES.
- These numbers are not comparable to serial or LR runs. With a red spectrum the ES part of rank 0 can be about 2x the serial value.
- Each rank also tracks its own `_best_validation_loss`. Only root saves, so this is harmless.
- `best_inference_ckpt.tar` uses area-weighted inference metrics and is fine.

**(d) Confirmed both ways.**

- `isotropic_noise` draws the full global `(lmax, mmax)` coefficient array on every rank and slices it.
- With identical RNG streams the tiles are bit-identical to serial.
- Any per-rank divergence silently decorrelates the tiles: a different seed, or one extra draw of the global torch/CUDA RNG on one rank. Cosine drops to 0.3-0.85 and the noise field gets seams at tile edges.
- In the planned run, `seed: 1` makes `set_seed` identical on all ranks. As far as I can see nothing in the training loop draws from the global RNG a rank-dependent number of times:
  - dataloader workers, samplers and var-masking use their own generators;
  - the SFNO has no dropout;
  - checkpointing preserves RNG state.

  So the noise should be consistent, but nothing enforces it. Broadcasting the drawn `alm` is too costly (2 members x 32 x 720 x 721 complex, about 270 MB per step at local batch 1). The cheap guards are a `broadcast_spatial` of a per-step seed for a dedicated generator, or an occasional check that a checksum of the noise is equal across the spatial group.
- In the old `hooks` mode, DDP spans the data group only, so the initial weights on spatial co-ranks also rely on identical init RNG. The `ddp` mode broadcasts from world rank 0.

**(e) New, real, small: corrector gradients.**

- `area_weighted_mean` inside the corrector goes through `_AutogradAllReduce`, whose backward is the identity.
- That is correct for the *final* loss reduction, where the downstream graph is replicated.
- It is wrong for a global mean that is applied back to each tile. Rank r only back-propagates `dℓ_r/dG` and drops `Σ_{q≠r} dℓ_q/dG`.
- Effect with realistic magnitudes: a gradient direction error of about 1e-3 (cos 0.9987-0.9994) and up to 4% pointwise.
- This is not training-breaking, but it is a systematic bias that depends on the tile layout.

## Practical implications for training (torch AdamW fused, wd 0.01, eps 1e-8, no clipping, no AMP)

- **Constant scale c (CRPS: c = HW = 4).**
  - Adam/AdamW updates `m/(sqrt(v)+eps)` and both moments scale with c, so the update is invariant except that eps is effectively eps/c (1e-8 -> 2.5e-9). That only matters for parameters whose RMS gradient is around 1e-8, so it is negligible.
  - `FusedAdam` here is `torch.optim.AdamW(fused=True)`, whose weight decay is **decoupled** (`θ -= lr·wd·θ`), so it is unaffected.
  - The coupled-L2 `Adam` optimizer_type would see its L2 effectively divided by c.
  - `max_grad_norm` is unset, so there is no clipping effect. If you set it, the threshold is effectively `max_grad_norm / c`.
  - Logged gradient norms are multiplied by c.
  - EMA and the LR schedule are unaffected.
  - **Resuming with a different (H,W)** changes c. The Adam `v` state is then off by (c_new/c_old)^2 for about 1/(1-β2) = 1000 steps, which gives a transient effective-LR jump (x2 for 1x1 -> 2x2).
- **Mixed term (0.9/0.1).** This is not a pure scale, so Adam cannot absorb it. The model trains on about 0.9 CRPS + 0.05 ES at 2x2, i.e. half the intended spectral (energy-score) pressure.
  - Config-only workaround, no code change: `energy_score_weight: 0.2` at 2x2, which in general is `0.1·sqrt(H·W)`. This restores the 0.9:0.1 ratio to within about 0.1%. The overall x4 scale is harmless. Re-derive it if the layout changes.
  - The cleaner route is the fix below.
- **Logging and selection.** Don't compare HR train/val loss against LR E01 values, and don't trust `best_ckpt.tar` (root-tile validation loss). Use `best_inference_ckpt.tar`. The fix below makes all of them global.
- **Corrector.** Leave it as is, or apply fix 7b. At about 1e-3 direction error it is not worth blocking the run on.

## 6. `FME_SPATIAL_GRAD_SYNC=ddp` (uncommitted change in the worktree) crashes

`model_torch_distributed.py` has `from __future__ import annotations`, so the comm hook's `bucket: torch.distributed.GradBucket` annotation is a string. DDP's `register_comm_hook` then raises `ValueError: Communication hook: bucket annotation should be dist.GradBucket.` on every `wrap_module` with spatial parallelism, including the existing `test_step.py`. Since `ddp` is the default, a spatial training launch from the current worktree would die at model wrap.

The fix, verified in a scratch copy, is to drop the annotation:

```diff
-    def hook(state, bucket: torch.distributed.GradBucket):
+    def hook(state, bucket):
```

With that change, `ddp` mode reproduced the `hooks` mode gradients exactly in every test here (sum over tiles, mean over data).

## 7. Proposed fixes (NOT applied to non-test code; verified in a scratch copy)

### 7a. Loss: global spatial mean for CRPS/ES; global spectral sizes and m=0 for ES (fixes a, b, c)

```diff
--- a/fme/core/loss.py
+++ b/fme/core/loss.py
@@ -9,6 +9,7 @@
 from fme.core.device import get_device
+from fme.core.distributed.distributed import Distributed
 from fme.core.ensemble import get_crps, get_energy_score
@@ -576,6 +577,22 @@
+def _spatial_mean(t: torch.Tensor) -> torch.Tensor:
+    """Mean over the last two dims of a ``(B, C, X, Y)`` tensor whose last two
+    dims are partitioned across the spatial group (grid points or spectral
+    modes). Each rank's autograd graph holds only its own share, which the
+    spatial gradient sync sums, while the value is the global mean on every
+    rank."""
+    dist = Distributed.get_instance()
+    if not dist.has_spatial_parallelism:
+        return t.mean(dim=(-2, -1))
+    local_sum = t.sum(dim=(-2, -1))
+    count = torch.tensor(
+        float(t.shape[-2] * t.shape[-1]), device=t.device, dtype=local_sum.dtype
+    )
+    return dist.spatial_reduce_sum(local_sum) / dist.spatial_reduce_sum(count)
+
+
 class EnergyScoreLoss(torch.nn.Module):
@@ -616,25 +633,24 @@
     def forward(self, x: torch.Tensor, y: torch.Tensor) -> list[LossComponent]:
         x_hat = self.sht(x)
         y_hat = self.sht(y)
-        n_l, n_m = x_hat.shape[-2], x_hat.shape[-1]
-        if self.scaling is None:
+        if self.mode_weights is None:
+            # global spectral extent; under spatial parallelism x_hat holds
+            # only this rank's (l, m) block
+            n_l, n_m = self.sht.lmax, self.sht.mmax
             self.scaling = 2 * (n_l * n_m) ** 0.5
             self.n_spectral = n_l * n_m
-        if self.mode_weights is None:
-            self.mode_weights = 2 * torch.ones(
-                (*([1] * (x_hat.ndim - 1)), n_l, n_m),
-                device=x_hat.device,
+            weights = 2 * torch.ones((n_l, n_m), device=x_hat.device)
+            weights[:, 0] = 1  # only the rank holding global m = 0
+            l_slice, m_slice = Distributed.get_instance().get_local_slices(
+                (n_l, n_m)
             )
-            self.mode_weights[..., 0] = 1
-        assert self.n_spectral is not None
+            self.mode_weights = weights[l_slice, m_slice]
+        assert self.n_spectral is not None and self.scaling is not None
         es = get_energy_score(x_hat, y_hat) * self.mode_weights
         if self._whitening is not None:
             es = es * self._whitening.factor(y_hat)
-        # Old path: .sum(dim=(-2,-1)).mean() / scaling
-        # New path: StandardLoss does .mean(dim=(-2,-1)) i.e. sum/(L*M)
-        # Multiply by L*M/scaling so mean gives the same result as sum/scaling.
-        pre_weighted = es * (self.n_spectral / self.scaling)
-        return [StandardLoss(pre_weighted)]
+        # sum over all (l, m) / scaling, as a global mean times L*M/scaling
+        return [StandardLoss(_spatial_mean(es) * (self.n_spectral / self.scaling))]
@@ -651,7 +667,7 @@
     def forward(self, x: torch.Tensor, y: torch.Tensor) -> list[LossComponent]:
-        return [StandardLoss(get_crps(x, y, alpha=self.alpha))]
+        return [StandardLoss(_spatial_mean(get_crps(x, y, alpha=self.alpha)))]
```

Notes on 7a:

- The components now return `(B, C)`, so `reduce_to_channel` passes them through. The serial values are unchanged: serial tests pass and checkpoints are unaffected.
- Because the forward value is global on every rank, it also fixes (c): the logged loss, the per-channel losses and the validation loss.
- It relies on the `(l, m)` block layout of `DistributedRealSHT` matching `get_local_slices((lmax, mmax))`. That is the same split torch_harmonics uses, and the same assumption `isotropic_noise` makes.
- It does not cover `_MSELoss`, `_L1Loss` (`_wrap_elementwise`), `FiniteDifferenceCRPS`, or `SpectralWhitening` under W>1. They have the same issue but are not in this config. A general fix would route `_wrap_elementwise` and `StandardLoss.reduce_to_channel` of full-field tensors through the same helper.

### 7b. Corrector: all-reduce in backward for in-graph global means (fixes e)

```diff
--- a/fme/core/distributed/model_torch_distributed.py
+++ b/fme/core/distributed/model_torch_distributed.py
+class _AutogradAllReduceShared(torch.autograd.Function):
+    """All-reduce (sum) whose backward is also an all-reduce (sum).
+
+    For a reduction used *inside* the graph (e.g. a global mean the corrector
+    applies back to each tile): every rank's downstream graph holds only its
+    tile's share of the objective, so the gradient w.r.t. the replicated
+    reduced value must be summed over the spatial group. The identity backward
+    of ``_AutogradAllReduce`` is only right for the final loss reduction.
+    """
+
+    @staticmethod
+    def forward(ctx, input, group):
+        ctx.group = group
+        output = input.clone()
+        torch.distributed.all_reduce(output, group=group)
+        return output
+
+    @staticmethod
+    def backward(ctx, grad_output):
+        grad = grad_output.contiguous().clone()
+        torch.distributed.all_reduce(grad, group=ctx.group)
+        return grad, None
@@ def weighted_mean(
         keepdim: bool = False,
+        in_graph: bool = False,
     ) -> torch.Tensor:
         ...
-        return self.spatial_reduce_sum(local_weighted_sum) / self.spatial_reduce_sum(
-            local_weight_sum
-        )
+        if in_graph and (self._h_size > 1 or self._w_size > 1):
+            numerator = _AutogradAllReduceShared.apply(
+                local_weighted_sum, self._spatial_group
+            )
+        else:
+            numerator = self.spatial_reduce_sum(local_weighted_sum)
+        return numerator / self.spatial_reduce_sum(local_weight_sum)
--- a/fme/core/distributed/distributed.py   (Distributed.weighted_mean: add in_graph, forward it)
--- a/fme/core/gridded_ops.py               (LatLonOperations.area_weighted_mean: add in_graph, forward it)
--- a/fme/core/corrector/atmosphere.py
-        area_weighted_mean = gridded_operations.area_weighted_mean
+        area_weighted_mean = functools.partial(
+            gridded_operations.area_weighted_mean, in_graph=True
+        )
```

For a real PR, `in_graph` should be added to `DistributedBackend.weighted_mean` (base, `NonDistributed`, `TorchDistributed`) and to `GriddedOperations` and `HEALPixOperations.area_weighted_mean`, instead of the `type: ignore` used in the scratch copy. The ocean corrector probably needs the same treatment if it is ever trained spatially. Full scratch diffs, node-local and ephemeral:

- `/tmp/claude-100358/-pscratch-sd-m-mahf708-ace/eeffdfa6-cdca-40be-b352-2aa36d7decd6/scratchpad/loss-fix.diff`
- `/tmp/claude-100358/-pscratch-sd-m-mahf708-ace/eeffdfa6-cdca-40be-b352-2aa36d7decd6/scratchpad/corrector-fix.diff`

The fixes are independent. 7a matters for the objective; 7b is about 1e-3. Either way, add `test_ensemble_loss_spatial.py` (or a trimmed version) as the regression test. It currently fails on (a)/(b)/(c)/(e) and passes with both fixes. Lint was not run: there is no `ruff`/`uv` on PATH on this node.
