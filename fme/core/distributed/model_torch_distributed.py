"""
Distributed backend using PhysicsNeMo's DistributedManager for
spatial (h, w) model parallelism.

Provides the same DistributedBackend interface as TorchDistributed
while splitting ranks across two spatial dimensions (h, w).
Any remaining ranks are used for data parallelism.

Process group hierarchy::

    world
    ├── spatial  (h * w ranks: model parallelism)
    │   ├── h
    │   └── w
    └── data     (residual ranks: data parallelism)
"""

from __future__ import annotations

import logging
import os
from typing import Any, TypeVar

import torch
import torch.distributed
import torch.nn as nn
import torch_harmonics.distributed as thd
from torch.nn import SyncBatchNorm
from torch.nn.parallel import DistributedDataParallel

from fme.core import metrics
from fme.core.device import in_dataloader_worker, using_gpu, using_srun

from ._gloo_patch import patch_gloo_alltoall
from .base import DistributedBackend
from .external.pnd_manager import DistributedManager
from .non_distributed import DummyWrapper
from .parameter_sharding import ParameterShard, named_parameter_shards
from .torch_distributed import _gather_irregular, _rank_metadata_from_env

logger = logging.getLogger(__name__)


T = TypeVar("T")


def _check_world_size_divisible(world_size: int, spatial_size: int) -> None:
    if world_size % spatial_size != 0:
        raise ValueError(
            f"world_size must be divisible by h_size * w_size, "
            f"got world_size={world_size} and h*w={spatial_size}"
        )


def _all_reduce_any_device(
    tensor: torch.Tensor,
    op: torch.distributed.ReduceOp.RedOpType = torch.distributed.ReduceOp.SUM,
    group: torch.distributed.ProcessGroup | None = None,
) -> torch.Tensor:
    """In-place all-reduce that also accepts CPU tensors under NCCL.

    Some aggregators (e.g. video) keep their accumulators on the CPU. Under
    the spatial backend the world always has more than one rank, so their
    data-group reductions reach NCCL even with a single data replica, and NCCL
    rejects CPU tensors ("No backend type associated with device type cpu").
    """
    if (
        tensor.device.type == "cpu"
        and torch.distributed.get_backend(group) == torch.distributed.Backend.NCCL
    ):
        on_device = tensor.to(torch.device("cuda", torch.cuda.current_device()))
        torch.distributed.all_reduce(on_device, op=op, group=group)
        tensor.copy_(on_device)
        return tensor
    torch.distributed.all_reduce(tensor, op=op, group=group)
    return tensor


def _sum_spatial_mean_data_hook(data_size: int):
    """DDP comm hook: all-reduce SUM over the world, divided by ``data_size``.

    DDP's default hook divides by the size of its process group (the whole
    world here), which would average over tiles as well; tiles hold disjoint
    pieces of the same sample, so their gradients must be summed instead.
    """

    # no annotation on `bucket`: under `from __future__ import annotations` it
    # is a string, which DDP's signature check rejects
    def hook(state, bucket):
        buffer = bucket.buffer()
        buffer.div_(data_size)
        fut = torch.distributed.all_reduce(buffer, async_op=True).get_future()
        return fut.then(lambda f: f.value()[0])

    return hook


def _ddp_ignore_names(names: set[str]) -> list[str]:
    """Parameter names for DDP to ignore. Part of DDP spells the root
    module's own parameters with a leading dot, part without.
    """
    return sorted(names | {f".{name}" for name in names if "." not in name})


def _register_reduce_hooks(
    params: list[nn.Parameter], group: torch.distributed.ProcessGroup, divisor: int
) -> None:
    """Sum the gradients of ``params`` over ``group``, divided by
    ``divisor``, in every backward.

    For spatially sharded parameters (DDP over the world cannot handle them:
    their shapes and values differ between spatial ranks): ``group`` holds
    the ranks with the same slice, its data-parallel replicas and any spatial
    ranks the slice is replicated over, and ``divisor`` is the data-parallel
    size. Each all-reduce starts as soon as the parameter's gradient is
    accumulated, overlapping the rest of the backward pass, and the backward
    pass waits for all of them before it returns, as DDP's does. Like the DDP
    sync, it assumes one backward pass per optimizer step.
    """
    pending: list[Any] = []

    def wait_for_all():
        for work in pending:
            work.wait()
        pending.clear()

    def hook(param: torch.Tensor) -> None:
        assert param.grad is not None
        if divisor != 1:
            param.grad.div_(divisor)
        pending.append(
            torch.distributed.all_reduce(param.grad, group=group, async_op=True)
        )
        # queued by every hook, not only the first of a backward: one that
        # raised drops its callbacks and leaves `pending` non-empty
        torch.autograd.Variable._execution_engine.queue_callback(wait_for_all)

    for param in params:
        param.register_post_accumulate_grad_hook(hook)


class _AutogradAllReduce(torch.autograd.Function):
    """Autograd-aware all-reduce (sum) for spatial parallelism.
    Forward: all-reduce (sum) the input across the given process group.
    Backward: identity — gradients pass through without communication.
    This makes ``spatial_reduce_sum`` differentiable so that gradients
    flow correctly through the loss computation path::
        AreaWeightedMSELoss → area_weighted_mean → weighted_mean
            → spatial_reduce_sum (uses this function)
    Without this, the raw ``torch.distributed.all_reduce`` would break
    the autograd graph because it is an in-place, non-differentiable op.
    """

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        group: torch.distributed.ProcessGroup,
    ) -> torch.Tensor:
        output = input.clone()
        torch.distributed.all_reduce(output, group=group)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None


class _AutogradAllReduceShared(torch.autograd.Function):
    """All-reduce (sum) whose backward is also an all-reduce (sum).

    For a reduction used *inside* the graph (e.g. a global mean the corrector
    applies back to each tile): every rank's downstream graph holds only its
    tile's share of the objective, so the gradient w.r.t. the replicated
    reduced value must be summed over the spatial group before flowing back
    into each rank's local inputs. The identity backward of
    ``_AutogradAllReduce`` is only right when the downstream graph is
    replicated, i.e. for the final loss reduction.
    """

    @staticmethod
    def forward(ctx, input, group):
        ctx.group = group
        output = input.clone()
        torch.distributed.all_reduce(output, group=group)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad = grad_output.contiguous().clone()
        torch.distributed.all_reduce(grad, group=ctx.group)
        return grad, None


class ModelTorchDistributed(DistributedBackend):
    """Distributed backend with spatial model parallelism.

    Splits ranks across two spatial dimensions (h, w).
    Residual ranks become data-parallel.

    Parameters
    ----------
    h_size : int
        Number of ranks along the *h* spatial dimension.
    w_size : int
        Number of ranks along the *w* spatial dimension.
    verbose : bool
        Print group layout at init time.
    """

    def __init__(
        self,
        h_size: int = 1,
        w_size: int = 1,
        verbose: bool = False,
    ):
        spatial_size = h_size * w_size
        if in_dataloader_worker():
            self._rank, self._world_size = _rank_metadata_from_env()
            _check_world_size_divisible(self._world_size, spatial_size)
            self._data_size = self._world_size // spatial_size
            self._data_rank = self._rank // spatial_size
            return

        # Keep the NCCL watchdog from std::terminate-ing the rank on a CUDA
        # error observed off the main thread; see the same setting in
        # TorchDistributed.__init__ for the full rationale.
        os.environ.setdefault("TORCH_NCCL_RETHROW_CUDA_ERRORS", "0")
        # Initialize PhysicsNeMo DistributedManager.
        DistributedManager.initialize()
        self._dm = DistributedManager()

        # Build a 3-D (data, h, w) DeviceMesh; data=-1 auto-sizes the
        # data-parallel dimension to absorb all remaining ranks.
        _check_world_size_divisible(self._dm.world_size, spatial_size)
        mesh = self._dm.initialize_mesh(
            mesh_shape=(-1, h_size, w_size),
            mesh_dim_names=("data", "h", "w"),
        )
        if verbose and self._dm.rank == 0:
            logger.info(
                "DeviceMesh initialized: data=%d h=%d w=%d",
                self._dm.world_size // spatial_size,
                h_size,
                w_size,
            )

        # Cache frequently-used values.
        self._rank = self._dm.rank
        self._local_rank = self._dm.local_rank
        self._world_size = self._dm.world_size

        # Derive per-axis process groups and sizes from the mesh.
        self._data_group = self._dm.get_mesh_group(mesh["data"])
        self._h_group = self._dm.get_mesh_group(mesh["h"])
        self._w_group = self._dm.get_mesh_group(mesh["w"])
        self._spatial_group = self._dm.get_mesh_group(mesh["h", "w"])

        self._data_size = torch.distributed.get_world_size(group=self._data_group)
        self._data_rank = torch.distributed.get_rank(group=self._data_group)
        self._h_size = torch.distributed.get_world_size(group=self._h_group)
        self._h_rank = torch.distributed.get_rank(group=self._h_group)
        self._w_size = torch.distributed.get_world_size(group=self._w_group)
        self._w_rank = torch.distributed.get_rank(group=self._w_group)

        # Keep a reference to the spatial sub-mesh for the combined group.
        # The (h, w) 2-D sub-mesh is obtained by slicing the global mesh
        # over the "data" dimension for this rank's data slice.
        self._mesh = mesh

        if using_gpu():
            self._device_id = self._local_rank
            torch.cuda.set_device(self._device_id)

        if not thd.is_initialized():
            thd.init(self._h_group, self._w_group)
        patch_gloo_alltoall()

        logger.info(
            "ModelTorchDistributed initialized: "
            "rank=%d/%d, data_rank=%d/%d, "
            "spatial h=%d w=%d",
            self._rank,
            self._world_size,
            self._data_rank,
            self._data_size,
            self._h_size,
            self._w_size,
        )

    @classmethod
    def is_available(cls) -> bool:
        """Check if the environment supports this backend."""
        return torch.distributed.is_available() and (
            "RANK" in os.environ or using_srun()
        )

    @property
    def rank(self) -> int:
        """Global rank of this process."""
        return self._rank

    @property
    def data_parallel_rank(self) -> int:
        """Rank within the data-parallel group."""
        return self._data_rank

    @property
    def total_ranks(self) -> int:
        """Total number of processes (world size)."""
        return self._world_size

    @property
    def total_data_parallel_ranks(self) -> int:
        """Number of data-parallel ranks."""
        return self._data_size

    def _get_local_spatial_shape(self, h: int, w: int):
        """Compute the local spatial slice for this rank.

        Args:
            h (int): Global height.
            w (int): Global width.

        Returns:
            tuple[slice, slice]: Slices for the local height and width.
        """
        from torch_harmonics.distributed import compute_split_shapes

        h_shapes = compute_split_shapes(h, self._h_size)
        w_shapes = compute_split_shapes(w, self._w_size)
        h_start = sum(h_shapes[: self._h_rank])
        w_start = sum(w_shapes[: self._w_rank])
        return (
            slice(h_start, h_start + h_shapes[self._h_rank]),
            slice(w_start, w_start + w_shapes[self._w_rank]),
        )

    def get_local_slices(self, tensor_shape, data_parallel_dim: int | None = None):
        """Return index slices for this rank's local chunk.

        Slices the ``data_parallel_dim`` across data-parallel ranks and
        the last two dimensions across spatial (h, w) model-parallel ranks.
        """
        if len(tensor_shape) < 2:
            raise ValueError(
                "expected tensor_shape with at least 2 dimensions for "
                "spatial slicing, "
                f"got shape {tensor_shape}"
            )
        if len(tensor_shape) == 2 and data_parallel_dim is not None:
            raise ValueError(
                "data_parallel_dim cannot be specified for 2D tensors, since the "
                "spatial slicing would consume both dimensions; got shape "
                f"{tensor_shape}"
            )
        if data_parallel_dim is not None and (
            data_parallel_dim in (-1, -2) or data_parallel_dim >= len(tensor_shape) - 2
        ):
            raise ValueError(
                "data_parallel_dim must be a non-negative integer less than "
                "the last two dimensions (reserved for spatial slicing), "
                f"got data_parallel_dim={data_parallel_dim} and shape "
                f"{tensor_shape}"
            )
        return_list = [slice(None, None) for _ in tensor_shape]
        if data_parallel_dim is not None:
            n_dp = self.total_data_parallel_ranks
            if tensor_shape[data_parallel_dim] % n_dp != 0:
                raise ValueError(
                    f"expected global data parallel dim to be "
                    f"divisible by data parallel ranks, got "
                    f"global shape {tensor_shape} with "
                    f"{n_dp} data parallel ranks"
                )
            per_rank = tensor_shape[data_parallel_dim] // n_dp
            return_list[data_parallel_dim] = slice(
                self._data_rank * per_rank, (self._data_rank + 1) * per_rank
            )
        # Spatial slicing on the last two dimensions (H, W).
        if len(tensor_shape) >= 2:
            return_list[-2], return_list[-1] = self._get_local_spatial_shape(
                tensor_shape[-2], tensor_shape[-1]
            )
        return tuple(return_list)

    def local_batch_size(self, batch_size: int) -> int:
        """Divide global batch among data-parallel ranks."""
        return batch_size // self.total_data_parallel_ranks

    def reduce_mean(self, tensor: torch.Tensor) -> torch.Tensor | None:
        _all_reduce_any_device(tensor, group=self._data_group)
        return tensor / self.total_data_parallel_ranks

    def reduce_sum(self, tensor: torch.Tensor) -> torch.Tensor | None:
        _all_reduce_any_device(tensor, group=self._data_group)
        return tensor

    def reduce_min(self, tensor: torch.Tensor) -> torch.Tensor | None:
        _all_reduce_any_device(
            tensor, op=torch.distributed.ReduceOp.MIN, group=self._data_group
        )
        return tensor

    def reduce_max(self, tensor: torch.Tensor) -> torch.Tensor | None:
        _all_reduce_any_device(
            tensor, op=torch.distributed.ReduceOp.MAX, group=self._data_group
        )
        return tensor

    def gather(
        self,
        tensor: torch.Tensor,
        gather_list: list[torch.Tensor] | None = None,
    ) -> list[torch.Tensor] | None:
        # NOTE: gather is performed globally, unlike reductions
        if gather_list is None and self.rank == 0:
            gather_list = [tensor] + [
                torch.empty_like(tensor) for _ in range(self.total_ranks - 1)
            ]
        torch.distributed.gather(tensor, gather_list)
        return gather_list

    def data_parallel_gather(
        self,
        tensor: torch.Tensor,
        gather_list: list[torch.Tensor] | None = None,
    ) -> list[torch.Tensor] | None:
        dst = torch.distributed.get_global_rank(self._data_group, 0)
        if gather_list is None and self._data_rank == 0:
            gather_list = [tensor] + [
                torch.empty_like(tensor) for _ in range(self._data_size - 1)
            ]
        torch.distributed.gather(tensor, gather_list, dst=dst, group=self._data_group)
        return gather_list if self._data_rank == 0 else None

    def data_parallel_scatter(
        self,
        tensor: torch.Tensor,
        scatter_list: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        src = torch.distributed.get_global_rank(self._data_group, 0)
        torch.distributed.scatter(tensor, scatter_list, src=src, group=self._data_group)
        return tensor

    def data_parallel_gather_object(self, obj: T) -> list[T] | None:
        dst = torch.distributed.get_global_rank(self._data_group, 0)
        gather_list: list[Any] | None = (
            [None for _ in range(self._data_size)] if self._data_rank == 0 else None
        )
        torch.distributed.gather_object(
            obj, gather_list, dst=dst, group=self._data_group
        )
        return gather_list if self._data_rank == 0 else None

    def data_parallel_broadcast_object(self, obj: T | None) -> T:
        if self._data_rank == 0:
            if obj is None:
                raise ValueError(
                    "Data-parallel root must provide an object to broadcast"
                )
            object_list: list[Any] | None = [obj for _ in range(self._data_size)]
        else:
            object_list = None
        output_list: list[Any] = [None]
        src = torch.distributed.get_global_rank(self._data_group, 0)
        torch.distributed.scatter_object_list(
            output_list, object_list, src=src, group=self._data_group
        )
        return output_list[0]

    def gather_object(self, obj: T) -> list[T] | None:
        """Gather a picklable object globally."""
        gather_list: list[Any] | None = (
            [None for _ in range(self.total_ranks)] if self._rank == 0 else None
        )
        torch.distributed.gather_object(obj, gather_list)
        return gather_list if self._rank == 0 else None

    def scatter_object(self, obj: T | None) -> T:
        """Scatter a picklable object from the root process to all processes."""
        if self._rank == 0:
            if obj is None:
                raise ValueError("Root process must provide an object to scatter")
            object_list = [obj for _ in range(self.total_ranks)]
        else:
            object_list = None
        output_list = [None]
        torch.distributed.scatter_object_list(
            scatter_object_output_list=output_list,
            scatter_object_input_list=object_list,
        )
        return output_list[0]  # type: ignore[return-value]

    # For now, let's just borrow the same gather_irregular implementation
    def gather_irregular(self, tensor: torch.Tensor) -> list[torch.Tensor] | None:
        return _gather_irregular(
            tensor,
            self.reduce_max,
            self.gather,
        )

    @property
    def _device_ids(self) -> list[int] | None:
        if using_gpu():
            return [self._device_id]
        return None

    def wrap_module(self, module: torch.nn.Module) -> torch.nn.Module:
        """Wrap ``module`` so that backward synchronizes its gradients.

        Without spatial parallelism this is DDP over the data process group.

        With it, the ranks of a spatial group hold tiles of the same samples
        (the forward pass communicates inside the distributed SHT/iSHT), so a
        replicated parameter's gradient is the sum over tiles and the mean
        over data-parallel replicas: one bucketed DDP all-reduce over the
        whole world, overlapped with backward, whose comm hook divides by the
        data size. Spatially sharded parameters (see ``parameter_sharding``)
        differ between tiles; DDP ignores them and hooks sync them instead.

        ``broadcast_buffers=False`` is required because the SHT/iSHT layers
        store precomputed Legendre polynomial buffers.  DDP's default
        buffer broadcast modifies these in-place between forward calls,
        which breaks autograd's tensor-version tracking.
        """
        if not any(p.requires_grad for p in module.parameters()):
            return DummyWrapper(module)
        output_device = [self._device_id] if using_gpu() else None
        if self._h_size == 1 and self._w_size == 1:
            return DistributedDataParallel(
                SyncBatchNorm.convert_sync_batchnorm(module),
                device_ids=self._device_ids,
                output_device=output_device,
                process_group=self._data_group,
                broadcast_buffers=False,
            )
        shards = named_parameter_shards(module)
        if shards:
            # DDP over the world would check that they have the same shape on
            # every rank, overwrite them with rank 0's slice and sum their
            # gradients over tiles
            DistributedDataParallel._set_params_and_buffers_to_ignore_for_model(
                module, _ddp_ignore_names(set(shards))
            )
            self._register_shard_grad_hooks(module, shards)
            if not any(
                p.requires_grad
                for name, p in module.named_parameters()
                if name not in shards
            ):
                return DummyWrapper(module)
        wrapped = DistributedDataParallel(
            SyncBatchNorm.convert_sync_batchnorm(module),
            device_ids=self._device_ids,
            output_device=output_device,
            process_group=None,
            broadcast_buffers=False,
            # grads alias the comm buckets instead of being a second
            # full-size copy (8.4 GB/GPU for the 2.1 B 0.25 deg SFNO)
            gradient_as_bucket_view=True,
        )
        wrapped.register_comm_hook(
            state=None, hook=_sum_spatial_mean_data_hook(self._data_size)
        )
        return wrapped

    def _register_shard_grad_hooks(
        self, module: nn.Module, shards: dict[str, ParameterShard]
    ) -> None:
        """Gradient sync of spatially sharded parameters: a sum over the
        spatial axes that do not split a parameter (its slice's replicas),
        and the mean over data-parallel replicas.
        """
        params: dict[tuple[str, ...], list[nn.Parameter]] = {}
        for name, param in module.named_parameters():
            if name not in shards or not param.requires_grad:
                continue
            split_h, split_w = shards[name].split
            axes = tuple(
                axis
                for axis, size, split in (
                    ("h", self._h_size, split_h),
                    ("w", self._w_size, split_w),
                )
                if size > 1 and not split
            )
            params.setdefault(axes, []).append(param)
        for axes, group_params in params.items():
            if axes or self._data_size > 1:
                _register_reduce_hooks(
                    group_params,
                    self._mesh_group(("data",) + axes),
                    divisor=self._data_size,
                )

    def _mesh_group(self, axes: tuple[str, ...]) -> torch.distributed.ProcessGroup:
        """The process group of the ranks that differ only along ``axes``."""
        if len(axes) == 1:
            return {"data": self._data_group, "h": self._h_group, "w": self._w_group}[
                axes[0]
            ]
        return self._dm.get_mesh_group(self._mesh[axes])

    def spatial_process_group(self) -> torch.distributed.ProcessGroup | None:
        """The (h, w) process group, or None without spatial parallelism."""
        if self._h_size > 1 or self._w_size > 1:
            return self._spatial_group
        return None

    def spatial_rank(self) -> tuple[int, int]:
        """This rank's (h, w) coordinates within its spatial group."""
        return (self._h_rank, self._w_rank)

    def spatial_size(self) -> tuple[int, int]:
        return (self._h_size, self._w_size)

    def _spatial_group_coords(self) -> list[tuple[int, int]]:
        """The (h, w) coordinates of each rank of the spatial group, indexed
        by its rank in the group. Collective (once) over the spatial group.
        """
        coords = getattr(self, "_cached_spatial_group_coords", None)
        if coords is None:
            gathered: list[Any] = [None] * (self._h_size * self._w_size)
            torch.distributed.all_gather_object(
                gathered, (self._h_rank, self._w_rank), group=self._spatial_group
            )
            coords = [tuple(c) for c in gathered]
            if coords[0] != (0, 0):
                raise RuntimeError(
                    "expected the spatial group's rank 0 to hold tile (0, 0), "
                    f"got {coords[0]}"
                )
            self._cached_spatial_group_coords = coords
        return coords

    def gather_spatial_to_root(
        self, tensor: torch.Tensor, img_shape: tuple[int, int]
    ) -> torch.Tensor | None:
        if self._h_size == 1 and self._w_size == 1:
            return tensor
        from torch_harmonics.distributed import compute_split_shapes

        h_shapes = compute_split_shapes(img_shape[0], self._h_size)
        w_shapes = compute_split_shapes(img_shape[1], self._w_size)
        # torch.distributed.gather needs equal shapes: pad every tile to the
        # largest one, and crop again on the root
        lead = tuple(tensor.shape[:-2])
        padded = tensor.new_zeros((*lead, max(h_shapes), max(w_shapes)))
        padded[..., : tensor.shape[-2], : tensor.shape[-1]] = tensor
        coords = self._spatial_group_coords()
        is_root = torch.distributed.get_rank(group=self._spatial_group) == 0
        gather_list = [torch.empty_like(padded) for _ in coords] if is_root else None
        torch.distributed.gather(
            padded,
            gather_list,
            dst=torch.distributed.get_global_rank(self._spatial_group, 0),
            group=self._spatial_group,
        )
        if gather_list is None:
            return None
        out = tensor.new_empty((*lead, *img_shape))
        for (h_rank, w_rank), part in zip(coords, gather_list):
            h0, h = sum(h_shapes[:h_rank]), h_shapes[h_rank]
            w0, w = sum(w_shapes[:w_rank]), w_shapes[w_rank]
            out[..., h0 : h0 + h, w0 : w0 + w] = part[..., :h, :w]
        return out

    def barrier(self):
        """Global barrier across all ranks."""
        logger.debug("Barrier on rank %d", self._rank)
        torch.distributed.barrier(device_ids=self._device_ids)

    def spatial_reduce_sum(self, tensor: torch.Tensor) -> torch.Tensor:
        if self._h_size > 1 or self._w_size > 1:
            return _AutogradAllReduce.apply(tensor, self._spatial_group)
        return tensor

    def broadcast_spatial(self, tensor: torch.Tensor) -> torch.Tensor:
        if self._h_size > 1 or self._w_size > 1:
            # Broadcast tensor over spatial group so all tiles get the root's value.
            src = torch.distributed.get_global_rank(self._spatial_group, 0)
            # NCCL does not support bool collectives; round-trip through uint8.
            if tensor.dtype == torch.bool:
                buffer = tensor.to(torch.uint8)
                torch.distributed.broadcast(buffer, src, group=self._spatial_group)
                return buffer.to(torch.bool)
            # Clone first: broadcast writes in place and would overwrite the
            # caller's tensor on non-root ranks.
            buffer = tensor.clone()
            torch.distributed.broadcast(buffer, src, group=self._spatial_group)
            return buffer
        return tensor

    def weighted_mean(
        self,
        data: torch.Tensor,
        weights: torch.Tensor,
        dim: tuple[int, ...],
        keepdim: bool = False,
        in_graph: bool = False,
    ) -> torch.Tensor:
        local_weighted_sum = metrics.weighted_sum(
            data, weights, dim=dim, keepdim=keepdim
        )
        local_weight_sum = weights.expand(data.shape).sum(dim=dim, keepdim=keepdim)
        if in_graph and (self._h_size > 1 or self._w_size > 1):
            numerator = _AutogradAllReduceShared.apply(
                local_weighted_sum, self._spatial_group
            )
        else:
            numerator = self.spatial_reduce_sum(local_weighted_sum)
        return numerator / self.spatial_reduce_sum(local_weight_sum)

    def zonal_mean(self, data: torch.Tensor) -> torch.Tensor:
        # If we have 1 rank along the longitude dimension, quickly return the local mean
        if self._w_size == 1:
            return data.nanmean(dim=-1)

        # Distributed nanmean over the longitude (w) dimension.
        local_sum = data.nansum(dim=-1)
        local_count = (~torch.isnan(data)).to(data.dtype).sum(dim=-1)
        torch.distributed.all_reduce(local_sum, group=self._w_group)
        torch.distributed.all_reduce(local_count, group=self._w_group)
        return local_sum / local_count

    def get_sht(self, nlat, nlon, lmax=None, mmax=None, grid="legendre-gauss"):
        return thd.DistributedRealSHT(
            nlat, nlon, lmax=lmax, mmax=mmax, grid=grid
        ).float()

    def get_isht(self, nlat, nlon, lmax=None, mmax=None, grid="legendre-gauss"):
        return thd.DistributedInverseRealSHT(
            nlat, nlon, lmax=lmax, mmax=mmax, grid=grid
        ).float()

    def get_disco_conv_s2(self, *args, **kwargs) -> nn.Module:
        return thd.DistributedDiscreteContinuousConvS2(*args, **kwargs)

    def shutdown(self):
        # only the log line is guarded: `cleanup` already skips the collective
        # when the manager is not initialized, and resets its shared state either
        # way, so returning early here would leave that state behind.
        if torch.distributed.is_initialized():
            logger.info("Shutting down rank %d", self._rank)
        DistributedManager.cleanup()

    def abort(self):
        if not torch.distributed.is_initialized():
            return
        # covers the manager's mesh subgroups too: passing no group aborts
        # every registered process group. The manager's own state is left
        # behind, but the process is about to exit.
        torch.distributed.distributed_c10d._abort_process_group()
