import contextlib
import logging
import os
from collections.abc import Generator, Iterator
from typing import TypeVar

import torch

from fme.core import metrics
from fme.core.device import get_device, using_gpu

from .base import DistributedBackend
from .model_torch_distributed import ModelTorchDistributed
from .non_distributed import NonDistributed
from .shutdown import abort_and_exit_on_termination, park_if_terminating
from .torch_distributed import TorchDistributed, _gather_irregular

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _release_torch_harmonics_caches() -> None:
    """Drop torch-harmonics' memoized Legendre tables.

    torch-harmonics wraps its quadrature/Legendre precomputations in an LRU
    cache (maxsize 20) that keeps every float64 table for the life of the
    process; each transform keeps its own copy anyway. At 720x1440 (lmax 720)
    one table is ~3 GB and an SFNO builds four, so every rank held ~12 GB of
    dead host memory -- with four ranks per node that OOM-killed a spatially
    parallel training start. Building a transform again only costs the
    recompute.
    """
    import torch_harmonics.legendre
    import torch_harmonics.quadrature

    for module in (torch_harmonics.legendre, torch_harmonics.quadrature):
        for value in vars(module).values():
            for cell in getattr(value, "__closure__", None) or ():
                cached = cell.cell_contents
                if callable(getattr(cached, "cache_clear", None)):
                    cached.cache_clear()


class SpatialParallelismNotImplemented(NotImplementedError):
    """Raised when a code path is incompatible with spatial parallelism."""

    pass


class Distributed:
    """
    A class to represent the distributed concerns for FME training.

    This should generally be initialized first, before any pytorch objects.
    This is important because it sets global variables such as the CUDA
    device for the local rank, which is used when initializing pytorch objects.

    This class uses the
    [Singleton pattern](https://en.wikipedia.org/wiki/Singleton_pattern) and should
    be initialized through get_instance. This pattern allows easy access to global
    variables without having to pass them around, and lets us put the initialization
    for this global state in the same place as the routines that use it.
    """

    _entered: bool = False

    def __init__(
        self,
        force_non_distributed: bool = False,
        spatial_parallelism: tuple[int, int] | None = None,
        verbose: bool = False,
    ):
        if force_non_distributed:
            self._distributed: DistributedBackend = NonDistributed()
        elif spatial_parallelism is not None:
            h_size, w_size = spatial_parallelism
            self._distributed = ModelTorchDistributed(
                h_size=h_size,
                w_size=w_size,
                verbose=verbose,
            )
        elif TorchDistributed.is_available():
            self._distributed = TorchDistributed()
        else:
            self._distributed = NonDistributed()
        self._seed = 0
        self._force_non_distributed = force_non_distributed

    @classmethod
    @contextlib.contextmanager
    def context(cls, handle_signals: bool = True) -> Generator[None, None, None]:
        """
        Context manager for initializing and shutting down the distributed backend.

        This should generally be used at the top level of the training script to
        wrap the entire training process, to ensure proper initialization and
        shutdown of the distributed backend.

        Termination signals are handled for the lifetime of the context, so that
        a preempted job aborts its communicators before exiting instead of
        dropping its NVLink peers. See `fme.core.distributed.shutdown`. A
        single-process job has no peers to protect and keeps the default signal
        behavior (Ctrl-C raises KeyboardInterrupt and unwinds normally).

        Args:
            handle_signals: Install the termination listener. Pass `False` only
                when something else owns the process's response to SIGTERM and
                SIGINT -- the test suite, which wraps the whole session in this
                context and needs Ctrl-C to keep interrupting pytest rather than
                becoming an abort-and-exit. Every entrypoint wants the default.
        """
        if cls._entered:
            raise RuntimeError("Nested Distributed.context() is not supported.")
        cls._entered = True
        instance = cls.get_instance()
        try:
            with contextlib.ExitStack() as stack:
                if handle_signals and instance.is_distributed():
                    stack.enter_context(
                        abort_and_exit_on_termination(
                            instance.abort, drain=instance._drain_local_work
                        )
                    )
                try:
                    yield
                except BaseException:
                    # exit immediately to avoid hanging other ranks
                    # the OS should clean up resources based on the non-zero exit
                    raise  # re-raise the exception to avoid masking it
                else:  # if no exception is raised, let root finish cleanup
                    # inside the stack: the teardown is itself a collective, so
                    # it must run while the listener still protects the process
                    instance.shutdown()
        finally:
            cls._entered = False

    @classmethod
    def get_instance(cls) -> "Distributed":
        """
        Get the singleton instance of the Distributed class.

        The backend can be overridden at the session level via environment
        variables, which is useful for switching between backends

        - ``FME_DISTRIBUTED_BACKEND=torch`` to force :class:`TorchDistributed`
          (the default when ``torch.distributed`` is available)
        - ``FME_DISTRIBUTED_BACKEND=model`` to force
          :class:`ModelTorchDistributed`. Requires ``FME_DISTRIBUTED_H`` and
          ``FME_DISTRIBUTED_W`` to also be set.
        - ``FME_DISTRIBUTED_BACKEND=none`` to force
          :class:`NonDistributed`
        """
        global singleton
        if singleton is None:
            backend_env = os.environ.get("FME_DISTRIBUTED_BACKEND")
            if backend_env == "model":
                h = int(os.environ["FME_DISTRIBUTED_H"])
                w = int(os.environ["FME_DISTRIBUTED_W"])
                singleton = cls(spatial_parallelism=(h, w))
            elif backend_env == "none":
                singleton = cls(force_non_distributed=True)
            elif backend_env == "torch":
                singleton = cls()
            elif backend_env is not None:
                raise ValueError(
                    f"Unknown FME_DISTRIBUTED_BACKEND value '{backend_env}'. "
                    "Valid values: 'torch', 'model', 'none'."
                )
            else:
                singleton = cls()
        if not cls._entered and singleton.world_size > 1:
            raise RuntimeError(
                "Distributed.get_instance() called before entering context. "
                "Please use Distributed.context() to wrap the training process, "
                "or ensure that get_instance() is only called within the context."
            )
        return singleton

    @classmethod
    @contextlib.contextmanager
    def force_non_distributed(cls) -> Iterator["Distributed"]:
        """
        Force the distributed singleton to be in non-distributed mode.
        """
        try:
            with cls.replace_backend(NonDistributed()) as instance:
                original_force_non_distributed = instance._force_non_distributed
                instance._force_non_distributed = True
                yield instance
        finally:
            instance._force_non_distributed = original_force_non_distributed

    @classmethod
    @contextlib.contextmanager
    def replace_backend(cls, backend: DistributedBackend) -> Iterator["Distributed"]:
        """
        Force the distributed singleton to be in non-distributed mode.
        """
        instance = cls.get_instance()
        original_backend = instance._distributed
        try:
            instance._distributed = backend
            yield instance
        finally:
            instance._distributed = original_backend

    @property
    def rank(self) -> int:
        """
        Get the global rank of this process.
        """
        return self._distributed.rank

    @property
    def data_parallel_rank(self) -> int:
        """
        Get the data parallel rank of this process.

        In the context of distributed learning, this is the "batch"
        rank of this process.
        """
        return self._distributed.data_parallel_rank

    @property
    def total_data_parallel_ranks(self) -> int:
        """
        Get the total number of data parallel ranks.

        This is the number of parallel splits along the "batch" dimension.
        """
        return self._distributed.total_data_parallel_ranks

    @property
    def world_size(self) -> int:
        """
        Get the total number of processes.
        """
        return self._distributed.total_ranks

    @property
    def has_spatial_parallelism(self) -> bool:
        """Whether spatial co-ranks exist (world_size !=
        total_data_parallel_ranks), meaning each rank holds only a shard of
        the model's spatial state rather than a full copy.
        """
        return self.world_size != self.total_data_parallel_ranks

    def get_spatial_process_group(self) -> torch.distributed.ProcessGroup | None:
        """The process group of this rank's spatial co-ranks, or None when
        there is no spatial parallelism.

        Every rank of a spatial group holds the same parameters and, after the
        gradient all-reduce, the same gradients, so per-parameter work (e.g.
        optimizer state, see ``OptimizationConfig.shard_optimizer_state``) can
        be divided among them. Each data-parallel replica has its own group.
        """
        if not self.has_spatial_parallelism:
            return None
        group = self._distributed.spatial_process_group()
        if group is None:
            raise RuntimeError(
                f"{type(self._distributed).__name__} has spatial co-ranks but "
                "does not expose their process group."
            )
        return group

    @property
    def spatial_rank(self) -> tuple[int, int]:
        """This rank's (h, w) coordinates within its spatial group, (0, 0)
        without spatial parallelism. ``get_local_slices`` splits the last two
        dimensions of a shape by these coordinates.
        """
        return self._distributed.spatial_rank()

    @property
    def spatial_size(self) -> tuple[int, int]:
        """The number of (h, w) tiles each field is split into; (1, 1)
        without spatial parallelism.
        """
        return self._distributed.spatial_size()

    def require_no_spatial_parallelism(self, msg: str) -> None:
        """Raise if spatial parallelism is active.

        Use this to guard code paths that are known to be incorrect
        when spatial co-ranks exist.
        """
        if self.has_spatial_parallelism:
            raise SpatialParallelismNotImplemented(msg)

    def get_sampler(
        self,
        dataset: torch.utils.data.Dataset,
        shuffle: bool,
        drop_last: bool = False,
    ) -> torch.utils.data.Sampler:
        return torch.utils.data.DistributedSampler(
            dataset,
            shuffle=shuffle,
            num_replicas=self._distributed.total_data_parallel_ranks,
            rank=self._distributed.data_parallel_rank,
            seed=self._seed,
            drop_last=drop_last,
        )

    def local_batch_size(self, batch_size: int) -> int:
        """
        Get the local batch size for the current process.
        """
        return self._distributed.local_batch_size(batch_size)

    def reduce_mean(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Reduce a tensor representing a mean across all processes.

        Whether the tensor represents a mean is important because to reduce a mean,
        we must divide by the number of processes. To reduce a sum, we must not.

        Modifies the input tensor in-place as a side effect.
        """
        return self._distributed.reduce_mean(tensor)

    def get_local_slices(
        self,
        tensor_shape,
        data_parallel_dim: int | None = None,
    ):
        """
        Gets the slice corresponding to the current rank within a global tensor_shape.

        Args:
            tensor_shape: the shape of the global tensor, which may or may not contain
                a data parallel (batch) dimension. Assume last dims are (H, W).
            data_parallel_dim: the index of the data parallel dimension, if it exists.
                by default, assumes the tensor does not have a data parallel dimension.
        """
        if data_parallel_dim is not None and (
            tensor_shape[data_parallel_dim] % self.total_data_parallel_ranks != 0
        ):
            raise ValueError(
                "expected global data parallel dim to be divisible by data parallel "
                f"ranks, got global shape {tensor_shape} with "
                f"{self.total_data_parallel_ranks} data parallel ranks"
            )
        return self._distributed.get_local_slices(
            tensor_shape, data_parallel_dim=data_parallel_dim
        )

    def reduce_sum(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Reduce a tensor representing a sum across all processes.

        Whether the tensor represents a mean is important because to reduce a mean,
        we must divide by the number of processes. To reduce a sum, we must not.

        Modifies the input tensor in-place as a side effect.
        """
        return self._distributed.reduce_sum(tensor)

    def reduce_min(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Reduce a tensor representing a min across all processes.

        Modifies the input tensor in-place as a side effect.
        """
        return self._distributed.reduce_min(tensor)

    def reduce_max(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Reduce a tensor representing a max across all processes.

        Modifies the input tensor in-place as a side effect.
        """
        return self._distributed.reduce_max(tensor)

    def gather_object(self, obj: T) -> list[T] | None:
        """
        Gather a picklable object from all processes to the root process.

        Args:
            obj: The object to gather.

        Returns:
            A list of objects, where the i-th element is the object
                from the i-th process.
        """
        return self._distributed.gather_object(obj)

    def scatter_object(self, obj: T | None) -> T:
        """
        Scatter a picklable object from the root process to all processes.

        Args:
            obj: The object to scatter.

        Returns:
            The object scattered from the root process.
        """
        scattered = self._distributed.scatter_object(obj)
        if scattered is None:  # necessary check due to mypy limitations
            raise RuntimeError("scatter_object returned None")
        return scattered

    def gather(self, tensor: T, gather_list: list[T] | None = None) -> list[T] | None:
        """
        Gather a tensor from all processes to the root process.

        Note: tensor shape is assumed to be equal across all processes; data will
            reshaped/filled/dropped to coerce non-root tensors to the shape
            of the root tensor if not. To avoid this behavior, use
            "gather_irregular" instead.

        Args:
            tensor: The tensor to gather.
            gather_list: A list of tensor buffers to gather into,
                one for each rank.

        Returns:
            A list of tensors, where the i-th element is the tensor
                from the i-th process.
        """
        return self._distributed.gather(tensor, gather_list=gather_list)

    def data_parallel_gather(
        self,
        tensor: torch.Tensor,
        gather_list: list[torch.Tensor] | None = None,
    ) -> list[torch.Tensor] | None:
        """Gather a tensor from all data-parallel ranks to the data-parallel root.

        Under spatial parallelism each spatial position gathers independently
        across its data-parallel co-ranks.

        Returns:
            A list of tensors on the data-parallel root, ``None`` on other
            data-parallel ranks.
        """
        return self._distributed.data_parallel_gather(tensor, gather_list=gather_list)

    def data_parallel_scatter(
        self,
        tensor: torch.Tensor,
        scatter_list: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Scatter tensors from the data-parallel root to all data-parallel ranks.

        Inverse of ``data_parallel_gather``. All ranks provide ``tensor`` (a
        pre-allocated receive buffer). The data-parallel root also provides
        ``scatter_list`` (one tensor per data-parallel rank).

        Returns:
            ``tensor``, populated with this rank's shard.
        """
        return self._distributed.data_parallel_scatter(
            tensor, scatter_list=scatter_list
        )

    def data_parallel_gather_object(self, obj: T) -> list[T] | None:
        """Gather a picklable object from all data-parallel ranks to the
        data-parallel root.
        """
        return self._distributed.data_parallel_gather_object(obj)

    def data_parallel_broadcast_object(self, obj: T | None) -> T:
        """Broadcast a picklable object from the data-parallel root to all
        data-parallel ranks.
        """
        result = self._distributed.data_parallel_broadcast_object(obj)
        if result is None:
            raise RuntimeError("data_parallel_broadcast_object returned None")
        return result

    def is_data_parallel_root(self) -> bool:
        """Whether this rank is the root of its data-parallel group."""
        return self._distributed.data_parallel_rank == 0

    def gather_global(
        self, tensor: torch.Tensor, global_shape, data_parallel_dim: int = 0
    ) -> torch.Tensor | None:
        """
        Gathers tensor data into a single tensor with the data from all ranks.

        Args:
            tensor: the tensor data to gather
            global_shape: the shape of the tensor containing data from all ranks
            data_parallel_dim: the dimension in global_shape corresponding to the
                data parallel (or "batch") dimension
        """
        if global_shape[data_parallel_dim] % self.total_data_parallel_ranks != 0:
            raise ValueError(
                "expected global data parallel dim to be divisible by data parallel "
                f"ranks, got global_shape {global_shape} with "
                f"{self.total_data_parallel_ranks} data parallel ranks"
            )
        local_slices = self.get_local_slices(
            global_shape, data_parallel_dim=data_parallel_dim
        )
        gathered_local_slices = self.gather_object(local_slices)
        gathered_tensors = self.gather(tensor.contiguous())
        if self.is_root():
            if gathered_local_slices is None or gathered_tensors is None:
                raise RuntimeError(
                    "gather_object or gather returned None on root process"
                )
            gathered_global = torch.zeros(
                *global_shape, dtype=tensor.dtype, device=tensor.device
            )
            for i in range(len(gathered_local_slices)):
                gathered_global[gathered_local_slices[i]] = gathered_tensors[i]
        else:
            gathered_global = None
        return gathered_global

    def gather_irregular(
        self,
        tensor: torch.Tensor,
    ) -> list[torch.Tensor] | None:
        """
        Gather a tensor from all processes to the root process. The rank tensors
        may have diferent dimension lengths, but must have the same number of
        dimensions.

        Args:
            tensor: The tensor to gather.

        Returns:
            A list of tensors of consistent shape, where the i-th element is the tensor
                from the i-th process.
        """
        return self._distributed.gather_irregular(tensor)

    def data_parallel_gather_irregular(
        self, tensor: torch.Tensor
    ) -> list[torch.Tensor] | None:
        """Like ``gather_irregular``, but over the data-parallel ranks only,
        for per-sample data that every spatial co-rank holds a copy of (e.g. a
        global or regional mean): each sample is gathered once, not once per
        spatial tile. Without spatial parallelism this is ``gather_irregular``.

        Returns:
            A list with the tensor of each data-parallel rank on the root
            process, None on the others.
        """
        gathered = _gather_irregular(tensor, self.reduce_max, self.data_parallel_gather)
        return gathered if self.is_root() else None

    def is_root(self) -> bool:
        """
        Returns True if this process is the root process.
        """
        return self._distributed.rank == 0

    def is_distributed(self) -> bool:
        """
        Returns True if this process is running in a distributed context
        with more than 1 worker.
        """
        return self._distributed.total_ranks > 1

    def wrap_module(self, module: torch.nn.Module) -> torch.nn.Module:
        """
        Wrap a model with DistributedDataParallel if running in a distributed context.
        """
        return self._distributed.wrap_module(module)

    def barrier(self):
        """
        Wait for all processes to reach this point.
        """
        return self._distributed.barrier()

    def set_seed(self, seed: int):
        """
        Set the random seed.
        """
        self._seed = seed

    def get_seed(self) -> int:
        """
        Get the random seed.
        """
        return self._seed

    def get_sht(self, nlat, nlon, lmax=None, mmax=None, grid="legendre-gauss"):
        sht = self._distributed.get_sht(nlat, nlon, lmax, mmax, grid)
        _release_torch_harmonics_caches()
        return sht

    def get_isht(self, nlat, nlon, lmax=None, mmax=None, grid="legendre-gauss"):
        isht = self._distributed.get_isht(nlat, nlon, lmax, mmax, grid)
        _release_torch_harmonics_caches()
        return isht

    def get_disco_conv_s2(self, *args, **kwargs):
        return self._distributed.get_disco_conv_s2(*args, **kwargs)

    def spatial_reduce_sum(self, tensor: torch.Tensor) -> torch.Tensor:
        return self._distributed.spatial_reduce_sum(tensor)

    def broadcast_spatial(self, tensor: torch.Tensor) -> torch.Tensor:
        return self._distributed.broadcast_spatial(tensor)

    def weighted_mean(
        self,
        data: torch.Tensor,
        weights: torch.Tensor,
        dim: tuple[int, ...],
        keepdim: bool = False,
        in_graph: bool = False,
    ) -> torch.Tensor:
        return self._distributed.weighted_mean(
            data, weights, dim=dim, keepdim=keepdim, in_graph=in_graph
        )

    def zonal_mean(self, data: torch.Tensor) -> torch.Tensor:
        return self._distributed.zonal_mean(data)

    def gradient_magnitude_percent_diff(
        self,
        truth: torch.Tensor,
        predicted: torch.Tensor,
        weights: torch.Tensor,
        dim: tuple[int, ...],
        img_shape: tuple[int, int],
    ) -> torch.Tensor:
        truth = self.gather_spatial_tensor(truth, img_shape)
        predicted = self.gather_spatial_tensor(predicted, img_shape)
        weights = self.gather_spatial_tensor(weights, img_shape)
        return metrics.gradient_magnitude_percent_diff(
            truth, predicted, weights=weights, dim=dim
        )

    def get_spatial_tile(
        self, img_shape: tuple[int, int]
    ) -> tuple[slice, slice] | None:
        """This rank's ``(h, w)`` slices of a global ``img_shape``, or None if
        this rank holds the whole grid (no spatial split).

        The slices are the ones ``get_local_slices`` and ``scatter_spatial``
        use, as plain ``slice(start, stop)`` objects that can be handed to a
        dataset (and through it to DataLoader workers) to load only the tile.
        """
        h_slice, w_slice = self.get_local_slices(img_shape)
        tile = (
            slice(*h_slice.indices(img_shape[0])[:2]),
            slice(*w_slice.indices(img_shape[1])[:2]),
        )
        if tile == (slice(0, img_shape[0]), slice(0, img_shape[1])):
            return None
        return tile

    def scatter_spatial(
        self, data: dict[str, torch.Tensor], img_shape: tuple[int, int]
    ) -> dict[str, torch.Tensor]:
        """Slice global tensors to the local spatial chunk for this rank.

        Tensors that already have the local chunk's shape, because they were
        loaded as this rank's tile (see ``get_spatial_tile``), are returned
        as they are. Any other shape is an error.
        """
        slices = self.get_local_slices(img_shape)
        global_hw = tuple(img_shape)
        local_hw = tuple(len(range(*s.indices(n))) for s, n in zip(slices, img_shape))
        out = {}
        for k, v in data.items():
            hw = tuple(v.shape[-2:])
            if hw == global_hw:
                out[k] = v[(..., *slices)].contiguous()
            elif hw == local_hw:
                out[k] = v
            else:
                raise ValueError(
                    f"Cannot scatter {k} of shape {tuple(v.shape)}: its last two "
                    f"dimensions match neither the global {global_hw} nor this "
                    f"rank's local {local_hw} spatial shape."
                )
        return out

    def gather_spatial(
        self, data: dict[str, torch.Tensor], img_shape: tuple[int, int]
    ) -> dict[str, torch.Tensor]:
        """Gather local spatial chunks back to global tensors via all-reduce."""
        return {k: self.gather_spatial_tensor(v, img_shape) for k, v in data.items()}

    def gather_spatial_tensor(
        self, tensor: torch.Tensor, img_shape: tuple[int, int]
    ) -> torch.Tensor:
        """Reassemble a spatially-sharded tensor on every rank via all-reduce.

        Args:
            tensor: Local spatial shard.
            img_shape: Global ``(H, W)`` spatial dimensions.
        """
        if img_shape == tensor.shape[-2:]:
            return tensor
        global_shape = (*tensor.shape[:-2], *img_shape)
        slices = self.get_local_slices(img_shape)
        buf = torch.zeros(global_shape, dtype=tensor.dtype, device=tensor.device)
        buf[(..., *slices)] = tensor
        return self.spatial_reduce_sum(buf)

    def is_spatial_root(self) -> bool:
        """Whether this rank holds the (0, 0) tile of its spatial group, i.e.
        receives ``gather_spatial_tile_to_root``'s result. Every rank is its
        own spatial root without spatial parallelism.
        """
        return self.spatial_rank == (0, 0)

    def _global_spatial_shape(self, local_hw: tuple[int, int]) -> tuple[int, int]:
        """The global ``(H, W)`` of a field of which this rank holds the
        ``local_hw`` tile. Collective over the spatial group.
        """
        h_size, w_size = self._distributed.spatial_size()
        sizes = torch.tensor(local_hw, dtype=torch.int64, device=get_device())
        # each tile row's height is counted once per w-rank, and each tile
        # column's width once per h-rank
        total = self.spatial_reduce_sum(sizes).tolist()
        img_shape = (total[0] // w_size, total[1] // h_size)
        local_slices = self.get_local_slices(img_shape)
        expected = tuple(
            len(range(*s.indices(n))) for s, n in zip(local_slices, img_shape)
        )
        if expected != tuple(local_hw):
            raise ValueError(
                f"A tile of shape {tuple(local_hw)} is not this rank's tile of "
                f"any global field: the tiles add up to {img_shape}, of which "
                f"this rank's tile is {expected}."
            )
        return img_shape

    def gather_spatial_tile(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reassemble, on every rank, the global field of which this rank
        holds its spatial tile in the last two dimensions.

        Use it to turn a tile-local map (e.g. a time mean) into the global
        map for plotting or saving. Area-weighted reductions must still be
        computed on the tile (``GriddedOperations`` reduces across tiles).
        The global shape is inferred from the tile sizes. The result is on
        the input's device. Collective over the spatial group; identity
        without spatial parallelism.
        """
        if not self.has_spatial_parallelism:
            return tensor
        img_shape = self._global_spatial_shape((tensor.shape[-2], tensor.shape[-1]))
        device = tensor.device
        return self.gather_spatial_tensor(tensor.to(get_device()), img_shape).to(device)

    def gather_spatial_tile_to_root(self, tensor: torch.Tensor) -> torch.Tensor | None:
        """Like ``gather_spatial_tile``, but the global field is assembled on
        the spatial root only (see ``is_spatial_root``); other ranks get
        None. Only the root holds a global-size buffer, so prefer this for
        large data that only the root uses, e.g. for writing files.
        """
        if not self.has_spatial_parallelism:
            return tensor
        img_shape = self._global_spatial_shape((tensor.shape[-2], tensor.shape[-1]))
        device = tensor.device
        gathered = self._distributed.gather_spatial_to_root(
            tensor.to(get_device()), img_shape
        )
        return None if gathered is None else gathered.to(device)

    def gather_spatial_lat(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reassemble, on every rank, a profile whose last dimension is this
        rank's latitude band and which is the same on every rank of a tile
        row (e.g. a zonal mean). Collective over the spatial group; identity
        without spatial parallelism.
        """
        if not self.has_spatial_parallelism:
            return tensor
        # each w-rank contributes a 1-wide column of the same profile
        return self.gather_spatial_tile(tensor.unsqueeze(-1))[..., 0]

    def shutdown(self):
        return self._distributed.shutdown()

    def abort(self):
        """Locally release the backend's communicators so this process can exit
        without faulting its peers. Not collective, unlike `shutdown`, and safe
        to call from a non-main thread.
        """
        return self._distributed.abort()

    def park_if_terminating(self):
        """
        If we've received a termination signal, stop work and wait for termination.

        The batch loops should call this once per batch or window. It returns
        immediately unless a termination signal has arrived; if one has, it
        drains this rank's device work, blocks, and the process exits from
        the termination listener's thread. See
        `fme.core.distributed.shutdown.park_if_terminating`.
        """
        park_if_terminating()

    def _drain_local_work(self):
        """Block until every kernel this rank has enqueued -- collectives
        included -- has completed. On CPU there is nothing asynchronous to
        drain.
        """
        if using_gpu():
            torch.cuda.synchronize()


singleton: Distributed | None = None
