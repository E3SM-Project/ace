"""Parameters that each rank of a spatial group holds only a slice of.

Under spatial parallelism (``FME_DISTRIBUTED_BACKEND=model``) some parameters
are only ever used on a slice that depends on the rank: a spectral filter
weight on the rank's range of spherical-harmonic degrees ``l``, a positional
embedding on the rank's (lat, lon) tile. A module can store just that slice
with :func:`shard_parameter`. Its optimizer and EMA state are then rank-local
too, and its gradient needs no sum over the tiles that hold other slices:
only over the ranks holding the same slice, if the slice is replicated along
a spatial axis (a spectral weight, split along l by the polar ranks, is used
by each azimuth rank on its own range of m), and the mean over data-parallel
replicas.

The serialized form stays that of an unsharded (serial) model:

- loading: ``module.load_state_dict`` accepts the full tensor and keeps this
  rank's slice (a tensor that already has the local shape is taken as is, so
  in-memory round trips through ``state_dict`` work);
- saving: inside :func:`gathered_parameter_shards`, which every rank enters
  together, ``module.state_dict()`` (and ``EMATracker.get_state()``, and the
  optimizer state consolidated by ``Optimization.consolidated_state``) return
  full tensors, gathered over the spatial group. Outside it they return this
  rank's slices.

The slices are the ones :meth:`Distributed.get_local_slices` gives, so they
handle uneven splits.
"""

import contextlib
import dataclasses
import math
from collections.abc import Iterable, Iterator, Mapping
from typing import Any

import torch
from torch import nn

_SHARDS_ATTR = "_fme_parameter_shards"

# destination ranks of the active gathered_parameter_shards contexts, innermost
# last; None means every rank
_gather_destinations: list[int | None] = []


@dataclasses.dataclass(frozen=True)
class ParameterShard:
    """How this rank's slice of a parameter relates to the full tensor.

    Attributes:
        global_shape: Shape of the full (serial) parameter.
        index: This rank's slice of the full tensor, one slice per dimension.
        split: Whether the slices differ along the polar (h) and the
            azimuth (w) spatial axis. Along an axis that does not split it,
            the ranks hold the same slice, and must sum its gradient.
        owner: Whether this rank contributes its slice when the full tensor is
            assembled. Ranks that hold the same slice (e.g. all ranks of an
            azimuth group, for a tensor split only along the polar axis) leave
            it to one of them.
    """

    global_shape: tuple[int, ...]
    index: tuple[slice, ...]
    split: tuple[bool, bool]
    owner: bool

    @property
    def local_shape(self) -> tuple[int, ...]:
        return tuple(
            len(range(*s.indices(n))) for s, n in zip(self.index, self.global_shape)
        )

    def take(self, full: torch.Tensor) -> torch.Tensor:
        """This rank's slice of a full tensor, as a new contiguous tensor."""
        return full[self.index].clone(memory_format=torch.contiguous_format)

    def localize(self, value: torch.Tensor) -> torch.Tensor:
        """This rank's slice of a full tensor; a local slice passes through.

        Any other shape also passes through, so that the caller's own shape
        check reports it.
        """
        if tuple(value.shape) == self.global_shape:
            return self.take(value)
        return value


def spatial_shard(
    global_shape: tuple[int, ...] | torch.Size,
    split_shape: tuple[int, int],
    dims: tuple[int | None, int | None],
) -> ParameterShard | None:
    """The shard of a tensor whose dimensions are split like a spatial array.

    Args:
        global_shape: Shape of the full tensor.
        split_shape: The 2D shape whose local slices
            (:meth:`Distributed.get_local_slices`) define the split, e.g. the
            image shape for a positional embedding, or ``(lmax, mmax)`` for a
            spectral weight.
        dims: The tensor dimensions split like the first (polar, h) and the
            second (azimuth, w) entry of ``split_shape``; None leaves the
            tensor whole along that axis (it is then replicated over the
            ranks of that axis).

    Returns:
        This rank's shard, or None when the layout leaves the tensor whole on
        every rank (e.g. no spatial parallelism), in which case it should stay
        an ordinary, replicated parameter.
    """
    from fme.core.distributed import Distributed

    dist = Distributed.get_instance()
    global_shape = tuple(global_shape)
    local_slices = dist.get_local_slices(split_shape)
    index = [slice(0, n) for n in global_shape]
    for axis, (dim, local_slice) in enumerate(zip(dims, local_slices)):
        if dim is None:
            continue
        if global_shape[dim] != split_shape[axis]:
            raise ValueError(
                f"dimension {dim} of shape {global_shape} does not match the "
                f"split shape {split_shape} along axis {axis}"
            )
        start, stop, _ = local_slice.indices(global_shape[dim])
        index[dim] = slice(start, stop)
    if all(s == slice(0, n) for s, n in zip(index, global_shape)):
        return None
    h_rank, w_rank = dist.spatial_rank
    owner = (dims[0] is not None or h_rank == 0) and (
        dims[1] is not None or w_rank == 0
    )
    return ParameterShard(
        global_shape=global_shape,
        index=tuple(index),
        split=(dims[0] is not None, dims[1] is not None),
        owner=owner,
    )


def shard_parameter(module: nn.Module, name: str, shard: ParameterShard | None):
    """Replace ``module.<name>`` by this rank's slice of it.

    The module's ``state_dict``/``load_state_dict`` keep the full tensor's
    format (see the module docstring). Initialization therefore equals
    slicing an unsharded model built from the same random state. Does nothing
    when ``shard`` is None.

    Must be called after any other ``load_state_dict`` pre-hook of the module
    that rewrites legacy checkpoint formats, so that those see full tensors.
    """
    if shard is None:
        return
    full = module._parameters[name]
    if full is None:
        raise ValueError(f"{name} is not a parameter of {type(module).__name__}")
    if tuple(full.shape) != shard.global_shape:
        raise ValueError(
            f"cannot shard {name} of shape {tuple(full.shape)} as a tensor of "
            f"shape {shard.global_shape}"
        )
    local = nn.Parameter(shard.take(full.detach()), requires_grad=full.requires_grad)
    setattr(module, name, local)
    shards: dict[str, ParameterShard] | None = module.__dict__.get(_SHARDS_ATTR)
    if shards is None:
        shards = {}
        setattr(module, _SHARDS_ATTR, shards)
        module.register_load_state_dict_pre_hook(_localize_loaded_state)
        module.register_state_dict_post_hook(_gather_saved_state)
    shards[name] = shard


def _module_shards(module: nn.Module) -> Mapping[str, ParameterShard]:
    return module.__dict__.get(_SHARDS_ATTR, {})


def named_parameter_shards(module: nn.Module) -> dict[str, ParameterShard]:
    """The shards of a module's sharded parameters, by parameter name."""
    out = {}
    for module_name, submodule in module.named_modules():
        for name, shard in _module_shards(submodule).items():
            out[f"{module_name}.{name}" if module_name else name] = shard
    return out


def parameter_shards(
    modules: Iterable[nn.Module],
) -> dict[nn.Parameter, ParameterShard]:
    """The shards of the modules' sharded parameters, by parameter."""
    out = {}
    for module in modules:
        for submodule in module.modules():
            for name, shard in _module_shards(submodule).items():
                out[submodule._parameters[name]] = shard
    return out


def get_parameter_shard(module: nn.Module, name: str) -> ParameterShard | None:
    """The shard of ``module``'s parameter ``name`` (a dotted path), or None if
    it is not sharded or does not exist.
    """
    module_name, _, parameter_name = name.rpartition(".")
    try:
        submodule = module.get_submodule(module_name) if module_name else module
    except AttributeError:
        return None
    return _module_shards(submodule).get(parameter_name)


def has_parameter_shards(modules: Iterable[nn.Module]) -> bool:
    return len(parameter_shards(modules)) > 0


@contextlib.contextmanager
def gathered_parameter_shards(dst: int | None = 0) -> Iterator[None]:
    """Within this context, state reads return full sharded parameters.

    Collective: every rank must enter it and read the same states in the same
    order (one gather over the spatial group per sharded tensor). With
    ``dst=None`` the ranks of all data-parallel replicas take part, each
    within its own spatial group; otherwise only the spatial group of ``dst``.

    Args:
        dst: The global rank that receives the full tensors (on the CPU); the
            other ranks get their local slices back. None: every rank receives
            them.
    """
    _gather_destinations.append(dst)
    try:
        yield
    finally:
        _gather_destinations.pop()


def parameter_shards_gathered() -> bool:
    """Whether a :func:`gathered_parameter_shards` context is active."""
    return len(_gather_destinations) > 0


def gather_shard(value: torch.Tensor, shard: ParameterShard) -> torch.Tensor:
    """The full tensor of a sharded parameter (or of state shaped like it).

    Inside :func:`gathered_parameter_shards` this is collective and returns
    the full tensor on the CPU of the destination rank(s); otherwise, and on
    other ranks, it returns ``value``. With a destination rank, only its
    spatial group takes part, and only the destination allocates the full
    tensor.
    """
    if not _gather_destinations:
        return value
    from fme.core.distributed import Distributed

    if tuple(value.shape) != shard.local_shape:
        raise ValueError(
            f"expected a local shard of shape {shard.local_shape}, "
            f"got {tuple(value.shape)}"
        )
    dist = Distributed.get_instance()
    dst = _gather_destinations[-1]
    group = dist.get_spatial_process_group()
    with torch.no_grad():
        if group is None:
            # no other tile: the full tensor holds only this rank's slice
            if dst is not None and dist.rank != dst:
                return value
            full = torch.zeros(shard.global_shape, dtype=value.dtype)
            if shard.owner:
                full[shard.index] = value
            return full
        if dst is None:
            full = torch.zeros(
                shard.global_shape, dtype=value.dtype, device=value.device
            )
            if shard.owner:
                full[shard.index] = value
            torch.distributed.all_reduce(full, group=group)
            return full.to("cpu", copy=full.device.type != "cpu")
        if dst not in torch.distributed.get_process_group_ranks(group):
            return value
        gathered = _gather_to(value, shard, group, dst)
    return value if gathered is None else gathered


def _real(tensor: torch.Tensor) -> torch.Tensor:
    return torch.view_as_real(tensor) if tensor.is_complex() else tensor


def _gather_to(
    value: torch.Tensor,
    shard: ParameterShard,
    group: torch.distributed.ProcessGroup,
    dst: int,
) -> torch.Tensor | None:
    """Gather the owners' slices into a full CPU tensor on ``dst``, a rank of
    ``group``; None on the other ranks.
    """
    # every rank's slice, to size the gather (padded to the largest slice)
    # and to place the slices on dst
    layout = torch.tensor(
        [int(shard.owner)]
        + [
            b for s, n in zip(shard.index, shard.global_shape) for b in s.indices(n)[:2]
        ],
        dtype=torch.int64,
        device=value.device,
    )
    layouts = [torch.empty_like(layout) for _ in range(group.size())]
    torch.distributed.all_gather(layouts, layout, group=group)
    ranks = [entry.tolist() for entry in layouts]
    real_per_element = 2 if value.is_complex() else 1
    sizes = [
        math.prod(stop - start for start, stop in zip(r[1::2], r[2::2]))
        * real_per_element
        for r in ranks
    ]
    send = _real(value).reshape(-1)
    if send.numel() < max(sizes):
        padded = send.new_empty(max(sizes))
        padded[: send.numel()] = send
        send = padded
    is_dst = torch.distributed.get_rank() == dst
    parts = [send.new_empty(max(sizes)) for _ in ranks] if is_dst else None
    torch.distributed.gather(send, parts, dst=dst, group=group)
    if parts is None:
        return None
    full = torch.empty(shard.global_shape, dtype=value.dtype)
    covered = 0
    for (owner, *bounds), size, part in zip(ranks, sizes, parts):
        if not owner:
            continue
        target = _real(full[tuple(slice(*b) for b in zip(bounds[::2], bounds[1::2]))])
        target.copy_(part[:size].view(target.shape))
        covered += size
    if covered != _real(full).numel():
        raise RuntimeError(
            f"the owners' slices cover {covered} of {_real(full).numel()} "
            f"elements of a tensor of shape {shard.global_shape}"
        )
    return full


def _localize_loaded_state(
    module: nn.Module,
    state_dict: dict[str, Any],
    prefix: str,
    local_metadata: dict,
    strict: bool,
    missing_keys: list[str],
    unexpected_keys: list[str],
    error_msgs: list[str],
) -> None:
    for name, shard in _module_shards(module).items():
        key = prefix + name
        if key in state_dict:
            state_dict[key] = shard.localize(state_dict[key])


def _gather_saved_state(
    module: nn.Module,
    state_dict: dict[str, Any],
    prefix: str,
    local_metadata: dict,
) -> None:
    if not _gather_destinations:
        return
    for name, shard in _module_shards(module).items():
        key = prefix + name
        if key in state_dict:
            state_dict[key] = gather_shard(state_dict[key], shard)
