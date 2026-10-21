"""Optimizer state sharded over the ranks that hold one model replica.

Under spatial parallelism every rank of a spatial group holds the same
parameters and, after the gradient all-reduce, the same gradients, so each
rank also kept a full copy of the optimizer state (two fp32 tensors per
parameter for Adam). :class:`SpatiallyShardedOptimizer` keeps only a
1/N share of that state on each of the N ranks of the group, using
:class:`torch.distributed.optim.ZeroRedundancyOptimizer` for the step: each
rank updates the parameters it owns and then broadcasts them to the group.

Checkpoints keep the format of an unsharded optimizer's ``state_dict()``: the
state is gathered onto the checkpoint-writing rank inside
:meth:`SpatiallyShardedOptimizer.gathered_on`, and :meth:`load_state_dict`
takes a full state dict (from a sharded or an unsharded run) on every rank and
keeps the share this rank owns.
"""

import contextlib
import dataclasses
from collections.abc import Iterator
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.optim import ZeroRedundancyOptimizer


@dataclasses.dataclass(frozen=True)
class _TensorSpec:
    """
    Placeholder for a state tensor that is sent separately from the
    pickled per-parameter state structure.
    """

    shape: tuple[int, ...]
    dtype: torch.dtype


def _is_sent_separately(value: Any) -> bool:
    # 0-dim tensors (Adam's ``step``) are tiny and may live on the CPU, which
    # NCCL cannot send; they travel inside the pickled structure instead.
    return isinstance(value, torch.Tensor) and value.dim() > 0


def _cpu_copy(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        # copy=True: on the CPU `.cpu()` would alias the live state, which the
        # next step updates in place
        return value.detach().to("cpu", copy=True)
    return value


class SpatiallyShardedOptimizer(ZeroRedundancyOptimizer):
    """A ZeroRedundancyOptimizer whose state dict is an unsharded optimizer's.

    ``state_dict()`` is only available on the gathering rank inside
    :meth:`gathered_on`, which is collective over the shard group. Parameter
    group hyperparameters (e.g. a scheduler's ``lr``) are set on this object's
    ``param_groups`` as for any optimizer; they reach the rank-local optimizer
    at every step.

    Args:
        params: The parameters to optimize, identical on every rank of
            ``process_group``.
        optimizer_class: The optimizer each rank runs on its share of the
            parameters, e.g. ``torch.optim.AdamW``.
        process_group: The ranks that share the optimizer state.
        **defaults: Keyword arguments for ``optimizer_class``.
    """

    def __init__(
        self,
        params: list[torch.nn.Parameter],
        optimizer_class: type[torch.optim.Optimizer],
        process_group: dist.ProcessGroup,
        **defaults: Any,
    ):
        super().__init__(
            params,
            optimizer_class=optimizer_class,
            process_group=process_group,
            **defaults,
        )
        self._gathered_state_dict: dict[str, Any] | None = None

    @staticmethod
    def _sync_param_groups(
        src_param_groups: list[dict[Any, Any]],
        dst_param_groups: list[dict[Any, Any]],
    ) -> None:
        """Make the destination groups' hyperparameters equal the source's.

        The base class only copies keys, so a key removed from the exposed
        groups (as fine-tuning does when it replaces a checkpoint's
        hyperparameters with the configured ones) would survive in the local
        optimizer and be copied back after the next step.
        """
        if len(src_param_groups) != len(dst_param_groups):
            raise ValueError(
                "Mismatch between number of source and destination parameter "
                f"groups: {len(src_param_groups)} vs {len(dst_param_groups)}"
            )
        for src, dst in zip(src_param_groups, dst_param_groups):
            for key in [k for k in dst if k != "params" and k not in src]:
                del dst[key]
            for key, value in src.items():
                if key != "params":
                    dst[key] = value

    def _local_state_by_global_index(self) -> dict[int, dict[str, Any]]:
        """
        This rank's per-parameter state, keyed by the parameter's index in
        an unsharded optimizer's state dict.
        """
        out = {}
        for group in self.optim.param_groups:
            for param in group["params"]:
                state = self.optim.state.get(param)
                if state:
                    out[self._param_to_index[param]] = state
        return out

    def _gather_state(self, dst_group_rank: int) -> dict[int, dict[str, Any]] | None:
        """Gather every rank's per-parameter state onto ``dst_group_rank``.

        Point-to-point, one rank at a time: the receiver holds at most one
        state tensor on the device at once and keeps the result on the CPU.
        """
        ranks = dist.get_process_group_ranks(self.process_group)
        local_state = self._local_state_by_global_index()
        device = self._default_device
        if self.rank != dst_group_rank:
            structure = {
                index: {
                    name: (
                        _TensorSpec(tuple(value.shape), value.dtype)
                        if _is_sent_separately(value)
                        else _cpu_copy(value)
                    )
                    for name, value in state.items()
                }
                for index, state in local_state.items()
            }
            dst = ranks[dst_group_rank]
            dist.send_object_list([structure], dst=dst, group=self.process_group)
            for state in local_state.values():
                for value in state.values():
                    if _is_sent_separately(value):
                        dist.send(
                            value.detach().to(device).contiguous(),
                            dst=dst,
                            group=self.process_group,
                        )
            return None
        gathered = {
            index: {name: _cpu_copy(value) for name, value in state.items()}
            for index, state in local_state.items()
        }
        for group_rank, src in enumerate(ranks):
            if group_rank == dst_group_rank:
                continue
            received: list[Any] = [None]
            dist.recv_object_list(received, src=src, group=self.process_group)
            for index, state in received[0].items():
                gathered[index] = {}
                for name, value in state.items():
                    # the structure was built by `_gather_state` on the sender:
                    # a spec marks a tensor that follows on its own
                    if isinstance(value, _TensorSpec):
                        buffer = torch.empty(
                            value.shape, dtype=value.dtype, device=device
                        )
                        dist.recv(buffer, src=src, group=self.process_group)
                        gathered[index][name] = buffer.cpu()
                    else:
                        gathered[index][name] = value
        return gathered

    @contextlib.contextmanager
    def gathered_on(self, global_rank: int) -> Iterator[None]:
        """Within this context, ``state_dict()`` works on ``global_rank``.

        Collective: every rank of the shard group must enter it at the same
        point. Ranks of other shard groups (other data-parallel replicas) may
        enter it too; they skip the gather. The gathered state lives on the
        CPU of ``global_rank`` until the context exits.
        """
        ranks = dist.get_process_group_ranks(self.process_group)
        if global_rank in ranks:
            # hyperparameters set on the exposed groups since the last step
            self._sync_param_groups(self.param_groups, self.optim.param_groups)
            gathered = self._gather_state(ranks.index(global_rank))
            if gathered is not None:
                state_dict = torch.optim.Optimizer.state_dict(self)
                state_dict["state"] = dict(sorted(gathered.items()))
                self._gathered_state_dict = state_dict
        try:
            yield
        finally:
            self._gathered_state_dict = None

    def state_dict(self) -> dict[str, Any]:
        """The full optimizer state, in the format of an unsharded optimizer.

        Only valid on the gathering rank inside :meth:`gathered_on`.
        """
        if self._gathered_state_dict is None:
            raise RuntimeError(
                "Sharded optimizer state is only available on the gathering "
                "rank, inside SpatiallyShardedOptimizer.gathered_on(), which "
                "every rank of the shard group must enter together."
            )
        return self._gathered_state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load this rank's share of a full (unsharded-format) state dict.

        Every rank is given the whole state dict. The local optimizer applies
        its own placement rules (fused AdamW keeps ``step`` on the parameter's
        device), and ``state_dict`` is not modified.
        """
        saved_groups = state_dict["param_groups"]
        if len(saved_groups) != len(self.param_groups):
            raise ValueError(
                "loaded state dict has a different number of parameter groups"
            )
        for saved, group in zip(saved_groups, self.param_groups):
            if len(saved["params"]) != len(group["params"]):
                raise ValueError(
                    "loaded state dict contains a parameter group that doesn't "
                    "match the size of optimizer's group"
                )
        # state dicts index parameters consecutively over the groups, the same
        # order as this optimizer's (unsharded) param_groups
        local_state = {}
        local_groups = []
        local_index = 0
        for saved, local_group in zip(saved_groups, self.optim.param_groups):
            indices = []
            for param in local_group["params"]:
                global_index = self._param_to_index[param]
                if global_index in state_dict["state"]:
                    local_state[local_index] = state_dict["state"][global_index]
                indices.append(local_index)
                local_index += 1
            hyperparameters = {k: v for k, v in saved.items() if k != "params"}
            local_groups.append({**hyperparameters, "params": indices})
        self.optim.load_state_dict({"state": local_state, "param_groups": local_groups})
        self._sync_param_groups(self.optim.param_groups, self.param_groups)
