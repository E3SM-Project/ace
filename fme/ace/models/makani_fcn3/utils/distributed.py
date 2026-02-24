# parallel helpers — distributed primitives for Makani / FCN3.
#
# Delegates to ``torch.distributed`` via the ACE ``comm`` shim so
# that operations like Welford-based DistributedInstanceNorm2d and
# DistributedRealFFT2 work correctly under spatial parallelism.
# When spatial parallelism is inactive (group size 1), every
# operation degrades to a no-op.

from __future__ import annotations

from typing import Any, List, Tuple

import torch
from torch_harmonics.distributed import compute_split_shapes as _thd_split_shapes


def _get_group(name: str):
    """Lazy import to avoid circular imports."""
    from fme.ace.models.makani_fcn3.utils import comm

    return comm.get_group(name)


def _get_size(name: str) -> int:
    from fme.ace.models.makani_fcn3.utils import comm

    return comm.get_size(name)


# -----------------------------------------------------------------------
# Autograd functions for correct gradient flow
# -----------------------------------------------------------------------


class _CopyToParallelRegion(torch.autograd.Function):
    """Identity in forward; all-reduce (SUM) in backward."""

    @staticmethod
    def forward(ctx, input_: torch.Tensor, group_name: str) -> torch.Tensor:
        ctx.group_name = group_name
        return input_

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        import torch.distributed as td

        pg = _get_group(ctx.group_name)
        if pg is not None and td.get_world_size(group=pg) > 1:
            td.all_reduce(grad_output, group=pg)
        return grad_output, None


class _GatherFromParallelRegion(torch.autograd.Function):
    """All-gather in forward; scatter (select local chunk) in backward."""

    @staticmethod
    def forward(
        ctx,
        input_: torch.Tensor,
        dim_: int,
        shapes_: Any,
        group_name: str,
    ) -> torch.Tensor:
        import torch.distributed as td

        ctx.dim = dim_
        ctx.group_name = group_name

        pg = _get_group(group_name)
        if pg is None or td.get_world_size(group=pg) <= 1:
            ctx.world_size = 1
            return input_

        world_size = td.get_world_size(group=pg)
        ctx.world_size = world_size
        ctx.rank = td.get_rank(group=pg)

        gathered = [torch.empty_like(input_) for _ in range(world_size)]
        td.all_gather(gathered, input_.contiguous(), group=pg)
        return torch.cat(gathered, dim=dim_)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if ctx.world_size <= 1:
            return grad_output, None, None, None
        chunks = grad_output.chunk(ctx.world_size, dim=ctx.dim)
        return chunks[ctx.rank].contiguous(), None, None, None


class _ReduceFromParallelRegion(torch.autograd.Function):
    """All-reduce (SUM) in forward; identity in backward."""

    @staticmethod
    def forward(ctx, input_: torch.Tensor, group_name: str) -> torch.Tensor:
        import torch.distributed as td

        pg = _get_group(group_name)
        if pg is not None and td.get_world_size(group=pg) > 1:
            td.all_reduce(input_, group=pg)
        return input_

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None


# -----------------------------------------------------------------------
# Public helper with the same API as the original NotDistributed stub
# -----------------------------------------------------------------------


class DistributedHelper:
    """Distributed primitives used by Makani / FCN3 layers."""

    @staticmethod
    def compute_split_shapes(size: int, num_chunks: int) -> List[int]:
        return _thd_split_shapes(size, num_chunks)

    @staticmethod
    def reduce_from_parallel_region(
        input_: torch.Tensor, group: str
    ) -> torch.Tensor:
        return _ReduceFromParallelRegion.apply(input_, group)

    @staticmethod
    def scatter_to_parallel_region(
        input_: torch.Tensor, dim: int, group: str
    ) -> torch.Tensor:
        import torch.distributed as td

        pg = _get_group(group)
        if pg is None or td.get_world_size(group=pg) <= 1:
            return input_
        rank = td.get_rank(group=pg)
        world_size = td.get_world_size(group=pg)
        chunks = input_.chunk(world_size, dim=dim)
        return chunks[rank].contiguous()

    @staticmethod
    def gather_from_parallel_region(
        input_: torch.Tensor, dim: int, shapes: Any, group: str
    ) -> torch.Tensor:
        return _GatherFromParallelRegion.apply(input_, dim, shapes, group)

    @staticmethod
    def copy_to_parallel_region(
        input_: torch.Tensor, group: str
    ) -> torch.Tensor:
        return _CopyToParallelRegion.apply(input_, group)

    @staticmethod
    def split_tensor_along_dim(
        tensor: torch.Tensor, dim: int, num_chunks: int
    ) -> Tuple[torch.Tensor, ...]:
        return tensor.chunk(num_chunks, dim=dim)


dist = DistributedHelper()
