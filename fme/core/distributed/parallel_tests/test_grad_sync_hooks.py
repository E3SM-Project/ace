"""Serial tests of the spatial backend's gradient sync hooks, with the
collective replaced by a fake that records when it is waited on."""

import dataclasses
import unittest.mock

import pytest
import torch
from torch import nn

from fme.core.distributed.model_torch_distributed import _register_reduce_hooks


@dataclasses.dataclass
class _Work:
    waited: bool = False

    def wait(self):
        self.waited = True


class _FailingBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        raise RuntimeError("backward failed")


def test_reduce_hooks_wait_after_a_failed_backward(monkeypatch):
    """A backward that raises after a hook fired skips its queued wait; the
    next backward must still wait for its own all-reduces before returning."""
    works: list[_Work] = []

    def all_reduce(tensor, group=None, async_op=False):
        works.append(_Work())
        return works[-1]

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    param = nn.Parameter(torch.ones(3))
    _register_reduce_hooks([param], group=unittest.mock.MagicMock(), divisor=2)

    # created first, so its backward runs after the parameter's hook
    failing = _FailingBackward.apply(torch.ones(3, requires_grad=True))
    with pytest.raises(RuntimeError, match="backward failed"):
        (failing.sum() + (2 * param).sum()).backward()
    assert len(works) == 1
    assert not works[0].waited

    (3 * param).sum().backward()
    assert len(works) == 2
    assert works[1].waited
