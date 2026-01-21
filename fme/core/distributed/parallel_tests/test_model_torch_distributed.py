import os
from unittest.mock import patch

import pytest
import torch.distributed

from fme.core.distributed import model_torch_distributed
from fme.core.distributed.model_torch_distributed import ModelTorchDistributed


@pytest.mark.parametrize(
    "torch_available,h_parallel,w_parallel,expected",
    [
        # Both torch distributed and spatial parallelism enabled
        (True, "2", "1", True),
        # Torch distributed NOT available, spatial parallelism enabled
        (False, "2", "2", False),
        # Torch distributed available, spatial parallelism NOT enabled
        (True, "1", "1", False),
        # Only W_PARALLEL_SIZE > 1
        (True, "1", "3", True),
        # Only H_PARALLEL_SIZE > 1
        (True, "4", "1", True),
        # Environment variables missing (defaults to 1)
        (True, None, None, False),
        # Zero values (should be treated as disabled)
        (True, "0", "0", False),
    ],
    ids=[
        "both_true",
        "torch_false_spatial_true",
        "torch_true_spatial_false",
        "w_parallel_only",
        "h_parallel_only",
        "env_vars_missing",
        "env_vars_zero",
    ],
)
@patch("torch.distributed.is_available")
def test_is_available(
    mock_torch_available, torch_available, h_parallel, w_parallel, expected
):
    """Test ModelTorchDistributed.is_available() with various configurations."""
    mock_torch_available.return_value = torch_available

    # Build environment dict
    env_dict = {}
    if h_parallel is not None:
        env_dict["H_PARALLEL_SIZE"] = h_parallel
    if w_parallel is not None:
        env_dict["W_PARALLEL_SIZE"] = w_parallel

    clear_env = h_parallel is None and w_parallel is None

    with patch.dict(os.environ, env_dict, clear=clear_env):
        result = ModelTorchDistributed.is_available()

    assert result is expected


def test_is_available_false_when_physicsnemo_missing(monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(model_torch_distributed, "pnd", None)
    with patch.dict(os.environ, {"H_PARALLEL_SIZE": "2", "W_PARALLEL_SIZE": "1"}):
        assert ModelTorchDistributed.is_available() is False


def test_is_available_spatial_disabled_without_physicsnemo(monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(model_torch_distributed, "pnd", None)
    with patch.dict(os.environ, {"H_PARALLEL_SIZE": "1", "W_PARALLEL_SIZE": "1"}):
        assert ModelTorchDistributed.is_available() is False


def test_wrap_module_returns_dummy_for_frozen(monkeypatch):
    """A module with no trainable parameters should be wrapped in DummyWrapper,
    even under spatial parallelism."""
    from fme.core.distributed.non_distributed import DummyWrapper

    monkeypatch.setenv("H_PARALLEL_SIZE", "1")
    monkeypatch.setenv("W_PARALLEL_SIZE", "1")

    # Use NonDistributed path (no GPUs required for this unit test)
    from fme.core.distributed.non_distributed import NonDistributed

    backend = NonDistributed()
    frozen_module = torch.nn.Linear(4, 4)
    for p in frozen_module.parameters():
        p.requires_grad = False

    wrapped = backend.wrap_module(frozen_module)
    assert isinstance(wrapped, DummyWrapper)
    assert wrapped.module is frozen_module
