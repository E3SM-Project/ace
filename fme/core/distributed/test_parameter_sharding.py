"""Serial tests of spatially sharded parameters. The behavior under spatial
parallelism is tested in ``parallel_tests/test_parameter_sharding.py``.
"""

import pytest
import torch
from torch import nn

from fme.ace.registry.stochastic_sfno import NoiseConditionedSFNOBuilder
from fme.core.dataset_info import DatasetInfo
from fme.core.distributed.parameter_sharding import (
    ParameterShard,
    gather_shard,
    gathered_parameter_shards,
    get_parameter_shard,
    named_parameter_shards,
    parameter_shards,
    parameter_shards_gathered,
    shard_parameter,
    spatial_shard,
)
from fme.core.ema import EMATracker
from fme.core.optimization import OptimizationConfig
from fme.core.weight_ops import overwrite_weights

# rows 1 and 2 of a 4 x 3 tensor, as a rank holding a polar slice would
SHARD = ParameterShard(
    global_shape=(4, 3),
    index=(slice(1, 3), slice(0, 3)),
    split=(True, False),
    owner=True,
)


def _sharded_linear() -> nn.Module:
    """A Linear(3, 4) whose weight holds rows 1 and 2 of the full weight."""
    torch.manual_seed(0)
    module = nn.Sequential(nn.Linear(3, 4, bias=False))
    shard_parameter(module[0], "weight", SHARD)
    return module


def test_no_shards_without_spatial_parallelism():
    assert spatial_shard((2, 8, 4), split_shape=(8, 4), dims=(1, None)) is None
    assert spatial_shard((1, 3, 8, 4), split_shape=(8, 4), dims=(-2, -1)) is None


def test_shard_flag_is_a_no_op_serially():
    def build(shard: bool) -> nn.Module:
        torch.manual_seed(0)
        return NoiseConditionedSFNOBuilder(
            embed_dim=8,
            noise_embed_dim=4,
            num_layers=2,
            context_pos_embed_dim=2,
            pos_embed=False,
            shard_spatial_parameters=shard,
        ).build(3, 2, DatasetInfo(img_shape=(9, 16), all_labels={"a", "b"}))

    sharded, unsharded = build(True), build(False)
    assert named_parameter_shards(sharded) == {}
    for (name, a), b in zip(sharded.named_parameters(), unsharded.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0, msg=name)


def test_shard_parameter_keeps_slice_of_initialization():
    torch.manual_seed(0)
    full = nn.Linear(3, 4, bias=False).weight.detach().clone()
    module = _sharded_linear()
    torch.testing.assert_close(module[0].weight.detach(), full[1:3])
    assert named_parameter_shards(module) == {"0.weight": SHARD}
    assert get_parameter_shard(module, "0.weight") == SHARD
    assert get_parameter_shard(module, "0") is None
    assert get_parameter_shard(module, "missing.weight") is None
    assert parameter_shards([module]) == {module[0].weight: SHARD}


def test_load_state_dict_takes_full_or_local_tensors():
    module = _sharded_linear()
    full = torch.arange(12.0).reshape(4, 3)
    module.load_state_dict({"0.weight": full})
    torch.testing.assert_close(module[0].weight.detach(), full[1:3])
    local = torch.ones(2, 3)
    module.load_state_dict({"0.weight": local})
    torch.testing.assert_close(module[0].weight.detach(), local)
    with pytest.raises(RuntimeError, match="size mismatch"):
        module.load_state_dict({"0.weight": torch.ones(3, 3)})


def test_state_dict_is_local_outside_and_full_inside_gather():
    module = _sharded_linear()
    assert module.state_dict()["0.weight"].shape == (2, 3)
    assert not parameter_shards_gathered()
    with gathered_parameter_shards(dst=0):
        assert parameter_shards_gathered()
        full = module.state_dict()["0.weight"]
    # serially there is no other rank: only this rank's rows are filled
    assert full.shape == (4, 3)
    torch.testing.assert_close(full[1:3], module[0].weight.detach())
    assert full[0].abs().sum() == 0
    with gathered_parameter_shards(dst=1):  # not this rank
        assert module.state_dict()["0.weight"].shape == (2, 3)


def test_gather_shard_rejects_wrong_shape():
    with gathered_parameter_shards(dst=None):
        with pytest.raises(ValueError, match="local shard"):
            gather_shard(torch.zeros(4, 3), SHARD)


def test_ema_state_is_full_and_loads_full():
    module = _sharded_linear()
    ema = EMATracker(module, decay=0.9)
    assert ema.get_state()["ema_params"]["0weight"].shape == (2, 3)
    with gathered_parameter_shards(dst=0):
        state = ema.get_state()
    assert state["ema_params"]["0weight"].shape == (4, 3)
    restored = EMATracker.from_state(state, module)
    torch.testing.assert_close(
        restored._ema_params["0weight"], module[0].weight.detach()
    )
    target = _sharded_linear()
    with torch.no_grad():
        target[0].weight.zero_()
    assert EMATracker.copy_params_from_state(state, target)
    torch.testing.assert_close(target[0].weight, module[0].weight)


def test_optimizer_state_round_trips_in_full_format():
    module = _sharded_linear()
    optimization = OptimizationConfig(optimizer_type="AdamW").build(
        nn.ModuleList([module]), max_epochs=1
    )
    optimization.accumulate_loss(module(torch.ones(5, 3)).pow(2).mean())
    optimization.step_weights()
    with pytest.raises(RuntimeError, match="consolidated_state"):
        optimization.get_state()
    with optimization.consolidated_state():
        state = optimization.get_state()
    weight_state = state["optimizer_state_dict"]["state"][0]
    assert weight_state["exp_avg"].shape == (4, 3)
    local_exp_avg = optimization.optimizer.state[module[0].weight]["exp_avg"]
    torch.testing.assert_close(weight_state["exp_avg"][1:3], local_exp_avg)
    other = _sharded_linear()
    other_optimization = OptimizationConfig(optimizer_type="AdamW").build(
        nn.ModuleList([other]), max_epochs=1
    )
    other_optimization.load_state(state)
    loaded = other_optimization.optimizer.state[other[0].weight]["exp_avg"]
    assert loaded.shape == (2, 3)
    torch.testing.assert_close(loaded, local_exp_avg)


def test_amp_is_rejected_with_sharded_parameters():
    with pytest.raises(ValueError, match="enable_automatic_mixed_precision"):
        OptimizationConfig(enable_automatic_mixed_precision=True).build(
            nn.ModuleList([_sharded_linear()]), max_epochs=1
        )


def test_overwrite_weights_takes_full_weights():
    module = _sharded_linear()
    full = torch.arange(12.0).reshape(4, 3)
    overwrite_weights({"0.weight": full}, module)
    torch.testing.assert_close(module[0].weight.detach(), full[1:3])
