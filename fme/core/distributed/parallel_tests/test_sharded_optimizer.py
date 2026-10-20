"""Sharded optimizer state (``OptimizationConfig.shard_optimizer_state``) must
train, checkpoint and resume exactly like an unsharded optimizer.

Run serially these check the flag is a no-op; under spatial parallelism
(``FME_DISTRIBUTED_BACKEND=model``) they exercise the sharded optimizer, with
and without data-parallel replicas.
"""

import copy
import dataclasses
import os
import tempfile
from typing import Any, Literal

import pytest
import torch
from torch import nn

from fme.core import get_device
from fme.core.distributed import Distributed
from fme.core.optimization import Optimization, OptimizationConfig
from fme.core.optimizer_sharding import SpatiallyShardedOptimizer
from fme.core.scheduler import SchedulerConfig

N_STEPS = 4
N_MORE_STEPS = 3
N_SUBSTEPS = 2  # losses accumulated per optimizer step


@dataclasses.dataclass
class Case:
    optimizer_type: Literal["Adam", "AdamW"] = "AdamW"
    kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
    scheduler: SchedulerConfig = dataclasses.field(default_factory=SchedulerConfig)
    max_grad_norm: float | None = None
    use_gradient_accumulation: bool = False


CASES = {
    "adamw": Case(kwargs={"weight_decay": 0.01}),
    "adamw_fused": Case(kwargs={"fused": True}),
    "adam": Case(optimizer_type="Adam"),
    "scheduler_clip_accumulate": Case(
        scheduler=SchedulerConfig(
            type="StepLR",
            kwargs={"step_size": 1, "gamma": 0.7},
            step_each_iteration=True,
        ),
        max_grad_norm=0.05,
        use_gradient_accumulation=True,
    ),
}


def _build_model() -> nn.Module:
    torch.manual_seed(0)
    # parameters of several sizes, so each rank of a 4-rank group owns some
    return nn.Sequential(
        nn.Linear(6, 32), nn.Tanh(), nn.Linear(32, 16), nn.Tanh(), nn.Linear(16, 3)
    ).to(get_device())


def _copy_model(model: nn.Module) -> nn.Module:
    copied = _build_model()
    copied.load_state_dict(model.state_dict())
    return copied


def _build_optimization(
    model: nn.Module, case: Case, shard: bool, lr: float = 0.01
) -> Optimization:
    return OptimizationConfig(
        optimizer_type=case.optimizer_type,
        lr=lr,
        kwargs=case.kwargs,
        scheduler=case.scheduler,
        max_grad_norm=case.max_grad_norm,
        use_gradient_accumulation=case.use_gradient_accumulation,
        shard_optimizer_state=shard,
    ).build(nn.ModuleList([model]), max_epochs=10)


def _train(module: nn.Module, optimization: Optimization, steps: range):
    """Steps whose input differs on every rank, as spatial tiles' would."""
    dist = Distributed.get_instance()
    wrapped = dist.wrap_module(module)
    for step in steps:
        for substep in range(N_SUBSTEPS):
            generator = torch.Generator().manual_seed(
                1000 * step + 100 * substep + dist.rank
            )
            x = torch.randn(5, 6, generator=generator).to(get_device())
            optimization.accumulate_loss(wrapped(x).pow(2).mean())
        optimization.step_weights()
        optimization.step_scheduler(is_iteration=True)


def _assert_same_params(a: nn.Module, b: nn.Module):
    for (name, pa), pb in zip(a.named_parameters(), b.parameters()):
        torch.testing.assert_close(pa, pb, rtol=1e-6, atol=1e-7, msg=name)


def _assert_same_optimizer_state(actual: dict, expected: dict):
    assert actual["param_groups"] == expected["param_groups"]
    assert list(actual["state"]) == list(expected["state"])
    for index, expected_state in expected["state"].items():
        assert list(actual["state"][index]) == list(expected_state)
        for name, value in expected_state.items():
            torch.testing.assert_close(
                actual["state"][index][name].cpu(),
                value.cpu(),
                rtol=1e-6,
                atol=1e-7,
                msg=f"state[{index}][{name}]",
            )


def _full_state_on_every_rank(optimization: Optimization) -> dict:
    """What every rank reads from a checkpoint the root wrote."""
    dist = Distributed.get_instance()
    with optimization.consolidated_state():
        state = copy.deepcopy(optimization.get_state()) if dist.is_root() else None
    return dist.scatter_object(state)


def _is_sharded(optimization: Optimization) -> bool:
    return isinstance(optimization.optimizer, SpatiallyShardedOptimizer)


def _local_optimizer(optimization: Optimization) -> torch.optim.Optimizer:
    """The optimizer that holds this rank's state."""
    if isinstance(optimization.optimizer, SpatiallyShardedOptimizer):
        return optimization.optimizer.optim
    return optimization.optimizer


@pytest.mark.parallel
@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
def test_sharded_optimizer_matches_unsharded(case: Case):
    dist = Distributed.get_instance()
    model = _build_model()
    reference = _build_optimization(model, case, shard=False)
    sharded_model = _copy_model(model)
    sharded = _build_optimization(sharded_model, case, shard=True)
    assert _is_sharded(sharded) == dist.has_spatial_parallelism
    if _is_sharded(sharded):
        # each rank holds only its share of the state
        _train(sharded_model, sharded, range(1))
        _train(model, reference, range(1))
        local = sum(
            v.numel()
            for state in _local_optimizer(sharded).state.values()
            for v in state.values()
            if v.dim() > 0
        )
        full = sum(
            v.numel()
            for state in reference.optimizer.state.values()
            for v in state.values()
            if v.dim() > 0
        )
        assert local < full
        start = 1
    else:
        start = 0
    _train(model, reference, range(start, N_STEPS))
    _train(sharded_model, sharded, range(start, N_STEPS))
    _assert_same_params(sharded_model, model)
    assert sharded.learning_rate == reference.learning_rate

    # the saved state has the unsharded format and values
    with sharded.consolidated_state():
        if dist.is_root():
            saved = sharded.get_state()
            expected = reference.get_state()
            _assert_same_optimizer_state(
                saved["optimizer_state_dict"], expected["optimizer_state_dict"]
            )
            assert saved["scheduler_state_dict"] == expected["scheduler_state_dict"]


@pytest.mark.parallel
@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
def test_sharded_optimizer_checkpoint_round_trip(case: Case):
    """Save -> load -> continue equals continuing, whichever of the saving
    and loading runs shards its optimizer state."""
    model = _build_model()
    reference = _build_optimization(model, case, shard=False)
    sharded_model = _copy_model(model)
    sharded = _build_optimization(sharded_model, case, shard=True)
    _train(model, reference, range(N_STEPS))
    _train(sharded_model, sharded, range(N_STEPS))
    sharded_state = _full_state_on_every_rank(sharded)
    unsharded_state = _full_state_on_every_rank(reference)

    resumed = {}
    for name, state, shard in [
        ("sharded->sharded", sharded_state, True),
        ("sharded->unsharded", sharded_state, False),
        ("unsharded->sharded", unsharded_state, True),
    ]:
        resumed_model = _copy_model(model)
        optimization = _build_optimization(resumed_model, case, shard=shard)
        optimization.load_state(copy.deepcopy(state))
        resumed[name] = (resumed_model, optimization)

    steps = range(N_STEPS, N_STEPS + N_MORE_STEPS)
    _train(model, reference, steps)
    _train(sharded_model, sharded, steps)
    _assert_same_params(sharded_model, model)
    for name, (resumed_model, optimization) in resumed.items():
        _train(resumed_model, optimization, steps)
        _assert_same_params(resumed_model, model)
        assert optimization.learning_rate == reference.learning_rate, name


@pytest.mark.parallel
def test_sharded_optimizer_finetune_from_checkpoint_file():
    """resume_optimizer_ckpt_path loads the moments into a sharded optimizer
    and keeps the configured hyperparameters, as for an unsharded one."""
    dist = Distributed.get_instance()
    case = CASES["scheduler_clip_accumulate"]
    source_model = _build_model()
    source = _build_optimization(source_model, case, shard=True)
    _train(source_model, source, range(N_STEPS))
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "ckpt.tar")
        with source.consolidated_state():
            if dist.is_root():
                torch.save({"optimization": source.get_state()}, path)
        path = dist.scatter_object(path if dist.is_root() else None)
        dist.barrier()  # the root's file is written before anyone reads it
        finetune_case = Case(kwargs={"weight_decay": 0.1})  # no scheduler
        models = {}
        optimizations = {}
        for shard in (True, False):
            models[shard] = _copy_model(source_model)
            optimizations[shard] = OptimizationConfig(
                optimizer_type="AdamW",
                lr=0.003,
                kwargs=finetune_case.kwargs,
                resume_optimizer_ckpt_path=path,
                shard_optimizer_state=shard,
            ).build(nn.ModuleList([models[shard]]), max_epochs=10)
        dist.barrier()  # every rank has read the file before the root deletes it
    for shard in (True, False):
        _train(models[shard], optimizations[shard], range(N_STEPS, 2 * N_STEPS))
    _assert_same_params(models[True], models[False])
    sharded = optimizations[True]
    local_groups = _local_optimizer(sharded).param_groups
    for group in [*sharded.optimizer.param_groups, *local_groups]:
        assert group["lr"] == 0.003
        assert group["weight_decay"] == 0.1
        # the checkpoint's scheduler key does not leak into the new run
        assert "initial_lr" not in group
