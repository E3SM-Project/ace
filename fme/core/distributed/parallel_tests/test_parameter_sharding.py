"""Spatially sharded parameters (``shard_spatial_parameters``) must initialize,
train, checkpoint, resume and infer exactly like the unsharded and the serial
model.

Each test trains a small NoiseConditionedSFNO (model, loss, optimizer and EMA,
as the trainer drives them) under the current backend, and compares it with
the same model unsharded and with a serial reference run in-process under
``Distributed.force_non_distributed``. The 9 x 16 grid (lmax 9) splits
unevenly over 2 polar ranks. Run serially, the flag is a no-op.
"""

import copy
import dataclasses
import unittest.mock
from typing import Any

import pytest
import torch
from torch import nn

from fme.ace.registry.stochastic_sfno import NoiseConditionedSFNOBuilder
from fme.core import get_device
from fme.core.dataset_info import DatasetInfo
from fme.core.distributed import Distributed
from fme.core.distributed.parameter_sharding import (
    gather_shard,
    gathered_parameter_shards,
    named_parameter_shards,
    spatial_shard,
)
from fme.core.ema import EMA_CHECKPOINT_KEY, EMAConfig, EMATracker
from fme.core.optimization import Optimization, OptimizationConfig
from fme.core.scheduler import SchedulerConfig

IMG_SHAPE = (9, 16)
N_IN = 3
N_OUT = 2
BATCH_PER_REPLICA = 2
N_STEPS = 3
N_MORE_STEPS = 2
LABELS = ["a", "b"]
# sharded vs unsharded under the same layout
RTOL = 1e-5
ATOL = 1e-6
# vs the serial model: the distributed transforms sum in another order, and
# Adam magnifies that for the smallest gradients
SERIAL_RTOL = 1e-4
SERIAL_ATOL = 1e-4


@dataclasses.dataclass
class Case:
    builder: dict[str, Any] = dataclasses.field(default_factory=dict)
    labels: bool = False
    shard_optimizer_state: bool = False
    max_grad_norm: float | None = None
    use_gradient_accumulation: bool = False
    n_substeps: int = 1
    scheduler: SchedulerConfig = dataclasses.field(default_factory=SchedulerConfig)


CASES = {
    "sfno_pos_embed": Case(),
    "context_pos_embed_labels": Case(
        builder={"pos_embed": False, "context_pos_embed_dim": 2}, labels=True
    ),
    "spectral_lora_zero": Case(
        builder={"spectral_lora_rank": 2}, shard_optimizer_state=True
    ),
    "clip_scheduler_zero": Case(
        shard_optimizer_state=True,
        max_grad_norm=0.05,
        scheduler=SchedulerConfig(
            type="StepLR",
            kwargs={"step_size": 1, "gamma": 0.7},
            step_each_iteration=True,
        ),
    ),
    "substeps_clip": Case(max_grad_norm=0.05, n_substeps=2),
}


@dataclasses.dataclass
class Run:
    """A model with its wrapper, optimization and EMA, as a trainer holds them."""

    model: nn.Module
    wrapped: nn.Module
    optimization: Optimization
    ema: EMATracker

    @property
    def modules(self) -> nn.ModuleList:
        return nn.ModuleList([self.wrapped])


def _build_run(case: Case, shard: bool, seed: int | None = 0) -> Run:
    """Build under the active backend, from the same random state everywhere
    (unless ``seed`` is None, which keeps each rank's state).
    """
    if seed is not None:
        torch.manual_seed(seed)
    builder = NoiseConditionedSFNOBuilder(
        embed_dim=8,
        noise_embed_dim=4,
        noise_type="isotropic",
        num_layers=2,
        affine_norms=True,
        shard_spatial_parameters=shard,
        **case.builder,
    )
    model = builder.build(
        N_IN,
        N_OUT,
        DatasetInfo(
            img_shape=IMG_SHAPE, all_labels=set(LABELS) if case.labels else None
        ),
    ).to(get_device())
    wrapped = Distributed.get_instance().wrap_module(model)
    modules = nn.ModuleList([wrapped])
    optimization = OptimizationConfig(
        optimizer_type="AdamW",
        lr=0.01,
        kwargs={"weight_decay": 0.01},
        scheduler=case.scheduler,
        max_grad_norm=case.max_grad_norm,
        use_gradient_accumulation=case.use_gradient_accumulation,
        shard_optimizer_state=case.shard_optimizer_state,
    ).build(modules, max_epochs=10)
    ema = EMAConfig(decay=0.9).build(modules)
    return Run(model, wrapped, optimization, ema)


def _global_batch(step: int, substep: int, case: Case, n_replicas: int):
    """The same global batch on every rank, and in the serial reference."""
    n = BATCH_PER_REPLICA * n_replicas
    generator = torch.Generator().manual_seed(1000 * step + substep)
    x = torch.randn(n, N_IN, *IMG_SHAPE, generator=generator)
    y = torch.randn(n, N_OUT, *IMG_SHAPE, generator=generator)
    labels = None
    if case.labels:
        labels = nn.functional.one_hot(
            torch.randint(len(LABELS), (n,), generator=generator), len(LABELS)
        ).float()
    return x, y, labels


def _local(tensor: torch.Tensor | None, spatial: bool = True) -> torch.Tensor | None:
    """This rank's samples (and tile) of a global batch tensor."""
    if tensor is None:
        return None
    dist = Distributed.get_instance()
    n = tensor.shape[0] // dist.total_data_parallel_ranks
    tensor = tensor[n * dist.data_parallel_rank : n * (dist.data_parallel_rank + 1)]
    if spatial:
        tensor = tensor[(..., *dist.get_local_slices(IMG_SHAPE))]
    return tensor.to(get_device())


def _forward(run: Run, x, labels, step: int):
    """The model's prediction, with the same noise for each sample in every
    layout and in the serial model: the (isotropic) noise is drawn for the
    global batch and each data-parallel replica takes its samples.
    """
    dist = Distributed.get_instance()
    generator = torch.Generator().manual_seed(10_000 + step)

    def randn(shape, dtype=None, device=None):
        n_local = shape[0]
        n_global = n_local * dist.total_data_parallel_ranks
        noise = torch.randn((n_global, *shape[1:]), generator=generator, dtype=dtype)
        start = n_local * dist.data_parallel_rank
        return noise[start : start + n_local].to(device)

    with unittest.mock.patch("fme.ace.registry.stochastic_sfno.randn", randn):
        if labels is None:
            return run.wrapped(x)
        return run.wrapped(x, labels=labels)


def _train(run: Run, case: Case, steps: range, n_replicas: int):
    """Train as on tiles: each rank sees its tile of its replica's samples.

    The loss is the mean over the replica's samples, the serial loss on every
    rank; replicas average their gradients.
    """
    dist = Distributed.get_instance()
    n_local = n_replicas * BATCH_PER_REPLICA // dist.total_data_parallel_ranks
    n_total = n_local * N_OUT * IMG_SHAPE[0] * IMG_SHAPE[1]
    for step in steps:
        for substep in range(case.n_substeps):
            x, y, labels = _global_batch(step, substep, case, n_replicas)
            pred = _forward(run, _local(x), _local(labels, spatial=False), step)
            local_sum = (pred - _local(y)).pow(2).sum()
            loss = dist.spatial_reduce_sum(local_sum) / n_total
            run.optimization.accumulate_loss(loss)
        run.optimization.step_weights()
        run.optimization.step_scheduler(is_iteration=True)
        run.ema(model=run.modules)


@dataclasses.dataclass
class Checkpoint:
    module: dict[str, torch.Tensor]
    ema: dict[str, Any]
    optimization: dict[str, Any]


def _checkpoint(run: Run) -> Checkpoint:
    """What the root writes (as the Trainer does), read back on every rank."""
    dist = Distributed.get_instance()
    state: Checkpoint | None = None
    with run.optimization.consolidated_state(), gathered_parameter_shards(dst=0):
        module = run.model.state_dict()
        ema = run.ema.get_state()
        if dist.is_root():
            state = copy.deepcopy(
                Checkpoint(
                    module={k: v.cpu() for k, v in module.items()},
                    ema=ema,
                    optimization=run.optimization.get_state(),
                )
            )
    return dist.scatter_object(state)


def _load(run: Run, checkpoint: Checkpoint):
    # a copy, as if read from disk: optimizers alias the tensors they load
    checkpoint = copy.deepcopy(checkpoint)
    run.model.load_state_dict(checkpoint.module)
    run.optimization.load_state(checkpoint.optimization)
    run.ema = EMATracker.from_state(checkpoint.ema, run.modules)


def _assert_close_dicts(
    actual: dict, expected: dict, msg: str, rtol: float, atol: float
):
    assert actual.keys() == expected.keys(), msg
    for key, value in expected.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(
                actual[key].cpu(),
                value.cpu(),
                rtol=rtol,
                atol=atol,
                msg=lambda m: f"{msg}[{key}]: {m}",
            )
        elif isinstance(value, dict):
            _assert_close_dicts(actual[key], value, f"{msg}[{key}]", rtol, atol)
        else:
            assert actual[key] == value, f"{msg}[{key}]"


def _assert_same_checkpoint(
    actual: Checkpoint,
    expected: Checkpoint,
    rtol: float = RTOL,
    atol: float = ATOL,
):
    _assert_close_dicts(actual.module, expected.module, "module", rtol, atol)
    _assert_close_dicts(actual.ema, expected.ema, "ema", rtol, atol)
    _assert_close_dicts(
        actual.optimization["optimizer_state_dict"],
        expected.optimization["optimizer_state_dict"],
        "optimizer",
        rtol,
        atol,
    )
    assert (
        actual.optimization["scheduler_state_dict"]
        == expected.optimization["scheduler_state_dict"]
    )


def _expects_shards() -> bool:
    return Distributed.get_instance().has_spatial_parallelism


def _expected_shard_names(model: nn.Module) -> set[str]:
    """pos_embeds by tile; spectral weights along l, by polar ranks only."""
    dist = Distributed.get_instance()
    # lmax equals the number of latitudes, so l splits as latitude does
    polar_split = dist.get_local_slices(IMG_SHAPE)[0].indices(IMG_SHAPE[0])[:2] != (
        0,
        IMG_SHAPE[0],
    )
    expected = set()
    for name, _ in model.named_parameters():
        base = name.rsplit(".", 1)[-1]
        if base in ("pos_embed", "label_pos_embed") and dist.has_spatial_parallelism:
            expected.add(name)
        elif ".filter.filter." in name and base in ("weight", "lora_A", "lora_B"):
            if polar_split:
                expected.add(name)
    return expected


@pytest.mark.parallel
@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
def test_sharded_init_is_sliced_serial_init(case: Case):
    run = _build_run(case, shard=True)
    with Distributed.force_non_distributed():
        serial = _build_run(case, shard=False)
    shards = named_parameter_shards(run.model)
    assert set(shards) == _expected_shard_names(run.model)
    serial_params = dict(serial.model.named_parameters())
    for name, param in run.model.named_parameters():
        expected = serial_params[name].detach()
        if name in shards:
            assert tuple(param.shape) == shards[name].local_shape
            expected = shards[name].take(expected)
        torch.testing.assert_close(param.detach(), expected, rtol=0, atol=0, msg=name)


@pytest.mark.parallel
def test_sharded_init_uses_the_root_random_state():
    """In a real run the ranks' random states differ when the model is built
    (the root does setup work the others skip); the slices must still come
    from one initialization, the root's.
    """
    case = CASES["sfno_pos_embed"]
    torch.manual_seed(Distributed.get_instance().rank)  # the root's seed is 0
    run = _build_run(case, shard=True, seed=None)
    with Distributed.force_non_distributed():
        serial = _build_run(case, shard=False, seed=0)
    shards = named_parameter_shards(run.model)
    serial_params = dict(serial.model.named_parameters())
    for name, shard in shards.items():
        torch.testing.assert_close(
            run.model.get_parameter(name).detach(),
            shard.take(serial_params[name].detach()),
            rtol=0,
            atol=0,
            msg=name,
        )


@pytest.mark.parallel
@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
def test_sharded_training_matches_unsharded_and_serial(case: Case):
    """Same parameters, EMA and optimizer state (checkpoint format and
    contents) as the unsharded model under the same layout, and as serial."""
    n_replicas = Distributed.get_instance().total_data_parallel_ranks
    sharded = _build_run(case, shard=True)
    unsharded = _build_run(case, shard=False)
    _train(sharded, case, range(N_STEPS), n_replicas)
    _train(unsharded, case, range(N_STEPS), n_replicas)
    actual = _checkpoint(sharded)
    expected = _checkpoint(unsharded)
    _assert_same_checkpoint(actual, expected)
    with Distributed.force_non_distributed():
        serial = _build_run(case, shard=False)
        _train(serial, case, range(N_STEPS), n_replicas)
        serial_checkpoint = _checkpoint(serial)
    _assert_same_checkpoint(
        actual, serial_checkpoint, rtol=SERIAL_RTOL, atol=SERIAL_ATOL
    )
    # the local state is the rank's slice of the full state
    for name, shard in named_parameter_shards(sharded.model).items():
        torch.testing.assert_close(
            sharded.model.get_parameter(name).detach().cpu(),
            shard.take(actual.module[name]),
            rtol=0,
            atol=0,
        )


@pytest.mark.parallel
@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
def test_sharded_checkpoint_resumes(case: Case):
    """A sharded run's checkpoint resumes into sharded, unsharded and serial
    runs, each continuing as the original run does."""
    n_replicas = Distributed.get_instance().total_data_parallel_ranks
    more = range(N_STEPS, N_STEPS + N_MORE_STEPS)
    original = _build_run(case, shard=True)
    _train(original, case, range(N_STEPS), n_replicas)
    checkpoint = _checkpoint(original)
    _train(original, case, more, n_replicas)
    expected = _checkpoint(original)
    for shard in (True, False):
        resumed = _build_run(case, shard=shard, seed=1)
        _load(resumed, checkpoint)
        _train(resumed, case, more, n_replicas)
        _assert_same_checkpoint(_checkpoint(resumed), expected)
    with Distributed.force_non_distributed():
        serial = _build_run(case, shard=False, seed=1)
        _load(serial, checkpoint)
        _train(serial, case, more, n_replicas)
        serial_checkpoint = _checkpoint(serial)
    _assert_same_checkpoint(
        serial_checkpoint, expected, rtol=SERIAL_RTOL, atol=SERIAL_ATOL
    )


@pytest.mark.parallel
def test_gradient_accumulation_is_rejected_under_spatial_parallelism():
    """Each backward pass would sum the already accumulated gradient over the
    tiles again."""
    case = Case(use_gradient_accumulation=True, n_substeps=2)
    if not Distributed.get_instance().has_spatial_parallelism:
        _build_run(case, shard=True)
        return
    with pytest.raises(ValueError, match="use_gradient_accumulation"):
        _build_run(case, shard=True)


def _predict(run: Run, case: Case, n_replicas: int) -> torch.Tensor:
    """This replica's prediction on the full grid."""
    x, _, labels = _global_batch(100, 0, case, n_replicas)
    run.model.eval()
    with torch.no_grad():
        pred = _forward(run, _local(x), _local(labels, spatial=False), 100)
    return Distributed.get_instance().gather_spatial_tensor(pred, IMG_SHAPE).cpu()


@pytest.mark.parallel
@pytest.mark.parametrize("use_ema", [False, True])
def test_inference_from_sharded_checkpoint(use_ema: bool):
    """Inference from a sharded run's checkpoint (raw or EMA weights) matches
    the serial model built from the same checkpoint."""
    case = CASES["context_pos_embed_labels"]
    dist = Distributed.get_instance()
    n_replicas = dist.total_data_parallel_ranks
    trained = _build_run(case, shard=True)
    _train(trained, case, range(N_STEPS), n_replicas)
    checkpoint = _checkpoint(trained)
    ema_checkpoint = {EMA_CHECKPOINT_KEY: checkpoint.ema}

    def load(run: Run):
        run.model.load_state_dict(checkpoint.module)
        if use_ema:
            assert EMATracker.copy_params_from_state(
                ema_checkpoint[EMA_CHECKPOINT_KEY], run.modules
            )

    predictions = []
    for shard in (True, False):
        run = _build_run(case, shard=shard, seed=1)
        load(run)
        predictions.append(_predict(run, case, n_replicas))
    with Distributed.force_non_distributed():
        serial = _build_run(case, shard=False, seed=1)
        load(serial)
        expected = _predict(serial, case, n_replicas)
    n = expected.shape[0] // n_replicas
    expected = expected[n * dist.data_parallel_rank : n * (dist.data_parallel_rank + 1)]
    torch.testing.assert_close(predictions[0], predictions[1], rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(
        predictions[0], expected, rtol=SERIAL_RTOL, atol=SERIAL_ATOL
    )


@pytest.mark.parallel
def test_state_dict_outside_gather_is_local_and_round_trips():
    """In-memory state (e.g. a stepper copy) keeps local slices, and loads."""
    case = CASES["sfno_pos_embed"]
    run = _build_run(case, shard=True)
    state = run.model.state_dict()
    for name, shard in named_parameter_shards(run.model).items():
        assert tuple(state[name].shape) == shard.local_shape
    other = _build_run(case, shard=True, seed=1)
    other.model.load_state_dict(state)
    for (name, a), b in zip(run.model.named_parameters(), other.model.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0, msg=name)


@pytest.mark.parallel
def test_gather_to_every_rank_then_overwrite_weights():
    """The pretrained-weights path: full weights on every rank (as
    ``load_weights_and_history`` reads them) initialize a sharded model."""
    from fme.core.weight_ops import overwrite_weights

    case = CASES["sfno_pos_embed"]
    source = _build_run(case, shard=True)
    with gathered_parameter_shards(dst=None):
        full = source.model.state_dict()
    with Distributed.force_non_distributed():
        serial = _build_run(case, shard=False)
    for name, value in serial.model.state_dict().items():
        torch.testing.assert_close(full[name].cpu(), value.cpu(), rtol=0, atol=0)
    target = _build_run(case, shard=True, seed=1)
    overwrite_weights(full, target.model)
    for (name, a), b in zip(source.model.named_parameters(), target.model.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0, msg=name)


@pytest.mark.parallel
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
@pytest.mark.parametrize("dims", [(0, 1), (0, None), (None, 1)])
def test_gather_shard_assembles_the_full_tensor_on_its_destination(dtype, dims):
    """On every rank (dst None), the root, or a rank of another replica."""
    dist = Distributed.get_instance()
    global_shape = (*IMG_SHAPE, 3)
    shard = spatial_shard(global_shape, IMG_SHAPE, dims)
    if shard is None:  # the layout does not split these dimensions
        return
    full = torch.arange(9 * 16 * 3, dtype=torch.float32).reshape(global_shape)
    if dtype.is_complex:
        full = torch.complex(full, -full)
    local = shard.take(full).to(get_device())
    for dst in (None, 0, dist.world_size - 1):
        with gathered_parameter_shards(dst=dst):
            gathered = gather_shard(local, shard)
        if dst is None or dist.rank == dst:
            assert gathered.device.type == "cpu"
            torch.testing.assert_close(gathered, full, rtol=0, atol=0)
        else:
            assert gathered is local
