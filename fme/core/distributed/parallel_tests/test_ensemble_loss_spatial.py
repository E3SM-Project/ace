"""
Serial-vs-spatial measurements for the ACE2S training loss.

Each test computes the same quantity twice inside one process:

- **serial**: under ``Distributed.force_non_distributed()``, on the full
  global field, with the non-distributed SHT;
- **spatial**: with whatever backend the session runs (``torchrun`` with
  ``FME_DISTRIBUTED_BACKEND=model``), on this rank's tile, with the
  distributed SHT and the spatial gradient hooks of ``wrap_module``.

and compares loss value, gradient norm and gradient direction.  The loss is
built exactly as ``Stepper.build_loss`` builds it (``StepLossConfig.build``
with the dataset's gridded operations, ``channel_dim=-3`` and ensemble-shaped
``[B, E, H, W]`` predictions).

Under the spatial backend the gradient each rank applies is the sum over ranks
of the gradient of each rank's *local* loss graph (distributed SHT transposes
route gradients between ranks, and ``wrap_module`` all-reduces parameter
gradients over the spatial group).  So the "effective" objective is the sum
over ranks of each rank's local loss value; that is what the gradient ratios
measure.

When ``FME_SPATIAL_LOSS_REPORT`` is set, the root rank appends one JSON line
per measured case to that path.  Assertions run only after every collective
has completed so a failing assertion on the root cannot hang the other ranks.
"""

import dataclasses
import datetime
import json
import os

import numpy as np
import pytest
import torch

import fme
from fme.ace.registry.stochastic_sfno import (
    NoiseConditionedSFNOBuilder,
    isotropic_noise,
)
from fme.core.coordinates import HybridSigmaPressureCoordinate, LatLonCoordinates
from fme.core.corrector.atmosphere import AtmosphereCorrectorConfig
from fme.core.dataset_info import DatasetInfo
from fme.core.distributed.distributed import Distributed
from fme.core.loss import StepLossConfig
from fme.core.normalizer import StandardNormalizer
from fme.core.registry import ModuleSelector
from fme.core.step.args import StepArgs
from fme.core.step.single_module import SingleModuleStepConfig
from fme.core.step.step import StepSelector
from fme.core.testing import trivial_network_and_loss_normalization

IMG_SHAPE = (16, 32)
NAMES = ["a", "b", "c", "d"]
N_BATCH = 2
N_ENSEMBLE = 2
TIMESTEP = datetime.timedelta(hours=6)

LOSS_CASES = {
    "crps": {"crps_weight": 1.0, "energy_score_weight": 0.0},
    "energy": {"crps_weight": 0.0, "energy_score_weight": 1.0},
    "combined": {"crps_weight": 0.9, "energy_score_weight": 0.1},
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _decomposition() -> tuple[int, int]:
    return (
        int(os.environ.get("FME_DISTRIBUTED_H", "1")),
        int(os.environ.get("FME_DISTRIBUTED_W", "1")),
    )


def _report(record: dict) -> None:
    path = os.environ.get("FME_SPATIAL_LOSS_REPORT")
    dist = Distributed.get_instance()
    if path and dist.is_root():
        h, w = _decomposition()
        record = {"H": h, "W": w, "world": dist.world_size, **record}
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")


def _gaussian_coordinates(img_shape: tuple[int, int]) -> LatLonCoordinates:
    nlat, nlon = img_shape
    x, _ = np.polynomial.legendre.leggauss(nlat)
    lat = torch.as_tensor(np.degrees(np.arcsin(x)), dtype=torch.float32)
    lon = torch.linspace(0.0, 360.0, nlon + 1)[:-1]
    device = fme.get_device()
    return LatLonCoordinates(lat=lat.to(device), lon=lon.to(device))


def _build_step_loss(kwargs: dict, coords: LatLonCoordinates):
    """Build the loss the way ``Stepper.build_loss`` does."""
    config = StepLossConfig(type="EnsembleLoss", kwargs=dict(kwargs))
    normalizer = StandardNormalizer(
        means={n: torch.tensor(0.0) for n in NAMES},
        stds={n: torch.tensor(1.0) for n in NAMES},
    )
    return config.build(
        coords.get_gridded_operations(),
        out_names=NAMES,
        normalizer=normalizer,
        channel_dim=-3,
    )


def _local(data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return Distributed.get_instance().scatter_spatial(data, IMG_SHAPE)


def _gather(t: torch.Tensor) -> torch.Tensor:
    return Distributed.get_instance().gather_spatial_tensor(t.detach(), IMG_SHAPE)


def _all_rank_values(value: torch.Tensor) -> list[float]:
    """Every rank's scalar, on every rank (all-gather via spatial all-reduce)."""
    dist = Distributed.get_instance()
    buf = torch.zeros(dist.world_size, dtype=torch.float64, device=value.device)
    buf[dist.rank] = value.detach().to(torch.float64)
    if dist.world_size > 1:
        torch.distributed.all_reduce(buf)
    return buf.tolist()


def _compare(serial: torch.Tensor, spatial: torch.Tensor) -> dict[str, float]:
    s = serial.detach().flatten().to(torch.float64)
    p = spatial.detach().flatten().to(torch.float64)
    return {
        "norm_ratio": float(p.norm() / s.norm()),
        "cosine": float(torch.dot(s, p) / (s.norm() * p.norm())),
        "max_abs_diff_rel": float((p - s).abs().max() / s.abs().max()),
    }


def _flat_grads(module: torch.nn.Module) -> torch.Tensor:
    return torch.cat(
        [
            (p.grad if p.grad is not None else torch.zeros_like(p)).flatten()
            for p in module.parameters()
        ]
    )


def _copy_parameters(src: torch.nn.Module, dst: torch.nn.Module) -> None:
    src_params = dict(src.named_parameters())
    dst_params = dict(dst.named_parameters())
    assert src_params.keys() == dst_params.keys()
    with torch.no_grad():
        for name, p in dst_params.items():
            p.copy_(src_params[name])


# ---------------------------------------------------------------------------
# (a), (b), (c): loss value and gradient w.r.t. predictions
# ---------------------------------------------------------------------------


@pytest.mark.parallel
@pytest.mark.parametrize("case", list(LOSS_CASES))
def test_ensemble_loss_prediction_gradient(case: str):
    dist = Distributed.get_instance()
    device = fme.get_device()
    torch.manual_seed(0)
    gen = {
        n: torch.randn(N_BATCH, N_ENSEMBLE, *IMG_SHAPE, device=device) for n in NAMES
    }
    target = {n: torch.randn(N_BATCH, 1, *IMG_SHAPE, device=device) for n in NAMES}

    with Distributed.force_non_distributed():
        loss_fn = _build_step_loss(LOSS_CASES[case], _gaussian_coordinates(IMG_SHAPE))
        gen_s = {k: v.clone().requires_grad_() for k, v in gen.items()}
        loss_s = loss_fn(gen_s, target, step=0).total()
        grad_s = torch.autograd.grad(loss_s, [gen_s[n] for n in NAMES])
    grad_s_all = torch.stack(grad_s)

    loss_fn = _build_step_loss(LOSS_CASES[case], _gaussian_coordinates(IMG_SHAPE))
    gen_p = {k: v.clone().requires_grad_() for k, v in _local(gen).items()}
    loss_p = loss_fn(gen_p, _local(target), step=0).total()
    grad_p = torch.autograd.grad(loss_p, [gen_p[n] for n in NAMES])
    grad_p_all = torch.stack([_gather(g) for g in grad_p])

    rank_losses = _all_rank_values(loss_p)
    # what the train aggregator logs: reduce_mean over the DATA group only
    logged = dist.reduce_mean(loss_p.detach().clone())
    logged_all = _all_rank_values(logged)
    serial = float(loss_s.detach())
    cmp = _compare(grad_s_all, grad_p_all)
    record = {
        "test": "prediction_gradient",
        "case": case,
        "serial_loss": serial,
        "rank_losses": rank_losses,
        "root_loss_ratio": rank_losses[0] / serial,
        "sum_rank_loss_ratio": sum(rank_losses) / serial,
        "mean_rank_loss_ratio": sum(rank_losses) / len(rank_losses) / serial,
        "logged_loss_per_rank": logged_all,
        **{f"grad_{k}": v for k, v in cmp.items()},
    }
    _report(record)
    if dist.is_root():
        print(json.dumps(record))
        assert rank_losses[0] == pytest.approx(serial, rel=1e-4)
        assert cmp["norm_ratio"] == pytest.approx(1.0, rel=1e-4)
        assert cmp["cosine"] == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# (a)/(b) end to end: parameter gradients of a tiny NoiseConditionedSFNO
# ---------------------------------------------------------------------------

N_IN = 3


def _build_sfno(coords: LatLonCoordinates) -> torch.nn.Module:
    builder = NoiseConditionedSFNOBuilder(
        embed_dim=8,
        noise_embed_dim=4,
        noise_type="isotropic",
        filter_type="linear",
        use_mlp=True,
        num_layers=2,
        affine_norms=True,
    )
    dataset_info = DatasetInfo(horizontal_coordinates=coords, timestep=TIMESTEP)
    torch.manual_seed(0)
    module = builder.build(
        n_in_channels=N_IN, n_out_channels=len(NAMES), dataset_info=dataset_info
    ).to(fme.get_device())
    return Distributed.get_instance().wrap_module(module)


def _sfno_loss(module, loss_fn, x: torch.Tensor, target, noise_seed: int):
    torch.manual_seed(noise_seed)
    out = module(x.reshape(-1, *x.shape[-3:]))
    out = out.reshape(N_BATCH, N_ENSEMBLE, len(NAMES), *out.shape[-2:])
    gen = {n: out[:, :, i] for i, n in enumerate(NAMES)}
    return loss_fn(gen, target, step=0).total(), out


def _area_weighted_mse_total(coords: LatLonCoordinates):
    """Control: a correctly spatially-reduced loss (all-reduce of local sums)."""
    ops = coords.get_gridded_operations()

    class _Total:
        def __call__(self, gen, target, step):
            vals = [
                ops.area_weighted_mean((gen[n] - target[n]) ** 2).mean() for n in NAMES
            ]

            class _Out:
                def total(self_inner):
                    return torch.stack(vals).mean()

            return _Out()

    return _Total()


SFNO_CASES = [*LOSS_CASES, "area_weighted_mse_control"]


@pytest.mark.parallel
@pytest.mark.parametrize("case", SFNO_CASES)
def test_ensemble_loss_parameter_gradient(case: str):
    dist = Distributed.get_instance()
    device = fme.get_device()
    torch.manual_seed(1)
    x = torch.randn(N_BATCH, N_ENSEMBLE, N_IN, *IMG_SHAPE, device=device)
    target = {n: torch.randn(N_BATCH, 1, *IMG_SHAPE, device=device) for n in NAMES}

    def get_loss_fn(coords):
        if case == "area_weighted_mse_control":
            return _area_weighted_mse_total(coords)
        return _build_step_loss(LOSS_CASES[case], coords)

    with Distributed.force_non_distributed():
        coords = _gaussian_coordinates(IMG_SHAPE)
        model_s = _build_sfno(coords)
        loss_s, out_s = _sfno_loss(model_s, get_loss_fn(coords), x, target, 7)
        loss_s.backward()
        grad_s = _flat_grads(model_s.module)

    coords = _gaussian_coordinates(IMG_SHAPE)
    model_p = _build_sfno(coords)
    _copy_parameters(model_s.module, model_p.module)
    loss_p, out_p = _sfno_loss(
        model_p,
        get_loss_fn(coords),
        _local({"x": x})["x"],
        _local(target),
        7,
    )
    loss_p.backward()
    grad_p = _flat_grads(model_p.module)
    out_cmp = _compare(out_s, _gather(out_p))
    rank_losses = _all_rank_values(loss_p)
    cmp = _compare(grad_s, grad_p)
    # the hook makes every rank's gradient identical; check it
    grad_norms = _all_rank_values(grad_p.norm())
    record = {
        "test": "parameter_gradient",
        "case": case,
        "serial_loss": float(loss_s.detach()),
        "rank_losses": rank_losses,
        "root_loss_ratio": rank_losses[0] / float(loss_s.detach()),
        "sum_rank_loss_ratio": sum(rank_losses) / float(loss_s.detach()),
        "forward_max_abs_diff_rel": out_cmp["max_abs_diff_rel"],
        "grad_norm_per_rank": grad_norms,
        **{f"grad_{k}": v for k, v in cmp.items()},
    }
    _report(record)
    if dist.is_root():
        print(json.dumps(record))
        assert out_cmp["max_abs_diff_rel"] < 1e-4
        assert cmp["norm_ratio"] == pytest.approx(1.0, rel=1e-3)
        assert cmp["cosine"] == pytest.approx(1.0, abs=1e-4)


# ---------------------------------------------------------------------------
# (d) isotropic noise consistency across spatial co-ranks
# ---------------------------------------------------------------------------


def _noise(seed_offset_by_rank: bool, extra_draw_on_rank: int | None):
    dist = Distributed.get_instance()
    coords = _gaussian_coordinates(IMG_SHAPE)
    model = _build_sfno(coords).module
    isht = model.conditional_model.itrans_up
    seed = 11 + (dist.rank if seed_offset_by_rank else 0)
    torch.manual_seed(seed)
    if (
        extra_draw_on_rank is not None
        and dist.world_size > 1
        and dist.rank == extra_draw_on_rank
    ):
        torch.randn(1)
    return isotropic_noise(
        (N_BATCH, 4), model._lmax, model._mmax, isht, device=fme.get_device()
    )


@pytest.mark.parallel
@pytest.mark.parametrize(
    "scenario",
    ["same_seed", "seed_plus_rank", "one_extra_draw_on_last_rank"],
)
def test_isotropic_noise_consistent_across_tiles(scenario: str):
    dist = Distributed.get_instance()
    with Distributed.force_non_distributed():
        noise_s = _noise(False, None)
    noise_p = _noise(
        seed_offset_by_rank=scenario == "seed_plus_rank",
        extra_draw_on_rank=(
            dist.world_size - 1 if scenario == "one_extra_draw_on_last_rank" else None
        ),
    )
    cmp = _compare(noise_s, _gather(noise_p))
    record = {"test": "isotropic_noise", "case": scenario, **cmp}
    _report(record)
    if dist.is_root():
        print(json.dumps(record))
        # Only identical RNG streams are expected to agree; the other
        # scenarios are measurements of how fragile that assumption is.
        if scenario == "same_seed" or dist.world_size == 1:
            assert cmp["max_abs_diff_rel"] < 1e-5


# ---------------------------------------------------------------------------
# (e) corrector: gradients through mid-graph global means
# ---------------------------------------------------------------------------

_CORRECTOR_IN = [
    "DSWRFtoa",
    "HGTsfc",
    *[f"air_temperature_{i}" for i in range(6)],
    *[f"specific_total_water_{i}" for i in range(6)],
    "PRESsfc",
]
_CORRECTOR_OUT = [
    *[f"air_temperature_{i}" for i in range(6)],
    *[f"specific_total_water_{i}" for i in range(6)],
    "PRESsfc",
    "PRATEsfc",
    "LHTFLsfc",
    "tendency_of_total_water_path_due_to_advection",
]


def _build_corrector_step(coords: LatLonCoordinates, corrector: bool):
    names = sorted(set(_CORRECTOR_IN).union(_CORRECTOR_OUT))
    selector = StepSelector(
        type="single_module",
        config=dataclasses.asdict(
            SingleModuleStepConfig(
                builder=ModuleSelector(
                    type="NoiseConditionedSFNO",
                    config=dataclasses.asdict(
                        NoiseConditionedSFNOBuilder(
                            embed_dim=8,
                            noise_embed_dim=4,
                            noise_type="isotropic",
                            num_layers=2,
                            affine_norms=True,
                        )
                    ),
                ),
                in_names=_CORRECTOR_IN,
                out_names=_CORRECTOR_OUT,
                normalization=trivial_network_and_loss_normalization(names),
                corrector=AtmosphereCorrectorConfig(
                    conserve_dry_air=True,
                    moisture_budget_correction="advection_and_precipitation",
                )
                if corrector
                else AtmosphereCorrectorConfig(),
            ),
        ),
    )
    device = fme.get_device()
    bk = torch.linspace(0.0, 1.0, 7, device=device)
    ak = (1.0 - bk) * 1.0e4
    dataset_info = DatasetInfo(
        horizontal_coordinates=coords,
        vertical_coordinate=HybridSigmaPressureCoordinate(ak=ak, bk=bk),
        timestep=TIMESTEP,
    )
    torch.manual_seed(0)
    return selector.get_step(dataset_info, lambda _: None)


def _corrector_data():
    device = fme.get_device()
    torch.manual_seed(3)
    n = N_BATCH * N_ENSEMBLE

    def field(name):
        if name == "PRESsfc":
            return 1.0e5 + 1.0e3 * torch.randn(n, *IMG_SHAPE, device=device)
        return torch.rand(n, *IMG_SHAPE, device=device)

    inp = {k: field(k) for k in _CORRECTOR_IN}
    nxt = {k: field(k) for k in ["DSWRFtoa", "HGTsfc"]}
    target = {k: field(k) for k in _CORRECTOR_OUT}
    return inp, nxt, target


def _corrector_loss(step, coords, inp, nxt, target, seed):
    ops = coords.get_gridded_operations()
    torch.manual_seed(seed)
    out = step.step(
        args=StepArgs(input=inp, next_step_input_data=nxt, labels=None),
        wrapper=lambda m: m,
    ).output
    # correctly reduced (all-reduce of local sums) normalized MSE
    scale = {k: target[k].std() for k in _CORRECTOR_OUT}
    loss = torch.stack(
        [
            ops.area_weighted_mean(((out[k] - target[k]) / scale[k]) ** 2).mean()
            for k in _CORRECTOR_OUT
        ]
    ).mean()
    return loss, out


@pytest.mark.parallel
@pytest.mark.parametrize("corrector", [False, True])
def test_corrector_parameter_gradient(corrector: bool):
    dist = Distributed.get_instance()
    inp, nxt, target = _corrector_data()
    # target std must be global, compute it from global data on every rank
    with Distributed.force_non_distributed():
        coords = _gaussian_coordinates(IMG_SHAPE)
        step_s = _build_corrector_step(coords, corrector)
        loss_s, out_s = _corrector_loss(step_s, coords, inp, nxt, target, 5)
        loss_s.backward()
        grad_s = torch.cat([_flat_grads(m) for m in step_s.modules])
        out_s_all = torch.stack([out_s[k] for k in _CORRECTOR_OUT])

    coords = _gaussian_coordinates(IMG_SHAPE)
    step_p = _build_corrector_step(coords, corrector)
    for m_s, m_p in zip(step_s.modules, step_p.modules):
        _copy_parameters(m_s.module, m_p.module)
    loc_target = _local(target)
    ops = coords.get_gridded_operations()
    torch.manual_seed(5)
    out_p = step_p.step(
        args=StepArgs(input=_local(inp), next_step_input_data=_local(nxt), labels=None),
        wrapper=lambda m: m,
    ).output
    loss_p = torch.stack(
        [
            ops.area_weighted_mean(
                ((out_p[k] - loc_target[k]) / target[k].std()) ** 2
            ).mean()
            for k in _CORRECTOR_OUT
        ]
    ).mean()
    loss_p.backward()
    grad_p = torch.cat([_flat_grads(m.module) for m in step_p.modules])
    out_p_all = torch.stack([_gather(out_p[k]) for k in _CORRECTOR_OUT])
    out_cmp = _compare(out_s_all, out_p_all)
    cmp = _compare(grad_s, grad_p)
    rank_losses = _all_rank_values(loss_p)
    record = {
        "test": "corrector_parameter_gradient",
        "case": "conserve_dry_air+advection_and_precipitation"
        if corrector
        else "no_corrector",
        "serial_loss": float(loss_s.detach()),
        "root_loss_ratio": rank_losses[0] / float(loss_s.detach()),
        "forward_max_abs_diff_rel": out_cmp["max_abs_diff_rel"],
        **{f"grad_{k}": v for k, v in cmp.items()},
    }
    _report(record)
    if dist.is_root():
        print(json.dumps(record))
        assert out_cmp["max_abs_diff_rel"] < 1e-4
        assert cmp["norm_ratio"] == pytest.approx(1.0, rel=1e-3)
        assert cmp["cosine"] == pytest.approx(1.0, abs=1e-4)


def _physical_fields(n: int, generator: torch.Generator) -> dict[str, torch.Tensor]:
    """Roughly realistic magnitudes for the fields the corrector touches."""
    device = fme.get_device()
    shape = (n, *IMG_SHAPE)

    def randn():
        return torch.randn(shape, generator=generator).to(device)

    def rand():
        return torch.rand(shape, generator=generator).to(device)

    fields = {
        "PRESsfc": 1.0e5 + 1.0e3 * randn(),
        "PRATEsfc": 3.0e-5 * rand(),
        "LHTFLsfc": 80.0 + 40.0 * randn(),
        "tendency_of_total_water_path_due_to_advection": 1.0e-5 * randn(),
    }
    for i in range(6):
        fields[f"specific_total_water_{i}"] = 1.0e-3 * (6 - i) * (0.5 + rand())
    return fields


def _corrector_function_loss(gen, inp, target, scale, coords, vc):
    ops = coords.get_gridded_operations()
    corrector = AtmosphereCorrectorConfig(
        conserve_dry_air=True,
        moisture_budget_correction="advection_and_precipitation",
    )._build(ops, vc, TIMESTEP)
    out = corrector(inp, gen, {}, None).corrected
    return torch.stack(
        [
            ops.area_weighted_mean(((out[k] - target[k]) / scale[k]) ** 2).mean()
            for k in target
        ]
    ).mean()


@pytest.mark.parallel
@pytest.mark.parametrize("regime", ["gen_near_target", "gen_independent"])
def test_corrector_function_gradient(regime: str):
    """Gradient w.r.t. the generated fields through the real AtmosphereCorrector.

    The corrector uses area-weighted global means mid-graph; under the spatial
    backend those are ``_AutogradAllReduce`` whose backward is the identity.
    """
    dist = Distributed.get_instance()
    device = fme.get_device()
    g = torch.Generator().manual_seed(17)
    inp = _physical_fields(N_BATCH, g)
    target = _physical_fields(N_BATCH, g)
    if regime == "gen_near_target":
        noise = _physical_fields(N_BATCH, g)
        gen = {k: target[k] + 0.1 * (noise[k] - noise[k].mean()) for k in target}
    else:
        gen = _physical_fields(N_BATCH, g)
    scale = {k: v.std() for k, v in target.items()}
    loss_names = [
        "PRESsfc",
        "PRATEsfc",
        "tendency_of_total_water_path_due_to_advection",
    ]
    target = {k: target[k] for k in loss_names}
    bk = torch.linspace(0.0, 1.0, 7, device=device)
    vc = HybridSigmaPressureCoordinate(ak=(1.0 - bk) * 1.0e4, bk=bk)
    names = sorted(gen)

    with Distributed.force_non_distributed():
        coords = _gaussian_coordinates(IMG_SHAPE)
        gen_s = {k: v.clone().requires_grad_() for k, v in gen.items()}
        loss_s = _corrector_function_loss(gen_s, inp, target, scale, coords, vc)
        loss_s.backward()
    coords = _gaussian_coordinates(IMG_SHAPE)
    gen_p = {k: v.clone().requires_grad_() for k, v in _local(gen).items()}
    loss_p = _corrector_function_loss(
        gen_p, _local(inp), _local(target), scale, coords, vc
    )
    loss_p.backward()
    # fields the corrector overwrites (advection) get no gradient
    names = [k for k in names if gen_s[k].grad is not None]
    per_field = {}
    for k in names:
        per_field[k] = _compare(gen_s[k].grad, _gather(gen_p[k].grad))
    all_s = torch.cat([gen_s[k].grad.flatten() / gen_s[k].grad.norm() for k in names])
    all_p = torch.cat(
        [_gather(gen_p[k].grad).flatten() / gen_s[k].grad.norm() for k in names]
    )
    cmp = _compare(all_s, all_p)
    record = {
        "test": "corrector_function_gradient",
        "case": regime,
        "loss_ratio": float(loss_p.detach()) / float(loss_s.detach()),
        **{f"grad_{k}": v for k, v in cmp.items()},
        "per_field": {
            k: per_field[k]
            for k in ["PRESsfc", "PRATEsfc", "LHTFLsfc", "specific_total_water_0"]
            if k in per_field
        },
    }
    _report(record)
    if dist.is_root():
        print(json.dumps(record))
        assert cmp["cosine"] == pytest.approx(1.0, abs=1e-4)
        assert cmp["norm_ratio"] == pytest.approx(1.0, rel=1e-3)
