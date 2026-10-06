import torch

from fme.ace.data_loading.batch_data import BatchData, PrognosticState
from fme.core.random_state import RandomState
from fme.coupled.data_loading.batch_data import CoupledPrognosticState


def _get_coupled_ic() -> CoupledPrognosticState:
    return CoupledPrognosticState(
        ocean_data=PrognosticState(
            BatchData.new_for_testing(names=["o_prog"], n_timesteps=1)
        ),
        atmosphere_data=PrognosticState(
            BatchData.new_for_testing(names=["a_prog"], n_timesteps=1)
        ),
    )


def _get_random_state(state: PrognosticState) -> RandomState | None:
    stepper_state = state.as_batch_data().stepper_state
    return None if stepper_state is None else stepper_state.random_state


def _draw(random_state: RandomState | None) -> torch.Tensor:
    assert random_state is not None
    return torch.randn(8, generator=random_state.generator)


def test_apply_config_seed_seeds_each_component_with_its_own_stream():
    seeded = _get_coupled_ic().apply_config_seed(0)
    atmos_draw = _draw(_get_random_state(seeded.atmosphere_data))
    ocean_draw = _draw(_get_random_state(seeded.ocean_data))
    # The atmosphere is seeded exactly as a standalone atmosphere run would be.
    assert torch.equal(atmos_draw, _draw(RandomState.from_seed(0)))
    # The ocean gets a distinct stream, so stochastic components in both realms
    # do not draw identical noise.
    assert not torch.equal(atmos_draw, ocean_draw)
    reseeded = _get_coupled_ic().apply_config_seed(0)
    assert torch.equal(ocean_draw, _draw(_get_random_state(reseeded.ocean_data)))


def test_apply_config_seed_none_is_a_no_op():
    ic = _get_coupled_ic()
    assert ic.apply_config_seed(None) is ic


def test_apply_config_seed_defers_to_a_restored_random_state_per_component():
    ic = _get_coupled_ic()
    restored = RandomState.from_seed(11)
    ic = CoupledPrognosticState(
        ocean_data=ic.ocean_data.with_random_state(restored),
        atmosphere_data=ic.atmosphere_data,
    )
    result = ic.apply_config_seed(0)
    assert _get_random_state(result.ocean_data) is restored
    assert _get_random_state(result.atmosphere_data) is not None
