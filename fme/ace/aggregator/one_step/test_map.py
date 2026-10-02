import torch
import xarray as xr

from fme.ace.aggregator.one_step.map import MapAggregator
from fme.core.device import get_device
from fme.core.testing import mock_identical_ranks


def test_map_aggregator_repeated_reduction_matches_single_process():
    """
    Validation flushes diagnostics (get_dataset) before computing the summary
    (get_logs), so the cross-rank reduction must not change recorded state.
    """
    batch_size, n_time, n_lat, n_lon = 3, 2, 4, 8
    agg = MapAggregator(dims=["lat", "lon"])
    for _ in range(2):
        shape = (batch_size, n_time, n_lat, n_lon)
        target = {"a": torch.randn(shape, device=get_device())}
        gen = {"a": torch.randn(shape, device=get_device())}
        agg.record_batch(
            loss=1.0,
            target_data=target,
            gen_data=gen,
            target_data_norm=target,
            gen_data_norm=gen,
        )
    expected = agg.get_dataset()
    with mock_identical_ranks(world_size=4):
        first = agg.get_dataset()
        second = agg.get_dataset()
    xr.testing.assert_allclose(first, expected)
    xr.testing.assert_allclose(second, expected)
