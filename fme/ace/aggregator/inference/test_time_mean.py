import numpy as np
import torch

from fme.ace.aggregator.inference.data import InferenceBatchData, make_dummy_time
from fme.ace.aggregator.inference.time_mean import (
    TimeMeanAggregator,
    TimeMeanEvaluatorAggregator,
    TimeMeanMetricConfig,
)
from fme.core.device import get_device
from fme.core.gridded_ops import LatLonOperations


def test_rmse_of_time_mean_all_channels():
    torch.manual_seed(0)
    area_weights = torch.ones(1, 1).to(get_device())
    agg = TimeMeanEvaluatorAggregator(
        LatLonOperations(area_weights),
        horizontal_dims=["lat", "lon"],
        target="norm",
    )
    target_data_norm = {
        "a": torch.ones([2, 3, 4, 4], device=get_device()),
        "b": torch.ones([2, 3, 4, 4], device=get_device()) * 3,
    }
    gen_data_norm = {
        "a": torch.ones([2, 3, 4, 4], device=get_device()) * 2.0,
        "b": torch.ones([2, 3, 4, 4], device=get_device()) * 5,
    }
    agg.record_batch(
        InferenceBatchData(
            prediction=gen_data_norm,
            prediction_norm=gen_data_norm,
            target=target_data_norm,
            target_norm=target_data_norm,
            time=make_dummy_time(2, 3),
            i_time_start=0,
        )
    )
    logs = agg.get_logs(label="time_mean_norm")
    assert logs["time_mean_norm/rmse/a"] == 1.0
    assert logs["time_mean_norm/rmse/b"] == 2.0
    assert logs["time_mean_norm/rmse/channel_mean"] == 1.5


def test_channel_mean_excludes_all_nan_target_channels():
    """A variable whose target is entirely NaN (e.g. filled by
    allow_missing_variables) has a NaN RMSE and is excluded from the channel
    mean rather than poisoning it."""
    torch.manual_seed(0)
    area_weights = torch.ones(1, 1).to(get_device())
    agg = TimeMeanEvaluatorAggregator(
        LatLonOperations(area_weights),
        horizontal_dims=["lat", "lon"],
        target="norm",
    )
    target_data_norm = {
        "a": torch.ones([2, 3, 4, 4], device=get_device()),
        "b": torch.ones([2, 3, 4, 4], device=get_device()) * 3,
        # "c" is missing from the data: entirely-NaN target.
        "c": torch.full([2, 3, 4, 4], torch.nan, device=get_device()),
    }
    gen_data_norm = {
        "a": torch.ones([2, 3, 4, 4], device=get_device()) * 2.0,
        "b": torch.ones([2, 3, 4, 4], device=get_device()) * 5,
        "c": torch.ones([2, 3, 4, 4], device=get_device()),
    }
    agg.record_batch(
        InferenceBatchData(
            prediction=gen_data_norm,
            prediction_norm=gen_data_norm,
            target=target_data_norm,
            target_norm=target_data_norm,
            time=make_dummy_time(2, 3),
            i_time_start=0,
        )
    )
    logs = agg.get_logs(label="time_mean_norm")
    # "c" is still recorded per-variable as NaN...
    assert np.isnan(logs["time_mean_norm/rmse/c"])
    # ...but excluded from channel_mean: mean of "a" (1) and "b" (2).
    assert logs["time_mean_norm/rmse/channel_mean"] == 1.5


def test_custom_channel_mean_names():
    torch.manual_seed(0)
    area_weights = torch.ones(1, 1).to(get_device())
    agg = TimeMeanEvaluatorAggregator(
        LatLonOperations(area_weights),
        horizontal_dims=["lat", "lon"],
        target="norm",
        channel_mean_names=["a"],
    )
    target_data_norm = {
        "a": torch.ones([2, 3, 4, 4], device=get_device()),
        "b": torch.ones([2, 3, 4, 4], device=get_device()) * 3,
    }
    gen_data_norm = {
        "a": torch.ones([2, 3, 4, 4], device=get_device()) * 2.0,
        "b": torch.ones([2, 3, 4, 4], device=get_device()) * 5,
    }
    agg.record_batch(
        InferenceBatchData(
            prediction=gen_data_norm,
            prediction_norm=gen_data_norm,
            target=target_data_norm,
            target_norm=target_data_norm,
            time=make_dummy_time(2, 3),
            i_time_start=0,
        )
    )
    logs = agg.get_logs(label="time_mean_norm")
    assert logs["time_mean_norm/rmse/a"] == 1.0
    assert logs["time_mean_norm/rmse/b"] == 2.0
    assert logs["time_mean_norm/rmse/channel_mean"] == 1.0


def test_mean_all_channels_not_in_denorm():
    area_weights = torch.ones(1, 1).to(get_device())
    agg = TimeMeanEvaluatorAggregator(
        LatLonOperations(area_weights),
        horizontal_dims=["lat", "lon"],
        target="denorm",
    )
    target_data = {
        "a": torch.ones([2, 3, 4, 4], device=get_device()),
        "b": torch.ones([2, 3, 4, 4], device=get_device()) * 3,
    }
    gen_data = {
        "a": torch.ones([2, 3, 4, 4], device=get_device()) * 2.0,
        "b": torch.ones([2, 3, 4, 4], device=get_device()) * 5,
    }
    agg.record_batch(
        InferenceBatchData(
            prediction=gen_data,
            prediction_norm=gen_data,
            target=target_data,
            target_norm=target_data,
            time=make_dummy_time(2, 3),
            i_time_start=0,
        )
    )
    logs = agg.get_logs(label="time_mean")
    assert "time_mean/rmse/channel_mean" not in list(logs.keys())
    ds = agg.get_dataset()
    assert "bias_map-a" in ds
    assert np.all(ds["bias_map-a"].values == 1.0)


def test_bias_values():
    area_weights = torch.ones(1, 1).to(get_device())
    agg = TimeMeanEvaluatorAggregator(
        LatLonOperations(area_weights),
        horizontal_dims=["lat", "lon"],
        target="denorm",
    )
    # use constant values so area-weighting doesn't matter
    target_data = {
        "a": (torch.rand(1) * torch.ones(size=[2, 3, 4, 5])).to(device=get_device()),
    }
    gen_data = {
        "a": (torch.rand(1) * torch.ones(size=[2, 3, 4, 5])).to(device=get_device()),
    }
    agg.record_batch(
        InferenceBatchData(
            prediction=gen_data,
            prediction_norm=gen_data,
            target=target_data,
            target_norm=target_data,
            time=make_dummy_time(2, 3),
            i_time_start=0,
        )
    )
    ds = agg.get_dataset()
    assert "bias_map-a" in ds
    np.testing.assert_array_equal(
        ds["bias_map-a"].values,
        (
            gen_data["a"].cpu().numpy().mean(axis=(0, 1))
            - target_data["a"].cpu().numpy().mean(axis=(0, 1))
        ),
    )
    assert "gen_map-a" in ds
    np.testing.assert_array_equal(
        ds["gen_map-a"].values,
        (gen_data["a"].cpu().numpy().mean(axis=(0, 1))),
    )


def test_log_variables_does_not_affect_channel_mean():
    torch.manual_seed(0)
    area_weights = torch.ones(1, 1).to(get_device())
    target_data_norm = {
        "a": torch.ones([2, 3, 4, 4], device=get_device()),
        "b": torch.ones([2, 3, 4, 4], device=get_device()) * 3,
    }
    gen_data_norm = {
        "a": torch.ones([2, 3, 4, 4], device=get_device()) * 2.0,
        "b": torch.ones([2, 3, 4, 4], device=get_device()) * 5,
    }
    batch = InferenceBatchData(
        prediction=gen_data_norm,
        prediction_norm=gen_data_norm,
        target=target_data_norm,
        target_norm=target_data_norm,
        time=make_dummy_time(2, 3),
        i_time_start=0,
    )
    agg = TimeMeanEvaluatorAggregator(
        LatLonOperations(area_weights),
        horizontal_dims=["lat", "lon"],
        target="norm",
        log_variables=frozenset(["a"]),
    )
    agg.record_batch(batch)
    logs = agg.get_logs(label="time_mean_norm")
    assert logs["time_mean_norm/rmse/a"] == 1.0
    assert "time_mean_norm/rmse/b" not in logs
    assert logs["time_mean_norm/rmse/channel_mean"] == 1.5


def test_empty_log_variables_still_computes_channel_mean():
    torch.manual_seed(0)
    area_weights = torch.ones(1, 1).to(get_device())
    agg = TimeMeanEvaluatorAggregator(
        LatLonOperations(area_weights),
        horizontal_dims=["lat", "lon"],
        target="norm",
        log_variables=frozenset(),
    )
    target_data_norm = {
        "a": torch.ones([2, 3, 4, 4], device=get_device()),
        "b": torch.ones([2, 3, 4, 4], device=get_device()) * 3,
    }
    gen_data_norm = {
        "a": torch.ones([2, 3, 4, 4], device=get_device()) * 2.0,
        "b": torch.ones([2, 3, 4, 4], device=get_device()) * 5,
    }
    agg.record_batch(
        InferenceBatchData(
            prediction=gen_data_norm,
            prediction_norm=gen_data_norm,
            target=target_data_norm,
            target_norm=target_data_norm,
            time=make_dummy_time(2, 3),
            i_time_start=0,
        )
    )
    logs = agg.get_logs(label="time_mean_norm")
    assert "time_mean_norm/rmse/a" not in logs
    assert "time_mean_norm/rmse/b" not in logs
    assert logs["time_mean_norm/rmse/channel_mean"] == 1.5


def test_log_variables_with_channel_mean_names():
    torch.manual_seed(0)
    area_weights = torch.ones(1, 1).to(get_device())
    agg = TimeMeanEvaluatorAggregator(
        LatLonOperations(area_weights),
        horizontal_dims=["lat", "lon"],
        target="norm",
        channel_mean_names=["a"],
        log_variables=frozenset(["b"]),
    )
    target_data_norm = {
        "a": torch.ones([2, 3, 4, 4], device=get_device()),
        "b": torch.ones([2, 3, 4, 4], device=get_device()) * 3,
    }
    gen_data_norm = {
        "a": torch.ones([2, 3, 4, 4], device=get_device()) * 2.0,
        "b": torch.ones([2, 3, 4, 4], device=get_device()) * 5,
    }
    agg.record_batch(
        InferenceBatchData(
            prediction=gen_data_norm,
            prediction_norm=gen_data_norm,
            target=target_data_norm,
            target_norm=target_data_norm,
            time=make_dummy_time(2, 3),
            i_time_start=0,
        )
    )
    logs = agg.get_logs(label="time_mean_norm")
    assert "time_mean_norm/rmse/a" not in logs
    assert logs["time_mean_norm/rmse/b"] == 2.0
    assert logs["time_mean_norm/rmse/channel_mean"] == 1.0


def test_log_variables_filters_dataset():
    torch.manual_seed(0)
    area_weights = torch.ones(1, 1).to(get_device())
    agg = TimeMeanEvaluatorAggregator(
        LatLonOperations(area_weights),
        horizontal_dims=["lat", "lon"],
        target="denorm",
        log_variables=frozenset(["a"]),
    )
    target_data = {
        "a": torch.ones([2, 3, 4, 4], device=get_device()),
        "b": torch.ones([2, 3, 4, 4], device=get_device()) * 3,
    }
    gen_data = {
        "a": torch.ones([2, 3, 4, 4], device=get_device()) * 2.0,
        "b": torch.ones([2, 3, 4, 4], device=get_device()) * 5,
    }
    agg.record_batch(
        InferenceBatchData(
            prediction=gen_data,
            prediction_norm=gen_data,
            target=target_data,
            target_norm=target_data,
            time=make_dummy_time(2, 3),
            i_time_start=0,
        )
    )
    ds = agg.get_dataset()
    assert "bias_map-a" in ds
    assert "gen_map-a" in ds
    assert "bias_map-b" not in ds
    assert "gen_map-b" not in ds


def test_aggregator_mean_values():
    area_weights = torch.ones(1, 1).to(get_device())
    agg = TimeMeanAggregator(LatLonOperations(area_weights))
    # use constant values so area-weighting doesn't matter
    data = {
        "a": (torch.rand(1) * torch.ones(size=[2, 3, 4, 5])).to(device=get_device()),
    }
    agg.record_batch(
        InferenceBatchData(
            prediction=data,
            prediction_norm=data,
            target=None,
            target_norm=None,
            time=make_dummy_time(2, 3),
            i_time_start=0,
        )
    )
    ds = agg.get_dataset()
    assert "gen_map-a" in ds
    np.testing.assert_allclose(
        ds["gen_map-a"].values,
        (data["a"].cpu().numpy().mean(axis=(0, 1))),
    )


def _paired_batch():
    """One batch with two channels whose bias is known, for the plot tests."""
    device = get_device()
    target = {
        "a": torch.ones([2, 3, 4, 4], device=device),
        "b": torch.ones([2, 3, 4, 4], device=device) * 3,
    }
    gen = {
        "a": torch.ones([2, 3, 4, 4], device=device) * 2.0,
        "b": torch.ones([2, 3, 4, 4], device=device) * 5,
    }
    return InferenceBatchData(
        prediction=gen,
        prediction_norm=gen,
        target=target,
        target_norm=target,
        time=make_dummy_time(2, 3),
        i_time_start=0,
    )


def _image_keys(logs):
    return {k for k in logs if "map" in k}


def test_report_plot_false_drops_maps_but_keeps_every_scalar():
    """The storage lever: no map images, identical scalars.

    Uploading a per-variable map for every channel every epoch is the largest
    thing an inline-inference run sends to the experiment tracker, and it is
    also the thing least worth having there -- the same fields go to disk via
    get_dataset. Turning the plots off must not cost a single number.
    """
    area_weights = torch.ones(1, 1).to(get_device())
    scalars = {}
    images = {}
    for report_plot in (True, False):
        agg = TimeMeanEvaluatorAggregator(
            LatLonOperations(area_weights),
            horizontal_dims=["lat", "lon"],
            target="denorm",
            report_plot=report_plot,
        )
        agg.record_batch(_paired_batch())
        logs = agg.get_logs(label="time_mean")
        images[report_plot] = _image_keys(logs)
        scalars[report_plot] = {
            k: v for k, v in logs.items() if isinstance(v, int | float)
        }

    assert images[True], "expected map images when report_plot is on"
    assert images[False] == set(), f"maps leaked: {sorted(images[False])}"
    assert scalars[True] == scalars[False]
    assert scalars[False]["time_mean/rmse/a"] == 1.0
    assert scalars[False]["time_mean/bias/b"] == 2.0


def test_report_plot_false_still_writes_maps_to_disk():
    """report_plot is an upload switch, not a compute switch."""
    area_weights = torch.ones(1, 1).to(get_device())
    agg = TimeMeanEvaluatorAggregator(
        LatLonOperations(area_weights),
        horizontal_dims=["lat", "lon"],
        target="denorm",
        report_plot=False,
    )
    agg.record_batch(_paired_batch())
    ds = agg.get_dataset()
    assert "bias_map-a" in ds
    assert "gen_map-a" in ds


def test_report_plot_config_default_is_on():
    """Existing checkpoints and configs keep the maps they had."""
    assert TimeMeanMetricConfig().report_plot is True


def test_plot_variables_narrows_maps_but_not_scalars():
    """ "A select few spatial plots" -- the middle setting between all and none.

    report_plot is the master switch; plot_variables narrows what it allows.
    Neither may touch a scalar, because the per-variable rmse and bias are what
    every run is compared on.
    """
    area_weights = torch.ones(1, 1).to(get_device())
    agg = TimeMeanEvaluatorAggregator(
        LatLonOperations(area_weights),
        horizontal_dims=["lat", "lon"],
        target="denorm",
        plot_variables=frozenset(["a"]),
    )
    agg.record_batch(_paired_batch())
    logs = agg.get_logs(label="time_mean")

    maps = _image_keys(logs)
    assert any(k.endswith("/a") for k in maps), sorted(maps)
    assert not any(k.endswith("/b") for k in maps), sorted(maps)
    # every scalar survives for both channels
    assert logs["time_mean/rmse/a"] == 1.0
    assert logs["time_mean/rmse/b"] == 2.0
    assert logs["time_mean/bias/b"] == 2.0


def test_report_plot_false_beats_plot_variables():
    """The master switch wins; a stale plot_variables list cannot re-enable maps."""
    area_weights = torch.ones(1, 1).to(get_device())
    agg = TimeMeanEvaluatorAggregator(
        LatLonOperations(area_weights),
        horizontal_dims=["lat", "lon"],
        target="denorm",
        report_plot=False,
        plot_variables=frozenset(["a", "b"]),
    )
    agg.record_batch(_paired_batch())
    assert _image_keys(agg.get_logs(label="time_mean")) == set()
