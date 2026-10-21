import dataclasses
from collections.abc import Mapping
from typing import Any

import cftime
import numpy as np
import torch
import xarray as xr

from fme.core.dataset.data_typing import VariableMetadata
from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.gridded_ops import GriddedOperations
from fme.core.typing_ import TensorDict, TensorMapping
from fme.core.wandb import Image

from ..plotting import gather_map_to_root, images_enabled, plot_paneled_data
from .build_context import MetricBuildContext, MetricNotSupportedError, maybe_filter
from .data import InferenceBatchData, MetricBuildResult, SubAggregator

# A fixed Julian year (365.25 days) is used for the slope's per-year units
# regardless of the data calendar; this introduces a small (~1.5%) scaling
# error for non-365.25-day calendars (e.g. CMIP6 360-day), which is fine for
# a diagnostic trend.
SECONDS_PER_YEAR = 365.25 * 24 * 60 * 60
# A fixed reference epoch is used (rather than each sample's start time) so the
# running sums stay consistent across batches and distributed ranks without
# communication. The regression slope is shift-invariant, so the epoch does not
# affect the result; it is chosen near the climate-data era only to keep the
# magnitude of the time regressor small, which improves the floating-point
# conditioning of the (float64) running sums.
_REFERENCE_UNITS = "seconds since 2000-01-01 00:00:00"


def _years_since_reference(time: xr.DataArray) -> np.ndarray:
    """Convert a ``(sample, time)`` datetime array to floating-point years
    since a fixed reference epoch.
    """
    calendar = time.dt.calendar
    seconds = cftime.date2num(
        time.values.ravel(), units=_REFERENCE_UNITS, calendar=calendar
    ).reshape(time.shape)
    return seconds / SECONDS_PER_YEAR


class TrendEvaluatorAggregator:
    """Per-grid-cell linear trend (least-squares slope vs. time) maps.

    The trend of each variable at each grid cell is the ordinary least-squares
    slope of its value against time. It is computed by accumulating the
    regression sufficient statistics -- the running sums of ``t``, ``t**2``,
    ``y``, ``t*y`` and the observation count ``n`` -- in a single streaming
    pass over the inference batches. Because only these sums are retained, the
    memory cost is independent of the number of timesteps, so trends can be
    computed for runs whose full time series does not fit in memory.

    For grid cell with values ``y`` at times ``t`` (in years), the slope is

        slope = (n * sum(t*y) - sum(t) * sum(y)) / (n * sum(t**2) - sum(t)**2)

    Both the predicted and target trend maps are produced, along with a bias
    map (generated - target) and the area-weighted RMSE between them. Slope
    units are ``<variable units> / year``.
    """

    _image_captions = {
        "maps": (
            "{name} linear trend; (left) target and (right) generated "
            "[{units} / year]"
        ),
        "difference_map": (
            "{name} linear trend difference (generated - target) [{units} / year]"
        ),
    }

    def __init__(
        self,
        gridded_operations: GriddedOperations,
        horizontal_dims: list[str],
        variable_metadata: Mapping[str, VariableMetadata] | None = None,
    ):
        """
        Args:
            gridded_operations: Computes gridded (area-weighted) operations.
            horizontal_dims: Names of the horizontal dimensions, used for the
                output dataset.
            variable_metadata: Mapping of variable names to metadata used in
                image captions and dataset attributes.
        """
        self._ops = gridded_operations
        self._horizontal_dims = horizontal_dims
        self._variable_metadata = variable_metadata or {}
        device = get_device()
        # time regressor statistics, shared across variables and grid cells
        self._n = torch.zeros((), dtype=torch.float64, device=device)
        self._sum_t = torch.zeros((), dtype=torch.float64, device=device)
        self._sum_tt = torch.zeros((), dtype=torch.float64, device=device)
        # per-variable [n_lat, n_lon] running sums of y and t*y
        self._target_sum_y: TensorDict = {}
        self._target_sum_ty: TensorDict = {}
        self._gen_sum_y: TensorDict = {}
        self._gen_sum_ty: TensorDict = {}

    @staticmethod
    def _add_running_sums(
        sum_y: TensorDict,
        sum_ty: TensorDict,
        tensor_data: TensorMapping,
        t: torch.Tensor,
    ) -> None:
        """Accumulate sum(y) and sum(t*y) over the (sample, time) dimensions.

        Args:
            sum_y: Running sum of y, mutated in place.
            sum_ty: Running sum of t*y, mutated in place.
            tensor_data: Mapping of variable name to a (sample, time, lat, lon)
                tensor, already sliced to the timesteps being recorded.
            t: Time in years with shape (sample, time), matching ``tensor_data``.
        """
        t_broadcast = t[:, :, None, None]
        for name, tensor in tensor_data.items():
            y = tensor.to(torch.float64)
            contrib_y = y.sum(dim=(0, 1))
            contrib_ty = (t_broadcast * y).sum(dim=(0, 1))
            if name in sum_y:
                sum_y[name] = sum_y[name] + contrib_y
                sum_ty[name] = sum_ty[name] + contrib_ty
            else:
                sum_y[name] = contrib_y
                sum_ty[name] = contrib_ty

    @torch.no_grad()
    def record_batch(self, data: InferenceBatchData) -> None:
        # Mirror TimeMeanAggregator: when the initial condition is the first
        # timestep of the first batch, skip it so it is not double-counted.
        time_slice = slice(1, None) if data.i_time_start == 0 else slice(None)
        years = _years_since_reference(data.time)[:, time_slice]
        t = torch.tensor(years, dtype=torch.float64, device=get_device())
        if t.shape[1] == 0:
            return
        self._n = self._n + t.numel()
        self._sum_t = self._sum_t + t.sum()
        self._sum_tt = self._sum_tt + (t * t).sum()
        gen = {name: tensor[:, time_slice] for name, tensor in data.prediction.items()}
        self._add_running_sums(self._gen_sum_y, self._gen_sum_ty, gen, t)
        target = {name: tensor[:, time_slice] for name, tensor in data.target.items()}
        self._add_running_sums(self._target_sum_y, self._target_sum_ty, target, t)

    @staticmethod
    def _reduce_running_sums(
        dist: Distributed, sum_y: TensorDict, sum_ty: TensorDict
    ) -> tuple[list[str], torch.Tensor, torch.Tensor]:
        """Reduce the per-variable running sums across ranks, stacked in a
        rank-deterministic (sorted) order.
        """
        names = sorted(sum_y.keys())
        reduced_y = dist.reduce_sum(torch.stack([sum_y[name] for name in names], dim=0))
        reduced_ty = dist.reduce_sum(
            torch.stack([sum_ty[name] for name in names], dim=0)
        )
        return names, reduced_y, reduced_ty

    def _get_trends(self) -> tuple[TensorDict, TensorDict]:
        """Reduce sufficient statistics across ranks and finalize the target
        and generated slopes, on every rank. Under spatial parallelism they
        are this rank's tile of the slope maps.

        All collectives are issued unconditionally and in the same order on
        every rank.
        """
        if not self._gen_sum_y:
            raise ValueError("No data has been recorded yet.")
        dist = Distributed.get_instance()
        # The time-regressor statistics are shared between target and generated,
        # so reduce them once. Clone first so repeated get_logs/get_dataset
        # calls stay correct (reduce_sum mutates its argument in place).
        n = dist.reduce_sum(self._n.clone())
        sum_t = dist.reduce_sum(self._sum_t.clone())
        sum_tt = dist.reduce_sum(self._sum_tt.clone())
        gen = self._reduce_running_sums(dist, self._gen_sum_y, self._gen_sum_ty)
        target = self._reduce_running_sums(
            dist, self._target_sum_y, self._target_sum_ty
        )
        denom = n * sum_tt - sum_t * sum_t

        def slopes(reduced: tuple[list[str], torch.Tensor, torch.Tensor]) -> TensorDict:
            names, sum_y, sum_ty = reduced
            return {
                name: (n * sum_ty[i] - sum_t * sum_y[i]) / denom
                for i, name in enumerate(names)
            }

        return slopes(target), slopes(gen)

    def _caption(self, key: str, name: str) -> str:
        if name in self._variable_metadata:
            caption_name = self._variable_metadata[name].display_long_name(name)
            units = self._variable_metadata[name].display_units("unknown units")
        else:
            caption_name, units = name, "unknown units"
        return self._image_captions[key].format(name=caption_name, units=units)

    @torch.no_grad()
    def get_logs(self, label: str) -> dict[str, float | Image]:
        """Returns the logs on the root process, and an empty dict elsewhere."""
        target_trends, gen_trends = self._get_trends()
        dist = Distributed.get_instance()
        plot = images_enabled()
        images: dict[str, Image] = {}
        metrics: dict[str, float] = {}
        for name in gen_trends:
            # collectives on every rank: the RMSE reduces over the spatial
            # tiles, and the maps are assembled from them
            metrics[f"weighted_rmse/{name}"] = float(
                self._ops.area_weighted_rmse(
                    predicted=gen_trends[name].to(torch.float32),
                    truth=target_trends[name].to(torch.float32),
                    name=name,
                )
                .cpu()
                .numpy()
            )
            if not plot:
                continue
            target_map = gather_map_to_root(target_trends[name])
            gen_map = gather_map_to_root(gen_trends[name])
            if target_map is None or gen_map is None:
                continue
            images[f"maps/{name}"] = plot_paneled_data(
                [[target_map, gen_map]],
                diverging=True,
                caption=self._caption("maps", name),
            )
            images[f"difference_map/{name}"] = plot_paneled_data(
                [[gen_map - target_map]],
                diverging=True,
                caption=self._caption("difference_map", name),
            )
        if not dist.is_root():
            return {}
        logs: dict[str, Any] = {}
        if len(label) > 0:
            label = label + "/"
        logs.update({f"{label}{key}": image for key, image in images.items()})
        logs.update({f"{label}{key}": metric for key, metric in metrics.items()})
        return logs

    def _var_attrs(self, name: str) -> dict[str, str]:
        if name in self._variable_metadata:
            long_name = self._variable_metadata[name].display_long_name(name)
            units = self._variable_metadata[name].display_units("unknown units")
        else:
            long_name, units = name, "unknown units"
        return {
            "long_name": f"{long_name} linear trend",
            "units": f"{units} / year",
        }

    def get_dataset(self) -> xr.Dataset:
        """Returns the dataset on the root process, and an empty one elsewhere."""
        target_trends, gen_trends = self._get_trends()
        target_maps = {
            name: gather_map_to_root(value) for name, value in target_trends.items()
        }
        gen_maps = {
            name: gather_map_to_root(value) for name, value in gen_trends.items()
        }
        if not Distributed.get_instance().is_root():
            return xr.Dataset()
        target_ds = xr.Dataset(
            {
                name: (self._horizontal_dims, value, self._var_attrs(name))
                for name, value in target_maps.items()
            }
        ).expand_dims({"source": ["target"]})
        gen_ds = xr.Dataset(
            {
                name: (self._horizontal_dims, value, self._var_attrs(name))
                for name, value in gen_maps.items()
            }
        ).expand_dims({"source": ["prediction"]})
        return xr.concat([target_ds, gen_ds], dim="source")


@dataclasses.dataclass
class TrendMetricConfig:
    """Configuration for the linear-trend-map inference metric.

    Attributes:
        variables: Variables to compute trends for. If ``None``, all available
            variables are included.
        name: Name used to label the metric's logs and diagnostics.
        enabled: Whether the metric is computed. Disabled by default.
        strict: If True, raise rather than skip when the metric is not
            supported for the current configuration.
    """

    variables: list[str] | None = None
    name: str = "trend"
    enabled: bool = False
    strict: bool = False

    def get_name(self) -> str:
        return self.name

    def build(self, ctx: MetricBuildContext) -> MetricBuildResult:
        # The initial condition is dropped when it is the first recorded
        # timestep, so require at least two forward steps to leave >= 2 points
        # for the regression (otherwise the slope is an undefined 0/0).
        if ctx.n_forward_steps < 2:
            raise MetricNotSupportedError(
                "trend metric requires at least 2 forward steps, got "
                f"{ctx.n_forward_steps}"
            )
        agg: SubAggregator = TrendEvaluatorAggregator(
            ctx.ops,
            horizontal_dims=list(ctx.horizontal_coordinates.dims),
            variable_metadata=ctx.variable_metadata,
        )
        return MetricBuildResult(aggregator=maybe_filter(agg, self.variables))
