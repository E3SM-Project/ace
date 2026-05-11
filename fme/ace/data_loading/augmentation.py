import abc
import dataclasses
import re
from collections.abc import Sequence
from typing import Literal

import torch

from fme.ace.data_loading.batch_data import BatchData


@dataclasses.dataclass
class AugmentationConfig:
    """
    Configuration for data augmentation.

    Attributes:
        rotate_probability: The probability of rotating the sphere by 180 degrees,
            as a value between 0.0 and 1.0.
        additional_directional_names: Names of variables whose sign is flipped when
            the poles are reversed. By default this includes known directional
            names as stored in RotateModifier.FLIP_NAMES.
    """

    rotate_probability: float = 0.0
    additional_directional_names: list[str] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        if not 0.0 <= self.rotate_probability <= 1.0:
            raise ValueError(
                "rotate_probability must be between 0.0 and 1.0, "
                f"got {self.rotate_probability}"
            )

    def build_modifier(self) -> "BatchModifierABC":
        if self.rotate_probability > 0.0:
            return RotateModifier(
                self.rotate_probability, self.additional_directional_names
            )
        return NullModifier()


class BatchModifierABC(abc.ABC):
    @abc.abstractmethod
    def __call__(self, batch: BatchData) -> BatchData: ...


class RotateModifier(BatchModifierABC):
    """
    Modifier that rotates the sphere by 180 degrees so that the poles swap
    places. This is the same as flipping both zonal and meridional axes.

    Also flips the sign of horizontal directional variables such as horizontal
    winds in specific directions, so their new values reflect the rotated axes.
    The names of such variables are stored in the `FLIP_NAMES` class variable.
    Variables not included in this list are not flipped.

    Specifically, the regex pattern r'{name}(_?[0-9]+m?)?$' is used to match the
    names of variables whose sign is flipped when the poles are reversed, for
    each name in `FLIP_NAMES`. This will match both names that end with something
    like "_0", "_1", etc. or something like "10m" or "2m".

    Note that seasons are handled by the fact that solar insolation is a data
    variable, but time is not modified. This means monthly or seasonal averages
    using this data will be affected by the rotation.
    """

    # names of variables whose sign is flipped when the poles are reversed
    FLIP_NAMES = [
        "eastward_wind",
        "northward_wind",
        "UGRD",
        "VGRD",
        "U",
        "V",
    ]

    def __init__(
        self,
        rotate_probability: float,
        additional_directional_names: list[str],
    ):
        self.rotate_probability = rotate_probability
        self.additional_directional_names = additional_directional_names
        self._pattern = re.compile(
            r"({})(_?[0-9]+m?)?$".format(
                "|".join(self.FLIP_NAMES + self.additional_directional_names)
            )
        )

    def __call__(self, batch: BatchData) -> BatchData:
        if batch.horizontal_dims != ["lat", "lon"]:
            raise NotImplementedError(
                "Horizontal dimensions must be lat and lon to rotate the sphere, got "
                f"{batch.horizontal_dims}"
            )
        example_value = next(iter(batch.data.values()))
        apply = (
            torch.rand(example_value.shape[0]).to(example_value.device)
            < self.rotate_probability
        )
        while len(apply.shape) < len(example_value.shape):
            apply = apply.unsqueeze(-1)
        new_data = {}
        for name, value in batch.data.items():
            new_value = torch.flip(value, dims=[-2, -1])
            if self._pattern.match(name):
                new_value = -1 * new_value
            new_data[name] = torch.where(apply, new_value, value)
        return BatchData(
            data=new_data,
            time=batch.time,
            horizontal_dims=batch.horizontal_dims,
            labels=batch.labels,
        )


class NullModifier(BatchModifierABC):
    def __call__(self, batch: BatchData) -> BatchData:
        return batch


@dataclasses.dataclass
class RoundtripConfig:
    """
    Configuration for a runtime spherical-harmonics roundtrip filter on data
    batches.

    A forward SHT truncated to ``fraction_modes_kept`` of the modes, followed by
    the corresponding inverse SHT, is applied to selected variables. This
    produces a band-limited version of the field on the same grid. It mirrors
    the offline ``xtorch_harmonics.roundtrip_filter`` used in the data
    preprocessing scripts, but moves the filter into the data loader so the
    cutoff can be changed without regenerating the dataset.

    Under spatial parallelism the modifier runs after ``scatter_spatial`` and
    uses the distributed SHT/ISHT implementations exposed by
    :class:`fme.core.distributed.Distributed`.

    The truncation strategy is selected by ``mode``:

    * ``"degree_only"`` (default): bit-equivalent to the offline impl. Truncates
      only the degree axis with cutoff ``min(round(n_lat * frac), default_lmax)``
      and leaves ``mmax`` at the grid-aware default. Building an SHT with reduced
      ``lmax`` and full ``mmax`` produces the same field as the offline approach
      of zeroing ``sht[..., round(n_lat * frac):, :]`` at full Nyquist (omitted
      rows ≡ zero rows).
    * ``"degree_and_order"``: scales both ``lmax`` and ``mmax`` by ``frac`` against
      the grid-aware Nyquist. Note that in current ``torch_harmonics`` (0.8.x),
      ``default_mmax = nlon // 2 + 1 >= default_lmax``, so the additional
      ``mmax`` truncation is vacuous (real-SHT coefficients already satisfy
      ``|m| <= l``) and this mode produces identical fields to ``degree_only``.
      The switch is kept for forward-compat with API versions where
      ``default_mmax < default_lmax``.

    Comparison to the offline ``xtorch_harmonics.roundtrip_filter`` defaults:

    * Offline ``forward_grid``/``inverse_grid``: ``"legendre-gauss"``. This
      config defaults ``grid`` to ``None`` (auto-detect from the dataset). Set
      ``grid: legendre-gauss`` explicitly for parity.
    * Offline ``fraction_modes_kept``: ``None`` (no truncation, identity in the
      representable spectral subspace); same here, where ``None`` short-
      circuits to a :class:`NullModifier`.
    * Offline truncation: degree-axis only, cutoff scaled by ``n_lat``. Default
      ``mode`` here matches that and also caps at the grid-aware ``default_lmax``
      (which is a no-op on legendre-gauss where ``default_lmax == n_lat``, and
      avoids the offline pitfall on equiangular where ``round(n_lat*frac)`` can
      silently exceed ``default_lmax == (n_lat+1)//2``).
    * Offline applies to every data variable containing the lat/lon dims; this
      config supports an explicit ``variables`` allow-list.

    Example YAML (reproduces the offline filter)::

        train_loader:
          roundtrip:
            fraction_modes_kept: 0.5
            grid: legendre-gauss   # match offline default; omit for auto-detect
            # mode: degree_only             # default, matches offline truncation
            # variables: [air_temperature_0, ...]   # optional, default: all

    Attributes:
        fraction_modes_kept: Fraction of spherical-harmonic modes to retain in
            (0, 1]. ``None`` (default) disables the filter.
        variables: Names of variables to filter. ``None`` (default) applies to
            every variable in the batch.
        grid: Quadrature grid passed to the SHT/ISHT, e.g. ``"equiangular"`` or
            ``"legendre-gauss"``. ``None`` (default) auto-detects from the
            dataset's horizontal coordinates at ``build_modifier`` time.
        mode: Truncation strategy. ``"degree_only"`` (default) matches the
            offline filter; ``"degree_and_order"`` truncates both axes.
    """

    fraction_modes_kept: float | None = None
    variables: Sequence[str] | None = None
    grid: str | None = None
    mode: Literal["degree_only", "degree_and_order"] = "degree_only"

    def __post_init__(self):
        if self.fraction_modes_kept is not None and not (
            0.0 < self.fraction_modes_kept <= 1.0
        ):
            raise ValueError(
                "fraction_modes_kept must be in (0, 1], "
                f"got {self.fraction_modes_kept}"
            )
        if self.mode not in ("degree_only", "degree_and_order"):
            raise ValueError(
                "mode must be 'degree_only' or 'degree_and_order', "
                f"got {self.mode!r}"
            )

    def build_modifier(
        self,
        global_shape: tuple[int, int],
        default_grid: str | None = None,
    ) -> "BatchModifierABC":
        if self.fraction_modes_kept is None:
            return NullModifier()
        from fme.core.distributed import Distributed

        grid = self.grid if self.grid is not None else default_grid
        if grid is None:
            raise ValueError(
                "RoundtripConfig.grid is None and no default_grid was provided; "
                "specify the grid explicitly or pass a default from the dataset."
            )
        if grid == "healpix":
            raise NotImplementedError(
                "RoundtripConfig is only supported on lat/lon grids "
                "('equiangular' or 'legendre-gauss'), got 'healpix'."
            )
        nlat, nlon = global_shape
        comm = Distributed.get_instance()
        # Probe the default (Nyquist) truncation for this grid type — it is
        # grid-dependent (e.g. (nlat+1)//2 for equiangular, nlat for
        # legendre-gauss in torch_harmonics 0.9+).
        probe = comm.get_sht(nlat, nlon, grid=grid)
        default_lmax = int(probe.lmax)
        default_mmax = int(probe.mmax)
        if self.mode == "degree_only":
            # Offline parity: degree cutoff is round(n_lat * frac), capped at the
            # grid-aware Nyquist lmax. mmax stays at default — building an SHT
            # with reduced lmax and full mmax is equivalent to taking the full
            # transform and zeroing the highest L rows.
            lmax = max(1, min(int(round(self.fraction_modes_kept * nlat)), default_lmax))
            mmax = default_mmax
        else:  # "degree_and_order"
            lmax = max(1, int(round(self.fraction_modes_kept * default_lmax)))
            mmax = max(1, int(round(self.fraction_modes_kept * default_mmax)))
        sht = comm.get_sht(nlat, nlon, lmax=lmax, mmax=mmax, grid=grid)
        isht = comm.get_isht(nlat, nlon, lmax=lmax, mmax=mmax, grid=grid)
        return RoundtripModifier(
            sht=sht,
            isht=isht,
            variables=(None if self.variables is None else list(self.variables)),
        )


class RoundtripModifier(BatchModifierABC):
    """Apply a spherical-harmonic roundtrip (forward + inverse SHT) to selected
    variables of each batch.

    The forward and inverse transforms must already be configured with matching
    truncated ``lmax``/``mmax``; the filtering happens implicitly via the
    truncation, so the caller is responsible for choosing modes.
    """

    def __init__(
        self,
        sht: torch.nn.Module,
        isht: torch.nn.Module,
        variables: list[str] | None,
    ):
        self._sht = sht
        self._isht = isht
        self._variables = variables
        self._initialized_device: torch.device | None = None

    def _ensure_on_device(self, device: torch.device) -> None:
        if self._initialized_device != device:
            self._sht = self._sht.to(device)
            self._isht = self._isht.to(device)
            self._initialized_device = device

    def _filter(self, value: torch.Tensor) -> torch.Tensor:
        self._ensure_on_device(value.device)
        leading_shape = value.shape[:-2]
        flat = value.reshape(-1, *value.shape[-2:])
        # SHT buffers are float32; cast to match and back, preserving dtype.
        coeffs = self._sht(flat.to(torch.float32))
        out = self._isht(coeffs).to(value.dtype)
        return out.reshape(*leading_shape, *out.shape[-2:])

    def __call__(self, batch: BatchData) -> BatchData:
        if batch.horizontal_dims != ["lat", "lon"]:
            raise NotImplementedError(
                "Horizontal dimensions must be lat and lon for SHT roundtrip, "
                f"got {batch.horizontal_dims}"
            )
        new_data = {}
        for name, value in batch.data.items():
            if self._variables is not None and name not in self._variables:
                new_data[name] = value
                continue
            new_data[name] = self._filter(value)
        return BatchData(
            data=new_data,
            time=batch.time,
            horizontal_dims=batch.horizontal_dims,
            labels=batch.labels,
            epoch=batch.epoch,
            n_ensemble=batch.n_ensemble,
        )
