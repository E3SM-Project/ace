import pathlib
import typing
from typing import Self

import cftime
import netCDF4
import numpy as np
import torch
import xarray as xr

from fme.core.coordinates import LatLonCoordinates, NullVerticalCoordinate
from fme.core.dataset.data_typing import VariableMetadata
from fme.core.dataset.dataset import DatasetABC, DatasetItem
from fme.core.dataset.properties import DatasetProperties
from fme.core.spatial_mask_provider import SpatialMaskProvider
from fme.core.typing_ import TensorMapping

_DATASET_ITEM_FIELDS = ("data", "time", "labels", "epoch", "missing_names")


def assert_dataset_item_length(item: DatasetItem):
    n_type_args = len(typing.get_args(DatasetItem))
    if len(item) != n_type_args:
        raise AssertionError(
            f"DatasetItem has {n_type_args} type args but item has "
            f"{len(item)} elements."
        )
    if len(item) != len(_DATASET_ITEM_FIELDS):
        raise AssertionError(
            f"DatasetItem now has {len(item)} elements but "
            f"_DATASET_ITEM_FIELDS only covers {len(_DATASET_ITEM_FIELDS)}. "
            f"Update _DATASET_ITEM_FIELDS and the tests that use it."
        )


class MockDataset(DatasetABC):
    START_DATE = "2000-01-01"
    START_DATE_CF = cftime.DatetimeGregorian(2000, 1, 1)
    CALENDAR = "gregorian"

    @classmethod
    def time_to_int(cls, time: cftime.DatetimeGregorian) -> int:
        """Convert a cftime index to integer representation (days since START_DATE)."""
        ref = cls.START_DATE_CF
        delta = time - ref
        return delta.days

    def __init__(
        self,
        time: xr.CFTimeIndex,
        data: TensorMapping,
        sample_n_times: int,
        labels: set[str] | None = None,
        properties: DatasetProperties | None = None,
        initial_epoch: int | None = None,
        missing_names: frozenset[str] | None = None,
    ):
        self.labels = labels
        self.time = time
        self.data = data
        self._sample_n_times = sample_n_times
        self.missing_names = missing_names
        if properties is not None:
            self._properties = properties
        else:
            self._properties = DatasetProperties(
                variable_metadata={
                    name: VariableMetadata(units="", long_name="")
                    for name in data.keys()
                },
                vertical_coordinate=NullVerticalCoordinate(),
                horizontal_coordinates=LatLonCoordinates(
                    lon=torch.arange(0, 1), lat=torch.arange(0, 1)
                ),
                spatial_mask_provider=SpatialMaskProvider(),
                timestep=None,
                is_remote=False,
                all_labels=self.labels,
            )
        self.epoch = initial_epoch

    @classmethod
    def new(
        cls,
        n_times: int,
        varnames: list[str],
        sample_n_times: int,
        labels: set[str] | None = None,
        properties: DatasetProperties | None = None,
        initial_epoch: int | None = None,
        missing_names: frozenset[str] | None = None,
    ) -> Self:
        time = xr.date_range(
            cls.START_DATE,
            periods=n_times,
            freq="D",
            calendar=cls.CALENDAR,
            use_cftime=True,
        )
        data = {varname: torch.arange(n_times) for varname in varnames}
        return cls(
            time=time,
            data=data,
            sample_n_times=sample_n_times,
            labels=labels,
            properties=properties,
            initial_epoch=initial_epoch,
            missing_names=missing_names,
        )

    def __getitem__(self, index) -> DatasetItem:
        time_slice = slice(index, index + self._sample_n_times)
        return self.get_sample_by_time_slice(time_slice)

    @property
    def all_times(self) -> xr.CFTimeIndex:
        return self.time

    @property
    def sample_start_times(self) -> xr.CFTimeIndex:
        return self.time[: len(self.time) - self._sample_n_times + 1]

    @property
    def sample_n_times(self) -> int:
        """The length of the time dimension of each sample."""
        return self._sample_n_times

    def get_sample_by_time_slice(self, time_slice: slice) -> DatasetItem:
        data = {k: v[time_slice] for k, v in self.data.items()}
        time = xr.DataArray(self.time[time_slice], dims=["time"])
        return data, time, self.labels, self.epoch, self.missing_names

    @property
    def properties(self) -> DatasetProperties:
        return self._properties

    def validate_inference_length(self, max_start_index: int, max_window_len: int):
        """
        Validate that this dataset supports inference, and that the
        number of forward steps is valid for the dataset.
        Raises a ValueError if the number of forward steps is not valid or
        if the dataset does not support inference.

        Parameters:
            max_start_index: The maximum valid start index for inference.
            max_window_len: The maximum window length including the
                start index for inference.
        """
        if max_start_index + max_window_len > len(self):
            raise ValueError("Inference length exceeds dataset length.")

    def enable_shared_memory(self):
        pass

    def set_global_epoch_tensor(self, tensor):
        pass

    def set_epoch(self, epoch: int):
        self.epoch = epoch


TILE_TEST_N_LAT, TILE_TEST_N_LON = 9, 15  # odd: tiles split unevenly
TILE_TEST_TIME_DEPENDENT_NAMES = ("foo", "bar", "sst", "ocnfrac")
TILE_TEST_TIME_INVARIANT_NAMES = ("static",)


def write_tile_test_files(
    directory: pathlib.Path,
    nc_format: str,
    n_files: int = 2,
    n_times_per_file: int = 6,
    seed: int = 0,
) -> pathlib.Path:
    """Write a small lat/lon dataset split over several files, for checking
    that loading a spatial tile matches slicing the global fields.

    ``nc_format`` is a netCDF4-python format: ``NETCDF4`` for the netcdf4
    engine, ``NETCDF3_64BIT_DATA`` (CDF-5) for the cdf5 engine. Every value
    is distinct, so a misplaced row or column cannot go unnoticed. The
    ``foo`` field has NaNs to exercise ``fill_nans``.
    """
    rng = np.random.default_rng(seed)
    directory.mkdir(parents=True, exist_ok=True)
    shape = (n_times_per_file, TILE_TEST_N_LAT, TILE_TEST_N_LON)
    for i_file in range(n_files):
        path = directory / f"data_{i_file}.nc"
        with netCDF4.Dataset(path, "w", format=nc_format) as ds:
            ds.createDimension("time", None)
            ds.createDimension("lat", TILE_TEST_N_LAT)
            ds.createDimension("lon", TILE_TEST_N_LON)
            lat = ds.createVariable("lat", "f8", ("lat",))
            lat[:] = np.linspace(-80, 80, TILE_TEST_N_LAT)
            lon = ds.createVariable("lon", "f8", ("lon",))
            lon[:] = np.linspace(0, 360, TILE_TEST_N_LON, endpoint=False)
            time = ds.createVariable("time", "f8", ("time",))
            time.units = "days since 2000-01-01 00:00:00"
            time.calendar = "noleap"
            time[:] = 0.25 * (np.arange(n_times_per_file) + i_file * n_times_per_file)
            for name in TILE_TEST_TIME_DEPENDENT_NAMES:
                var = ds.createVariable(name, "f4", ("time", "lat", "lon"))
                var.units = "K"
                values = rng.random(shape, dtype=np.float32)
                if name == "foo":
                    values[:, 1, 2] = np.nan
                var[:] = values
            static = ds.createVariable("static", "f4", ("lat", "lon"))
            static[:] = rng.random(shape[1:], dtype=np.float32)
    return directory
