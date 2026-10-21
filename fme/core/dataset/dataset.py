import abc
import logging
from typing import Any, final

import xarray as xr

from fme.core.dataset.properties import DatasetProperties
from fme.core.distributed import Distributed
from fme.core.generics.dataset import GenericDataset
from fme.core.typing_ import TensorDict

DatasetItem = tuple[
    TensorDict, xr.DataArray, set[str] | None, int | None, frozenset[str] | None
]


class DatasetABC(GenericDataset[DatasetItem], abc.ABC):
    @property
    @abc.abstractmethod
    def sample_start_times(self) -> xr.CFTimeIndex:
        pass

    @property
    @abc.abstractmethod
    def all_times(self) -> xr.CFTimeIndex:
        """
        Like sample_start_times, but includes all times in the dataset, including
        final times which are not valid as a start index.

        This is relevant for inference, where we may use get_sample_by_time_slice to
        retrieve time windows directly.

        If this dataset does not support inference,
        this will raise a NotImplementedError.
        """
        pass

    @abc.abstractmethod
    def __getitem__(self, index) -> DatasetItem:
        pass

    @final
    def __len__(self) -> int:
        return len(self.sample_start_times)

    @property
    @final
    def first_time(self) -> Any:
        return self.sample_start_times[0]

    @property
    @final
    def last_time(self) -> Any:
        return self.sample_start_times[-1]

    @property
    @abc.abstractmethod
    def sample_n_times(self) -> int:
        """The length of the time dimension of each sample."""
        pass

    @abc.abstractmethod
    def get_sample_by_time_slice(self, time_slice: slice) -> DatasetItem:
        pass

    @property
    @abc.abstractmethod
    def properties(self) -> DatasetProperties:
        pass

    @abc.abstractmethod
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
        pass

    @property
    def supports_spatial_tile(self) -> bool:
        """Whether ``set_spatial_tile`` can restrict this dataset's samples to a
        tile of the horizontal grid. A dataset that cannot keeps returning
        global fields, which its callers then slice themselves.
        """
        return False

    def set_spatial_tile(self, tile: tuple[slice, slice]) -> None:
        """Return only the ``(tile[0], tile[1])`` slab of the last two
        horizontal dimensions in every sample from now on, reading no more of
        the data than the backend requires to produce it.

        ``properties`` (coordinates, masks, metadata) stay global: only the
        sample tensors are restricted. The tile is plain data, so it reaches
        DataLoader workers with the dataset and needs no distributed state
        there.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support spatial tiles."
        )

    @abc.abstractmethod
    def enable_shared_memory(self):
        pass

    @abc.abstractmethod
    def set_global_epoch_tensor(self, tensor):
        pass


def restrict_to_local_spatial_tile(dataset: DatasetABC) -> tuple[slice, slice] | None:
    """Make ``dataset`` load only this rank's tile of the horizontal grid,
    under spatial parallelism and if it can.

    The tile is computed here, where the distributed backend knows the rank's
    (h, w) position, and handed to the dataset as plain slices, so DataLoader
    workers load the same tile without consulting the backend.

    Returns:
        The tile, or None if the dataset keeps loading the global grid.
    """
    shape = dataset.properties.horizontal_coordinates.shape
    tile = Distributed.get_instance().get_spatial_tile((shape[-2], shape[-1]))
    if tile is None:
        return None
    if not dataset.supports_spatial_tile:
        logging.info(
            "Loading global fields on every spatial rank: "
            f"{type(dataset).__name__} cannot load a spatial tile."
        )
        return None
    logging.info(f"Loading only this rank's spatial tile {tile} of the data.")
    dataset.set_spatial_tile(tile)
    return tile
