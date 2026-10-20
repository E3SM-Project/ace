"""Reassembling global fields from spatial tiles.

Run under torchrun with the model backend, e.g. (H, W) in (2, 1), (1, 2),
(2, 2); serially every gather is the identity.
"""

import pytest
import torch

from fme.core.device import get_device
from fme.core.distributed import Distributed

# 9 x 18 does not divide evenly over 2 tiles: exercises unequal tile shapes
IMG_SHAPES = [(8, 16), (9, 18)]


def _global_field(img_shape: tuple[int, int], dtype=torch.float32) -> torch.Tensor:
    n = 2 * 3 * img_shape[0] * img_shape[1]
    field = torch.arange(n, dtype=dtype).reshape(2, 3, *img_shape)
    field[0, 0, 0, 0] = float("nan")  # NaN must stay where it is
    return field.to(get_device())


def _tile(field: torch.Tensor) -> torch.Tensor:
    slices = Distributed.get_instance().get_local_slices(field.shape[-2:])
    return field[(..., *slices)].contiguous()


@pytest.mark.parallel
@pytest.mark.parametrize("img_shape", IMG_SHAPES)
@pytest.mark.parametrize("on_cpu", [False, True])
def test_gather_spatial_tile_matches_global(img_shape, on_cpu):
    field = _global_field(img_shape)
    tile = _tile(field)
    if on_cpu:
        tile = tile.cpu()
    gathered = Distributed.get_instance().gather_spatial_tile(tile)
    assert gathered.device == tile.device
    torch.testing.assert_close(
        gathered.cpu(), field.cpu(), rtol=0, atol=0, equal_nan=True
    )


@pytest.mark.parallel
@pytest.mark.parametrize("img_shape", IMG_SHAPES)
def test_gather_spatial_tile_to_root_matches_global(img_shape):
    dist = Distributed.get_instance()
    field = _global_field(img_shape, dtype=torch.float64)
    gathered = dist.gather_spatial_tile_to_root(_tile(field).cpu())
    if dist.is_spatial_root():
        assert gathered is not None
        assert gathered.device == torch.device("cpu")
        torch.testing.assert_close(
            gathered, field.cpu(), rtol=0, atol=0, equal_nan=True
        )
    else:
        assert gathered is None


@pytest.mark.parallel
@pytest.mark.parametrize("img_shape", IMG_SHAPES)
def test_gather_spatial_lat_matches_global(img_shape):
    dist = Distributed.get_instance()
    field = _global_field(img_shape)
    # a zonal mean: the same latitude band profile on every rank of a tile row
    profile = dist.zonal_mean(_tile(field))
    gathered = dist.gather_spatial_lat(profile)
    torch.testing.assert_close(
        gathered.cpu(), field.nanmean(dim=-1).cpu(), equal_nan=True
    )


@pytest.mark.parallel
def test_gather_spatial_tile_rejects_inconsistent_tiles():
    dist = Distributed.get_instance()
    if not dist.has_spatial_parallelism:
        pytest.skip("every shape is a tile without spatial parallelism")
    # tile heights that no split of a global field produces
    n_lat = 1 if dist.is_spatial_root() else 8
    with pytest.raises(ValueError, match="not this rank's tile"):
        dist.gather_spatial_tile(torch.zeros(n_lat, 16, device=get_device()))


@pytest.mark.parallel
def test_area_weighted_mean_dict_does_not_depend_on_dict_order():
    """Each variable's area-weighted mean is a collective over the tiles, and
    a dict of data can be ordered differently on different ranks (names come
    from sets, which every process hashes differently).
    """
    from fme.core.gridded_ops import LatLonOperations

    dist = Distributed.get_instance()
    img_shape = (9, 18)
    ops = LatLonOperations(torch.ones(img_shape))
    fields = {
        name: _tile(_global_field(img_shape) * (i + 1)).nan_to_num()
        for i, name in enumerate(["a", "b", "c"])
    }
    names = list(fields)
    if dist.rank % 2 == 1:
        names = names[::-1]
    means = ops.area_weighted_mean_dict({name: fields[name] for name in names})
    for i, name in enumerate(["a", "b", "c"]):
        expected = (_global_field(img_shape) * (i + 1)).nan_to_num().mean(dim=(-2, -1))
        torch.testing.assert_close(means[name].cpu(), expected.cpu())
