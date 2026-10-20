"""Helpers for tests that compare spatially parallel outputs with serial ones.

Such tests run under torchrun with the model backend: a serial reference is
computed in-process (under ``Distributed.force_non_distributed``) and the
root rank compares it with the parallel outputs.
"""

import pathlib
import re
from typing import Any

import numpy as np
import xarray as xr

from fme.core.distributed import Distributed

_NUMBER = re.compile(r"-?\d+\.?\d*(?:e[-+]?\d+)?")


def shared_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    """The root rank's ``tmp_path``, the same directory on every rank."""
    return Distributed.get_instance().scatter_object(tmp_path)


def _assert_caption_close(serial: str, parallel: str, key: str):
    """Captions agree in their text, and in their numbers (e.g. the color
    scale limits, which are the data's min and max) to 4 digits.
    """
    assert _NUMBER.sub("#", serial) == _NUMBER.sub("#", parallel), key
    np.testing.assert_allclose(
        [float(x) for x in _NUMBER.findall(parallel)],
        [float(x) for x in _NUMBER.findall(serial)],
        rtol=1e-3,
        err_msg=key,
    )


def assert_logs_close(
    serial: list[dict[str, Any]],
    parallel: list[dict[str, Any]],
    ignore: tuple[str, ...] = (),
):
    """The same keys; the same scalars; images and videos with the same
    captions. Keys containing any of ``ignore`` (e.g. timings) are skipped.
    """
    assert len(serial) == len(parallel)
    for i, (s, p) in enumerate(zip(serial, parallel)):
        assert set(s) == set(p), (i, sorted(set(s) ^ set(p)))
        for key, value in s.items():
            if any(pattern in key for pattern in ignore):
                continue
            if hasattr(value, "_caption"):  # an image or a video
                _assert_caption_close(value._caption, p[key]._caption, key)
            elif isinstance(value, float | int | np.floating | np.ndarray):
                np.testing.assert_allclose(
                    p[key], value, rtol=1e-5, atol=1e-6, err_msg=f"{i} {key}"
                )


def assert_netcdf_files_close(serial_dir: pathlib.Path, parallel_dir: pathlib.Path):
    """The same netCDF files, recursively, with the same variables and values."""
    serial_files = sorted(
        str(p.relative_to(serial_dir)) for p in serial_dir.rglob("*.nc")
    )
    assert len(serial_files) > 0
    parallel_files = sorted(
        str(p.relative_to(parallel_dir)) for p in parallel_dir.rglob("*.nc")
    )
    assert serial_files == parallel_files
    for name in serial_files:
        with (
            xr.open_dataset(serial_dir / name, decode_timedelta=False) as s,
            xr.open_dataset(parallel_dir / name, decode_timedelta=False) as p,
        ):
            assert set(s.data_vars) == set(p.data_vars), name
            xr.testing.assert_allclose(s.load(), p.load(), rtol=1e-5, atol=1e-6)
