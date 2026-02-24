"""
Compatibility shim for torch_harmonics distributed transforms.

Some versions of torch_harmonics do not expose ``lmax_local``,
``mmax_local``, ``lpad_local``, and ``mpad_local`` as attributes on
``DistributedRealSHT`` / ``DistributedInverseRealSHT``.  The codebase
(e.g. ``s2convolutions.py``) reads these attributes on the transform
objects, so we patch them in after construction when they are missing.
"""

from __future__ import annotations

import torch.nn as nn
import torch_harmonics.distributed as thd


def _patch_sht_local_attrs(transform: nn.Module) -> nn.Module:
    """Add ``lmax_local``, ``mmax_local``, ``lpad_local``, ``mpad_local``
    to a distributed SHT / ISHT module if they are missing.

    The values are derived from ``l_shapes`` / ``m_shapes`` which *are*
    always present (set by all torch_harmonics versions we support).

    This is a no-op when the attributes already exist.
    """
    if not isinstance(
        transform, (thd.DistributedRealSHT, thd.DistributedInverseRealSHT)
    ):
        return transform

    if not hasattr(transform, "lmax_local"):
        transform.lmax_local = transform.l_shapes[transform.comm_rank_polar]

    if not hasattr(transform, "mmax_local"):
        transform.mmax_local = transform.m_shapes[transform.comm_rank_azimuth]

    if not hasattr(transform, "lpad_local"):
        # Pad = 0 because compute_split_shapes always sums to the total.
        transform.lpad_local = 0

    if not hasattr(transform, "mpad_local"):
        transform.mpad_local = 0

    return transform


def patch_distributed_sht(*transforms: nn.Module) -> None:
    """Convenience: patch multiple transforms in one call."""
    for t in transforms:
        _patch_sht_local_attrs(t)
