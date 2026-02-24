# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Communicator shim for Makani / FCN3.

Delegates to the ACE :class:`~fme.core.distributed.Distributed` singleton so
that the rest of the vendored Makani code works unchanged.

Supported group names: ``"h"``, ``"w"``, ``"spatial"``, ``"matmul"``.
``"matmul"`` parallelism is not used (always returns size 1 / rank 0).
"""

from __future__ import annotations


def _dist():
    """Lazy import to avoid circular imports at module level."""
    from fme.core.distributed import Distributed

    return Distributed.get_instance()


def get_size(name: str) -> int:
    """Return the number of ranks in the named group."""
    d = _dist()
    if name == "h":
        return d.h_size
    if name == "w":
        return d.w_size
    if name == "spatial":
        return d.h_size * d.w_size
    if name == "matmul":
        return 1
    raise ValueError(f"Unknown comm group name: {name!r}")


def get_rank(name: str) -> int:
    """Return the rank of this process in the named group."""
    d = _dist()
    if name == "h":
        return d.h_rank
    if name == "w":
        return d.w_rank
    if name == "spatial":
        # Row-major linearised rank within the (h, w) tile.
        return d.h_rank * d.w_size + d.w_rank
    if name == "matmul":
        return 0
    raise ValueError(f"Unknown comm group name: {name!r}")


def get_group(name: str):
    """Return the ``torch.distributed`` process group for the named group."""
    d = _dist()
    if name == "h":
        return d.h_group
    if name == "w":
        return d.w_group
    if name == "spatial":
        return d.spatial_group
    if name == "matmul":
        return None
    raise ValueError(f"Unknown comm group name: {name!r}")


def is_distributed(name: str) -> bool:
    """Return whether the named group has more than one rank."""
    return get_size(name) > 1
