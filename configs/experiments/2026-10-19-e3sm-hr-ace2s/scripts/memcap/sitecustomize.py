"""Emulate a smaller GPU: cap the caching allocator of every torchrun rank.

Active only when FME_GPU_MEM_CAP_GB is set; put this directory first on
PYTHONPATH. The cap leaves room for the CUDA context and NCCL buffers, which
the caching allocator does not count.
"""

import os

_cap = os.environ.get("FME_GPU_MEM_CAP_GB")
if _cap and "LOCAL_RANK" in os.environ:
    import torch

    _dev = int(os.environ["LOCAL_RANK"])
    _total = torch.cuda.get_device_properties(_dev).total_memory / 1e9
    torch.cuda.set_per_process_memory_fraction(float(_cap) / _total, _dev)
