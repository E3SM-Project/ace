"""GPU memory instrumentation for torchrun ranks (opt-in by environment).

FME_GPU_MEM_CAP_GB=<GB>  cap this rank's caching allocator (emulate a smaller
                         card; the CUDA context and NCCL buffers come on top).
FME_MEMLOG_DIR=<dir>     every 5 s, append "time allocated reserved
                         max_allocated max_reserved" (GiB) for this rank to
                         <dir>/rank<R>.txt.
Loaded through PYTHONPATH (scripts/env.sh adds this directory when
FME_MEMTOOLS=1).
"""

import os

_rank = os.environ.get("LOCAL_RANK")
if _rank is not None and (
    os.environ.get("FME_GPU_MEM_CAP_GB") or os.environ.get("FME_MEMLOG_DIR")
):
    import threading
    import time

    import torch

    _dev = int(_rank)
    _cap = os.environ.get("FME_GPU_MEM_CAP_GB")
    if _cap:
        _total = torch.cuda.get_device_properties(_dev).total_memory / 2**30
        torch.cuda.set_per_process_memory_fraction(float(_cap) / _total, _dev)
    _dir = os.environ.get("FME_MEMLOG_DIR")
    if _dir:
        os.makedirs(_dir, exist_ok=True)
        _path = os.path.join(_dir, f"rank{os.environ.get('RANK', _rank)}.txt")

        def _log():
            g = 2**30
            while True:
                time.sleep(5)
                try:
                    if not torch.cuda.is_initialized():
                        continue
                    line = (
                        f"{time.strftime('%H:%M:%S')} "
                        f"{torch.cuda.memory_allocated(_dev) / g:.2f} "
                        f"{torch.cuda.memory_reserved(_dev) / g:.2f} "
                        f"{torch.cuda.max_memory_allocated(_dev) / g:.2f} "
                        f"{torch.cuda.max_memory_reserved(_dev) / g:.2f}\n"
                    )
                    with open(_path, "a") as f:
                        f.write(line)
                except Exception:
                    pass

        threading.Thread(target=_log, daemon=True).start()
