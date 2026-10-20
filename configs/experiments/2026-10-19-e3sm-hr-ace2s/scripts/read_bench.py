"""Per-reader read throughput of one HR EAM file: raw, netCDF4, xarray.

python read_bench.py FILE
"""

import os
import sys
import time

import netCDF4 as nc
import numpy as np
import xarray as xr

path = sys.argv[1]
names = ["PS", "TS", "T_0", "U_3", "V_5", "STW_2", "LHFLX", "FLUT"]
size = os.path.getsize(path)
res = {}

# raw sequential read of 4 GB from the middle of the file
with open(path, "rb", buffering=0) as f:
    f.seek(size // 3)
    t = time.time()
    n = 0
    while n < 4 * 2**30:
        b = f.read(64 * 2**20)
        n += len(b)
    res["raw_seq_MBps"] = n / (time.time() - t) / 1e6

ds = nc.Dataset(path)
ds.set_auto_mask(False)
for label, sl in [("nc4_1rec", slice(40, 41)), ("nc4_8rec", slice(50, 58))]:
    t = time.time()
    n = 0
    for k in names:
        n += np.asarray(ds.variables[k][sl]).nbytes
    res[f"{label}_MBps"] = n / (time.time() - t) / 1e6

x = xr.open_dataset(path, decode_times=False, mask_and_scale=False)
t = time.time()
n = 0
for i in (70, 71):  # what the loader does per sample: two timesteps, many vars
    for k in names:
        n += x[k].isel(time=i).values.nbytes
res["xarray_2rec_MBps"] = n / (time.time() - t) / 1e6
print(os.uname().nodename, {k: round(v) for k, v in res.items()})
