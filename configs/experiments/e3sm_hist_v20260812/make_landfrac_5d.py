"""Write LANDFRAC / sea_surface_fraction on the ocean 5-day axis, one file per year.

LANDFRAC is an EAM field and is absent from the MPAS streams, but the coupled
ocean needs it (sea ice fraction of a grid cell = ocean_sea_ice_fraction *
(1 - LANDFRAC)). Merge members must share sample_start_times, so it cannot be
taken from the 6-hourly EAM stream directly and is materialised here instead.
It is time-invariant, so the files compress to almost nothing.
"""

import glob
import os
import re
import sys

import numpy as np
import xarray as xr

R = "/global/cfs/cdirs/e3smdata/simulations/v3.LR.historical_0101.aigo/run/"
OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)

atm = xr.open_dataset(
    sorted(glob.glob(R + "*eam.h0.1940-01.nc"))[0], decode_timedelta=False
)
lf = atm["LANDFRAC"].isel(time=0).clip(0.0, 1.0).values.astype("float32")

files = sorted(glob.glob(R + "*fmeDerivedFields5D.*.remapped.nc"))
by_year = {}
for p in files:
    y = re.search(r"5D\.(\d{4})-\d{2}\.remapped\.nc$", p).group(1)
    by_year.setdefault(y, []).append(p)


def enc():
    return {"zlib": True, "complevel": 4}


for y in sorted(by_year):
    times = []
    for p in by_year[y]:
        d = xr.open_dataset(p, decode_timedelta=False)
        times.append(d.time.values)
        d.close()
    t = np.concatenate(times)
    nt = len(t)
    arr = np.broadcast_to(lf, (nt,) + lf.shape)
    ds = xr.Dataset(
        {
            "LANDFRAC": (("time", "lat", "lon"), arr),
            "sea_surface_fraction": (
                ("time", "lat", "lon"),
                (1.0 - arr).astype("float32"),
            ),
        },
        coords={"time": t, "lat": atm.lat, "lon": atm.lon},
    )
    ds["LANDFRAC"].attrs = {"long_name": "land fraction", "units": "unitless"}
    ds["sea_surface_fraction"].attrs = {
        "long_name": "sea surface fraction",
        "units": "unitless",
    }
    ds.to_netcdf(f"{OUT}/landfrac5d.{y}.nc", encoding={v: enc() for v in ds.data_vars})
print("years written:", len(by_year))
os.system(f"du -sh {OUT}")
