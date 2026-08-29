"""Write LANDFRAC / sea_surface_fraction on an ocean time axis, one file per year.

    ./make_landfrac_ocn.py <outdir>              # 5-day axis (O5), the default
    ./make_landfrac_ocn.py <outdir> --cadence 1d # 1-day axis (O1)

LANDFRAC is an EAM field and is absent from the MPAS streams, but the coupled
ocean needs it (sea ice fraction of a grid cell = ocean_sea_ice_fraction *
(1 - LANDFRAC)). Merge members must share sample_start_times, so it cannot be
taken from the 6-hourly EAM stream directly and is materialised here instead.
It is time-invariant, so the files compress to almost nothing -- the only thing
that changes between cadences is which time axis it is broadcast onto, which is
read from the corresponding fmeDerivedFields stream.
"""

import argparse
import glob
import os
import re

import numpy as np
import xarray as xr

R = "/global/cfs/cdirs/e3smdata/simulations/v3.LR.historical_0101.aigo/run/"

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("outdir")
parser.add_argument("--cadence", choices=["5d", "1d"], default="5d",
                    help="ocean time axis to materialise onto (default 5d)")
args = parser.parse_args()
OUT = args.outdir
# The 1-day stream is the un-suffixed one; 5-day carries the 5D suffix.
SUFFIX = "5D" if args.cadence == "5d" else ""
PREFIX = "landfrac5d" if args.cadence == "5d" else "landfrac1d"
os.makedirs(OUT, exist_ok=True)

atm = xr.open_dataset(
    sorted(glob.glob(R + "*eam.h0.1940-01.nc"))[0], decode_timedelta=False
)
lf = atm["LANDFRAC"].isel(time=0).clip(0.0, 1.0).values.astype("float32")

files = sorted(glob.glob(R + f"*fmeDerivedFields{SUFFIX}.*.remapped.nc"))
by_year = {}
for p in files:
    y = re.search(rf"{SUFFIX}\.(\d{{4}})-\d{{2}}\.remapped\.nc$", p).group(1)
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
    ds.to_netcdf(f"{OUT}/{PREFIX}.{y}.nc", encoding={v: enc() for v in ds.data_vars})
print(f"cadence {args.cadence}: years written:", len(by_year))
os.system(f"du -sh {OUT}")
