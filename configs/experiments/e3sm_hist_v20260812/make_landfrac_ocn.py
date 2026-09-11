"""Write LANDFRAC / sea_surface_fraction on an ocean time axis, one file per year.

    ./make_landfrac_ocn.py <outdir>              # 5-day axis (O5), the default
    ./make_landfrac_ocn.py <outdir> --cadence 1d # 1-day axis (O1)

LANDFRAC is an EAM field and is absent from the MPAS streams, but the coupled
ocean needs it (sea ice fraction of a grid cell = ocean_sea_ice_fraction *
(1 - LANDFRAC)). The files also carry atmosphere_flux_fraction, a static
data-only field the coupled stepper multiplies into the atmosphere-to-ocean
fluxes: the fraction of each ocean cell's atmosphere-ocean flux the ocean
actually receives, which is 1 everywhere except under Antarctic ice shelves
(MPAS-Ocean has ocean there, the atmosphere sees land, the ocean-side flux is
zero). It is derived from the data as the ratio of the MPAS ocean-side
downwelling longwave to EAM's cell-mean FLDS times the ice-free fraction,
summed over --flux-fraction-years (2026-09-10, see AGENTS.md).
Merge members must share sample_start_times, so it cannot be
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
parser.add_argument(
    "--cadence",
    choices=["5d", "1d"],
    default="5d",
    help="ocean time axis to materialise onto (default 5d)",
)
parser.add_argument(
    "--flux-fraction-years",
    default="1990,1991",
    help="years whose EAM/MPAS FLDS ratio defines "
    "atmosphere_flux_fraction (default 1990,1991)",
)
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


def atmosphere_flux_fraction(years):
    """Ratio of MPAS ocean-side FLDS to EAM cell-mean FLDS x (1 - ice fraction),
    accumulated over 5-day windows of the given years; 1 where undefined.
    """
    num = np.zeros(lf.shape)
    den = np.zeros(lf.shape)
    wet = np.zeros(lf.shape, dtype=bool)
    for y in years:
        for m in range(1, 13):
            ym = f"{y}-{m:02d}"
            ocn = xr.open_dataset(
                R
                + f"*fmeDerivedFields5D.{ym}.remapped.nc".replace(
                    "*", "v3.LR.historical_0101.aigo.mpaso.hist.am."
                ),
                decode_timedelta=False,
                use_cftime=True,
            )
            ice = xr.open_dataset(
                R
                + "v3.LR.historical_0101.aigo.mpassi.hist.am."
                + f"fmeSeaiceDerivedFields5D.{ym}.remapped.nc",
                decode_timedelta=False,
                use_cftime=True,
            )
            months = [(y - 1, 12)] if m == 1 else []
            months += [(y, m)]
            eam = xr.concat(
                [
                    xr.open_dataset(
                        R + f"v3.LR.historical_0101.aigo.eam.h0.{yy}-{mm:02d}.nc",
                        decode_timedelta=False,
                        use_cftime=True,
                    )[["FLDS", "time_bnds"]].load()
                    for yy, mm in months
                ],
                "time",
            )
            atb = eam.time_bnds.values
            for i, (lo, hi) in enumerate(ocn.time_bnds.values):
                sel = (atb[:, 0] >= lo) & (atb[:, 1] <= hi)
                if sel.sum() == 0:
                    continue
                flds = eam.FLDS.values[sel].mean(0)
                mpas = ocn.longWaveHeatFluxDown.values[i]
                wet |= np.isfinite(mpas)
                ow = 1.0 - np.nan_to_num(ice.iceAreaTotal.values[i])
                num += np.nan_to_num(mpas)
                den += flds * ow
    frac = np.where(wet & (den > 0), num / np.maximum(den, 1e-6), 1.0)
    return np.clip(frac, 0.0, 1.0).astype("float32")


afx = atmosphere_flux_fraction([int(y) for y in args.flux_fraction_years.split(",")])
print("atmosphere_flux_fraction < 0.9 in", int((afx < 0.9).sum()), "cells")

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
            "atmosphere_flux_fraction": (
                ("time", "lat", "lon"),
                np.broadcast_to(afx, (nt,) + afx.shape),
            ),
        },
        coords={"time": t, "lat": atm.lat, "lon": atm.lon},
    )
    ds["LANDFRAC"].attrs = {"long_name": "land fraction", "units": "unitless"}
    ds["sea_surface_fraction"].attrs = {
        "long_name": "sea surface fraction",
        "units": "unitless",
    }
    ds["atmosphere_flux_fraction"].attrs = {
        "long_name": "fraction of the atmosphere-ocean flux received by the "
        "ocean (ice-shelf cavities excluded)",
        "units": "unitless",
    }
    ds.to_netcdf(f"{OUT}/{PREFIX}.{y}.nc", encoding={v: enc() for v in ds.data_vars})
print(f"cadence {args.cadence}: years written:", len(by_year))
os.system(f"du -sh {OUT}")
