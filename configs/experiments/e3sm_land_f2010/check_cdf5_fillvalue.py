#!/usr/bin/env python3
"""Compare declared vs stored fill values in a CDF5 netCDF file, stdlib only.

Parses the header bytes directly from the netCDF CDF5 format spec
(https://docs.unidata.ucar.edu/netcdf-c/current/file_format_specifications.html)
and reads the stored float32/float64 values straight from disk, so the result
does not depend on netCDF4, xarray or their fill-value decoding.

For every (lat, lon) variable it reports the declared ``_FillValue`` and
``missing_value``, and how many cells of one time record hold the sentinel
``|x| >= 1e19`` versus the declared fill value.

    python3 check_cdf5_fillvalue.py <file.nc> [--record N] [--var NAME ...]

Exit status is 1 if any variable stores large sentinels that do not match its
declared ``_FillValue``, so it can be run over a directory as a check.
"""

import argparse
import struct
import sys

# nc_type -> (struct code, size in bytes)
_TYPES = {
    1: ("b", 1),
    2: ("c", 1),
    3: ("h", 2),
    4: ("i", 4),
    5: ("f", 4),
    6: ("d", 8),
    7: ("B", 1),
    8: ("H", 2),
    9: ("I", 4),
    10: ("q", 8),
    11: ("Q", 8),
}
_SENTINEL = 1e19


class _Cdf5:
    def __init__(self, path):
        self.f = open(path, "rb")
        if self.f.read(4) != b"CDF\x05":
            raise ValueError(f"{path} is not CDF5 (64-bit data) format")
        self.numrecs = self._i64()
        self._i32()
        self.dims = [(self._name(), self._i64()) for _ in range(self._i64())]
        self.global_attrs = self._attrs()
        self._i32()
        self.vars = {}
        for _ in range(self._i64()):
            name = self._name()
            dims = [self.dims[self._i64()] for _ in range(self._i64())]
            attrs = self._attrs()
            nc_type, vsize, begin = self._i32(), self._i64(), self._i64()
            self.vars[name] = dict(
                dims=dims, attrs=attrs, type=nc_type, vsize=vsize, begin=begin
            )
        # record variables are interleaved; one record is all of them in turn
        self.recsize = sum(v["vsize"] for v in self.vars.values() if self._is_rec(v))

    @staticmethod
    def _is_rec(var):
        return bool(var["dims"]) and var["dims"][0][1] == 0

    def _i32(self):
        return struct.unpack(">i", self.f.read(4))[0]

    def _i64(self):
        return struct.unpack(">q", self.f.read(8))[0]

    def _skip_pad(self, n):
        self.f.read((4 - n % 4) % 4)

    def _name(self):
        n = self._i64()
        s = self.f.read(n).decode()
        self._skip_pad(n)
        return s

    def _attrs(self):
        self._i32()
        out: dict[str, str | tuple] = {}
        for _ in range(self._i64()):
            key = self._name()
            nc_type, n = self._i32(), self._i64()
            code, size = _TYPES[nc_type]
            raw = self.f.read(n * size)
            self._skip_pad(n * size)
            if nc_type == 2:
                out[key] = raw.decode(errors="replace")
            else:
                out[key] = struct.unpack(f">{n}{code}", raw)
        return out

    def read_2d(self, name, record=0):
        """Return the flat (lat*lon) values of one record of a 2D/3D variable."""
        var = self.vars[name]
        code, size = _TYPES[var["type"]]
        shape = dict(self.dims)
        n = shape["lat"] * shape["lon"]
        offset = var["begin"] + (record * self.recsize if self._is_rec(var) else 0)
        self.f.seek(offset)
        return struct.unpack(f">{n}{code}", self.f.read(n * size))


def _fmt(value):
    return "-" if value is None else f"{value:.4g}"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path")
    parser.add_argument("--record", type=int, default=0)
    parser.add_argument("--var", action="append", help="limit to these variables")
    args = parser.parse_args()

    ds = _Cdf5(args.path)
    print(f"{args.path}\nnumrecs={ds.numrecs} record={args.record}")
    print(
        f"{'variable':<16}{'_FillValue':>14}{'missing_value':>15}"
        f"{'n(|x|>=1e19)':>14}{'n(==_FillValue)':>17}  status"
    )
    mismatched = 0
    for name, var in ds.vars.items():
        dim_names = [d[0] for d in var["dims"]]
        if dim_names[-2:] != ["lat", "lon"] or var["type"] not in (5, 6):
            continue
        if args.var and name not in args.var:
            continue
        fill = var["attrs"].get("_FillValue", (None,))[0]
        missing = var["attrs"].get("missing_value", (None,))[0]
        values = ds.read_2d(name, args.record)
        n_sentinel = sum(1 for x in values if abs(x) >= _SENTINEL)
        n_fill = 0 if fill is None else sum(1 for x in values if x == fill)
        bad = n_sentinel > n_fill
        mismatched += bad
        print(
            f"{name:<16}{_fmt(fill):>14}{_fmt(missing):>15}"
            f"{n_sentinel:>14}{n_fill:>17}  {'MISMATCH' if bad else 'ok'}"
        )
    print(f"{mismatched} variable(s) store sentinels that differ from _FillValue")
    sys.exit(1 if mismatched else 0)


if __name__ == "__main__":
    main()
