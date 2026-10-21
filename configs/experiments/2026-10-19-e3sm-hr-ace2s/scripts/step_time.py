"""Median s/step (5-step intervals after step 15) and allocator peak of a run.

Usage: python scripts/step_time.py RUN [RUN ...]
"""

import datetime
import glob
import os
import re
import statistics
import sys

for run in sys.argv[1:]:
    log = os.path.join(os.environ["PSCRATCH"], "hr-runs", run, "out.log")
    pts = []
    for line in open(log):
        m = re.match(r"(\S+ \S+),\d+ .*Step (\d+):", line)
        if m:
            t = datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
            pts.append((t, int(m.group(2))))
    d = [
        (t2 - t1).total_seconds() / (s2 - s1)
        for (t1, s1), (t2, s2) in zip(pts, pts[1:])
        if s1 >= 15 and s2 > s1
    ]
    peak = 0.0
    memlogs = os.path.join(os.environ["PSCRATCH"], "hr-memlog", run, "rank*.txt")
    for f in glob.glob(memlogs):
        for line in open(f):
            peak = max(peak, float(line.split()[3]))
    step = f"{statistics.median(d):.2f} s/step (n={len(d)})" if d else "no steps"
    print(f"{run}: {step}, peak allocated {peak:.1f} GiB")
