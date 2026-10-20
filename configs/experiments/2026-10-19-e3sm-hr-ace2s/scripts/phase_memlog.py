"""Per-phase allocator memory from out.log + a memtools rank log.

python phase_memlog.py OUT_LOG RANK_LOG
"""

import re
import sys

PHASES = [
    (r"Beginning epoch after (\d+)", lambda m: f"train epoch {int(m.group(1)) + 1}"),
    (r"Starting validation loop", lambda m: "validation"),
    (r"Starting inference step", lambda m: "inline inference"),
    (r"Saving checkpoints for epoch", lambda m: "checkpoint saves"),
    (r"DONE", lambda m: "shutdown"),
]
ev = []
for line in open(sys.argv[1]):
    m = re.match(r"\S+ (\d\d:\d\d:\d\d),\d+ - root - INFO - (.*)", line)
    if m:
        for pat, name in PHASES:
            mm = re.search(pat, m.group(2))
            if mm:
                ev.append((m.group(1), name(mm)))
                break
mem = [line.split() for line in open(sys.argv[2])]
bounds = ev + [("23:59:59", "")]
for (t0, name), (t1, _) in zip(bounds, bounds[1:]):
    w = [(float(a), float(r)) for t, a, r, _, _ in mem if t0 <= t < t1]
    if w:
        print(
            f"{t0}  {name:18s} allocated max {max(a for a, _ in w):6.2f}"
            f"  reserved max {max(r for _, r in w):6.2f} GiB"
        )
print(
    "run max_allocated / max_reserved:",
    max(float(x[3]) for x in mem),
    max(float(x[4]) for x in mem),
)
