"""Max GPU / host memory per training phase, from out.log + a sampler log.

    python phase_memory.py OUT_LOG SAMPLER_LOG
Sampler lines: "HH:MM:SS gpu=<MiB> host_used=<GB>".
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
events = []
for line in open(sys.argv[1]):
    m = re.match(r"\S+ (\d\d:\d\d:\d\d),\d+ - root - INFO - (.*)", line)
    if not m:
        continue
    for pat, name in PHASES:
        mm = re.search(pat, m.group(2))
        if mm:
            events.append((m.group(1), name(mm)))
samples = []
for line in open(sys.argv[2]):
    g = re.search(r"(\d\d:\d\d:\d\d) gpu=(\d+) host_used=(\d+)", line)
    if g:
        samples.append((g.group(1), int(g.group(2)), int(g.group(3))))
bounds = events + [("23:59:59", "")]
for (t0, name), (t1, _) in zip(bounds, bounds[1:]):
    w = [(gpu, host) for t, gpu, host in samples if t0 <= t < t1]
    if w:
        gpus = sorted(g for g, _ in w)
        print(
            f"{t0}  {name:18s} GPU max {gpus[-1]:6d}  p50 {gpus[len(gpus) // 2]:6d} MiB"
            f"   host max {max(h for _, h in w):4d} GB"
        )
