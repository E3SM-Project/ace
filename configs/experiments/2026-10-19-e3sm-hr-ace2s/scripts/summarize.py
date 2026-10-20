"""Print bench_model.py JSONL results as a table: summarize.py logs/*.jsonl"""

import json
import sys

cols = ["tag", "world", "fme_h", "fme_w", "batch", "params_M", "peak_GB", "s_step"]
print(" | ".join(cols))
for path in sys.argv[1:]:
    # concurrent writers can glue two records onto one line
    text = open(path).read().replace("RESULT ", "").replace("}{", "}\n{")
    for line in text.splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        s = d.get("s_per_step_median")
        print(
            " | ".join(
                [
                    d["tag"],
                    str(d.get("world", 1)),
                    str(d.get("fme_h", 1)),
                    str(d.get("fme_w", 1)),
                    str(d["batch"]),
                    f"{d['params_M']:.0f}",
                    f"{d.get('peak_GB', 0):.1f}" + ("" if d["ok"] else " OOM"),
                    f"{s:.2f}" if s else "-",
                ]
            )
        )
