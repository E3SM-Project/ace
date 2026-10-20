"""Print the communication decomposition table from bench JSONL files."""

import json
import sys

rows = {}
for path in sys.argv[1:]:
    text = open(path).read().replace("RESULT ", "").replace("}{", "}\n{")
    for line in text.splitlines():
        if line.strip():
            d = json.loads(line)
            rows[d["tag"]] = d
for t, d in rows.items():
    s = d.get("s_per_step_median")
    print(
        f"{t:30s} ok={d['ok']!s:5} fwd={d.get('fwd_s', 0):.3f} "
        f"bwd={d.get('bwd_s', 0):.3f} opt={d.get('opt_s', 0):.3f} "
        f"step={(s or 0):.3f} peak={d.get('peak_GB', 0):.1f}"
    )
