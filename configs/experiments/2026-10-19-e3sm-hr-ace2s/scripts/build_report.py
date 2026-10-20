"""Build report/index.html from report/narrative.html + the benchmark JSONL logs.

narrative.html is hand-written HTML with placeholders of the form
{{table:<jsonl stem>[,<jsonl stem>...]}} which are replaced by result tables.

    python scripts/build_report.py
"""

import datetime
import html
import json
import os
import re

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(stem: str) -> list[dict]:
    path = os.path.join(HERE, "logs", f"{stem}.jsonl")
    if not os.path.exists(path):
        return []
    text = open(path).read().replace("RESULT ", "").replace("}{", "}\n{")
    out, seen = [], set()
    for line in text.splitlines():
        if line.strip():
            d = json.loads(line)
            if d["tag"] in seen:  # failed ranks each print; keep one
                continue
            seen.add(d["tag"])
            out.append(d)
    return out


def knobs(d: dict) -> str:
    parts = [f"embed {d['embed_dim']}", f"ckpt {d['checkpointing']}"]
    if d.get("spectral_ratio", 1) != 1:
        parts.append(f"spectral_ratio {d['spectral_ratio']}")
    if d.get("htf", 1) != 1:
        parts.append(f"modes x{d['htf']}")
    if d.get("filter_num_groups", 1) != 1:
        parts.append(f"groups {d['filter_num_groups']}")
    if d.get("no_pos_embed"):
        parts.append("no pos_embed")
    if d.get("bf16"):
        parts.append("bf16")
    if (d["nlat"], d["nlon"]) != (720, 1440):
        parts.append(f"grid {d['nlat']}x{d['nlon']}")
    return ", ".join(parts)


def table(stems: list[str]) -> str:
    rows = [d for s in stems for d in load(s)]
    if not rows:
        return '<p class="pending">Running; no results yet.</p>'
    head = (
        "<tr><th>run</th><th>model</th><th class=n>GPUs</th><th>tiles HxW</th>"
        "<th class=n>batch</th><th class=n>params</th><th class=n>peak GB</th>"
        "<th class=n>s/step</th><th class=n>samples/s</th></tr>"
    )
    body = []
    for d in rows:
        world = d.get("world", 1)
        h, w = d.get("fme_h", 1), d.get("fme_w", 1)
        s = d.get("s_per_step_median")
        ok = d["ok"]
        # one "sample" = one initial condition with its n_ensemble=2 members
        sps = f"{(d['batch'] / 2) / s:.2f}" if ok and s else "&ndash;"
        peak = f"{d.get('peak_GB', 0):.1f}"
        status = "" if ok else ' <span class="pill bad">OOM</span>'
        body.append(
            f"<tr class='{'' if ok else 'oom'}'><td>{html.escape(d['tag'])}</td>"
            f"<td>{html.escape(knobs(d))}</td><td class=n>{world}</td>"
            f"<td>{h}&times;{w}</td><td class=n>{d['batch']}</td>"
            f"<td class=n>{d['params_M'] / 1000:.2f} B</td>"
            f"<td class=n>{peak}{status}</td>"
            f"<td class=n>{f'{s:.2f}' if s else '&ndash;'}</td><td class=n>{sps}</td></tr>"
        )
    return "<div class=tw><table>" + head + "".join(body) + "</table></div>"


def main():
    src = open(os.path.join(HERE, "report", "narrative.html")).read()
    src = re.sub(
        r"\{\{table:([\w,\-.]+)\}\}", lambda m: table(m.group(1).split(",")), src
    )
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M %Z").strip()
    src = src.replace("{{updated}}", now or "")
    with open(os.path.join(HERE, "report", "index.html"), "w") as f:
        f.write(src)
    print("wrote report/index.html")


if __name__ == "__main__":
    main()
