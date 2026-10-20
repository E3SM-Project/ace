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
    text = open(path).read().replace("RESULT ", "")
    out, seen = [], set()
    # concurrent writers can interleave records; every record starts with
    # the same key, so split on it and keep only the pieces that parse
    for piece in text.split('{"nlat"')[1:]:
        piece = '{"nlat"' + piece.strip()
        piece = piece[: piece.rfind("}") + 1]
        try:
            d = json.loads(piece)
        except json.JSONDecodeError:
            continue
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
            f"<td class=n>{f'{s:.2f}' if s else '&ndash;'}</td>"
            f"<td class=n>{sps}</td></tr>"
        )
    return "<div class=tw><table>" + head + "".join(body) + "</table></div>"


def decomp(stems: list[str]) -> str:
    """Group runs tagged <layout>-{compute,real-nosync,fake,real} into one row:
    pure split compute, + transposes, + grad sync, and the comm share.
    """
    rows: dict[str, dict[str, dict]] = {}
    for d in (d for s in stems for d in load(s)):
        aliases = {"-fake-nosync": "compute"}
        for kind in ("-real-nosync", "-fake-nosync", "-compute", "-fake", "-real"):
            if d["tag"].endswith(kind) and d["ok"]:
                base = d["tag"][: -len(kind)]
                rows.setdefault(base, {})[aliases.get(kind, kind[1:])] = d
                break
    if not rows:
        return '<p class="pending">Running; no results yet.</p>'
    head = (
        "<tr><th>layout</th><th class=n>GPUs</th><th>tiles HxW</th>"
        "<th class=n>compute s</th><th class=n>+ transposes</th>"
        "<th class=n>+ grad sync</th><th class=n>step s</th><th class=n>comm %</th></tr>"
    )
    body = []
    for base, r in rows.items():
        any_d = next(iter(r.values()))
        step = r.get("real", {}).get("s_per_step_median")
        comp = r.get("compute", {}).get("s_per_step_median")
        nosync = r.get("real-nosync", {}).get("s_per_step_median")

        def f(v):
            return "&ndash;" if v is None else f"{v:.2f}"

        tr = None if (nosync is None or comp is None) else nosync - comp
        gs = None if (step is None or nosync is None) else step - nosync
        share = None if (step is None or comp is None) else 100 * (step - comp) / step
        body.append(
            f"<tr><td>{html.escape(base)}</td><td class=n>{any_d.get('world', 1)}</td>"
            f"<td>{any_d.get('fme_h', 1)}&times;{any_d.get('fme_w', 1)}</td>"
            f"<td class=n>{f(comp)}</td><td class=n>{f(tr)}</td><td class=n>{f(gs)}</td>"
            f"<td class=n>{f(step)}</td>"
            f"<td class=n>{'&ndash;' if share is None else f'{share:.0f}'}</td></tr>"
        )
    return "<div class=tw><table>" + head + "".join(body) + "</table></div>"


def main():
    src = open(os.path.join(HERE, "report", "narrative.html")).read()
    src = re.sub(
        r"\{\{table:([\w,\-.]+)\}\}", lambda m: table(m.group(1).split(",")), src
    )
    src = re.sub(
        r"\{\{decomp:([\w,\-.]+)\}\}", lambda m: decomp(m.group(1).split(",")), src
    )
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M %Z").strip()
    src = src.replace("{{updated}}", now or "")
    with open(os.path.join(HERE, "report", "index.html"), "w") as f:
        f.write(src)
    print("wrote report/index.html")


if __name__ == "__main__":
    main()
