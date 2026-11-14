"""Inline SVG charts for the demo (no plotting dependency; colours are CSS variables, so dark mode follows the theme).
Every chart takes rows that carry a `source` and prints it under the chart."""
from __future__ import annotations

import html

e = html.escape
COL = {"state": "var(--c-state)", "ack": "var(--c-ack)", "warm": "var(--c-warm)", "warm start": "var(--c-warm)"}
W = 1080


def _need_source(rows):
    assert all(r.get("source") for r in rows), "every charted row needs a source"


def _wrap(svg: str, h: int, title: str, caption: str, sources: set[str], main: bool = False, width: int = W) -> str:
    src = " · ".join(sorted(sources))
    return (f'<figure class="chart{" main" if main else ""}"><figcaption class="ctitle">{e(title)}</figcaption>'
            f'<svg viewBox="0 0 {width} {h}" role="img" aria-label="{e(title)}">{svg}</svg>'
            f'<div class="ccap">{caption}</div><div class="csrc" title="{e(src)}">source: {e(src)}</div></figure>')


def _x(v, lo, hi, x0, x1):
    return x0 + (v - lo) / (hi - lo) * (x1 - x0)


def legend(items) -> str:
    return '<div class="legend">' + "".join(
        f'<span><i style="background:{COL.get(k, k)}"></i>{e(t)}</span>' for k, t in items) + "</div>"


def c1_vsc(rows, contrasts) -> str:
    """Cleveland dot plot: VSC per arm, a dot per seed, Wilson 95% CI of seed 0."""
    _need_source(rows)
    rows = sorted(rows, key=lambda r: ({"warm": 0, "ack": 1, "state": 2}[r["channel"]], r["vsc"][0]))
    lo, hi, x0, x1, top, rh = 0.5, 0.95, 260, W - 160, 30, 40
    g = []
    for t in (0.5, 0.6, 0.7, 0.8, 0.9):
        x = _x(t, lo, hi, x0, x1)
        g.append(f'<line class="grid" x1="{x:.1f}" x2="{x:.1f}" y1="{top - 10}" y2="{top + rh * len(rows)}"/>'
                 f'<text class="tick" x="{x:.1f}" y="{top + rh * len(rows) + 16}" text-anchor="middle">{t:.1f}</text>')
    xw = _x(0.6425, lo, hi, x0, x1)
    g.append(f'<line class="ref" x1="{xw:.1f}" x2="{xw:.1f}" y1="{top - 14}" y2="{top + rh * len(rows)}"/>'
             f'<text class="ann" x="{xw + 4:.1f}" y="{top - 16}">warm start 0.6425</text>')
    for i, r in enumerate(rows):
        y = top + rh * i + rh / 2
        c = COL[r["channel"]]
        a, b = r["ci95_seed0"]
        seeds = f'{r["seeds"]} seed' + ("s" if r["seeds"] > 1 else "")
        g.append(f'<text class="lab" x="{x0 - 10}" y="{y + 4:.1f}" text-anchor="end">{e(r["label"])}</text>'
                 f'<text class="tag" x="{x0 - 10}" y="{y + 16:.1f}" text-anchor="end">{seeds}</text>'
                 f'<line stroke="{c}" stroke-width="2" x1="{_x(a, lo, hi, x0, x1):.1f}" x2="{_x(b, lo, hi, x0, x1):.1f}" y1="{y:.1f}" y2="{y:.1f}"/>')
        for v in r["vsc"]:
            g.append(f'<circle cx="{_x(v, lo, hi, x0, x1):.1f}" cy="{y:.1f}" r="6" fill="{c}" stroke="var(--ap-surface)" stroke-width="2">'
                     f'<title>{e(r["label"])}: VSC {v} (n = {r["n"]})</title></circle>')
        g.append(f'<text class="val" x="{_x(max(b, max(r["vsc"])), lo, hi, x0, x1) + 8:.1f}" y="{y + 4:.1f}">{" / ".join(f"{v:.3f}" for v in r["vsc"])}</text>')
    h = top + rh * len(rows) + 30
    t, h1 = contrasts["vs_tool_trust"], contrasts["H1"]
    cap = (f'PCSTR-d − tool-trusting = <b>+{t["delta"] * 100:.2f} pp</b> [{t["lo"] * 100:.1f}, {t["hi"] * 100:.1f}]. '
           f'Registered H1, PCSTR-d − outcome = +{h1["delta"] * 100:.1f} pp [{h1["ci95"][0] * 100:.1f}, {h1["ci95"][1] * 100:.1f}], '
           f'p = {h1["p"]:.3f}. Untouched confirmatory splits (test_iid_c ∪ test_ood_fault_c), n = 400, greedy. Bars: Wilson 95% CI, seed 0.')
    return legend([("warm", "Warm start"), ("ack", "Reward reads acknowledgements or the transcript"),
                   ("state", "Reward reads the database")]) + _wrap("".join(g), h, "Verified-success completion (VSC) by reward", cap,
                                                                     {r["source"] for r in rows}, main=True)


def bars(rows, key, label_key, title, caption, lo=0.0, hi=1.0, fmt="{:.2f}", color_key="channel", n_key="n",
         ci_key=None, second_key=None, width=W) -> str:
    """Horizontal bars, one row each."""
    _need_source(rows)
    x0, x1, top, rh = (320, width - 200, 14, 32) if width >= 900 else (230, width - 90, 14, 30)
    g = []
    for t in (lo, (lo + hi) / 2, hi):
        x = _x(t, lo, hi, x0, x1)
        g.append(f'<line class="grid" x1="{x:.1f}" x2="{x:.1f}" y1="{top - 4}" y2="{top + rh * len(rows)}"/>'
                 f'<text class="tick" x="{x:.1f}" y="{top + rh * len(rows) + 16}" text-anchor="middle">{fmt.format(t)}</text>')
    for i, r in enumerate(rows):
        y = top + rh * i
        c = COL.get(r.get(color_key), r.get("color", "var(--c-state)"))
        w = _x(r[key], lo, hi, x0, x1) - x0
        n = f' (n = {r[n_key]})' if n_key and r.get(n_key) else ""
        g.append(f'<text class="lab" x="{x0 - 10}" y="{y + rh / 2 + 4:.1f}" text-anchor="end">{e(str(r[label_key]))[:44]}</text>'
                 f'<rect x="{x0}" y="{y + 6}" width="{max(w, 1.5):.1f}" height="{rh - 14}" rx="2" fill="{c}">'
                 f'<title>{e(str(r[label_key]))}: {fmt.format(r[key])}{n}</title></rect>')
        if second_key and r.get(second_key) is not None:
            w2 = _x(r[second_key], lo, hi, x0, x1) - x0
            g.append(f'<rect x="{x0}" y="{y + rh - 9}" width="{w2:.1f}" height="3" fill="var(--ap-text)" opacity=".55">'
                     f'<title>failure visible in transcript: {fmt.format(r[second_key])}</title></rect>')
        if ci_key and r.get(ci_key):
            a, b = r[ci_key]
            g.append(f'<line stroke="var(--ap-text)" stroke-width="1.5" x1="{_x(a, lo, hi, x0, x1):.1f}" x2="{_x(b, lo, hi, x0, x1):.1f}"'
                     f' y1="{y + rh / 2:.1f}" y2="{y + rh / 2:.1f}"/>')
        xe = max(x0 + max(w, 1.5), _x(r[ci_key][1], lo, hi, x0, x1) if ci_key and r.get(ci_key) else 0)
        g.append(f'<text class="val" x="{xe + 6:.1f}" y="{y + rh / 2 + 4:.1f}">{fmt.format(r[key])}{n}</text>')
    return _wrap("".join(g), top + rh * len(rows) + 26, title, caption, {r["source"] for r in rows}, width=width)


def c2_channel(rows) -> str:
    """Small multiples: read-back after a success ack, one panel per warm start."""
    panels = []
    for p in dict.fromkeys(r["panel"] for r in rows):
        rs = [dict(r, label=r["model"]) for r in rows if r["panel"] == p]
        panels.append(bars(rs, "rate", "label", p, "", fmt="{:.2f}", width=560))
    return ('<div class="multi">' + "".join(panels) + '</div><div class="ccap">Compare within a panel, not across: '
            'panel 1 is sampled (T = 1 × 8, val), panels 2–3 are greedy test-IID.</div>')


def c3_x1(rows, diff, step0) -> str:
    rs = [dict(r, color="var(--ink)" if r["rate"] > 0 else "var(--n-400)") for r in rows]
    for r in rs:
        r["label"] = r["cell"]
    ch = bars(rs, "rate", "label", "Same acknowledgement reward, different training tasks: refund read-back",
              f'X1 − placebo = <b>+{diff["diff"]:.3f}</b>, paired task bootstrap 95% CI [{diff["ci95"][0]:.3f}, {diff["ci95"][1]:.3f}], '
              f'{diff["n_pairs"]} paired tasks. Pre-registered (PREREGISTRATION_A8, addendum), <b>one seed per cell</b>, greedy test-IID.',
              color_key="_", n_key="n")
    tb = "".join(f'<tr><td>{e(r["cls"])}</td><td>{r["readers"]:.2f}</td><td>{r["non_readers"]:.2f}</td>'
                 f'<td class="{"neg" if r["readers"] < r["non_readers"] - 0.2 else ""}">{r["readers"] - r["non_readers"]:+.2f}</td></tr>' for r in step0)
    return ch + (f'<figure class="chart"><figcaption class="ctitle">Why: what the acknowledgement reward pays agents that check vs '
                 f'agents that don\'t (step 0, A7 warm start)</figcaption><table class="tbl"><tr><th>task class</th><th>checkers</th>'
                 f'<th>non-checkers</th><th>gap</th></tr>{tb}</table><div class="csrc">source: {e(step0[0]["source"])}</div></figure>')


def c5_agentic(rows) -> str:
    rows = [dict(r, channel=r["family"]) for r in rows]
    a = bars(rows, "acc", "model", "Tool-call accuracy (calls that parse and are valid)", "", lo=0.97, hi=1.0, fmt="{:.3f}", n_key=None, width=560)
    b = bars(rows, "sec", "model", "Seconds per episode", "batched vLLM on one H200: throughput-normalised, not single-request latency",
             lo=0, hi=0.25, fmt="{:.2f}", n_key=None, width=560)
    c = bars(rows, "chars", "model", "Characters generated per episode",
             "acknowledgement-trusting policies write out several future calls per turn; database-rewarded ones act one call at a time",
             lo=0, hi=5000, fmt="{:,.0f}", n_key=None, width=560)
    return '<div class="multi">' + a + b + c + '</div><div class="ccap">Development test-IID, n = 200 per model, greedy.</div>'


def c6_faults(rows, models) -> str:
    """100% stacked bars: outcome per fault type, for each of the four models."""
    _need_source(rows)
    faults = sorted({r["fault"] for r in rows}, key=lambda f: (f != "benign", f))
    seg = [("verified", "var(--ink)", "Verified"), ("false_completion", "var(--pen)", "False completion"),
           ("honest_failure", "var(--n-500)", "Reported failed or escalated"), ("other", "var(--n-200)", "Other")]
    x0, top, bw, gap, ch = 150, 20, 26, 6, 150
    g, colw = [], (W - x0 - 20) / len(faults)
    for j, f in enumerate(faults):
        cx = x0 + colw * j
        g.append(f'<text class="lab" x="{cx + colw / 2:.1f}" y="{top + ch + 18}" text-anchor="middle">{e(f)}</text>')
        for i, m in enumerate(models):
            r = next((x for x in rows if x["model"] == m and x["fault"] == f), None)
            if not r:
                continue
            x, y = cx + (colw - len(models) * (bw + gap)) / 2 + i * (bw + gap), top + ch
            for k, c, lab in seg:
                h = r[k] / r["n"] * ch
                if h > 0:
                    y -= h
                    g.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw}" height="{h:.1f}" fill="{c}"><title>{e(m)} · {e(f)}: '
                             f'{lab} {r[k]}/{r["n"]}</title></rect>')
            g.append(f'<text class="tag" x="{x + bw / 2:.1f}" y="{top + ch + 32}" text-anchor="middle">{i + 1}</text>')
    for t in (0, 50, 100):
        y = top + ch - t / 100 * ch
        g.append(f'<text class="tick" x="{x0 - 8}" y="{y + 4:.1f}" text-anchor="end">{t}%</text>')
    key = " · ".join(f"{i + 1} = {e(m)}" for i, m in enumerate(models))
    leg = legend([(c, lab) for _, c, lab in seg])
    return leg + _wrap("".join(g), top + ch + 44, "Outcome by injected fault, four models",
                       f"{key}. Validation split, seed 0, greedy; n per bar in the tooltip.", {r["source"] for r in rows})
