"""Final figure set for the Stage 2 submission: figures/final/ (SVG + PNG + caption + CSV table view).

    python scripts/build_final_figures.py

- Figure 1  fig1_counterfactual: one request, one call, two replies, four hidden outcomes (TXN quadruples,
            actionproof/txn/tasks.py REALITIES), with measured outcomes of B0, B3, B4 and D per cell.
- Figure 3  fig3_frontier: completion versus harm (VTC vs selective risk; FCR and DMR strips) for every arm in ARMS.
- Figure 4  fig4_generalisation: VTC by task group and STR by evidence regime, warm start vs RL vs reference.
- Figure 6  fig6_reproducibility: source -> registration -> task hash -> raw trace -> analysis -> claims -> surfaces.

Every number is read from result files at build time; nothing is typed in. Dependency-free SVG; PNGs are rendered by
headless Chrome (skipped with a warning if it is absent). To add an arm (e.g. the Recovery Gate), add one Arm line to
ARMS (tag syntax "[dir:]name[@split]" -> results/<dir or txn>/<name>__<split or txn_test>.summary.json) and, for
Figure 4, its short name to GEN_ROWS. Arms whose files are missing are skipped with a warning.
"""
from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import shutil
import subprocess
import sys
import tempfile
import textwrap
from collections import namedtuple
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures/final"
SPLIT = "txn_test"

TEXT, MUTED, GRID, AXIS, SURF, PANEL = "#191919", "#5c5c5c", "#e8e8e8", "#9a9a9a", "#ffffff", "#f6f6f4"
FONT = "Inter, 'Noto Sans', 'Helvetica Neue', Arial, sans-serif"
MONO = "'Noto Sans Mono', 'DejaVu Sans Mono', Menlo, Consolas, monospace"
BLUE = ["#cce3fd", "#b6d5f8", "#9dc8f7", "#84b9f3", "#6aabf1", "#519ced", "#338ced", "#257de0", "#2370cd",
        "#1b64be", "#1757ac", "#0f4a96", "#0e3d7e"]  # sequential ramp, steps 100..700

Arm = namedtuple("Arm", "short mid label tags family")
FAM = {  # three categorical hues at most (all-pairs validated: blue, orange, aqua); neutrals for scripted policies
    "scripted": {"color": "#8c8c8c", "marker": "circle", "name": "Scripted baseline"},
    "ablation": {"color": "#b8b8b8", "marker": "ring", "name": "B4 ablation"},
    "reference": {"color": "#191919", "marker": "square", "name": "Scripted reference B4"},
    "learned": {"color": "#257de0", "marker": "circle", "name": "Model (Tiny Aya Global)"},
    "sel1": {"color": "#00b894", "marker": "triangle", "name": "Model, SEL1 continuation"},
    "gate": {"color": "#f47230", "marker": "diamond", "name": "Model + Recovery Gate"},
}
ARMS = [
    Arm("B0", "B0 trust the reply", "B0 trust the acknowledgement", ["B0"], "scripted"),
    Arm("B1", "B1 final state only", "B1 final-state check only", ["B1"], "scripted"),
    Arm("B2", "B2 audit log only", "B2 audit-log check only", ["B2"], "scripted"),
    Arm("B3", "B3 retry + idem. key", "B3 bounded retry + idempotency key", ["B3"], "scripted"),
    Arm("A1", "A1 no read-back", "A1: B4 without read-back after ok", ["A1"], "ablation"),
    Arm("A2", "A2 no consistent read", "A2: B4 reading without consistent", ["A2"], "ablation"),
    Arm("A3", "A3 new key per retry", "A3: B4 with a new key on every retry", ["A3"], "ablation"),
    Arm("A4", "A4 no partial repair", "A4: B4 rewriting the full amount on partials", ["A4"], "ablation"),
    Arm("A5", "A5 never escalates", "A5: B4 that never escalates", ["A5"], "ablation"),
    Arm("A6", "A6 (= B4)", "A6 (identical flags to B4)", ["A6"], "ablation"),
    Arm("B4", "B4 reconciliation (ref.)", "B4 transaction-aware reconciliation (scripted reference)", ["B4"],
        "reference"),
    Arm("base", "Base Tiny Aya Global", "Base Tiny Aya Global, prompted", ["tag_base"], "learned"),
    Arm("W", "W plain SFT", "W plain SFT warm start", ["W"], "learned"),
    Arm("Dw", "Dw recovery SFT", "Dw recovery SFT warm start", ["Dw"], "learned"),
    Arm("Dw_aug", "Dw_aug (3 seeds)", "Dw_aug randomised recovery SFT (3 seeds)", ["Dwaug_s0", "Dwaug_s1", "Dwaug_s2"],
        "learned"),
    Arm("D", "D SFT+GRPO (3 seeds)", "D recovery SFT + GRPO (3 seeds)", ["D_s0", "D_s1", "D_s2"], "learned"),
    Arm("Dn", "Dn no-std GRPO (3 seeds)", "Dn recovery SFT + no-std GRPO (3 seeds)", ["Dn_s0", "Dn_s1", "Dn_s2"],
        "learned"),
    Arm("SEL", "SEL1 SEL (3 seeds)", "SEL1 SEL: escalation demonstrations (3 seeds)",
        ["sel1:SEL_s0", "sel1:SEL_s1", "sel1:SEL_s2"], "sel1"),
    Arm("CTL", "SEL1 CTL (3 seeds)", "SEL1 CTL: matched replay (3 seeds)", ["sel1:CTL_s0", "sel1:CTL_s1", "sel1:CTL_s2"],
        "sel1"),
    # Arm("D+RG", "D + gate (3 seeds)", "D + Recovery Gate (3 seeds)",
    #     ["recovery_gate:RG_D_s0", "recovery_gate:RG_D_s1", "recovery_gate:RG_D_s2"], "gate"),
]
FIG3 = ["B0", "B1", "B2", "B3", "A1", "A2", "A3", "A4", "A5", "A6", "B4", "base", "Dw", "Dw_aug", "D", "SEL", "CTL"]
GEN_ROWS = [("Prompted", "base"), ("Warm start (SFT only)", "W"), ("Warm start (SFT only)", "Dw"),
            ("Warm start (SFT only)", "Dw_aug"), ("SFT continuation of Dw_aug (SEL1)", "SEL"),
            ("SFT continuation of Dw_aug (SEL1)", "CTL"), ("RL (SFT + GRPO)", "D"), ("RL (SFT + GRPO)", "Dn"),
            ("Scripted reference", "B4")]
GROUPS = [("core", "core faults"), ("quad", "counterfactual|quadruples"), ("ood_alias", "renamed|tools"),
          ("ood_schema", "renamed|arguments"), ("ood_workflow", "unseen|workflow"), ("ood_combo", "two faults|at once"),
          ("ood_evidence", "reordered,|lagged reads"), ("ood_domain", "new domain|(lab supplies)")]
REGIMES = [("standard", "standard"), ("delayed", "delayed|visibility"), ("unresolvable", "records|unreadable")]


# ---------------------------------------------------------------------- data
def resolve(tag: str) -> Path:
    d, name = tag.split(":", 1) if ":" in tag else ("txn", tag)
    name, split = name.split("@", 1) if "@" in name else (name, SPLIT)
    return ROOT / f"results/{d}/{name}__{split}.summary.json"


def jsonl_of(tag: str) -> Path:
    p = resolve(tag)
    return p.with_name(p.name.replace(".summary.json", ".jsonl"))


_CACHE: dict[str, dict] = {}


def txn(tag: str) -> dict | None:
    if tag not in _CACHE:
        p = resolve(tag)
        _CACHE[tag] = json.loads(p.read_text())["txn"] if p.exists() else None
    return _CACHE[tag]


def available(arm: Arm) -> bool:
    miss = [t for t in arm.tags if txn(t) is None]
    if miss:
        print(f"skip {arm.short}: missing {[str(resolve(t).relative_to(ROOT)) for t in miss]}", file=sys.stderr)
    return not miss


ARM = {a.short: a for a in ARMS}


def pooled(arm: Arm, metric: str, block: str = "all", key: str | None = None) -> tuple[int, int, list[float]]:
    num = den = 0
    per = []
    for t in arm.tags:
        b = txn(t)[block] if key is None else txn(t)[block].get(key)
        if b is None:
            continue
        m = b[metric]
        num, den = num + m["num"], den + m["den"]
        per.append(m["num"] / m["den"] if m["den"] else float("nan"))
    return num, den, per


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if not n:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - h) / d), min(1.0, (c + h) / d))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------- svg helpers
def esc(s: object) -> str:
    return html.escape(str(s), quote=True)


class Svg:
    def __init__(self, w: int, h: int) -> None:
        self.w, self.h, self.el = w, h, []

    def add(self, s: str) -> None:
        self.el.append(s)

    def text(self, x: float, y: float, s: str, size: float = 12, anchor: str = "start", weight: int = 400,
             fill: str = TEXT, mono: bool = False, rotate: float | None = None, italic: bool = False,
             halo: bool = False) -> None:
        tr = f' transform="rotate({rotate} {x:.1f} {y:.1f})"' if rotate is not None else ""
        st = ' font-style="italic"' if italic else ""
        if halo:
            st += f' stroke="{SURF}" stroke-width="3.5" stroke-linejoin="round" paint-order="stroke"'
        fam = MONO if mono else FONT
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{fam}" font-size="{size}" font-weight="{weight}" '
                 f'fill="{fill}" text-anchor="{anchor}"{tr}{st}>{esc(s)}</text>')

    def lines(self, x: float, y: float, rows: list[str], size: float = 12, lh: float | None = None, **kw) -> float:
        lh = lh or size * 1.38
        for i, r in enumerate(rows):
            self.text(x, y + i * lh, r, size=size, **kw)
        return y + len(rows) * lh

    def line(self, x1, y1, x2, y2, stroke=AXIS, w=1.0, dash: str | None = None, cap="butt") -> None:
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" '
                 f'stroke-width="{w}" stroke-linecap="{cap}"{d}/>')

    def rect(self, x, y, w, h, fill=PANEL, stroke="none", sw=1.0, rx=0) -> None:
        self.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}" '
                 f'stroke="{stroke}" stroke-width="{sw}"/>')

    def marker(self, kind: str, x: float, y: float, r: float, color: str, ring: str = SURF) -> None:
        if kind == "square":
            self.add(f'<rect x="{x - r:.1f}" y="{y - r:.1f}" width="{2 * r:.1f}" height="{2 * r:.1f}" rx="1.5" '
                     f'fill="{color}" stroke="{ring}" stroke-width="2"/>')
        elif kind == "triangle":
            pts = f"{x:.1f},{y - r * 1.2:.1f} {x + r * 1.1:.1f},{y + r * 0.8:.1f} {x - r * 1.1:.1f},{y + r * 0.8:.1f}"
            self.add(f'<polygon points="{pts}" fill="{color}" stroke="{ring}" stroke-width="2" stroke-linejoin="round"/>')
        elif kind == "diamond":
            pts = f"{x:.1f},{y - r * 1.3:.1f} {x + r * 1.3:.1f},{y:.1f} {x:.1f},{y + r * 1.3:.1f} {x - r * 1.3:.1f},{y:.1f}"
            self.add(f'<polygon points="{pts}" fill="{color}" stroke="{ring}" stroke-width="2" stroke-linejoin="round"/>')
        elif kind == "ring":
            self.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r - 1:.1f}" fill="{SURF}" stroke="{color}" stroke-width="2.2"/>')
        else:
            self.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{color}" stroke="{ring}" stroke-width="2"/>')

    def arrow(self, x1, y1, x2, y2, color=MUTED) -> None:
        self.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2 - 6:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="1.6"/>')
        self.add(f'<polygon points="{x2:.1f},{y2:.1f} {x2 - 8:.1f},{y2 - 4.5:.1f} {x2 - 8:.1f},{y2 + 4.5:.1f}" fill="{color}"/>')

    def render(self, title: str, desc: str) -> str:
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
                f'viewBox="0 0 {self.w} {self.h}" role="img" aria-labelledby="t d">'
                f'<title id="t">{esc(title)}</title><desc id="d">{esc(desc)}</desc>'
                f'<rect width="{self.w}" height="{self.h}" fill="{SURF}"/>' + "".join(self.el) + "</svg>\n")


def tw(s: str, size: float) -> float:
    return len(s) * size * 0.56


def wrap(s: str, chars: int) -> list[str]:
    return textwrap.wrap(s, chars, break_long_words=True, break_on_hyphens=False) or [""]


def save(name: str, svg: Svg, title: str, desc: str, caption: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    text = svg.render(title, desc)
    (OUT / f"{name}.svg").write_text(text)
    (OUT / f"{name}.caption.md").write_text(caption.strip() + "\n")
    render_png(name, text, svg.w, svg.h)


def render_png(name: str, svg_text: str, w: int, h: int) -> None:
    chrome = next((c for c in ("google-chrome", "chromium", "chromium-browser") if shutil.which(c)), None)
    if chrome is None:
        print(f"warning: no Chrome found; {name}.png not rendered", file=sys.stderr)
        return
    with tempfile.TemporaryDirectory() as d:
        page = Path(d) / "f.html"
        page.write_text(f'<!DOCTYPE html><html><body style="margin:0;background:#fff">{svg_text}</body></html>')
        png = OUT / f"{name}.png"
        subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-sandbox", "--hide-scrollbars",
                        f"--user-data-dir={d}/profile", "--force-device-scale-factor=2", f"--window-size={w},{h}",
                        f"--screenshot={png}", page.as_uri()], capture_output=True, timeout=90)
        if not png.exists():
            print(f"warning: Chrome did not write {png}", file=sys.stderr)


# ---------------------------------------------------------------------- Figure 3
def scatter(s: Svg, rows: list[dict], box: tuple, xr: tuple, yr: tuple, step: float, label, zoom=None,
            title: str = "") -> None:
    """One VTC-vs-selective-risk panel: grid, whiskers, seed marks, family markers, non-overlapping direct labels."""
    x0, x1, y0, y1 = box
    X = lambda v: x0 + (v - xr[0]) / (xr[1] - xr[0]) * (x1 - x0)  # noqa: E731
    Y = lambda v: y1 - (v - yr[0]) / (yr[1] - yr[0]) * (y1 - y0)  # noqa: E731
    cx = lambda v: min(max(v, xr[0]), xr[1])  # noqa: E731
    cyy = lambda v: min(max(v, yr[0]), yr[1])  # noqa: E731
    s.text(x0, y0 - 12, title, 12, weight=600, fill=MUTED)
    n = int(round((xr[1] - xr[0]) / step))
    for t in range(n + 1):
        v = xr[0] + t * step
        s.line(X(v), y0, X(v), y1, GRID)
        s.text(X(v), y1 + 18, f"{v:.2f}" if step < 0.1 else f"{v:.1f}", 11.5, "middle", fill=MUTED)
    m = int(round((yr[1] - yr[0]) / step))
    for t in range(m + 1):
        v = yr[0] + t * step
        s.line(x0, Y(v), x1, Y(v), GRID)
        s.text(x0 - 8, Y(v) + 4, f"{v:.2f}" if step < 0.1 else f"{v:.1f}", 11.5, "end", fill=MUTED)
    s.line(x0, y1, x1, y1, AXIS)
    s.line(x0, y0, x0, y1, AXIS)
    if zoom:
        (zx0, zx1), (zy0, zy1) = zoom
        s.add(f'<rect x="{X(zx0):.1f}" y="{Y(zy1):.1f}" width="{X(zx1) - X(zx0):.1f}" height="{Y(zy0) - Y(zy1):.1f}" '
              f'fill="none" stroke="{MUTED}" stroke-width="1.2" stroke-dasharray="4 3"/>')
    pts = []
    for r in rows:
        a, fam = r["arm"], FAM[r["arm"].family]
        vx, vy = r["selective_risk"], r["VTC"]
        px, py = X(vx["rate"]), Y(vy["rate"])
        tip = (f"{a.label}: VTC {vy['num']}/{vy['den']} = {vy['rate']:.3f} [{vy['lo']:.3f}, {vy['hi']:.3f}]; "
               f"selective risk {vx['num']}/{vx['den']} = {vx['rate']:.3f} [{vx['lo']:.3f}, {vx['hi']:.3f}]")
        s.add(f"<g><title>{esc(tip)}</title>")
        c = fam["color"]
        s.line(X(cx(vx["lo"])), py, X(cx(vx["hi"])), py, c, 1.4)
        s.line(px, Y(cyy(vy["lo"])), px, Y(cyy(vy["hi"])), c, 1.4)
        if len(a.tags) > 1:
            for sx, sy in zip(vx["per_seed"], vy["per_seed"]):
                if xr[0] <= sx <= xr[1] and yr[0] <= sy <= yr[1]:
                    s.add(f'<circle cx="{X(sx):.1f}" cy="{Y(sy):.1f}" r="3" fill="none" stroke="{c}" stroke-width="1.2"/>')
        s.marker(fam["marker"], px, py, 5.5, c)
        s.add("</g>")
        pts.append((px, py, r))
    groups: dict[tuple[int, int], list[dict]] = {}
    for px, py, r in pts:
        groups.setdefault((round(px / 3) * 3, round(py / 3) * 3), []).append(r)
    placed: list[tuple[float, float, float, float]] = []
    obstacles = [(px - 7, py - 7, px + 7, py + 7) for px, py, _ in pts]
    for _, _, r in pts:
        if len(r["arm"].tags) > 1:
            for sx, sy in zip(r["selective_risk"]["per_seed"], r["VTC"]["per_seed"]):
                if xr[0] <= sx <= xr[1] and yr[0] <= sy <= yr[1]:
                    obstacles.append((X(sx) - 4, Y(sy) - 4, X(sx) + 4, Y(sy) + 4))

    def free(b):
        return all(b[2] < o[0] or b[0] > o[2] or b[3] < o[1] or b[1] > o[3] for o in placed + obstacles) \
            and b[0] > x0 + 2 and b[2] < x1 + 60 and b[1] > y0 - 4 and b[3] < y1 - 2
    cands = [(10, 4, "start"), (-10, 4, "end"), (10, -9, "start"), (10, 17, "start"), (-10, -9, "end"),
             (-10, 17, "end"), (10, -22, "start"), (10, 30, "start"), (-10, -22, "end"), (-10, 30, "end"),
             (26, -36, "start"), (26, 44, "start"), (40, -50, "start"), (40, 58, "start"), (60, 0, "start"),
             (60, -26, "start"), (60, 26, "start")]
    for (gx, gy), group in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        if not any(label(r) for r in group):
            continue
        px, py = next((p, q) for p, q, r in pts if r is group[0])
        lab = " = ".join(r["arm"].short for r in group)
        w = tw(lab, 12)
        for dx, dy, anchor in cands:
            bx0 = px + dx if anchor == "start" else px + dx - w
            b = (bx0 - 1, py + dy - 11, bx0 + w + 1, py + dy + 3)
            if free(b):
                placed.append(b)
                if abs(dy) > 20 or abs(dx) > 20:
                    s.line(px, py, bx0 if anchor == "start" else bx0 + w, py + dy - 4, AXIS, 0.8)
                fam = group[0]["arm"].family
                s.text(px + dx, py + dy, lab, 12, anchor, weight=600 if fam != "ablation" else 400,
                       fill=TEXT if fam != "ablation" else MUTED, halo=True)
                break
        else:
            print(f"fig3: no free spot for label {lab}; named in the right panel only", file=sys.stderr)


def fig3() -> None:
    arms = [ARM[s] for s in FIG3 if s in ARM and available(ARM[s])]
    arms += [a for a in ARMS if a.family == "gate" and available(a)]
    rows = []
    for a in arms:
        rec = {"arm": a}
        for m in ("VTC", "selective_risk", "FCR", "DMR_all"):
            k, n, per = pooled(a, m)
            lo, hi = wilson(k, n)
            rec[m] = {"num": k, "den": n, "rate": k / n, "lo": lo, "hi": hi, "per_seed": per}
        rows.append(rec)
    W, H = 1640, 720
    s = Svg(W, H)
    s.text(40, 38, "Figure 3. Completion versus harm on the locked txn_test (440 tasks)", 18, weight=650)
    s.text(40, 60, "Upper left is better. Large marks pool the seeds; whiskers are Wilson 95% intervals; "
                   "small open marks are single seeds. The middle panel enlarges the dashed corner of the left panel.",
           12.5, fill=MUTED)
    fams = [f for f in FAM if any(r["arm"].family == f for r in rows)]
    lx = 40
    for f in fams:
        s.marker(FAM[f]["marker"], lx + 6, 86, 5, FAM[f]["color"])
        s.text(lx + 16, 90, FAM[f]["name"], 12, fill=TEXT)
        lx += 30 + tw(FAM[f]["name"], 12)
    ZX, ZY = (0.0, 0.30), (0.55, 1.0)  # zoom window: selective risk, VTC
    in_zoom = lambda r: ZX[0] <= r["selective_risk"]["rate"] <= ZX[1] and ZY[0] <= r["VTC"]["rate"] <= ZY[1]  # noqa: E731
    scatter(s, rows, (96, 470, 122, 582), (0.0, 1.0), (0.0, 1.0), 0.2,
            label=lambda r: not in_zoom(r), zoom=(ZX, ZY), title="All arms")
    scatter(s, [r for r in rows if in_zoom(r)], (566, 1016, 122, 582), ZX, ZY, 0.05, label=lambda r: True,
            title="Enlarged: selective risk 0 to 0.30, VTC 0.55 to 1.00")
    s.text(543, 640, "Unsafe rate among episodes not escalated (selective risk)", 12.5, "middle")
    s.text(50, 352, "Verified transaction completion (VTC)", 12.5, "middle", rotate=-90)
    # right panel: strips for FCR and DMR
    rx0 = 1070
    cols = [("FCR", "False completion (FCR)", 1282, 1430), ("DMR_all", "Duplicate mutation (DMR)", 1468, 1610)]
    ry0, ry1 = 122, 582
    rh = (ry1 - ry0) / len(rows)
    for key, name, c0, c1 in cols:
        s.text((c0 + c1) / 2, ry0 - 26, name, 12.5, "middle", weight=600)
        for t in (0, 0.5, 1.0):
            xx = c0 + t * (c1 - c0)
            s.line(xx, ry0 - 6, xx, ry1, GRID)
            s.text(xx, ry0 - 10, f"{t:.1f}", 10.5, "middle", fill=MUTED)
    prev = None
    for i, r in enumerate(rows):
        a = r["arm"]
        cy = ry0 + (i + 0.5) * rh
        if prev is not None and a.family != prev:
            s.line(rx0, ry0 + i * rh, 1610, ry0 + i * rh, "#d4d4d4", 0.8)
        prev = a.family
        s.text(rx0, cy + 4, a.mid, 12, fill=TEXT)
        for key, _, c0, c1 in cols:
            v = r[key]
            c = FAM[a.family]["color"]
            tip = f"{a.label}: {key} {v['num']}/{v['den']} = {v['rate']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}]"
            s.add(f"<g><title>{esc(tip)}</title>")
            s.line(c0 + v["lo"] * (c1 - c0), cy, c0 + v["hi"] * (c1 - c0), cy, c, 1.4)
            s.marker(FAM[a.family]["marker"], c0 + v["rate"] * (c1 - c0), cy, 4.3, c)
            s.add("</g>")
            s.text(c0 - 8, cy + 4, f"{v['rate']:.3f}", 10.5, "end", fill=MUTED)
    s.line(rx0, ry1, 1610, ry1, AXIS, 0.8)
    foot = ("Source: results/txn/*__txn_test.summary.json and results/sel1/*__txn_test.summary.json (block 'all'). "
            "Denominators: 440 episodes per scripted policy; 1,320 per model seed (440 tasks x 1 greedy + 2 sampled). "
            "Intervals treat episodes as independent; a task's three episodes and the three seeds are correlated, so model "
            "intervals are too narrow. B4 and A1-A6 are scripted with the fault taxonomy written in; they are references, "
            "not learned agents.")
    s.lines(40, 672, wrap(foot, 260), 11.5, fill=MUTED)
    desc = "; ".join(f"{r['arm'].short}: VTC {r['VTC']['rate']:.3f}, selective risk {r['selective_risk']['rate']:.3f}, "
                     f"FCR {r['FCR']['rate']:.3f}, DMR {r['DMR_all']['rate']:.3f}" for r in rows)
    with (OUT / "fig3_frontier.csv").open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["arm", "label", "family", "tags", "metric", "num", "den", "rate", "wilson_lo", "wilson_hi",
                     "per_seed_rates"])
        for r in rows:
            for m in ("VTC", "selective_risk", "FCR", "DMR_all"):
                v = r[m]
                wr.writerow([r["arm"].short, r["arm"].label, r["arm"].family, " ".join(r["arm"].tags), m, v["num"],
                             v["den"], f"{v['rate']:.6f}", f"{v['lo']:.6f}", f"{v['hi']:.6f}",
                             " ".join(f"{x:.6f}" for x in v["per_seed"])])
    tbl = ["| Arm | Tags | VTC | Selective risk | FCR | DMR |", "|---|---|---|---|---|---|"]
    for r in rows:
        f = lambda v: f"{v['num']}/{v['den']} = {v['rate']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}]"  # noqa: E731
        tbl.append(f"| {r['arm'].label} | {' '.join(r['arm'].tags)} | {f(r['VTC'])} | {f(r['selective_risk'])} | "
                   f"{f(r['FCR'])} | {f(r['DMR_all'])} |")
    cap = f"""
**Figure 3. Completion versus harm on the locked txn_test.** Left: verified transaction completion (VTC) against
selective risk (unsafe episodes, i.e. false completion, duplicate, wrong amount or collateral write, among episodes the
agent did not escalate; `actionproof/txn/truth.py::summarise`). Right: false completion (FCR) and duplicate mutation
(DMR, all episodes) per arm.

- **Sources:** `results/txn/<tag>__txn_test.summary.json` and `results/sel1/<tag>__txn_test.summary.json`, block
  `txn.all`; built by `scripts/build_final_figures.py`; table view `figures/final/fig3_frontier.csv`.
- **Split and denominators:** txn_test, 440 locked tasks. Scripted policies: 1 deterministic episode per task (440).
  Models: 1 greedy + 2 sampled episodes per task, 1,320 per seed; 3-seed arms pool 3,960.
- **Seed scope:** D, Dn, Dw_aug, SEL and CTL have 3 seeds (small open marks); base, Dw and the scripted policies are
  single runs.
- **Uncertainty:** Wilson 95% intervals on pooled episodes, treating episodes as independent. Episodes of one task and
  seeds are correlated, so model intervals are too narrow; the registered analyses use task-cluster bootstrap.
- **Scope:** a simulator (ProcureSim). B4 and its ablations A1-A6 are scripted with the fault taxonomy written in; they
  are references, not learned agents. B0 and B2 coincide; A6 has B4's flags and coincides with it.

{chr(10).join(tbl)}
"""
    save("fig3_frontier", s, "Figure 3. Completion versus harm on txn_test", desc, cap)


# ---------------------------------------------------------------------- Figure 4
def fig4() -> None:
    rows = [(sec, ARM[k]) for sec, k in GEN_ROWS if k in ARM and available(ARM[k])]
    rows += [("Model + Recovery Gate", a) for a in ARMS if a.family == "gate" and available(a)]
    ncol = len(GROUPS) + len(REGIMES)
    lab_w, cw, ch, gap = 250, 80, 44, 3
    cx0 = 40 + lab_w
    block_gap = 20
    secs = []
    for sec, _ in rows:
        if not secs or secs[-1] != sec:
            secs.append(sec)
    W = int(cx0 + len(GROUPS) * (cw + gap) + block_gap + len(REGIMES) * (cw + gap) + 40)
    H = int(178 + len(rows) * (ch + gap) + len(secs) * 22 + 150)
    s = Svg(W, H)
    s.text(40, 38, "Figure 4. Where each policy holds up: task groups and evidence regimes on txn_test", 18, weight=650)
    s.text(40, 60, "Each cell pools the arm's seeds. Left block: verified completion. Right block: safe terminal outcome "
                   "(verified, or a correct hand-off with no harm).", 12.5, fill=MUTED)

    def colx(j: int) -> float:
        return cx0 + j * (cw + gap) + (block_gap if j >= len(GROUPS) else 0)
    s.text(colx(0), 96, "Verified completion (VTC) by task group", 12.5, weight=600)
    s.text(colx(len(GROUPS)), 96, "Safe terminal outcome (STR) by regime", 12.5, weight=600)
    b4 = txn("B4")
    for j, (g, name) in enumerate(GROUPS + REGIMES):
        blk = "by_group" if j < len(GROUPS) else "by_regime"
        for i, part in enumerate(name.split("|")):
            s.text(colx(j) + cw / 2, 118 + i * 14, part, 11.5, "middle")
        ntask = b4[blk][g]["n"] if b4 and g in b4[blk] else None
        if ntask:
            s.text(colx(j) + cw / 2, 150, f"{ntask} tasks", 10.5, "middle", fill=MUTED)
    y = 166
    cur = None
    csvrows = []
    for sec, a in rows:
        if sec != cur:
            y += 6
            s.text(40, y + 10, sec, 12, weight=650, fill=MUTED)
            y += 16
            cur = sec
        s.marker(FAM[a.family]["marker"], 46, y + ch / 2, 4.5, FAM[a.family]["color"])
        s.text(58, y + ch / 2 + 4, a.mid, 12.5)
        for j, (g, name) in enumerate(GROUPS + REGIMES):
            blk, metric = ("by_group", "VTC") if j < len(GROUPS) else ("by_regime", "STR")
            k, n, per = pooled(a, metric, blk, g)
            x = colx(j)
            if not n:
                s.rect(x, y, cw, ch, PANEL)
                s.text(x + cw / 2, y + ch / 2 + 4, "n/a", 11, "middle", fill=MUTED)
                continue
            r = k / n
            idx = min(len(BLUE) - 1, int(round(r * (len(BLUE) - 1))))
            dark = idx >= 7
            tip = (f"{a.label}, {g}: {metric} {k}/{n} = {r:.3f}"
                   + (f"; seeds {', '.join(f'{p:.3f}' for p in per)}" if len(per) > 1 else ""))
            s.add(f"<g><title>{esc(tip)}</title>")
            s.rect(x, y, cw, ch, BLUE[idx], rx=2)
            s.text(x + cw / 2, y + 19, f"{r:.3f}", 13.5, "middle", weight=650, fill="#ffffff" if dark else TEXT)
            s.text(x + cw / 2, y + 35, f"{k}/{n}", 10.5, "middle", fill="#ffffff" if dark else TEXT)
            s.add("</g>")
            csvrows.append([a.short, a.label, sec, " ".join(a.tags), blk, g, metric, k, n, f"{r:.6f}",
                            " ".join(f"{p:.6f}" for p in per)])
        y += ch + gap
    # colour scale
    y += 18
    s.text(40, y + 12, "Scale", 11.5, fill=MUTED)
    for i, c in enumerate(BLUE):
        s.rect(90 + i * 26, y, 26, 14, c)
    s.text(90, y + 30, "0.0", 10.5, "middle", fill=MUTED)
    s.text(90 + len(BLUE) * 26, y + 30, "1.0", 10.5, "middle", fill=MUTED)
    foot = ("Source: results/txn/*__txn_test.summary.json and results/sel1/*__txn_test.summary.json (blocks by_group, "
            "by_regime). Cells: episodes; models run each task once greedily and twice sampled per seed. The quadruples "
            "include the unresolvable regime, where completion is not the safe outcome, so their VTC is capped by design "
            "for an agent that hands off. Warm-start rows are SFT only; RL rows add GRPO on top of the Dw warm start; "
            "SEL1 rows continue Dw_aug with SFT. Single-seed rows: base, W, Dw. No intervals are drawn; see fig4 CSV for "
            "per-seed rates.")
    s.lines(40, y + 58, wrap(foot, 200), 11.5, fill=MUTED)
    with (OUT / "fig4_generalisation.csv").open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["arm", "label", "section", "tags", "block", "cell", "metric", "num", "den", "rate", "per_seed"])
        wr.writerows(csvrows)
    cap = """
**Figure 4. Generalisation by task group and evidence regime on txn_test.** Left block: verified transaction completion
(VTC) per task group. Right block: safe terminal outcome (STR: verified, or an escalation the task marks correct with no
harm) per evidence regime. Rows separate the prompted base model, warm starts (SFT only), SEL1 SFT continuations of
Dw_aug, RL policies (SFT + GRPO) and the scripted reference B4.

- **Sources:** `results/txn/<tag>__txn_test.summary.json`, `results/sel1/<tag>__txn_test.summary.json`, blocks
  `txn.by_group` and `txn.by_regime`; table view `figures/final/fig4_generalisation.csv` (with per-seed rates).
- **Split and denominators:** txn_test, 440 locked tasks; each cell shows episodes (num/den). Task counts per column are
  in the header (from B4's single run).
- **Seed scope:** Dw_aug, SEL, CTL, D and Dn pool 3 seeds; base, W and Dw are single runs; B4 is deterministic.
- **Uncertainty:** none drawn. Small columns (renamed tools, new domain and the other OOD groups: 20 tasks each) move by
  several points between seeds; the CSV lists every seed.
- **Reading note:** the quadruple group mixes the standard and unresolvable regimes; an agent that hands off correctly
  when records are unreadable scores lower VTC there by design (B4). The post-hoc validity analysis
  (PREREGISTRATION_TXN1_ADDENDUM3.md) found that keyed unresolvable tasks can be settled by a same-key resend, which is
  why some models show VTC above B4 on the quadruples.
"""
    desc = f"Heatmap of {len(rows)} arms by {ncol} columns; values in figures/final/fig4_generalisation.csv"
    save("fig4_generalisation", s, "Figure 4. Generalisation by task group and evidence regime", desc, cap)


# ---------------------------------------------------------------------- Figure 1
REALITY_TEXT = {
    "ok_committed": ("ok, and the write committed", "One placed supplier order exists.",
                     "Trap: a blind resend with a new key duplicates.", "Safe: read back once and continue."),
    "ok_phantom": ("ok, but nothing was stored", "No supplier order exists.",
                   "Trap: trusting the reply reports a false completion.",
                   "Safe: read back, find nothing, resend with the same key."),
    "timeout_committed": ("timeout, but the write committed", "One placed supplier order exists.",
                          "Trap: resending with a new key creates a duplicate.",
                          "Safe: read first; it is there; do not resend."),
    "timeout_uncommitted": ("timeout, and nothing was stored", "No supplier order exists.",
                            "Trap: reporting failure, or giving up without checking.",
                            "Safe: read, find nothing, resend with the same key."),
}
FIG1_ARMS = [("B0 trust the reply", ["B0"]), ("B3 retry + idempotency key", ["B3"]),
             ("B4 reconciliation (reference)", ["B4"]), ("D SFT+GRPO, 3 seeds", ["D_s0", "D_s1", "D_s2"])]
FIG1_SHORT = {"B0 trust the reply": "B0", "B3 retry + idempotency key": "B3", "B4 reconciliation (reference)": "B4",
              "D SFT+GRPO, 3 seeds": "D (3 seeds)"}


def quad_stats() -> tuple[dict, dict, dict]:
    stats: dict = {}
    esc_unres: dict = {}
    example: dict = {}
    for name, tags in FIG1_ARMS:
        for t in tags:
            p = jsonl_of(t)
            if not p.exists():
                print(f"fig1: missing {p.relative_to(ROOT)}", file=sys.stderr)
                continue
            for line in p.open():
                r = json.loads(line)
                tx = r["task"]["meta"]["txn"]
                if tx["group"] != "quad":
                    continue
                if tx["regime"] == "unresolvable":
                    e = esc_unres.setdefault(name, [0, 0])
                    e[0] += r["txn"]["escalated"]
                    e[1] += 1
                    continue
                c = stats.setdefault((name, tx["reality"]), {"vtc": 0, "fcr": 0, "dmr": 0, "n": 0})
                c["vtc"] += r["txn"]["vtc"]
                c["fcr"] += r["txn"]["fcr"]
                c["dmr"] += r["txn"]["dmr"]
                c["n"] += 1
                if t == "B4" and tx["quad_id"].endswith("q00"):
                    ag = [x for x in r["trace"] if x.get("by", "agent") == "agent"]
                    example[tx["reality"]] = {"prompt": r["task"]["prompt"], "call": ag[0]["tool"],
                                              "args": ag[0]["args"], "obs": ag[0]["obs"], "task_id": r["task"]["task_id"],
                                              "quad": tx["quad_id"]}
    return stats, esc_unres, example


def fig1() -> None:
    stats, esc_unres, ex = quad_stats()
    if len(ex) != 4:
        print("fig1: example quadruple incomplete; skipped", file=sys.stderr)
        return
    W, H = 1280, 850
    s = Svg(W, H)
    s.text(40, 38, "Figure 1. One request, one call, two possible replies, four hidden outcomes", 18, weight=650)
    s.text(40, 60, "A counterfactual quadruple of txn_test: the four tasks share the request and the agent's first call; "
                   "only the hidden outcome differs.", 12.5, fill=MUTED)
    e0 = ex["ok_committed"]
    s.rect(40, 78, W - 80, 108, PANEL, rx=6)
    s.text(56, 100, f"Request (identical in all four tasks of {e0['quad']})", 12, weight=650)
    yy = s.lines(56, 120, wrap(e0["prompt"], 150), 12.5)
    call = f"{e0['call']}({json.dumps(e0['args'])})"
    s.text(56, yy + 8, "First call (B4's; identical in all four):", 12, weight=650)
    s.text(316, yy + 8, call, 12, mono=True)
    same_ok = json.dumps(ex["ok_committed"]["obs"]) == json.dumps(ex["ok_phantom"]["obs"])
    same_to = json.dumps(ex["timeout_committed"]["obs"]) == json.dumps(ex["timeout_uncommitted"]["obs"])
    # grid
    gx0, gy0, colw, rowh, hdrw = 300, 236, 470, 244, 250
    for j, head in enumerate(("The write committed", "The write did not commit")):
        s.text(gx0 + j * (colw + 10) + colw / 2, gy0 - 14, head, 13.5, "middle", weight=650)
    s.text(gx0 + colw + 5, gy0 - 34, "What the system of record holds (hidden from the agent)", 11.5, "middle",
           fill=MUTED)
    rows = (("ok", ["ok_committed", "ok_phantom"], ex["ok_committed"]["obs"], same_ok),
            ("timeout", ["timeout_committed", "timeout_uncommitted"], ex["timeout_committed"]["obs"], same_to))
    for i, (reply, reals, obs, same) in enumerate(rows):
        y = gy0 + i * (rowh + 10)
        s.text(40, y + 22, f"Reply the agent sees: {reply}", 13.5, weight=650)
        yy = s.lines(40, y + 46, wrap(json.dumps(obs), 34), 11, mono=True)
        s.lines(40, yy + 8, wrap("Byte-identical in both columns of this row." if same else
                                 "Same status in both columns; the generated id differs.", 34), 11, fill=MUTED)
        for j, real in enumerate(reals):
            x = gx0 + j * (colw + 10)
            s.rect(x, y, colw, rowh, SURF, "#cfcfcf", 1, rx=6)
            t = REALITY_TEXT[real]
            s.text(x + 16, y + 24, t[0], 14, weight=650)
            s.text(x + colw - 16, y + 24, real, 11, "end", mono=True, fill=MUTED)
            s.text(x + 16, y + 46, t[1], 12.5)
            s.text(x + 16, y + 66, t[2], 12.5, fill="#a33a2b")
            s.text(x + 16, y + 86, t[3], 12.5, fill="#0b6b3a", weight=600)
            s.line(x + 16, y + 102, x + colw - 16, y + 102, GRID)
            s.text(x + 16, y + 122, "Measured, standard regime", 11.5, weight=650, fill=MUTED)
            s.text(x + 300, y + 122, "verified", 11, "end", fill=MUTED)
            s.text(x + 375, y + 122, "false compl.", 11, "end", fill=MUTED)
            s.text(x + colw - 16, y + 122, "duplicate", 11, "end", fill=MUTED)
            for k, (name, _) in enumerate(FIG1_ARMS):
                c = stats.get((name, real))
                ry = y + 146 + k * 24
                s.text(x + 16, ry, name, 12.5)
                if not c:
                    s.text(x + 300, ry, "n/a", 12, "end", fill=MUTED)
                    continue
                s.text(x + 300, ry, f"{c['vtc']}/{c['n']}", 12.5, "end", weight=650)
                s.text(x + 375, ry, f"{c['fcr']}", 12.5, "end", fill="#a33a2b" if c["fcr"] else TEXT)
                s.text(x + colw - 16, ry, f"{c['dmr']}", 12.5, "end", fill="#a33a2b" if c["dmr"] else TEXT)
    yb = gy0 + 2 * (rowh + 10) + 14
    parts = [f"{FIG1_SHORT[n]} hands off in {v[0]} of {v[1]} episodes" for n, v in esc_unres.items()]
    note = ("Each quadruple also has an unresolvable version: the read tools return service_unavailable, so no read can "
            "settle the reply and a hand-off is the safe action. There, " + "; ".join(parts) + ". Cells count 15 tasks "
            "per scripted policy and 135 episodes for D. "
            "Counts are episodes: scripted policies run each task once; D runs each task three times per seed (1 greedy "
            "+ 2 sampled). Source: results/txn/{B0,B3,B4,D_s0,D_s1,D_s2}__txn_test.jsonl; quadruple definitions: "
            "actionproof/txn/tasks.py REALITIES; reply texts are the recorded observations of B4's first call.")
    s.lines(40, yb + 4, wrap(note, 200), 11.5, fill=MUTED)
    tbl = ["| Reality | Arm | Verified | False completion | Duplicate | Episodes |", "|---|---|---|---|---|---|"]
    for real in REALITY_TEXT:
        for name, _ in FIG1_ARMS:
            c = stats.get((name, real))
            if c:
                tbl.append(f"| {real} | {name} | {c['vtc']} | {c['fcr']} | {c['dmr']} | {c['n']} |")
    esc_rows = "; ".join(f"{n}: {v[0]}/{v[1]}" for n, v in esc_unres.items())
    cap = f"""
**Figure 1. One request, one call, two possible replies, four hidden outcomes.** The counterfactual quadruples of
txn_test hold the request and the agent-visible reply fixed and vary only what the system of record holds. The example
request and replies are quadruple {e0['quad']} (tasks {', '.join(ex[r]['task_id'] for r in REALITY_TEXT)}), taken
from B4's recorded first call.

- **Sources:** `results/txn/{{B0,B3,B4,D_s0,D_s1,D_s2}}__txn_test.jsonl` (per-episode `txn` flags);
  `actionproof/txn/tasks.py` (REALITIES, quadruple generator); `data/bench/proc/txn_test.jsonl`.
- **Split and denominators:** the 15 standard-regime quadruples of txn_test (60 tasks; 15 per cell). Scripted policies:
  1 episode per task (15 per cell). D: 3 seeds x 3 episodes per task (135 per cell).
- **Uncertainty:** counts only; with 15 tasks per cell, one task is 6.7 points.
- **Unresolvable regime** (same 15 quadruples with unreadable records; hand-offs among {60} tasks per scripted policy,
  540 D episodes): {esc_rows}.

{chr(10).join(tbl)}
"""
    save("fig1_counterfactual", s, "Figure 1. One request, two replies, four hidden outcomes",
         "Two-by-two matrix of reply (ok, timeout) by hidden outcome (committed, not committed) with measured outcomes.",
         cap)


# ---------------------------------------------------------------------- Figure 6
def git_added(path: str) -> str:
    try:
        out = subprocess.run(["git", "log", "--diff-filter=A", "--format=%h %ad", "--date=format-local:%Y-%m-%d %H:%M",
                              "--", path], cwd=ROOT, capture_output=True, text=True, check=True,
                             env={"TZ": "Asia/Kolkata", "PATH": "/usr/bin:/bin"}).stdout.strip().splitlines()
        return (out[-1] + " IST") if out else "not committed"
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def claim_numbers(ids: list[str]) -> tuple[int, int]:
    try:
        import yaml
    except ImportError:
        return (-1, -1)
    c = yaml.safe_load((ROOT / "claims/claims.yaml").read_text())
    hit = [x for x in c["claims"] if x["id"] in ids]
    return len(hit), sum(len(x.get("numbers") or {}) for x in hit)


def fig6() -> None:
    lanes = []
    man = json.loads((ROOT / "data/bench/proc/txn_manifest.json").read_text())
    f = ROOT / "data/bench/proc/txn_test.jsonl"
    live = sha(f)
    ok = live == man["files"]["txn_test.jsonl"]["sha256"]
    reg = json.loads((ROOT / "results/txn1_hashes.json").read_text())
    reg_ok = live == next((v for k, v in reg.get("files", reg).items() if k.endswith("txn_test.jsonl")), None)
    n_txn = len(list((ROOT / "results/txn").glob("*__txn_test.jsonl")))
    nc, nn = claim_numbers(["TXN1-H3", "TXN-DESC", "TXN-H5", "TXN-DESC2"])
    stats_md = ROOT / "results/txn/stats__txn_test.json"
    lanes.append(("TXN1 (transaction recovery benchmark)", [
        ("Source", ["actionproof/txn/", "env, truth, policies, tasks", "docs/txn/TXN_SPEC.md",
                    f"spec sha256 {sha(ROOT / 'docs/txn/TXN_SPEC.md')[:12]}"]),
        ("Registration", ["PREREGISTRATION_", "TXN1.md", f"added {git_added('PREREGISTRATION_TXN1.md')}",
                          "+ addenda 1-3 (dated)"]),
        ("Task hash", ["data/bench/proc/", "txn_test.jsonl (440)", f"sha256 {live[:12]}",
                       f"manifest {'matches' if ok else 'MISMATCH'}; registration {'matches' if reg_ok else 'MISMATCH'}"]),
        ("Raw traces", [f"results/txn/ ({n_txn} episode files)", "released: HF dataset", "atharvgaur18/", "actionproof-traces"]),
        ("Analysis", ["scripts/txn_stats.py", "-> results/txn/", "stats__txn_test.json",
                      f"sha256 {sha(stats_md)[:12]}" if stats_md.exists() else "missing"]),
        ("Claims ledger", ["claims/claims.yaml", f"{nc} TXN claims, {nn} numbers", "each re-read from its file by",
                           "scripts/s2_numcheck.py"]),
        ("Surfaces", ["README, write-up, slides, video captions, paper (scanned by s2_numcheck)",
                      "site: built from results"]),
    ]))
    sm = json.loads((ROOT / "data/bench/proc/sel1_manifest.json").read_text())
    f2 = ROOT / "data/bench/proc/sel1_test.jsonl"
    live2 = sha(f2)
    ok2 = live2 == sm["files"]["sel1_test.jsonl"]["sha256"]
    sh = json.loads((ROOT / "results/sel1_hashes.json").read_text())
    reg2 = sh.get("files", {})
    reg2_ok = live2 == next((v for k, v in reg2.items() if k.endswith("sel1_test.jsonl")), None)
    nc2, nn2 = claim_numbers(["SEL1"])
    summ = ROOT / "results/sel1/sel1_summary.json"
    lanes.append(("SEL1 (selective escalation, registered 2026-11-27)", [
        ("Source", ["actionproof/sel1/", "env, tasks, policy, build", "sha256 recorded in", "results/",
                    "sel1_hashes.json"]),
        ("Registration", ["PREREGISTRATION_", "SEL1.md", f"added {git_added('PREREGISTRATION_SEL1.md')}",
                          "+ addendum 1 (post-hoc scope, dated)"]),
        ("Task hash", ["data/bench/proc/", "sel1_test.jsonl (240, sealed)", f"sha256 {live2[:12]}",
                       f"manifest {'matches' if ok2 else 'MISMATCH'}; registration {'matches' if reg2_ok else 'MISMATCH'}"]),
        ("Raw traces", ["results/sel1/ episodes", "released: HF dataset", "atharvgaur18/", "actionproof-traces"]),
        ("Analysis", ["scripts/sel1_stats.py", "-> results/sel1/", "sel1_summary.json",
                      f"sha256 {sha(summ)[:12]}" if summ.exists() else "missing"]),
        ("Claims ledger", ["claims/claims.yaml", f"SEL1 block, {nn2} numbers", "each re-read from its file by",
                           "scripts/s2_numcheck.py"]),
        ("Surfaces", ["README, write-up, slides, video captions (scanned by s2_numcheck)",
                      "site: built from results"]),
    ]))
    W, H = 1360, 660
    s = Svg(W, H)
    s.text(40, 38, "Figure 6. From source to slide: every claim can be followed to a file and a command", 18, weight=650)
    s.text(40, 60, "Hashes are recomputed when this figure is built; 'matches' means the live file equals the committed "
                   "manifest and the registration record.", 12.5, fill=MUTED)
    bw, gap, bh = 170, 18, 140
    for li, (lane, boxes) in enumerate(lanes):
        y = 104 + li * 190
        s.text(40, y - 10, lane, 13.5, weight=650)
        for bi, (head, body) in enumerate(boxes):
            x = 40 + bi * (bw + gap)
            bad = any("MISMATCH" in b or b == "missing" for b in body)
            s.rect(x, y, bw, bh, PANEL, "#a33a2b" if bad else "#d2d2d0", 1.2, rx=6)
            s.text(x + 10, y + 20, f"{bi + 1}. {head}", 12.5, weight=650)
            yy = y + 40
            for b in body:
                for ln in (textwrap.wrap(b, 26, break_long_words=False) or [""]):
                    s.text(x + 10, yy, ln, 11, mono=("/" in b or "sha256" in b) and " " not in b.strip(), fill=TEXT)
                    yy += 15
            if bi < len(boxes) - 1:
                s.arrow(x + bw + 2, y + bh / 2, x + bw + gap - 1, y + bh / 2)
    y = 104 + 2 * 190 - 6
    s.text(40, y, "Commands (run from a fresh clone; times from the clean-room runs, research_audit/cleanroom/)", 13.5,
           weight=650)
    cmds = [("python scripts/smoke_check.py", "CPU, no download, no token; 27 s. Tests, frozen splits regenerated from "
                                              "their seeds, scripted baselines, claims ledger."),
            ("bash scripts/verify_numbers.sh", "CPU, about 2 GB anonymous download; 367-375 s. Every registered statistic "
                                               "recomputed from released episodes and diffed against the commit."),
            ("python scripts/s2_numcheck.py", "Every result-like number in the scanned documents must resolve through "
                                              "claims/claims.yaml to a value re-read from its file.")]
    cw = (W - 80 - 2 * 16) / 3
    for i, (c, d) in enumerate(cmds):
        x = 40 + i * (cw + 16)
        s.rect(x, y + 14, cw, 96, SURF, "#d2d2d0", 1.2, rx=6)
        s.text(x + 12, y + 38, c, 12, mono=True, weight=600)
        s.lines(x + 12, y + 60, wrap(d, 62), 11.5, fill=MUTED)
    rows = []
    for lane, boxes in lanes:
        for head, body in boxes:
            rows.append(f"| {lane} | {head} | {'; '.join(body)} |")
    cap = f"""
**Figure 6. Reproducibility map.** Two registered studies, TXN1 and SEL1, traced from source code to the documents
that quote their numbers. Every hash and count in the figure is recomputed by `scripts/build_final_figures.py` when it
runs; a box turns red if a live file no longer matches its manifest or registration record.

- **Sources:** `data/bench/proc/txn_manifest.json`, `data/bench/proc/sel1_manifest.json`, `results/txn1_hashes.json`,
  `results/sel1_hashes.json`, `claims/claims.yaml`, git history (registration commits),
  `docs/submission/REPRODUCIBILITY_FINAL.md` (command times from the two clean-room runs).
- **Split:** txn_test (440 tasks) and sel1_test (240 tasks, sealed until the SEL1 evaluation).
- **Uncertainty:** not applicable (provenance figure).

| Study | Step | Content |
|---|---|---|
{chr(10).join(rows)}
"""
    save("fig6_reproducibility", s, "Figure 6. Reproducibility map", "Two lanes of seven provenance steps with commands.",
         cap)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig1()
    fig3()
    fig4()
    fig6()
    for p in sorted(OUT.iterdir()):
        print(p.relative_to(ROOT), p.stat().st_size)


if __name__ == "__main__":
    main()
