"""Source-freshness check (docs/quality/RESULT_INTEGRITY_REPORT.md). Stops an old data file or dashboard being
presented as the latest result.

    python -m actionproof.reporting.freshness      # exit 1 if any generated surface is stale

**Generated surfaces.** Each is declared with its builder. Its sources are every `results/...`, `data/...`,
`demo/...` or `docs/...` path or glob that appears as a string literal in the builder (and in modules the builder
names below). A surface is **stale** if any of those source files is newer than the surface itself, or if the
builder script is.

**The canonical file** is also checked by hash: it records each source's sha256, and any change in a source makes
it stale.

**Hand-written documents** cannot be regenerated; `consistency.py` checks their numbers instead.
"""
from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SURFACES = {
    "demo/static/index.html": ["scripts/build_site.py", "scripts/site_upgrade.py", "scripts/site_procure.py"],
    "demo/data.json": ["scripts/build_demo_data.py"],
    "product/data/dashboard.json": ["product/build_data.py"],
    "product/data/replays.json": ["product/build_data.py"],
    "docs/figures/figures_data.json": ["scripts/make_figures.py"],
}
LIT = re.compile(r"""["']((?:results|data|demo|docs)/[^"'\s{}]+)["']""")


def sources(builders: list[str]) -> list[Path]:
    out = set()
    for b in builders:
        p = ROOT / b
        if not p.exists():
            continue
        out.add(p)
        for lit in LIT.findall(p.read_text()):
            out.update(Path(x) for x in glob.glob(str(ROOT / lit), recursive=True) if Path(x).is_file())
    return sorted(out)


def check() -> dict:
    rep = {}
    for surf, builders in SURFACES.items():
        s = ROOT / surf
        if not s.exists():
            rep[surf] = {"status": "missing"}
            continue
        newer = [str(p.relative_to(ROOT)) for p in sources(builders) if p.exists() and p != s and p.stat().st_mtime > s.stat().st_mtime]
        rep[surf] = {"status": "stale" if newer else "fresh", "newer_sources": newer[:12], "n_newer": len(newer)}
    can = ROOT / "results/canonical/results.json"
    if can.exists():
        import hashlib
        d = json.loads(can.read_text())
        changed = [f for f, h in d["sources"].items()
                   if h != (hashlib.sha256((ROOT / f).read_bytes()).hexdigest() if (ROOT / f).is_file() else None)]
        rep["results/canonical/results.json"] = {"status": "stale" if changed else "fresh", "newer_sources": changed[:12],
                                                 "n_newer": len(changed)}
    return rep


def main() -> None:
    rep = check()
    for k, v in rep.items():
        print(f"{v['status']:7s} {k}" + (f"  ({v['n_newer']} newer sources, e.g. {', '.join(v['newer_sources'][:3])})" if v.get("n_newer") else ""))
    sys.exit(1 if any(v["status"] != "fresh" for v in rep.values()) else 0)


if __name__ == "__main__":
    main()
