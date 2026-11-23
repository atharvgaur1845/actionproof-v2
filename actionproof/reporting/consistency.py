"""Publication consistency (docs/quality/PUBLICATION_CONSISTENCY_REPORT.md): every result-like number on every
published surface, checked against the claim ledger's `shown` strings and `allow` list.

    python -m actionproof.reporting.consistency     # writes results/canonical/consistency.json

The ledger's own documents (VIDEO_SCRIPT, PRODUCT, README, paper) are enforced by `scripts/s2_numcheck.py`. This
check extends the same token rule to the other surfaces, and reports instead of failing:
- the write-up and article;
- the final claims and review documents;
- the upgrade reports;
- the site template's hand-written text;
- the generated site with its scripts and styles stripped.

A number that is not in the ledger is not necessarily wrong. It is **unverified**: no entry ties it to a result
file, so a stale or mistyped value could not be caught.
"""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SURFACES = ["docs/submission/WRITEUP.md", "docs/ARTICLE_DRAFT.md", "docs/final/FINAL_CLAIMS.md", "docs/final/THESIS.md",
            "docs/upgrade/FINAL_REVIEW.md", "docs/upgrade/ROI.md", "docs/upgrade/CAPABILITY_RETENTION.md",
            "docs/upgrade/JUDGES.md", "demo/site_template.html", "demo/static/index.html",
            "docs/stage2/VIDEO_SCRIPT.md", "docs/stage2/PRODUCT.md", "README.md", "paper/main.tex", "paper/appendix.tex"]


def text_of(p: Path) -> str:
    t = p.read_text()
    if p.suffix == ".html":  # visible text only: drop scripts, styles, tags and embedded JSON
        t = re.sub(r"<script.*?</script>|<style.*?</style>", " ", t, flags=re.S)
        t = re.sub(r"<[^>]+>", " ", t)
    return t


def main() -> None:
    spec = importlib.util.spec_from_file_location("s2_numcheck", ROOT / "scripts/s2_numcheck.py")
    nc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nc)
    cfg = yaml.safe_load((ROOT / "claims/claims.yaml").read_text())
    shown = {str(s).replace("−", "-") for c in cfg["claims"] if c["status"] != "pending"
             for n in (c.get("numbers") or {}).values() for s in n.get("shown", [])}
    allow = set(map(str, cfg.get("allow", [])))
    enforced = set(cfg["documents"])
    rep = {}
    for doc in SURFACES:
        p = ROOT / doc
        if not p.exists():
            rep[doc] = {"exists": False}
            continue
        toks = [t for t in nc.TOKEN.findall(text_of(p).replace("−", "-")) if nc.is_result_token(t)]
        bad = sorted({t for t in toks if t not in shown and t not in allow})
        rep[doc] = {"exists": True, "enforced_by_numcheck": doc in enforced, "result_tokens": len(toks),
                    "unverified_distinct": len(bad), "unverified_examples": bad[:40]}
    out = ROOT / "results/canonical/consistency.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1))
    for d, r in rep.items():
        if r.get("exists"):
            print(f"{d:45s} enforced={str(r['enforced_by_numcheck']):5s} tokens={r['result_tokens']:5d} unverified={r['unverified_distinct']}")


if __name__ == "__main__":
    main()
