"""Canonical results (docs/quality/RESULT_INTEGRITY_REPORT.md): one machine-readable file every surface is built from.

    python -m actionproof.reporting.canonical        # writes results/canonical/results.json

**Inputs.**
- The claim ledger `claims/claims.yaml`. Every number is re-read from its result file, exactly as
  `scripts/s2_numcheck.py` does.
- The registered study summaries that exist: P1, P2, AD-C1.

**Each metric entry carries:**
- value (as re-read), stated value, match flag;
- source path and key, experiment (claim) id, registration, and evidence label;
- model, split, seed count, task count, statistical status and interval type where the ledger's `meta` block
  states them; otherwise the explicit string "not_recorded".

**Evidence labels.**

| Label | Rule |
|---|---|
| confirmatory | registered (PREREGISTRATION_*) and supported |
| unsupported | registered, and its registered rule was not met |
| failed | a candidate method dropped by its own criterion ("refuted") |
| development | a candidate method's development result |
| exploratory | registered after results were seen, or descriptive by its registration |
| descriptive | no registration: a description of recorded data |
| incomplete | a registered study whose summary does not exist yet |
| future | named in a registration but not run |

The file records the sha256 of every source it read, and its own content hash. `freshness.py` uses both.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/canonical/results.json"
EXPLORATORY = {"J1-A1", "R1-T", "S4-DIAG"}
STUDIES = {  # registered studies tracked outside the claim ledger until they report
    "P1": ("PREREGISTRATION_P1.md", "results/p1/p1_summary.json"),
    "P2": ("PREREGISTRATION_P2.md", "results/p2/p2_summary.json"),
    "AD-C1": ("PREREGISTRATION_AD_C1.md", "results/ad/c1_summary.json"),
}


def _numcheck():
    spec = importlib.util.spec_from_file_location("s2_numcheck", ROOT / "scripts/s2_numcheck.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def sha(p: Path) -> str | None:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() and p.is_file() else None


def label(c: dict) -> str:
    pre = c.get("prereg")
    pres = pre if isinstance(pre, list) else [pre] if pre else []
    registered = any(str(p).startswith("PREREGISTRATION") for p in pres)
    if c["id"] in EXPLORATORY:
        return "exploratory"
    if c["status"] == "refuted":
        return "failed"
    if c["status"] == "pending":
        return "incomplete"
    if c["status"] == "not_run":
        return "future"
    if not pres:
        return "descriptive"
    if not registered:
        return "development"
    return "confirmatory" if c["status"] == "supported" else "unsupported"


def build() -> dict:
    nc = _numcheck()
    cfg = yaml.safe_load((ROOT / "claims/claims.yaml").read_text())
    sources: dict[str, str | None] = {"claims/claims.yaml": sha(ROOT / "claims/claims.yaml")}
    metrics, claims = [], []
    for c in cfg["claims"]:
        lab = label(c)
        meta = c.get("meta") or {}
        claims.append({"id": c["id"], "statement": c.get("statement"), "status": c["status"], "label": lab,
                       "registration": c.get("prereg"), "stats": c.get("stats")})
        for name, n in (c.get("numbers") or {}).items():
            files = n["file"] if isinstance(n["file"], list) else [n["file"]]
            for f in files:
                sources.setdefault(f, sha(ROOT / f))
            try:
                v, err = nc.file_value(n), None
            except Exception as e:  # a missing file or key is reported, never silently skipped
                v, err = None, f"{type(e).__name__}: {e}"
            metrics.append({
                "claim": c["id"], "metric": name, "label": lab, "registration": c.get("prereg"),
                "value": v, "stated": n.get("value"), "match": (v is not None and nc.matches(n.get("value"), v)),
                "error": err, "source": n["file"], "key": n.get("key"), "agg": n.get("agg"), "shown": n.get("shown", []),
                **{k: meta.get(k, "not_recorded") for k in ("model", "split", "seeds", "tasks", "stat_status", "ci_type")}})
    studies = {}
    for sid, (reg, summ) in STUDIES.items():
        p = ROOT / summ
        sources.setdefault(summ, sha(p))
        studies[sid] = {"registration": reg, "summary": summ, "label": "incomplete" if not p.exists() else "registered",
                        "registration_exists": (ROOT / reg).exists(),
                        "summary_data": json.loads(p.read_text()) if p.exists() else None}
    body = {"claims": claims, "metrics": metrics, "studies": studies, "sources": sources,
            "label_counts": {lab: sum(m["label"] == lab for m in metrics) for lab in sorted({m["label"] for m in metrics})},
            "mismatches": [f"{m['claim']}:{m['metric']}" for m in metrics if not m["match"]]}
    body["content_sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()
    return body


def main() -> None:
    b = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(b, indent=1, default=str))
    print(json.dumps({"metrics": len(b["metrics"]), "claims": len(b["claims"]), "labels": b["label_counts"],
                      "mismatches": len(b["mismatches"]), "studies": {k: v["label"] for k, v in b["studies"].items()},
                      "sha": b["content_sha256"][:12]}))


if __name__ == "__main__":
    main()
