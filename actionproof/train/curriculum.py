"""Learnable-band curriculum (spec §13.3).

Given samples of the shared init policy on the training split (run_eval with --samples 8,
T=1.0), keep tasks whose per-task VSC rate lies in [lo, hi]. Tasks the policy always or never
solves give zero-variance GRPO groups and no gradient. The SAME task file is used by every
GRPO arm, so the curriculum cannot favour one reward.

    python -m actionproof.train.curriculum --samples results/curriculum/rft_tiny_train.jsonl \
        --out data/curriculum/band_10_90.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def band(records, lo: float, hi: float, metric: str = "vsc") -> tuple[list[dict], dict]:
    by_task: dict[str, list[float]] = defaultdict(list)
    meta: dict[str, dict] = {}
    for r in records:
        by_task[r["task_id"]].append(float(r[metric]) if metric == "vsc" else r["verdict"][metric])
        meta[r["task_id"]] = {"faults": r.get("faults", []), "workflow": r.get("workflow")}
    kept, rates = [], {}
    for t, xs in sorted(by_task.items()):
        rate = sum(xs) / len(xs)
        rates[t] = rate
        if lo <= rate <= hi:
            kept.append({"task_id": t, "rate": rate, **meta[t]})
    hist = Counter(round(r, 1) for r in rates.values())
    stats = {"tasks": len(by_task), "kept": len(kept), "lo": lo, "hi": hi,
             "rate_histogram": dict(sorted(hist.items())),
             "kept_faults": dict(Counter(f for k in kept for f in (k["faults"] or ["benign"])))}
    return kept, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lo", type=float, default=0.10)
    ap.add_argument("--hi", type=float, default=0.90)
    ap.add_argument("--min-kept", type=int, default=120, help="widen to [0, 1) band if fewer survive")
    a = ap.parse_args()
    records = [json.loads(l) for f in a.samples for l in open(f)]
    kept, stats = band(records, a.lo, a.hi)
    if len(kept) < a.min_kept:  # too strict for a weak init: keep anything not always-solved
        kept, stats = band(records, 0.0, 0.99)
        stats["widened"] = True
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(k) + "\n" for k in kept))
    out.with_suffix(".stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
