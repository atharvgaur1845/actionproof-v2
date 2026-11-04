"""Footprint metrics for episode files (spec §12): mean excess writes, USER = P(excess > 0), NWCG,
over episodes whose final state meets every goal. Replays each recorded episode (no model needed).

    python -m actionproof.eval.footprint_report results/eval/r3/*__val.jsonl
"""
from __future__ import annotations

import json
import sys

from actionproof.bench.generator import make_split
from actionproof.env.counterfactual import footprint, replay
from actionproof.eval.stats import wilson


def report(path: str) -> dict:
    eps = [json.loads(l) for l in open(path)]
    split = eps[0]["task_id"].rsplit("_", 1)[0]
    tasks = {t.task_id: t for t in make_split(split)}
    fps = []
    for e in eps:
        t = tasks[e["task_id"]]
        fps.append(footprint(t, replay(t, e)))
    ok = [f for f in fps if f["excess"] is not None]
    k = sum(f["excess"] > 0 for f in ok)
    p, lo, hi = wilson(k, len(ok))
    return {"file": path.rsplit("/", 1)[-1], "n_goal_met": len(ok),
            "mean_excess": round(sum(f["excess"] for f in ok) / max(len(ok), 1), 3),
            "USER": round(p, 3) if ok else None, "USER_ci95": [round(lo, 3), round(hi, 3)] if ok else None,
            "NWCG": round(sum(f["nwcg"] for f in ok) / max(len(ok), 1), 3)}


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(json.dumps(report(p)))
