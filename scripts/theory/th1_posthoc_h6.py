"""POST HOC (not registered): why did TH1-H6 fail? Lists the H6 cells where GRPO's regret exceeds Dr. GRPO's + 0.01,
with the predicted and simulated pi_T, regrets, delta_q, vote, G, pi0.

    python scripts/theory/th1_posthoc_h6.py --split heldout   # -> results/theory/th1_heldout_posthoc_h6.json
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/theory"))
from th1_grid import grid  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="heldout")
a = ap.parse_args()
cells = {c["id"]: c for c in grid(a.split)}
rows = defaultdict(dict)
for f in sorted((ROOT / f"results/theory/th1_{a.split}").glob("shard_*.jsonl")):
    for line in f.open():
        r = json.loads(line)
        if r["est"] in ("grpo", "nostd"):
            rows[r["cell"]][r["est"]] = r


def sgn(x):
    return 0 if abs(x) < 1e-9 else (1 if x > 0 else -1)


out = []
for cid, rr in rows.items():
    c, g, n = cells[cid], rr["grpo"], rr["nostd"]
    if not (c["law"] == "shared" and g["deterministic"] and g["min_abs_beta"] >= 0.005 and abs(g["vote"]) > 1e-9):
        continue
    if sgn(g["vote"]) != sgn(g["delta_q"]):
        continue
    rg = float(np.mean([s["regret"] for s in g["sim"]]))
    rn = float(np.mean([s["regret"] for s in n["sim"]]))
    out.append({"cell": cid, "fail": rg > rn + 0.01, "regret_grpo": rg, "regret_nostd": rn,
                "pi_T_grpo": float(np.mean([s["pi_T"] for s in g["sim"]])), "pred_grpo": g["pred"]["pi_T"],
                "pi_T_nostd": float(np.mean([s["pi_T"] for s in n["sim"]])), "pred_nostd": n["pred"]["pi_T"],
                "delta_q": g["delta_q"], "vote": g["vote"], "G": c["G"], "pi0": c["pi0"],
                "endpoint_grpo": g["pred"]["endpoint"], "endpoint_nostd": n["pred"]["endpoint"]})
fails = [o for o in out if o["fail"]]
summ = {"n": len(out), "fails": len(fails),
        "fails_with_vote_positive": sum(o["vote"] > 0 for o in fails),
        "fails_where_grpo_endpoint_is_bayes_corner": sum((o["endpoint_grpo"] > 0.5) == (o["delta_q"] > 0) for o in fails),
        "fails_mean_abs_pi_gap_to_bayes_grpo": float(np.mean([abs(o["pi_T_grpo"] - (1.0 if o["delta_q"] > 0 else 0.0)) for o in fails])) if fails else None,
        "fails_mean_abs_pi_gap_to_bayes_nostd": float(np.mean([abs(o["pi_T_nostd"] - (1.0 if o["delta_q"] > 0 else 0.0)) for o in fails])) if fails else None,
        "fails_mean_regret_grpo": float(np.mean([o["regret_grpo"] for o in fails])) if fails else None,
        "fails_mean_regret_nostd": float(np.mean([o["regret_nostd"] for o in fails])) if fails else None,
        "fails_max_pred_error_grpo": max(abs(o["pi_T_grpo"] - o["pred_grpo"]) for o in fails) if fails else None}
p = ROOT / f"results/theory/th1_{a.split}_posthoc_h6.json"
p.write_text(json.dumps({"label": "POST HOC, not registered", "summary": summ, "cells": out}, indent=1))
print(json.dumps(summ, indent=1))
for o in sorted(fails, key=lambda x: -x["regret_grpo"])[:12]:
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in o.items()})
