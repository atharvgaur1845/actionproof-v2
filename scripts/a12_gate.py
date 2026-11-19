"""Amendment 12 warm-start gate and H3 step-0 predictions, from the sampled gate file (T = 1 x 4, 100 proc_train tasks).

Gate (all must hold; PREREGISTRATION_A12 §4):
  phantom read-back in [0.10, 0.60]; verified completion in [0.25, 0.75]; tool-call validity >= 0.95;
  share of tasks whose 4 samples are not all equal under the reward >= 0.30, for R_V and for R_A
  (a conservative stand-in for the share of informative GRPO groups of 8).
H3 predictions (committed before GRPO step 1): for each fault class and reward, the mean group-relative advantage
of samples that read back a write on the task PO minus that of samples that did not. Its sign is the predicted
direction of selection on reading under that reward in that class.

    python scripts/a12_gate.py --gate results/eval/a12/init__neutral__proc_train_T1x4.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def reads_back(r: dict) -> bool:
    s = r["score"]
    return any(s[k][0] > 0 for k in ("phantom_readback", "benign_readback", "lost_ack_readback"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", required=True)
    ap.add_argument("--out", default="results/a12/gate.json")
    a = ap.parse_args()
    recs = [json.loads(l) for l in open(a.gate)]
    groups = defaultdict(list)
    for r in recs:
        groups[r["task_id"]].append(r)
    pr_n = sum(r["score"]["phantom_readback"][0] for r in recs)
    pr_d = sum(r["score"]["phantom_readback"][1] for r in recs)
    calls = sum(r["n_model_calls"] for r in recs)
    gate = {
        "n_episodes": len(recs), "n_tasks": len(groups),
        "phantom_readback": pr_n / pr_d if pr_d else None, "phantom_readback_counts": [pr_n, pr_d],
        "vsc": sum(r["score"]["vsc"] for r in recs) / len(recs),
        "tool_call_validity": 1 - sum(r["invalid"] for r in recs) / max(calls, 1),
        "nonzero_variance_share": {k: sum(len({r["score"][k] for r in g}) > 1 for g in groups.values()) / len(groups)
                                   for k in ("r_v", "r_a")},
        "read_back_episode_share": sum(reads_back(r) for r in recs) / len(recs),
    }
    gate["checks"] = {
        "phantom_readback_in_band": gate["phantom_readback"] is not None and 0.10 <= gate["phantom_readback"] <= 0.60,
        "vsc_in_band": 0.25 <= gate["vsc"] <= 0.75,
        "validity": gate["tool_call_validity"] >= 0.95,
        "informative_groups": all(v >= 0.30 for v in gate["nonzero_variance_share"].values()),
    }
    gate["pass"] = all(gate["checks"].values())
    diff: dict = {}
    for reward in ("r_v", "r_a"):
        by_cls = defaultdict(lambda: {"read": [], "noread": []})
        for g in groups.values():
            mean = sum(r["score"][reward] for r in g) / len(g)
            for r in g:
                by_cls[r["fault_class"]]["read" if reads_back(r) else "noread"].append(r["score"][reward] - mean)
        diff[reward] = {}
        for cls, d in sorted(by_cls.items()):
            if d["read"] and d["noread"]:
                v = sum(d["read"]) / len(d["read"]) - sum(d["noread"]) / len(d["noread"])
                sign = "+" if v > 0.01 else "-" if v < -0.01 else "0"
            else:
                v, sign = None, "n/a (no contrast in gate samples)"
            diff[reward][cls] = {"advantage_read_minus_noread": v, "predicted_sign": sign,
                                 "n_read": len(d["read"]), "n_noread": len(d["noread"])}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(gate, indent=2))
    out.with_name("h3_predictions.json").write_text(json.dumps(
        {"source": a.gate, "note": "Committed before GRPO step 1 (PREREGISTRATION_A12 H3).", "selection": diff}, indent=2))
    print(json.dumps(gate, indent=2))
    print(json.dumps(diff, indent=2))


if __name__ == "__main__":
    main()
