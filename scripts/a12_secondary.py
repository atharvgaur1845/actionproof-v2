"""A12 addendum 2 §3 sensitivity analyses and Case B diagnostics on the confirmatory outputs
(results/eval/a12/conf/a12_<arm>_s<seed>__neutral__<split>.jsonl), per seed and arm:
  task_level_phantom_readback  share of tasks with a phantom write in which the agent read back after it at least once
                               (not affected by how many phantom attempts each arm makes)
  phantom_readback_by_step     attempt-level phantom read-back per step (pay, order, receive, post)
  observation_equivalence      phantom vs benign read-back (attempt level)
  case_b                       lost-ack read-back, precommit recovery, false success, duplicate rate
    python scripts/a12_secondary.py --out results/s2/a12_secondary.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEP = {"pay_supplier": "pay", "place_supplier_order": "order", "record_goods_receipt": "receive",
        "post_journal_entry": "post"}
READ = {"pay": "list_payments", "order": "find_supplier_order", "receive": "list_goods_receipts",
        "post": "get_journal_entries"}


def phantom_attempts(r: dict) -> list[tuple[str, bool]]:
    """(step, read back before the next write or report) for every agent attempt that was acknowledged ok but
    committed nothing or only part, following VERIFIER_SPEC §7."""
    po = r["task"]["meta"]["po_id"]
    trace = [s for s in r["trace"] if s.get("by", "agent") == "agent"]
    atts = [a for a in r["attempts"] if not a.get("from_idempotency_store")]
    out = []
    for a in atts:
        if a["po_id"] != po or a["tool"] not in STEP or a["obs_status"] != "ok" or a["committed"] not in ("none", "partial"):
            continue
        step = STEP[a["tool"]]
        idx = next((i for i, s in enumerate(trace) if s["idx"] == a["idx"]), None)
        rb = False
        if idx is not None:
            for s in trace[idx + 1:]:
                if s["tool"] in STEP or s["tool"] == "submit_report":
                    break
                if s["tool"] == READ[step] and (s["args"] or {}).get("po_id") == po:
                    rb = True
                    break
        out.append((step, rb))
    return out


def ratio(rs: list[dict], key: str) -> dict:
    n = sum(r["score"][key][0] for r in rs)
    d = sum(r["score"][key][1] for r in rs)
    return {"num": n, "den": d, "rate": n / d if d else None}


def analyse(rs: list[dict]) -> dict:
    tasks = [phantom_attempts(r) for r in rs]
    with_ph = [t for t in tasks if t]
    by_step: dict[str, list[bool]] = {}
    for t in tasks:
        for step, rb in t:
            by_step.setdefault(step, []).append(rb)
    n = len(rs)
    return {"task_level_phantom_readback": {"num": sum(any(rb for _, rb in t) for t in with_ph), "den": len(with_ph)},
            "phantom_readback_by_step": {k: {"num": sum(v), "den": len(v)} for k, v in sorted(by_step.items())},
            "observation_equivalence": {"phantom": ratio(rs, "phantom_readback"), "benign": ratio(rs, "benign_readback")},
            "case_b": {"lost_ack_readback": ratio(rs, "lost_ack_readback"),
                       "precommit_recovery": ratio(rs, "precommit_recovery"),
                       "false_success": sum(r["score"]["false_success"] for r in rs) / n,
                       "duplicate": sum(r["score"]["duplicate_irreversible"] for r in rs) / n}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--out", default="results/s2/a12_secondary.json")
    a = ap.parse_args()
    res: dict = {}
    for sp in ("proc_conf", "proc_combo", "proc_stress"):
        for s in a.seeds.split(","):
            for arm in ("rv", "ra"):
                f = ROOT / f"results/eval/a12/conf/a12_proc_{arm}_s{s}__neutral__{sp}.jsonl"
                if f.exists():
                    res.setdefault(sp, {})[f"{arm}_s{s}"] = analyse([json.loads(line) for line in f.open()])
    out = ROOT / a.out
    out.write_text(json.dumps(res, indent=2))
    for sp, runs in res.items():
        for k, v in runs.items():
            t, oe, cb = v["task_level_phantom_readback"], v["observation_equivalence"], v["case_b"]
            print(sp, k, "task-level", f"{t['num']}/{t['den']}", "phantom", oe["phantom"]["rate"], "benign",
                  round(oe["benign"]["rate"], 3) if oe["benign"]["rate"] is not None else None,
                  "lostack_rb", cb["lost_ack_readback"]["rate"], "precommit_rec", cb["precommit_recovery"]["rate"],
                  "false_succ", round(cb["false_success"], 3), "dup", cb["duplicate"])


if __name__ == "__main__":
    main()
