"""P2 smoke check (not analysed): P2's evaluator with no ledger and no guard must reproduce P1's evaluator.
Compares results/p2/smoke/A_equiv__p1_dev_rec.jsonl with results/p1/dev/A_s0__p1_dev_rec.jsonl episode by episode
(same model, tasks, seeds): share of identical action sequences and of identical verdicts, greedy and sampled.
vLLM batching can change sampled tokens slightly, so identity is expected for most, not all, sampled episodes."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(p):
    return {(r["task_id"], r["sample"]): r for r in map(json.loads, (ROOT / p).read_text().splitlines())}


a = load("results/p1/dev/A_s0__p1_dev_rec.jsonl")
b = load("results/p2/smoke/A_equiv__p1_dev_rec.jsonl")
out = {}
for mode, sel in (("greedy", lambda k: k[1] < 0), ("sampled", lambda k: k[1] >= 0)):
    ks = [k for k in a if sel(k) and k in b]
    same_act = sum([(s["tool"], json.dumps(s["args"], sort_keys=True)) for s in a[k]["trace"]] ==
                   [(s["tool"], json.dumps(s["args"], sort_keys=True)) for s in b[k]["trace"]] for k in ks)
    same_v = sum(a[k]["vsc"] == b[k]["vsc"] for k in ks)
    out[mode] = {"n": len(ks), "identical_actions": same_act / len(ks), "identical_vsc": same_v / len(ks),
                 "vsc_p1": sum(a[k]["vsc"] for k in ks) / len(ks), "vsc_p2": sum(b[k]["vsc"] for k in ks) / len(ks)}
print(json.dumps(out, indent=1))
