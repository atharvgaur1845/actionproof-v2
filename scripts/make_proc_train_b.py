"""A12 addendum 1: training split `proc_train_b` with a rebalanced fault mix (registered before any GRPO run).

Why: the registered gate failed only its informative-group check (R_V 0.15, R_A 0.17 of tasks with reward variation),
because 48% of `proc_train` tasks are classes where reading cannot change either reward (reward-truth table: clean and
precommit timeout). This split keeps every class but shifts weight to the classes where the rewards differ.
Same generator (`actionproof/bench/generator_proc.py`, unchanged: its TRAIN_MIX is overridden at run time here),
seed 109, 400 tasks, PO ids disjoint from every other split (asserted).
    python scripts/make_proc_train_b.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import actionproof.bench.generator_proc as G

MIX_B = (("clean", 0.05), ("phantom", 0.35), ("partial", 0.15), ("lost_ack", 0.25),
         ("precommit_timeout", 0.05), ("never_commit", 0.15))
SEED, N = 109, 400

if __name__ == "__main__":
    G.TRAIN_MIX = MIX_B
    rng = random.Random(SEED)
    tasks = [G.make_task("proc_train_b", i, rng) for i in range(N)]
    seen = {p["po_id"] for s in G.SPLITS for t in G.load_tasks(G.DATA / f"{s}.jsonl") for p in t.initial_state["pos"]}
    for t in tasks:
        for p in t.initial_state["pos"]:
            assert p["po_id"] not in seen, p["po_id"]
            seen.add(p["po_id"])
    out = G.DATA / "proc_train_b.jsonl"
    out.write_text("".join(json.dumps(t.to_json(), sort_keys=True) + "\n" for t in tasks))
    from collections import Counter
    print(out, Counter(t.meta["fault_class"] for t in tasks))
