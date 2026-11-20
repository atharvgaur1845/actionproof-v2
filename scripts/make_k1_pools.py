"""K1 task-composition control (PREREGISTRATION_K1): two R_A training pools of equal size cut from the frozen
`proc_train_b` (400 tasks).
  proc_train_k1t (targeted): drops every lost_ack and never_commit task (the classes where reading lowers R_A).
  proc_train_k1p (placebo):  drops the same number of tasks chosen uniformly at random (seed 113).
    python scripts/make_k1_pools.py
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/bench/proc/proc_train_b.jsonl"
DROP = {"lost_ack", "never_commit"}
SEED = 113

if __name__ == "__main__":
    rows = [l for l in SRC.read_text().splitlines() if l.strip()]
    cls = [json.loads(l)["meta"]["fault_class"] for l in rows]
    keep_t = [l for l, c in zip(rows, cls) if c not in DROP]
    keep_p = sorted(random.Random(SEED).sample(range(len(rows)), len(keep_t)))
    pools = {"proc_train_k1t": keep_t, "proc_train_k1p": [rows[i] for i in keep_p]}
    for name, rs in pools.items():
        (ROOT / f"data/bench/proc/{name}.jsonl").write_text("".join(r + "\n" for r in rs))
        print(name, len(rs), dict(Counter(json.loads(r)["meta"]["fault_class"] for r in rs)))
