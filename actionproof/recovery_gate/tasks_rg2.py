"""RG2's fresh sealed split `rg2_test` (PREREGISTRATION_RG2.md): SEL1's test composition (actionproof/sel1/tasks.py,
six groups of 10 quadruples: u_in, u_receive, u_ood_wf, t2, audit_down, std; 240 tasks) with a new seed. Instance-
disjoint from every existing split; same templates as sel1_test.

    python -m actionproof.recovery_gate.tasks_rg2     # writes data/bench/proc/rg2_test.jsonl and rg2_manifest.json
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import actionproof.sel1.tasks as S
import actionproof.txn.tasks as T
from actionproof.bench.generator_proc import DATA, ProcTask

SEED = 26112802
SPLIT = "rg2_test"


def make_split() -> list[ProcTask]:
    rng = random.Random(SEED)
    out: list[ProcTask] = []
    for g in S.TEST_GROUPS:  # the sel1_test construction, verbatim, under a new split name and seed
        for q in range(S.TEST_QUADS):
            qid = f"{SPLIT}-{g}-q{q:02d}"
            if g == "u_in":
                spec = (T.TRAINED_WF[q % 3], [("unresolvable", 0)], S.TRAIN_STEPS)
            elif g == "u_receive":
                spec = ("receive_pay", [("unresolvable", 0)], ("receive",))
            elif g == "u_ood_wf":
                spec = (S.OOD_WFS[q % 2], [("unresolvable", 0)], None)
            elif g == "t2":
                spec = (T.TRAINED_WF[q % 3], [("transient", 2)], None)
            elif g == "audit_down":
                spec = (T.TRAINED_WF[q % 3], [("audit_down", 0)], None)
            else:
                spec = (T.TRAINED_WF[q % 3], [("standard", 0)], None)
            wf, regs, steps = spec
            out += S.quad(SPLIT, len(out), rng, wf, qid, g, regs, steps)
    return out


def check_disjoint(tasks: list[ProcTask]) -> None:
    tids, pos, prompts = T.existing(exclude_prefix="rg2_")
    po_owner: dict[str, str] = {}
    seen: set[str] = set()
    for t in tasks:
        owner = t.meta["txn"].get("quad_id") or t.task_id
        assert t.task_id not in tids and t.task_id not in seen, t.task_id
        seen.add(t.task_id)
        assert t.prompt not in prompts, f"{t.task_id}: prompt collides with an existing split"
        for p in t.initial_state["pos"]:
            assert p["po_id"] not in pos, f"{t.task_id}: PO id collides"
            assert po_owner.setdefault(p["po_id"], owner) == owner, f"{t.task_id}: PO id reused"


def write(out: Path = DATA) -> dict[str, Any]:
    tasks = make_split()
    check_disjoint(tasks)
    path, man = out / f"{SPLIT}.jsonl", out / "rg2_manifest.json"
    if path.exists() or man.exists():
        raise SystemExit("refusing to overwrite rg2_test")
    text = "".join(T.task_line(t) for t in tasks)
    path.write_text(text)
    m = {"registration": "PREREGISTRATION_RG2.md", "generator": "actionproof/sel1/tasks.py (test construction)",
         "seed": SEED, "files": {path.name: {"n": len(tasks), "sha256": hashlib.sha256(text.encode()).hexdigest()}},
         "groups": {g: sum(t.meta["txn"]["group"] == g for t in tasks) for g in S.TEST_GROUPS}}
    man.write_text(json.dumps(m, indent=1, sort_keys=True) + "\n")
    return m["files"]


if __name__ == "__main__":
    print(json.dumps(write(), indent=1))
