"""RG1's fresh sealed split `rg1_test` (PREREGISTRATION_RG1.md): the TXN v1 generator (actionproof/txn/tasks.py)
with a new seed and txn_test's composition (25 core tasks per fault class, 15 counterfactual quadruples, 20 tasks per
OOD group; 440 tasks). Instance-disjoint from every existing split (task ids, PO ids and prompt texts); it reuses
the TXN templates, so it is not template-disjoint.

    python -m actionproof.recovery_gate.tasks          # writes data/bench/proc/rg1_test.jsonl and rg1_manifest.json
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import actionproof.txn.tasks as T
from actionproof.bench.generator_proc import DATA, ProcTask

SEEDS = {"rg1_test": 26112801}
SIZES = {"rg1_test": T.SIZES["txn_test"]}


def make_split(split: str = "rg1_test") -> list[ProcTask]:
    saved = T.SEEDS, T.SIZES
    T.SEEDS, T.SIZES = {**T.SEEDS, **SEEDS}, {**T.SIZES, **SIZES}
    try:
        return T.make_split(split)
    finally:
        T.SEEDS, T.SIZES = saved


def check_disjoint(tasks: list[ProcTask]) -> None:
    tids, pos, prompts = T.existing(exclude_prefix="rg1_")
    po_owner: dict[str, str] = {}
    seen: set[str] = set()
    for t in tasks:
        owner = t.meta["txn"].get("quad_id") or t.task_id
        assert t.task_id not in tids and t.task_id not in seen, f"task id {t.task_id} collides"
        seen.add(t.task_id)
        assert t.prompt not in prompts, f"{t.task_id}: prompt text collides with an existing split"
        for p in t.initial_state["pos"]:
            assert p["po_id"] not in pos, f"{t.task_id}: PO id {p['po_id']} collides with an existing split"
            assert po_owner.setdefault(p["po_id"], owner) == owner, f"{t.task_id}: PO id {p['po_id']} reused"


def write(out: Path = DATA) -> dict[str, Any]:
    tasks = make_split()
    check_disjoint(tasks)
    path, man = out / "rg1_test.jsonl", out / "rg1_manifest.json"
    clash = [str(p) for p in (path, man) if p.exists()]
    if clash:
        raise SystemExit(f"refusing to overwrite: {clash}")
    text = "".join(T.task_line(t) for t in tasks)
    path.write_text(text)
    manifest = {"registration": "PREREGISTRATION_RG1.md", "generator": "actionproof/txn/tasks.py", "seeds": SEEDS,
                "sizes": {k: list(v) for k, v in SIZES.items()}, "env_version": "proc3",
                "files": {path.name: {"n": len(tasks), "sha256": hashlib.sha256(text.encode()).hexdigest()}},
                "tasks": {t.task_id: {"template_id": t.meta["txn"]["template_id"], "group": t.meta["txn"]["group"],
                                      "regime": t.meta["txn"]["regime"], "fault_class": t.meta["fault_class"],
                                      "quad_id": t.meta["txn"].get("quad_id"),
                                      "sha256": hashlib.sha256(T.task_line(t).encode()).hexdigest()} for t in tasks}}
    man.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return manifest["files"]


if __name__ == "__main__":
    print(json.dumps(write(), indent=1))
