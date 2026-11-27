"""SEL1 task pools and the sealed test split (PREREGISTRATION_SEL1.md). Built with TXN v1's generator functions
(`actionproof/txn/tasks.py`, frozen), with fresh seeds; disjoint from every existing `data/bench/proc/*.jsonl`
(task ids, PO ids, prompt text).

Evidence regimes (all on the faulted step of a counterfactual quadruple; the four realities are TXN's):
- `standard`: no outage;
- `unresolvable` (permanent outage): the faulted step's read tool and `get_audit_log` are unavailable for the whole
  episode (TXN's construction); `escalation_correct = True`;
- `transient`: the faulted step's read tool is unavailable for its first n agent calls, then served normally
  (`actionproof/sel1/env.py`); the audit log is available; `escalation_correct = False`;
- `audit_down`: only `get_audit_log` is unavailable; `escalation_correct = False`.

Training (`sel1_train`): core classes (TXN's 8, trained workflows) and quadruples in three regimes (standard,
unresolvable, transient with n = 1). The faulted step of every training quadruple is `order` or `pay`, never
`receive`.

Control pool (`sel1_ctl_pool`): resolvable tasks only (core classes and standard quadruples), used to give the CTL
arm the same number of rows.

Sealed test (`sel1_test`), six groups of 10 quadruples (40 tasks each):
- `u_in`: unresolvable, trained workflows, faulted step order or pay (the training construction, fresh tasks);
- `u_receive`: unresolvable with the faulted step `receive` (held out from training);
- `u_ood_wf`: unresolvable on TXN's OOD workflows `order_receive_pay_post` and `order_pay_receive` (held out);
- `t2`: transient with n = 2 (held out: training only has n = 1), any non-post faulted step;
- `audit_down`: only the audit log unavailable (held out: never in training);
- `std`: standard quadruples (resolvable reference).
"""
from __future__ import annotations

import copy
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import actionproof.bench.generator_proc as G
from actionproof.bench.generator_proc import DATA, ProcTask
from actionproof.integrity.sim.env import STEP_READ
from actionproof.txn import tasks as T

SEEDS = {"sel1_train": 26112711, "sel1_ctl_pool": 26112713, "sel1_test": 26112712}
TRAIN_CORE_PER_CLASS = 12
TRAIN_QUADS = 20
CTL_CORE_PER_CLASS = 20
CTL_QUADS = 30
TEST_QUADS = 10
TRAIN_STEPS = ("order", "pay")
TEST_GROUPS = ("u_in", "u_receive", "u_ood_wf", "t2", "audit_down", "std")
OOD_WFS = ("order_receive_pay_post", "order_pay_receive")


def _regime(t: ProcTask, regime: str, step: str, n: int = 0) -> None:
    tx = t.meta["txn"]
    if regime == "unresolvable":
        tx.update(regime="unresolvable", unavailable_reads=[STEP_READ[step], "get_audit_log"], escalation_correct=True)
    elif regime == "transient":
        tx.update(regime="transient", unavailable_reads=[STEP_READ[step]], transient_unavailable={STEP_READ[step]: n},
                  escalation_correct=False)
    elif regime == "audit_down":
        tx.update(regime="audit_down", unavailable_reads=["get_audit_log"], escalation_correct=False)
    else:
        assert regime == "standard"


def quad(split: str, i0: int, rng: random.Random, workflow: str, qid: str, group: str,
         regimes: list[tuple[str, int]], steps: tuple[str, ...] | None) -> list[ProcTask]:
    """One world and prompt; for each regime the four TXN realities on one faulted step."""
    base = T._base(split, i0, rng, workflow, group)
    allowed = [s for s in base.meta["requested"] if s != "post" and (steps is None or s in steps)]
    assert allowed, (workflow, steps)
    step = rng.choice(allowed)
    out: list[ProcTask] = []
    for regime, n in regimes:
        for reality, kind in T.REALITIES:
            t = copy.deepcopy(base)
            t.task_id = f"{split}-{i0 + len(out):04d}"
            t.meta["faults"] = [T._fault(rng, kind, step)] if kind else []
            t.meta["txn"].update(quad_id=qid, reality=reality, quad_step=step)
            _regime(t, regime, step, n)
            label = regime if regime != "transient" else f"transient{n}"
            out.append(T._finish(t, f"quad:{reality}"))
            out[-1].meta["txn"]["template_id"] = f"{group}/{workflow}/{reality}/{label}"
    return out


def make_split(split: str) -> list[ProcTask]:
    rng = random.Random(SEEDS[split])
    out: list[ProcTask] = []
    if split in ("sel1_train", "sel1_ctl_pool"):
        per_class, n_quads = ((TRAIN_CORE_PER_CLASS, TRAIN_QUADS) if split == "sel1_train"
                              else (CTL_CORE_PER_CLASS, CTL_QUADS))
        for cls in T.CORE:
            for j in range(per_class):
                out.append(T.core_task(split, len(out), rng, cls, T.TRAINED_WF[j % len(T.TRAINED_WF)]))
        regimes = ([("standard", 0), ("unresolvable", 0), ("transient", 1)] if split == "sel1_train"
                   else [("standard", 0)])
        for q in range(n_quads):
            wf = T.TRAINED_WF[q % len(T.TRAINED_WF)]
            out += quad(split, len(out), rng, wf, f"{split}-q{q:02d}", "quad", regimes, TRAIN_STEPS)
        return out
    assert split == "sel1_test"
    qn = 0
    for g in TEST_GROUPS:
        for q in range(TEST_QUADS):
            qid = f"{split}-{g}-q{q:02d}"
            if g == "u_in":
                spec = (T.TRAINED_WF[q % 3], [("unresolvable", 0)], TRAIN_STEPS)
            elif g == "u_receive":
                spec = ("receive_pay", [("unresolvable", 0)], ("receive",))
            elif g == "u_ood_wf":
                spec = (OOD_WFS[q % 2], [("unresolvable", 0)], None)
            elif g == "t2":
                spec = (T.TRAINED_WF[q % 3], [("transient", 2)], None)
            elif g == "audit_down":
                spec = (T.TRAINED_WF[q % 3], [("audit_down", 0)], None)
            else:
                spec = (T.TRAINED_WF[q % 3], [("standard", 0)], None)
            wf, regs, steps = spec
            out += quad(split, len(out), rng, wf, qid, g, regs, steps)
            qn += 1
    return out


def existing() -> tuple[set[str], set[str], set[str]]:
    """Task ids, PO ids and prompts of every existing split, TXN's included."""
    tids, pos, prompts = set(), set(), set()
    for f in sorted(DATA.glob("*.jsonl")):
        if f.name.startswith("sel1_"):
            continue
        for line in f.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                tids.add(d.get("task_id"))
                prompts.add(d.get("prompt"))
                pos |= {p["po_id"] for p in d.get("initial_state", {}).get("pos", [])}
    return tids, pos, prompts


def check_disjoint(splits: dict[str, list[ProcTask]]) -> None:
    tids, pos, prompts = existing()
    seen: set[str] = set()
    po_owner: dict[str, str] = {}
    prompt_owner: dict[str, str] = {}
    for tasks in splits.values():
        for t in tasks:
            owner = t.meta["txn"].get("quad_id") or t.task_id
            assert t.task_id not in tids and t.task_id not in seen, f"task id {t.task_id} collides"
            seen.add(t.task_id)
            assert t.prompt not in prompts, f"{t.task_id}: prompt text collides with an existing split"
            assert prompt_owner.setdefault(t.prompt, owner) == owner, f"{t.task_id}: prompt shared across quads"
            for p in t.initial_state["pos"]:
                assert p["po_id"] not in pos, f"{t.task_id}: PO id {p['po_id']} collides with an existing split"
                assert po_owner.setdefault(p["po_id"], owner) == owner, f"{t.task_id}: PO id reused"


def load(split: str) -> list[ProcTask]:
    return G.load_tasks(DATA / f"{split}.jsonl")


def write_splits() -> dict[str, Any]:
    splits = {s: make_split(s) for s in SEEDS}
    check_disjoint(splits)
    paths = {s: DATA / f"{s}.jsonl" for s in splits}
    man = DATA / "sel1_manifest.json"
    clash = [str(p) for p in [*paths.values(), man] if p.exists()]
    if clash:
        raise SystemExit(f"refusing to overwrite: {clash}")
    manifest: dict[str, Any] = {"registration": "PREREGISTRATION_SEL1.md", "seeds": SEEDS, "files": {}, "tasks": {}}
    for s, tasks in splits.items():
        text = "".join(T.task_line(t) for t in tasks)
        paths[s].write_text(text)
        manifest["files"][paths[s].name] = {"n": len(tasks), "sha256": hashlib.sha256(text.encode()).hexdigest()}
        for t in tasks:
            tx = t.meta["txn"]
            manifest["tasks"][t.task_id] = {"split": s, "group": tx["group"], "regime": tx["regime"],
                                            "template_id": tx["template_id"], "quad_id": tx.get("quad_id"),
                                            "escalation_correct": tx["escalation_correct"]}
    man.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return manifest["files"]


if __name__ == "__main__":
    print(json.dumps(write_splits(), indent=1))
