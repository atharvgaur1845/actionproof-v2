"""Amendment 4 splits (PREREGISTRATION_A4 §3.2, §4.2, §4.4). Builds on Domain A's world generator, unchanged.

Training pools `a4_train_A` / `a4_train_B` share ONE random stream per world: the world, F5 status, environment
class (OK / RP / UR), fault type and persistence are drawn identically. The regime changes only which tool the
single fault event lands on (A: `issue_refund`; B: uniform over the seen write tools, from a separate stream).

A fault event (tool, type, k_max) fails calls k = 0..k_max of that tool (RP, k_max in {0,1,2}); a UR event fails
every call. Types: F1 (phantom) and F4 (timeout, commits). F3 is excluded so the type mix is identical across
regimes (only composite tools support it).

    python -m actionproof.bench.generator_a4 --out data/bench/a4
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from actionproof.bench.generator import F5_SHARE, PROMPTS, Task, _goal, _world
from actionproof.env.faults import Fault, FaultSchedule
from actionproof.env.tools_a4 import HELD_OUT_TOOL
from actionproof.env.verifier import GoalSpec

SEEN_WRITES = ("issue_refund", "add_audit_note", "close_ticket")  # refund-workflow writes
MAX_TURNS = 18
PERSISTENT_K = range(MAX_TURNS)  # UR: every call of the tool fails
CREDIT_PROMPT = ("Ticket {ticket}: customer {cust} returned order {order}. Process the refund to the original payment "
                 "method, apply a goodwill store credit of {credit} cents to the order, add an audit note describing "
                 "what you did, and close the ticket.")

TRAIN_POOLS = {"a4_train_A": "A", "a4_train_B": "B"}
TRAIN_N, TRAIN_SEED = 400, 21
EVAL_SPLITS: dict[str, dict[str, Any]] = {  # name -> seed, composition
    "e_seen_refund":   {"seed": 22, "n_rp": 150, "n_ok": 50, "workflow": "refund", "tools": ("issue_refund",)},
    "e_seen_other":    {"seed": 23, "n_rp": 150, "n_ok": 50, "workflow": "refund", "tools": ("add_audit_note", "close_ticket")},
    "e_heldout":       {"seed": 24, "n_rp": 150, "n_ok": 50, "workflow": "credit", "tools": (HELD_OUT_TOOL,)},
    "e_heldout_ctrl":  {"seed": 25, "n_rp": 150, "n_ok": 50, "workflow": "credit", "tools": SEEN_WRITES},
    "e_unrecoverable": {"seed": 26, "n_ur": 150},
}
ALL_A4 = (*TRAIN_POOLS, *EVAL_SPLITS)


def _schedule(tool: str, ftype: Fault, k_max: int | None) -> FaultSchedule:
    ks = PERSISTENT_K if k_max is None else range(k_max + 1)
    return FaultSchedule({(tool, k): ftype for k in ks})


def _goal_credit(task_id: str, ids: dict[str, Any], credit: int) -> GoalSpec:
    g = _goal(task_id, "refund", ids, feasible=True)
    post = (*g.postconditions[:2], {"p": "credit", "order_id": ids["order"], "amount_cents": credit},
            *g.postconditions[2:])
    writes = (g.required_writes[0], (HELD_OUT_TOOL, ids["order"]), *g.required_writes[1:])
    return GoalSpec(task_id, True, post, writes, g.allowed_entities)


def _task(split: str, i: int, rng: random.Random, workflow: str, f5: bool, env_class: str,
          tool: str | None, ftype: Fault | None, k_max: int | None, regime: str | None, eval_env: bool) -> Task:
    state, ids = _world(rng, "refund", f5)
    task_id = f"{split}_{i:04d}"
    credit = rng.randrange(500, 5_000, 100)  # drawn for every world so streams stay aligned across workflows
    if workflow == "credit":
        goal = _goal_credit(task_id, ids, credit)
        prompt = CREDIT_PROMPT.format(ticket=ids["ticket"], cust=ids["cust"], order=ids["order"], credit=credit)
    else:
        goal = _goal(task_id, "refund", ids, feasible=not f5)
        prompt = PROMPTS["refund"].format(ticket=ids["ticket"], cust=ids["cust"], order=ids["order"])
    sched = _schedule(tool, ftype, k_max) if tool else FaultSchedule()
    faults = ["F5"] if f5 else ([ftype.value] if ftype else [])
    meta = {"faults": faults, "services": [], "ids": {**ids, "credit": credit}, "benign": not faults,
            "language": "en",
            "a4": {"env_class": env_class, "fault_tool": tool, "fault_type": ftype.value if ftype else None,
                   "k_max": k_max, "regime": regime, "workflow": workflow, "max_turns": MAX_TURNS}}
    if eval_env:
        meta["domain"] = "a4"  # -> EcomEnvA4 + verify_a4 (held-out tool listed)
    return Task(task_id, split, "refund", prompt, state, sched, goal, meta=meta)


def make_train_pool(split: str) -> list[Task]:
    regime = TRAIN_POOLS[split]
    out = []
    for i in range(TRAIN_N):
        rng = random.Random(f"actionproof-a4/train/{TRAIN_SEED}/{i}")      # identical for A and B
        tool_rng = random.Random(f"actionproof-a4/train-tool/{TRAIN_SEED}/{i}")  # used only by regime B
        f5 = rng.random() < F5_SHARE
        u, ftype, k_max = rng.random(), rng.choice((Fault.PHANTOM, Fault.TIMEOUT)), rng.randrange(3)
        b_tool = tool_rng.choice(SEEN_WRITES)
        tool = "issue_refund" if regime == "A" else b_tool
        if f5:
            out.append(_task(split, i, rng, "refund", True, "F5", None, None, None, regime, False))
        elif u < 0.20:
            out.append(_task(split, i, rng, "refund", False, "OK", None, None, None, regime, False))
        elif u < 0.85:
            out.append(_task(split, i, rng, "refund", False, "RP", tool, ftype, k_max, regime, False))
        else:
            out.append(_task(split, i, rng, "refund", False, "UR", tool, Fault.PHANTOM, None, regime, False))
    return out


def make_eval_split(split: str) -> list[Task]:
    c = EVAL_SPLITS[split]
    out = []
    if split == "e_unrecoverable":
        for i in range(c["n_ur"]):
            rng = random.Random(f"actionproof-a4/{split}/{c['seed']}/{i}")
            group = i // 50
            tool = ("issue_refund", rng.choice(("add_audit_note", "close_ticket")), HELD_OUT_TOOL)[group]
            wf = "credit" if group == 2 else "refund"
            out.append(_task(split, i, rng, wf, False, "UR", tool, Fault.PHANTOM, None, None, True))
        return out
    for i in range(c["n_rp"] + c["n_ok"]):
        rng = random.Random(f"actionproof-a4/{split}/{c['seed']}/{i}")
        ftype, k_max, tool = rng.choice((Fault.PHANTOM, Fault.TIMEOUT)), rng.randrange(3), rng.choice(c["tools"])
        if i < c["n_rp"]:
            out.append(_task(split, i, rng, c["workflow"], False, "RP", tool, ftype, k_max, None, True))
        else:
            out.append(_task(split, i, rng, c["workflow"], False, "OK", None, None, None, None, True))
    return out


def make_a4_split(split: str) -> list[Task]:
    return make_train_pool(split) if split in TRAIN_POOLS else make_eval_split(split)


def split_bytes(split: str) -> bytes:
    return "".join(json.dumps(t.to_json(), sort_keys=True) + "\n" for t in make_a4_split(split)).encode()


def load_tasks(path: str) -> list[Task]:
    return [Task.from_json(json.loads(l)) for l in open(path)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/bench/a4")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for s in ALL_A4:
        (out / f"{s}.jsonl").write_bytes(split_bytes(s))
        print(s, len(make_a4_split(s)))


if __name__ == "__main__":
    main()
