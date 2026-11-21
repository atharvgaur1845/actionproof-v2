"""Recovery-start task pools (candidate F, docs/stage3/RESEARCH_REPORT.md §4.F).

A recovery-start task is an ordinary ProcureSim task plus `meta["start_prefix"]`: scripted calls (the oracle's own
argument and key conventions) that the rollout replays before the policy acts. The fault sits on attempt 0 of the
start step's write, so the replayed state is the real faulted state. Start types:

  ack      prefix ends at the start step's acknowledgement: the policy must decide whether to check
  read     prefix also contains the authoritative read: the policy must act on the evidence
  done     every requested step written and read back: the policy must recognise that nothing is left (stop)

Faults at the start step: phantom, partial (ratio 1/4, 1/2 or 3/4), lost_ack, dup_delivery, none (complete), and
never_commit (every attempt of that write acknowledges ok and commits nothing). Prompts are the generator's and never
depend on the start type or fault. Base tasks are fresh per pool (their own seed), so prompts stay unique.
"""
from __future__ import annotations

import copy
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from actionproof.bench.generator_proc import DATA, ProcTask, load_tasks, make_task
from actionproof.integrity.sim.agents import _args, _po
from actionproof.integrity.sim.env import STEP_READ, STEP_TOOL

START_MIX = (("phantom", 0.15), ("partial", 0.25), ("lost_ack", 0.10), ("dup_delivery", 0.15),
             ("complete", 0.15), ("never_commit", 0.10), ("done", 0.10))
RATIOS = (0.25, 0.5, 0.75)


def prefix(task: ProcTask, step: str | None, with_read: bool) -> list[list[Any]]:
    """Oracle calls for the steps before `step` (write + read), then `step`'s write (+ read). `step=None`: all."""
    po = _po(task)
    acts: list[list[Any]] = []
    for s in task.meta["requested"]:
        if s == step:
            acts.append([STEP_TOOL[s], _args(s, po, f"{s}-{po['po_id']}")])
            if with_read:
                acts.append([STEP_READ[s], {"po_id": po["po_id"]}])
            return acts
        acts += [[STEP_TOOL[s], _args(s, po, f"{s}-{po['po_id']}")], [STEP_READ[s], {"po_id": po["po_id"]}]]
    return acts


def _pick(rng: random.Random) -> str:
    x, acc = rng.random(), 0.0
    for k, p in START_MIX:
        acc += p
        if x < acc:
            return k
    return START_MIX[-1][0]


def recovery_task(base: ProcTask, rng: random.Random) -> ProcTask:
    t = copy.deepcopy(base)
    kind = _pick(rng)
    req = t.meta["requested"]
    with_read = rng.random() < 0.5
    t.meta.update(faults=[], never_commit=None)
    if kind == "done":
        t.meta.update(start_prefix=prefix(t, None, True), start_type="done", fault_class="s3:done")
        return t
    # partial: repairable on pay (refund, re-pay) and receive (record the rest); a duplicate only on pay (refund);
    # ProcureSim has no tool that reverses a goods receipt, so a duplicate receipt would be an unwinnable start
    choices = ([s for s in req if s in ("pay", "receive")] if kind == "partial" else ["pay"] if kind == "dup_delivery"
               else [s for s in req if s != "post"])
    step = rng.choice(choices)
    tool = STEP_TOOL[step]
    if kind == "never_commit":
        t.meta["never_commit"] = tool
    elif kind != "complete":
        f: dict[str, Any] = {"tool": tool, "k": 0, "kind": kind}
        if kind == "partial":
            f["ratio"] = rng.choice(RATIOS)
        t.meta["faults"] = [f]
    stype = "read" if with_read else "ack"
    t.meta.update(start_prefix=prefix(t, step, with_read), start_type=stype, start_step=step,
                  fault_class=f"s3:{kind}:{stype}")
    return t


def make_pool(split: str, n: int, seed: int) -> list[ProcTask]:
    rng = random.Random(seed)
    return [recovery_task(make_task(split, i, rng), rng) for i in range(n)]


def seen_ids() -> tuple[set[str], set[str]]:
    tids, pos = set(), set()
    for f in DATA.glob("*.jsonl"):
        for t in load_tasks(f):
            tids.add(t.task_id)
            pos |= {p["po_id"] for p in t.initial_state["pos"]}
    return tids, pos


def write(name: str, tasks: list[ProcTask]) -> Path:
    tids, pos = seen_ids()
    mine = [p["po_id"] for t in tasks for p in t.initial_state["pos"]]
    assert len(mine) == len(set(mine)) and not set(mine) & pos, f"{name}: PO ids collide"
    assert not {t.task_id for t in tasks} & tids, f"{name}: task ids collide"
    out = DATA / f"{name}.jsonl"
    out.write_text("".join(json.dumps(t.to_json(), sort_keys=True) + "\n" for t in tasks))
    print(name, len(tasks), dict(Counter(t.meta["fault_class"] for t in tasks)))
    return out
