"""A15: amount-ratio forced-evidence forks (PREREGISTRATION_A15). Extends A14's design (`scripts/cef_eval.py`,
frozen, imported unchanged) from one partial ratio (1/2, the only one ProcureSim produces) to q = 0, 1/4, 1/2, 3/4, 1.
The partial ratio is applied by `RatioProcEnv`, a subclass that rescales the committed payment; at q = 1/2 it equals
ProcEnv exactly (tested). Branches share the prefix and the acknowledgement; only the stored payment and the forced
read differ.

    python scripts/a15_eval.py make-split                       # proc_mech (200, seed 111), proc_mech_dev (60, seed 112)
    python scripts/a15_eval.py scripted --split proc_mech_dev --n-tasks 20
    python scripts/a15_eval.py model --model runs/rft_proc/merged --adapter runs/a12_proc_rv_s0/final --tag rv_s0
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

from actionproof.bench.generator_proc import ProcTask, load_tasks, make_task
from actionproof.integrity.sim.env import STEP_TOOL, ProcEnv

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("cef_eval", ROOT / "scripts/cef_eval.py")
cef = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cef)

SPLITS = {"proc_mech": (200, 111), "proc_mech_dev": (60, 112)}
QUARTERS = {"Q0": 0, "Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}   # q = quarters / 4 of the purchase-order amount stored
BRANCHES = {"Q0": "phantom", "Q1": "partial", "Q2": "partial", "Q3": "partial", "Q4": None}
COMMITTED = {"Q0": "none", "Q1": "partial", "Q2": "partial", "Q3": "partial", "Q4": "full"}
CORRECT = {"Q0": {"repay_same_key", "repay_new_key"}, "Q1": {"refund", "escalate"}, "Q2": {"refund", "escalate"},
           "Q3": {"refund", "escalate"}, "Q4": {"next_step"}}
UNSEEN, TRAINED, BINARY = ("Q1", "Q3"), ("Q2",), ("Q0", "Q4")


class RatioProcEnv(ProcEnv):
    """ProcEnv whose partial pay commits `quarters`/4 of the requested amount instead of half."""

    def __init__(self, *args: Any, quarters: int = 2, **kw: Any) -> None:
        super().__init__(*args, **kw)
        self.quarters = quarters

    def _write(self, tool, po_id, key, prefix, rows, id_field):  # type: ignore[override]
        if tool != STEP_TOOL["pay"]:
            return super()._write(tool, po_id, key, prefix, rows, id_field)

        def scaled(i: str, part: bool) -> list[tuple[str, dict[str, Any]]]:
            if not part:
                return rows(i, False)
            return [(t, {**r, "amount_cents": r["amount_cents"] * self.quarters // 4}) for t, r in rows(i, False)]
        return super()._write(tool, po_id, key, prefix, scaled, id_field)


def make_env_ratio(task: Any, rollout_id: str, seed: int | None = None) -> RatioProcEnv:
    m = task.meta
    return RatioProcEnv(task.initial_state, m.get("faults"), rollout_id=rollout_id,
                        seed=m["env_seed"] if seed is None else seed, never_commit=m.get("never_commit"),
                        quarters=m.get("partial_quarters", 2))


def branch_task(task: ProcTask, branch: str) -> ProcTask:
    t = copy.deepcopy(task)
    kind = BRANCHES[branch]
    t.meta["faults"] = [] if kind is None else [{"tool": STEP_TOOL["pay"], "k": 0, "kind": kind}]
    t.meta["never_commit"] = None
    t.meta["partial_quarters"] = QUARTERS[branch]
    t.meta["fault_class"] = f"mech:{branch}"
    return t


# A14's harness, pointed at the ratio environment and the five ratio branches (its code is not modified).
cef.BRANCHES, cef.COMMITTED, cef.CORRECT = BRANCHES, COMMITTED, CORRECT
cef.branch_task, cef.make_env_proc = branch_task, make_env_ratio


def aligned(amount: int) -> bool:
    """Paid amounts at q = 1/4 .. 1 have the same number of digits (needed for token-aligned A16 pairs)."""
    return len({len(str(amount * q // 4)) for q in (1, 2, 3, 4)}) == 1


def make_split(name: str) -> list[ProcTask]:
    n, seed = SPLITS[name]
    rng = random.Random(seed)
    tasks, i = [], 0
    while len(tasks) < n:
        t = make_task(name, i, rng)
        i += 1
        po = next(p for p in t.initial_state["pos"] if p["po_id"] == t.meta["po_id"])
        if "pay" not in t.meta["requested"] or t.meta["requested"][-1] != "post" or not aligned(po["amount_cents"]):
            continue
        t.meta = {**t.meta, "faults": [], "never_commit": None, "fault_class": "mech"}
        tasks.append(t)
    return tasks


_a14_record = cef.record


def record(task: ProcTask, branch: str, rep: str, env: Any, n_prefix: int, conv: list, extra: dict) -> dict:
    r = _a14_record(task, branch, rep, env, n_prefix, conv, extra)
    pay_id = next(x["obs"].get("payment_id") for x in env.trace[:n_prefix] if x["tool"] == STEP_TOOL["pay"])
    first = r["first"]
    strict = first["action"] == "escalate" or (first["action"] == "refund" and
                                               (first.get("args") or {}).get("payment_id") == pay_id)
    if branch in ("Q0", "Q4"):
        strict = r["correct"]
    return {**r, "correct_a14_rule": r["correct"], "correct": bool(strict), "q": QUARTERS[branch] / 4}


cef.record = record  # run_scripted and run_model call the module-level record


def summary(recs: list[dict]) -> dict:
    def rate(bs: tuple[str, ...], key: str = "correct") -> float | None:
        per = [[r[key] for r in recs if r["branch"] == b] for b in bs]
        return None if not all(per) else sum(sum(p) / len(p) for p in per) / len(per)
    return {"n_tasks": len({r["task_id"] for r in recs}), "unseen_partial": rate(UNSEEN), "trained_partial": rate(TRAINED),
            "binary_bca": rate(BINARY), "unseen_partial_a14_rule": rate(UNSEEN, "correct_a14_rule"),
            "by_branch": {b: {"correct": rate((b,)), "correct_a14_rule": rate((b,), "correct_a14_rule"),
                              "goal_met": rate((b,), "goal_met"), "duplicate": rate((b,), "duplicate"),
                              "report_accurate": rate((b,), "report_accurate"),
                              "first_actions": dict(Counter(r["first"]["action"] for r in recs if r["branch"] == b))}
                          for b in BRANCHES}}


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("make-split")
    for c in ("scripted", "model"):
        p = sub.add_parser(c)
        p.add_argument("--split", default="proc_mech")
        p.add_argument("--n-tasks", type=int, default=None)
        p.add_argument("--out-dir", default="results/eval/a15")
        p.add_argument("--tag", default="scripted")
        if c == "model":
            p.add_argument("--model", required=True)
            p.add_argument("--adapter", default=None)
            p.add_argument("--max-turns", type=int, default=16)
            p.add_argument("--max-tokens", type=int, default=256)
            p.add_argument("--gpu-mem", type=float, default=0.85)
    a = ap.parse_args()
    if a.cmd == "make-split":
        seen_tasks, seen_pos = set(), set()
        for f in (ROOT / "data/bench/proc").glob("*.jsonl"):
            if f.stem in SPLITS:
                raise SystemExit(f"{f} exists")
            for t in load_tasks(f):
                seen_tasks.add(t.task_id)
                seen_pos |= {p["po_id"] for p in t.initial_state["pos"]}
        for name in SPLITS:
            tasks = make_split(name)
            ids, pos = {t.task_id for t in tasks}, {p["po_id"] for t in tasks for p in t.initial_state["pos"]}
            assert not ids & seen_tasks and not pos & seen_pos, name
            seen_tasks |= ids
            seen_pos |= pos
            (ROOT / f"data/bench/proc/{name}.jsonl").write_text("".join(json.dumps(t.to_json()) + "\n" for t in tasks))
            print(f"wrote {len(tasks)} tasks to data/bench/proc/{name}.jsonl")
        return
    tasks = load_tasks(ROOT / f"data/bench/proc/{a.split}.jsonl")[: a.n_tasks]
    out_dir = ROOT / a.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if a.cmd == "scripted":
        recs = cef.run_scripted(tasks, "agent")
        groups = {p: [r for r in recs if r["policy"] == p] for p in ("oracle", "naive", "fixed_retry")}
    else:
        from vllm import LLM
        from actionproof.train.merge import merge
        llm = LLM(merge(a.adapter, a.model) if a.adapter else a.model, gpu_memory_utilization=a.gpu_mem,
                  max_model_len=8192, dtype="bfloat16", seed=0)
        recs = cef.run_model(llm, tasks, "agent", list(BRANCHES), a.max_turns, a.max_tokens)
        groups = {a.tag: recs}
    res = {}
    for name, rs in groups.items():
        with (out_dir / f"{name}__{a.split}.jsonl").open("w") as f:
            for r in rs:
                f.write(json.dumps({**r, "policy": name}) + "\n")
        res[name] = summary(rs)
    res["_meta"] = {"split": a.split, "seconds": round(time.time() - t0), "n_episodes": len(recs)}
    (out_dir / f"summary__{a.tag}__{a.split}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: {kk: v[kk] for kk in ("unseen_partial", "trained_partial", "binary_bca")} if k != "_meta" else v
                      for k, v in res.items()}, indent=1))


if __name__ == "__main__":
    main()
