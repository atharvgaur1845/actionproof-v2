"""Scripted-expert demonstrations for RFT v2 (deviation 2026-11-04, see SPEC §27).

Finding that motivates it: the Aya 32B teacher, explicitly told to confirm every write,
never re-reads state after a SUCCESS ack (0 post-write checks in 560 phantom-ack episodes),
so the verifier-curated teacher data contains no F1/F3 recoveries and GRPO groups on those
faults are all-zero (no gradient). `lean_expert` uses only agent-visible tools; its
actions are replayed through the real `run_episode` loop on a fresh env with the same
seed, so the conversation is byte-identical to what a policy would produce, then the
records go through the same VSC filter as teacher data.

    python -m actionproof.train.expert_demos --split train --frac 0.5 --out results/teacher/expert_refund.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from actionproof.bench.generator import make_split
from actionproof.bench.prompts import NEUTRAL
from actionproof.rewards.sanity_agents import lean_expert
from actionproof.rollout.agenttune_rollout import episode_record, make_env
from actionproof.rollout.episode import run_episode


def demo(task, seed: int = 0, max_turns: int = 10) -> dict:
    ids = task.meta["ids"]
    probe = make_env(task, f"expert-plan-{task.task_id}", seed=seed)
    lean_expert(probe, ids["order"], ids["ticket"])
    actions = [json.dumps([{"tool_name": s["tool"], "parameters": s["args"]}]) for s in probe.trace]
    if len(actions) > max_turns:
        raise ValueError(f"{task.task_id}: expert needs {len(actions)} turns > {max_turns}")
    it = iter(actions)
    env = make_env(task, f"expert-{task.task_id}", seed=seed)
    conv, st = run_episode(env, lambda conv: next(it), NEUTRAL, task.prompt, max_turns=max_turns)
    if [s["obs"] for s in env.trace] != [s["obs"] for s in probe.trace]:
        raise AssertionError(f"{task.task_id}: replay diverged from the planning run")
    return {**episode_record(task, env), "conversation": conv, "ended_by": st.ended_by,
            "workflow": task.workflow, "sample": 0, "source": "lean_expert"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--frac", type=float, default=0.5,
                    help="share of refund tasks that get a demo: <1 keeps post-write verification "
                         "stochastic in the student, i.e. inside GRPO's learnable band")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tasks = [t for t in make_split(a.split) if t.workflow == "refund"]
    tasks = random.Random(a.seed).sample(tasks, round(a.frac * len(tasks)))
    recs = [demo(t) for t in tasks]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    by = {}
    for r in recs:
        k = ",".join(r["faults"]) or "benign"
        n, s = by.get(k, (0, 0))
        by[k] = (n + 1, s + r["vsc"])
    print(json.dumps({"n": len(recs), "vsc": sum(r["vsc"] for r in recs) / max(len(recs), 1),
                      "by_fault": {k: f"{s}/{n}" for k, (n, s) in by.items()}}))


if __name__ == "__main__":
    main()
