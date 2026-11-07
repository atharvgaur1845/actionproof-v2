"""A5 RFT demonstrations: `a5_oracle` (agent-visible tools only) on a stratified FRACTION of `a5_train` tasks per
scenario, replayed through the real `run_episode` loop (byte-identical conversations), then filtered on proof
validity. Held-out combinations are absent from a5_train by construction (generator_a5.HELD_OUT).

    python -m actionproof.train.a5_demos --frac 0.5 --out results/teacher/a5_oracle_f050.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from actionproof.bench.generator_a5 import HELD_OUT, make_a5_split
from actionproof.bench.prompts import NEUTRAL
from actionproof.rewards.sanity_agents_a5 import a5_oracle
from actionproof.rollout.agenttune_rollout import episode_record, make_env
from actionproof.rollout.episode import run_episode

MAX_TURNS = 18


def demo(task) -> dict:
    probe = make_env(task, f"a5-plan-{task.task_id}")
    a5_oracle(probe, task)
    actions = [json.dumps([{"tool_name": s["tool"], "parameters": s["args"]}]) for s in probe.trace]
    assert len(actions) <= MAX_TURNS, (task.task_id, len(actions))
    it = iter(actions)
    env = make_env(task, f"a5-demo-{task.task_id}")
    conv, st = run_episode(env, lambda conv: next(it), NEUTRAL, task.prompt, max_turns=MAX_TURNS)
    if [s["obs"] for s in env.trace] != [s["obs"] for s in probe.trace]:
        raise AssertionError(f"{task.task_id}: replay diverged from the planning run")
    return {**episode_record(task, env), "conversation": conv, "ended_by": st.ended_by,
            "workflow": task.workflow, "sample": 0, "source": "a5_oracle"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    by = defaultdict(list)
    for t in make_a5_split("a5_train"):
        assert t.meta["a5"]["scenario"] not in HELD_OUT
        by[t.meta["a5"]["scenario"]].append(t)
    rng = random.Random(a.seed)
    tasks = [t for sc in sorted(by) for t in rng.sample(by[sc], round(a.frac * len(by[sc])))]
    recs = [demo(t) for t in tasks]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(json.dumps({"n": len(recs), "proof_valid": sum(r["vsc"] for r in recs) / len(recs),
                      "by_scenario": {sc: sum(r["a5"]["scenario"] == sc for r in recs) for sc in sorted(by)}}))


if __name__ == "__main__":
    main()
