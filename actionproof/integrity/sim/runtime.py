"""ProcureSim env factory and episode record, shared by GRPO rollouts and evaluation so both score identically."""
from __future__ import annotations

import copy
from typing import Any

from actionproof.integrity.sim.env import READ_TOOLS, ProcEnv
from actionproof.integrity.sim.truth import score


def make_env_proc(task: Any, rollout_id: str, seed: int | None = None) -> ProcEnv:
    m = task.meta
    return ProcEnv(task.initial_state, m.get("faults"), rollout_id=rollout_id,
                   seed=m["env_seed"] if seed is None else seed, never_commit=m.get("never_commit"))


def episode_record_proc(task: Any, env: ProcEnv) -> dict[str, Any]:
    rec = {"task": task.to_json(), "trace": copy.deepcopy(env.trace), "attempts": copy.deepcopy(env.attempts),
           "report": copy.deepcopy(env.final_report), "final_state": copy.deepcopy(env.state)}
    s = score(rec)
    agent = [x for x in env.trace if x.get("by", "agent") == "agent"]
    return {**rec, "score": s, "rollout_id": env.rollout_id, "task_id": task.task_id, "task_prompt": task.prompt,
            "vsc": s["vsc"], "fault_class": task.meta["fault_class"],
            "n_verify": sum(x["tool"] in READ_TOOLS for x in agent), "n_model_calls": len(agent),
            "receipt_rollout_ids": [env.rollout_id] if any(a["committed"] != "none" for a in env.attempts) else []}
