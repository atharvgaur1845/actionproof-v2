"""A5 env factory and episode record (shared by GRPO rollouts and evaluation, so both score identically)."""
from __future__ import annotations

import copy
from dataclasses import asdict
from typing import Any

from actionproof.env.tools_a5 import A5_EVIDENCE_TOOLS, EcomEnvA5


def make_env_a5(task, rollout_id: str, seed: int = 0, cls: type = EcomEnvA5) -> EcomEnvA5:
    return cls(task.initial_state, task.schedule, rollout_id=rollout_id, seed=seed,
               stale_status=task.stale_status, ambiguous_text=task.ambiguous_text, a5=copy.deepcopy(task.meta["a5"]))


def episode_record_a5(task, env: EcomEnvA5) -> dict[str, Any]:
    from actionproof.env.verifier_a5 import verify_a5
    trace = copy.deepcopy(env.trace)
    report = copy.deepcopy(env.final_report)
    v = verify_a5(task, env)  # settles the hidden horizon
    model_steps = [s for s in trace if not s.get("by")]
    resolvable = task.meta["a5"]["resolvable"]
    return {
        "rollout_id": env.log.rollout_id, "task_id": task.task_id, "task_prompt": task.prompt, "vsc": v["proof_valid"],
        # verdict-shaped fields so generic summaries (run_eval.summarize) still work; A5 metrics live in "a5v"
        "verdict": {"G": v["goal_final"], "P": float(v["goal_final"]), "V": 1.0,
                    "I": float(any(v["violations"].values())), "D": float(v["duplicate"]),
                    "F_over": v["false_completion"], "F_under": v["underclaim"], "reported": v["status"],
                    "expected": "completed" if resolvable else "uncertain", "feasible": True},
        "violations": v["violations"], "a5v": v, "a5": task.meta["a5"], "goal": task.goal.to_json(),
        "trace": trace, "report": report,
        "n_verify": sum(s["tool"] in A5_EVIDENCE_TOOLS for s in model_steps),
        "n_harness": sum(bool(s.get("by")) for s in trace), "n_model_calls": len(model_steps),
        "faults": task.meta.get("faults", []), "receipt_rollout_ids": sorted({r.rollout_id for r in env.log.receipts}),
        "ops": [{k: o[k] for k in ("op_id", "tool", "entity", "actor", "status", "accepted_at", "committed_at")}
                for o in env.ops.values()],
        "attempts": [{"action_idx": at.action_idx, "op": at.op, "entity": at.entity, "fault": at.fault,
                      "committed": at.committed, "obs_status": at.observation.get("status")}
                     for at in env.log.attempts],
    }
