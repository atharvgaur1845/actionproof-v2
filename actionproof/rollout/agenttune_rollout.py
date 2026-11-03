"""Per-rollout environments for AgentTune GRPO (spec §13.3).

AgentTune's built-in tool loop shares one set of tool callables across all rollouts, which
would let GRPO's group samples see each other's side effects. This `rollout_func` instead
builds a fresh `EcomEnv` (own SQLite DB, own receipt log, harness-generated UUID) for every
sampled completion, runs AgentTune's own tool loop on it, and attaches the verifier output as
the extra field `ap_episodes`. TRL forwards extra fields to reward functions per sample.
"""
from __future__ import annotations

import uuid
from dataclasses import asdict
from typing import Any, Callable

from actionproof.bench.generator import Task
from actionproof.bench.prompts import TOOL_RESULT_FORMAT
from actionproof.env.db import snapshot
from actionproof.env.tools_ecom import VERIFICATION_TOOLS, EcomEnv
from actionproof.env.verifier import verify, vsc

LIST_KEYS = ("prompt_ids", "completion_ids", "logprobs", "env_mask", "responses", "rewards",
             "trajectories", "conversations", "tool_call_counts", "queries")


def prompt_key(prompt: Any) -> str:
    """The task's user message; unique per task because it embeds the entity ids."""
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, dict):
        prompt = prompt.get("prompt", prompt)
    users = [m["content"] for m in prompt if isinstance(m, dict) and m.get("role") == "user"]
    return users[-1] if users else str(prompt)


def make_env(task: Task, rollout_id: str, seed: int = 0) -> EcomEnv:
    return EcomEnv(task.initial_state, task.schedule, rollout_id=rollout_id, seed=seed,
                   stale_status=task.stale_status, ambiguous_text=task.ambiguous_text)


def episode_record(task: Task, env: EcomEnv) -> dict[str, Any]:
    v = verify(task.goal, snapshot(env.conn), env.log, env.final_report)
    return {
        "rollout_id": env.log.rollout_id, "task_id": task.task_id, "vsc": vsc(v),
        "verdict": {k: val for k, val in asdict(v).items() if k != "details"},
        "violations": v.details["violations"], "trace": env.trace, "report": env.final_report,
        "n_verify": sum(s["tool"] in VERIFICATION_TOOLS for s in env.trace),
        "receipt_rollout_ids": sorted({r.rollout_id for r in env.log.receipts}),
        "faults": task.meta.get("faults", []),
        # Per-write ground truth for probes (spec §14): did the acknowledged effect land?
        "attempts": [{"action_idx": at.action_idx, "op": at.op, "entity": at.entity, "fault": at.fault,
                      "committed": at.committed, "obs_status": at.observation.get("status")}
                     for at in env.log.attempts],
    }


def make_rollout_func(tasks: list[Task], max_steps: int = 10, **rollout_kwargs: Any) -> Callable:
    from agenttune.agentic.rollout_engines.rollout_factory import (
        DEFAULT_TOOLS_FALLBACK_PROMPT, create_rollout_fn)

    by_prompt = {t.prompt: t for t in tasks}
    if len(by_prompt) != len(tasks):
        raise ValueError("task prompts must be unique (they key the env lookup)")

    def rollout_func(prompts: list[Any], trainer: Any = None, **gen_kwargs: Any) -> dict[str, Any]:
        merged: dict[str, Any] = {k: [] for k in LIST_KEYS}
        merged["ap_episodes"] = []
        for p in prompts:
            task = by_prompt[prompt_key(p)]
            rid = f"ro-{uuid.uuid4().hex[:12]}"
            env = make_env(task, rid, seed=uuid.UUID(rid[3:].ljust(32, "0")).int % 2**31)
            fn = create_rollout_fn(tools=env.tools(), max_steps=max_steps,
                                   tools_fallback_prompt=DEFAULT_TOOLS_FALLBACK_PROMPT,
                                   tool_result_format=TOOL_RESULT_FORMAT, **rollout_kwargs)
            out = fn([p], trainer, **gen_kwargs)
            for k in LIST_KEYS:
                if k in out:
                    merged[k].extend(out[k] if isinstance(out[k], list) else [out[k]])
            merged["ap_episodes"].append(episode_record(task, env))
        merged["tools_used"] = True
        n = len(prompts)
        ids = [e["rollout_id"] for e in merged["ap_episodes"]]
        assert len(set(ids)) == n, "duplicate rollout ids"
        assert all(e["receipt_rollout_ids"] in ([], [e["rollout_id"]]) for e in merged["ap_episodes"]), \
            "receipt written by a different rollout"
        assert len(merged["completion_ids"]) == n, (len(merged["completion_ids"]), n)
        return {k: v for k, v in merged.items() if not (isinstance(v, list) and not v and k != "ap_episodes")}

    return rollout_func
