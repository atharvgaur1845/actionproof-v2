"""Build p1_eval-format episode records from scripted assistant turns (no model needed)."""
from __future__ import annotations

from typing import Any

from actionproof.bench.generator_proc import DATA, load_tasks
from actionproof.selfheal.envs import spec_for
from actionproof.selfheal.replay import act_text, execute_turn


def tasks(split: str) -> list[Any]:
    return load_tasks(DATA / f"{split}.jsonl")


def find(split: str, pred) -> Any:
    return next(t for t in tasks(split) if pred(t))


def system_text(env: Any) -> str:
    """The evaluation system prompt where transformers is installed (cluster); a stand-in otherwise. Replay copies
    the recorded start messages verbatim, so the text does not affect any test."""
    try:
        from actionproof.integrity.sim.prompts import PROC_NEUTRAL
        from actionproof.rollout.episode import system_prompt
        return system_prompt(env, PROC_NEUTRAL)
    except ImportError:
        return "SYSTEM PROMPT (stand-in)"


def simulate(task: Any, calls: list[tuple[str, dict[str, Any]] | str], tag: str = "fake",
             sample: int = -1) -> dict[str, Any]:
    """Run turns (a (tool, args) pair or raw text) the way scripts/p1_eval.py does and return its record."""
    from actionproof.selfheal.replay import tool_text
    spec = spec_for(task)
    env = spec.make(task, f"p1-{task.task_id}")
    conv = [{"role": "system", "content": system_text(env)}, {"role": "user", "content": task.prompt}]
    for tool, args in task.meta.get("start_prefix") or []:
        obs = env.call(tool, dict(args))
        conv += [{"role": "assistant", "content": act_text(tool, args)}, {"role": "user", "content": tool_text(tool, obs)}]
    ended, turns, invalid = "max_turns", 0, 0
    for c in calls:
        text = c if isinstance(c, str) else act_text(*c)
        turns += 1
        conv.append({"role": "assistant", "content": text})
        res, inv, had = execute_turn(env, text)
        invalid += inv
        if not had:
            ended = "no_call"
            break
        conv.append({"role": "user", "content": "\n\n".join(res)})
        if env.done:
            ended = "report"
            break
    rec = spec.record(task, env)
    rec.update(sample=sample, turns=turns, invalid=invalid, ended_by=ended, conversation=conv, tag=tag,
               workflow=task.workflow)
    return rec
