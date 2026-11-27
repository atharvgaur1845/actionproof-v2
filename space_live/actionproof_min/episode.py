"""The scripts/p1_eval.py episode loop (functions `start`, `tool_text`, `run`), as a single-episode generator.

Replicated: system prompt = `system_prompt(env, PROC_NEUTRAL)` (rollout/episode.py), tool-call format
`[{"tool_name": ..., "parameters": {...}}]`, tool results in TOOL_RESULT_FORMAT, parse with the lite parser (what
`_agenttune()` returns where AgentTune is not installed), every parsed call in a turn is executed, a turn with no
parseable call ends the episode, `submit_report` ends it, 16 turns, 256 new tokens, greedy, context guard at 8192.
`generate(conv) -> str` is injectable. It may expose `token_len(conv) -> int` for the context guard.
Not replicated: `--recon` (arm B, off in the reported runs) and sampled runs.
"""
from __future__ import annotations

import json
from typing import Any, Callable, Iterator

from actionproof_min import lite_format as L
from actionproof_min.env_s3 import make_env_s3
from actionproof_min.prompts import PROC_NEUTRAL, TOOL_RESULT_FORMAT
from actionproof_min.runtime import episode_record_proc

MAX_TURNS, MAX_NEW_TOKENS, CTX = 16, 256, 8192


def system_prompt(env: Any, base: str) -> str:
    if hasattr(env, "adapt_prompt"):
        base = env.adapt_prompt(base)
    schemas = [L.build_schema(t) for t in env.tools()]
    tools = json.dumps(schemas, indent=2)
    try:
        block = L.FALLBACK_PROMPT.format(tools=tools)
    except (KeyError, ValueError, IndexError):
        block = L.FALLBACK_PROMPT.replace("{tools}", tools)
    return f"{base}\n\n{block}"


def act_text(tool: str, args: dict[str, Any]) -> str:
    return json.dumps([{"tool_name": tool, "parameters": args}])


def tool_text(name: str, obs: dict) -> str:
    return TOOL_RESULT_FORMAT.format(name=name, content=json.dumps(obs))


def start(env: Any, task: Any, system_fn: Callable[[Any, str], str] = system_prompt) -> list[dict]:
    conv = [{"role": "system", "content": system_fn(env, PROC_NEUTRAL)}, {"role": "user", "content": task.prompt}]
    for tool, args in task.meta.get("start_prefix") or []:
        obs = env.call(tool, dict(args))
        conv.append({"role": "assistant", "content": act_text(tool, args)})
        conv.append({"role": "user", "content": tool_text(tool, obs)})
    return conv


def stream_episode(task: Any, generate: Callable[[list[dict]], str], rollout_id: str,
                   max_turns: int = MAX_TURNS, max_tokens: int = MAX_NEW_TOKENS, ctx: int = CTX,
                   system_fn: Callable[[Any, str], str] = system_prompt, gate_rules: Any = None) -> Iterator[dict]:
    """Yield events: start, text (raw model output), call (tool, args, reply), gate_call (a call the Recovery Gate
    made itself; only with `gate_rules`), end (carries the scored record).

    Events are plain dicts (safe to pass out of a ZeroGPU worker). `system_fn` exists so tests can run without
    transformers; the app always uses the default. With `gate_rules`, the Recovery Gate (actionproof_min/gate.py)
    sits in front of the tools, and an episode that ends without a report is finalised by the gate."""
    env = make_env_s3(task, rollout_id)
    gate_events: list[dict] = []
    if gate_rules is not None:
        from actionproof_min.gate_env import gate_env
        gate_env(env, task, gate_rules, gate_events)
    conv = start(env, task, system_fn)
    n0 = len(env.trace)
    token_len = getattr(generate, "token_len", None)
    turns = invalid = 0
    ended_by = "max_turns"
    yield {"type": "start", "system": conv[0]["content"], "prompt": task.prompt}
    for _ in range(max_turns):
        if token_len and token_len(conv) + max_tokens > ctx:
            ended_by = "context"
            break
        text = generate(conv)
        turns += 1
        conv.append({"role": "assistant", "content": text})
        yield {"type": "text", "turn": turns, "raw": text}
        calls = L.extract_tool_calls_from_text(text) or []
        if not calls:
            ended_by = "no_call"
            break
        res = []
        for c in calls:
            name = c["function"]["name"]
            try:
                args = json.loads(c["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                args = None
            obs = env.call(name, args) if isinstance(args, dict) else {"status": "error", "error": "invalid_arguments"}
            invalid += obs.get("error") in ("unknown_tool", "invalid_arguments")
            res.append(tool_text(name, obs))
            yield {"type": "call", "turn": turns, "tool": name, "args": args, "reply": obs}
            while gate_events:
                yield {**gate_events.pop(0), "turn": turns}
        conv.append({"role": "user", "content": "\n\n".join(res)})
        if env.done:
            ended_by = "report"
            break
    if gate_rules is not None:
        env.gate_finalize()
        while gate_events:
            yield {**gate_events.pop(0), "turn": turns}
    rec = episode_record_proc(task, env)
    if gate_rules is not None:
        from actionproof_min.gate_env import gate_record
        rec["gate"] = gate_record(env)
    yield {"type": "end", "ended_by": ended_by, "rec": rec, "turns": turns, "invalid": invalid, "n0": n0}
