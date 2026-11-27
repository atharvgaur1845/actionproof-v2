"""The Recovery Gate in front of the Space's proc2 env (actionproof_min.env_s3): the instance-level form of
actionproof/recovery_gate/env.py. The gate gets a callback, never the env. This simulator's reads have no
`consistent` flag (it has no replica copy), so the freshness rule is off here (SPACE_RULES); every other rule is the
one RG1 registered."""
from __future__ import annotations

import copy
from typing import Any

from actionproof_min.gate import Gate, Rules, intent_of

SPACE_RULES = Rules(fresh=False)


def gate_env(env: Any, task: Any, rules: Rules = SPACE_RULES, events: list | None = None) -> Any:
    """Put the gate in front of `env` (in place). Calls the gate makes itself are appended to `events` as
    {"type": "gate_call", "tag", "tool", "args", "reply"} and tagged `"gate": tag` in the trace."""
    gate = Gate(intent_of(task), rules)
    raw = env.call

    def execute(tool: str, args: dict, tag: str) -> dict:
        n = len(env.trace)
        obs = raw(tool, copy.deepcopy(args), "agent")
        if tag != "agent":
            for e in env.trace[n:]:
                if e.get("by", "agent") == "agent":
                    e["gate"] = tag
            if events is not None:
                events.append({"type": "gate_call", "tag": tag, "tool": tool, "args": copy.deepcopy(args),
                               "reply": copy.deepcopy(obs)})
        return obs

    def call(tool: str, args: dict | None = None, by: str = "agent") -> dict:
        if by != "agent":
            return raw(tool, args, by)
        return gate.handle(tool, dict(args or {}), execute)

    def finalize() -> None:
        if not env.reports:
            gate.finalize(execute)

    env.call = call
    env.gate = gate
    env.gate_finalize = finalize
    return env


def gate_record(env: Any) -> dict:
    g = getattr(env, "gate", None)
    if g is None:
        return {}
    return {"rules": {k: getattr(g.rules, k) for k in g.rules.__dataclass_fields__}, "log": copy.deepcopy(g.log),
            "rejections": g.rejections, "gate_calls": sum(1 for e in env.trace if e.get("gate"))}
