"""Deterministic replay of a recorded episode, turn by turn.

The simulator is deterministic given the task (its `env_seed` drives every id, and faults are keyed on the k-th
call of a tool). Replaying the recorded assistant turns through the evaluation loop's own parser and call path
(`scripts/p1_eval.py::run`) therefore reproduces the recorded trace exactly; `replay(..., check=True)` asserts it.

A cut at turn T keeps the start messages (system, request, scripted prefix) and the first T assistant turns with
their tool-result messages. A continuation is then appended by a teacher; `verify_continuation` replays prefix plus
continuation from scratch, parsing the continuation's text with the same parser, and scores the result with the
verifier. Only that score decides whether a correction is kept.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from actionproof.selfheal.envs import EnvSpec, task_from_record

MAX_TURNS = 16  # scripts/p1_eval.py and scripts/txn_eval.py


class ReplayMismatch(AssertionError):
    pass


def parser():
    from actionproof.rollout.episode import _agenttune
    return _agenttune()[2]


def tool_text(name: str, obs: dict[str, Any]) -> str:
    """The tool-result message format of p1_eval/txn_eval without the reconciliation line."""
    from actionproof.bench.prompts import TOOL_RESULT_FORMAT
    return TOOL_RESULT_FORMAT.format(name=name, content=json.dumps(obs))


def act_text(tool: str, args: dict[str, Any]) -> str:
    return json.dumps([{"tool_name": tool, "parameters": args}])


def execute_turn(env: Any, text: str, parse=None) -> tuple[list[str], int, bool]:
    """Run one assistant turn exactly as p1_eval does: (tool-result texts, invalid count, had a parseable call)."""
    parse = parse or parser()
    calls = parse(text) or []
    if not calls:
        return [], 0, False
    res, invalid = [], 0
    for c in calls:
        name = c["function"]["name"]
        try:
            args = json.loads(c["function"]["arguments"] or "{}")
        except json.JSONDecodeError:
            args = None
        obs = env.call(name, args) if isinstance(args, dict) else {"status": "error", "error": "invalid_arguments"}
        invalid += obs.get("error") in ("unknown_tool", "invalid_arguments")
        res.append(tool_text(name, obs))
    return res, invalid, True


def _norm(x: Any) -> Any:
    return json.loads(json.dumps(x, default=str))


@dataclass
class Replayed:
    task: Any
    env: Any
    conv: list[dict[str, str]]
    n_start: int                    # messages before the first policy turn
    turn_ends: list[int] = field(default_factory=list)  # len(env.trace) after the start, then after each turn
    turns: int = 0

    @property
    def turns_left(self) -> int:
        return MAX_TURNS - self.turns


def start(rec: dict[str, Any], spec: EnvSpec, rid: str | None = None) -> Replayed:
    task = task_from_record(rec)
    env = spec.make(task, rid or rec.get("rollout_id") or f"selfheal-{task.task_id}")
    prefix = task.meta.get("start_prefix") or []
    for tool, args in prefix:
        env.call(tool, dict(args))
    n_start = 2 + 2 * len(prefix)
    conv = [dict(m) for m in rec["conversation"][:n_start]]
    return Replayed(task=task, env=env, conv=conv, n_start=n_start, turn_ends=[len(env.trace)])


def replay(rec: dict[str, Any], spec: EnvSpec, upto_turn: int | None = None, check: bool = True,
           parse=None) -> Replayed:
    """Replay the recorded turns (all of them, or the first `upto_turn`). With `check`, the replayed trace must equal
    the recorded one call for call (tool, args, observation, actor) and each rebuilt tool-result message must equal
    the recorded message."""
    parse = parse or parser()
    r = start(rec, spec)
    conv = rec["conversation"]
    i = r.n_start
    while i < len(conv) and (upto_turn is None or r.turns < upto_turn):
        msg = conv[i]
        if msg["role"] != "assistant":
            raise ReplayMismatch(f"message {i} is {msg['role']}, expected assistant")
        res, _, had = execute_turn(r.env, msg["content"], parse)
        if not had:
            break  # a final turn with no parseable call ends the episode and has no result message
        r.conv.append(dict(msg))
        if i + 1 < len(conv):
            if check and conv[i + 1]["content"] != "\n\n".join(res):
                raise ReplayMismatch(f"tool-result message {i + 1} differs on replay")
            r.conv.append(dict(conv[i + 1]))
        else:
            r.conv.append({"role": "user", "content": "\n\n".join(res)})
        r.turns += 1
        r.turn_ends.append(len(r.env.trace))
        i += 2
        if r.env.done:
            break
    if check:
        got = _norm(r.env.trace)
        want = rec["trace"] if upto_turn is None else rec["trace"][:len(got)]
        keys = ("tool", "args", "obs")
        if len(got) != len(want) or any({k: a[k] for k in keys} != {k: b[k] for k in keys} or
                                         a.get("by", "agent") != b.get("by", "agent") for a, b in zip(got, want)):
            raise ReplayMismatch(f"{rec.get('task_id')}: replayed trace differs from the record")
    return r


def turn_before(turn_ends: list[int], trace_idx: int | None, turns: int) -> int:
    """The largest cut T such that every call of turns 1..T has a trace index below `trace_idx` (None: the end)."""
    if trace_idx is None:
        return turns
    t = 0
    for j in range(1, turns + 1):
        if turn_ends[j] <= trace_idx:
            t = j
    return t


def verify_continuation(rec: dict[str, Any], spec: EnvSpec, cut: int, continuation: list[dict[str, str]],
                        parse=None) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Replay the recorded prefix to `cut`, then the continuation's assistant turns (parsed, not trusted), and score
    the finished env. Returns (scored record, full conversation). The continuation's tool-result messages must be
    reproduced exactly by the replay, so the SFT text matches what the simulator does."""
    parse = parse or parser()
    r = replay(rec, spec, upto_turn=cut, check=True, parse=parse)
    if r.turns != cut:
        raise ReplayMismatch(f"cut {cut} beyond the recorded turns ({r.turns})")
    conv = list(r.conv)
    turns = r.turns
    for j in range(0, len(continuation), 2):
        msg = continuation[j]
        if r.env.done or turns >= MAX_TURNS:
            break
        res, _, had = execute_turn(r.env, msg["content"], parse)
        turns += 1
        conv.append(dict(msg))
        if not had:
            break
        text = "\n\n".join(res)
        if j + 1 < len(continuation) and continuation[j + 1]["content"] != text:
            raise ReplayMismatch(f"continuation result {j + 1} differs on replay")
        conv.append({"role": "user", "content": text})
    out = spec.record(r.task, r.env)
    out.update(turns=turns, ended_by="report" if r.env.done else "max_turns", sample=rec.get("sample"))
    return out, conv
