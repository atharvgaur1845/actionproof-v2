"""Model-agnostic multi-turn episode runner in AgentTune's Cohere fallback format.

Reuses AgentTune's own prompt (DEFAULT_TOOLS_FALLBACK_PROMPT), schema builder and tool-call
parser so evaluation sees byte-for-byte the format GRPO trains on. `generate` is any
`conversation -> text` function (HF Inference, vLLM, transformers), which keeps the runner
usable for G0 smoke tests, leaderboard evaluation and the demo.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from actionproof.bench.prompts import TOOL_RESULT_FORMAT
from actionproof.env.tools_ecom import EcomEnv

Generate = Callable[[list[dict[str, str]]], str]


def _agenttune():
    """AgentTune's prompt, schema builder and parser; the lite copy where AgentTune can't install (demo Space)."""
    try:
        from agenttune.agentic.rollout_engines.rollout_factory import (
            DEFAULT_TOOLS_FALLBACK_PROMPT, _build_tool_schema)
        from agenttune.agentic.rollout_engines.tool_call_parse import extract_tool_calls_from_text
        return DEFAULT_TOOLS_FALLBACK_PROMPT, _build_tool_schema, extract_tool_calls_from_text
    except ImportError:
        from actionproof.rollout import lite_format as L
        return L.FALLBACK_PROMPT, L.build_schema, L.extract_tool_calls_from_text


@dataclass
class EpisodeStats:
    turns: int = 0
    calls: int = 0
    unparsed_turns: int = 0      # assistant turns with no parseable tool call
    invalid_calls: int = 0       # parsed but rejected (unknown tool / bad args)
    ended_by: str = "max_turns"  # report | no_call | max_turns
    raw: list[str] = field(default_factory=list)


def system_prompt(env: EcomEnv, base: str) -> str:
    fallback, build_schema, _ = _agenttune()
    schemas = [build_schema(t) for t in env.tools()]
    tools = json.dumps(schemas, indent=2)
    try:
        block = fallback.format(tools=tools)
    except (KeyError, ValueError, IndexError):
        block = fallback.replace("{tools}", tools)
    return f"{base}\n\n{block}"


def run_episode(env: EcomEnv, generate: Generate, base_system: str, user_prompt: str,
                max_turns: int = 10) -> tuple[list[dict[str, str]], EpisodeStats]:
    _, _, parse = _agenttune()
    conv = [{"role": "system", "content": system_prompt(env, base_system)},
            {"role": "user", "content": user_prompt}]
    st = EpisodeStats()
    for _ in range(max_turns):
        st.turns += 1
        text = generate(conv)
        st.raw.append(text)
        conv.append({"role": "assistant", "content": text})
        calls = parse(text) or []
        if not calls:
            st.unparsed_turns += 1
            st.ended_by = "no_call"  # same as AgentTune: a plain-text turn ends the episode
            break
        results = []
        for c in calls:
            name = c["function"]["name"]
            try:
                args = json.loads(c["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                args = None
            obs = env.call(name, args) if isinstance(args, dict) else \
                {"status": "error", "error": "invalid_arguments", "detail": "arguments are not a JSON object"}
            st.calls += 1
            st.invalid_calls += obs.get("error") in ("unknown_tool", "invalid_arguments")
            results.append(TOOL_RESULT_FORMAT.format(name=name, content=json.dumps(obs)))
        conv.append({"role": "user", "content": "\n\n".join(results)})
        if env.done:
            st.ended_by = "report"
            break
    return conv, st
