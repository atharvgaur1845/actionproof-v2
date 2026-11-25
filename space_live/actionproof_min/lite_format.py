"""Dependency-free version of the Cohere tool-call format for the demo Space.

ZeroGPU cannot install AgentTune (it pins torch 2.11 + vLLM). This module reproduces the
format our policies are trained and evaluated in: tool schemas from transformers'
get_json_schema, AgentTune's fallback system-prompt text, and a parser for the
`[{"tool_name": ..., "parameters": {...}}]` list the models emit. Parity with AgentTune's
parser on real model outputs is checked on the cluster (scripts/check_lite_parity.py).
"""
from __future__ import annotations

import json
import re
from typing import Any

FALLBACK_PROMPT = (
    "You can call these tools:\n{tools}\n\n"
    "To call tools, reply with only a JSON list, one object per call, and no other text: "
    '[{{"tool_name": "<name>", "parameters": {{"<argument>": <value>}}}}]. '
    "Use a list even for one call. "
    'A user turn that starts with "Tool result" is the output of your call, not a new '
    "message from the user: use it to answer the user's original question as plain text."
)


def build_schema(fn) -> dict[str, Any]:
    from transformers.utils import get_json_schema
    return get_json_schema(fn)


def _calls(obj: Any) -> list[dict[str, Any]] | None:
    items = obj if isinstance(obj, list) else [obj] if isinstance(obj, dict) else None
    if items is None:
        return None
    out = []
    for c in items:
        if not isinstance(c, dict):
            continue
        fn = c.get("function") if isinstance(c.get("function"), dict) else {}
        name = fn.get("name") or c.get("tool_name") or c.get("name") or c.get("tool")
        args = fn.get("arguments", c.get("parameters", c.get("arguments", c.get("args", {}))))
        if isinstance(name, str) and name.strip() and name.strip() != "directly_answer":
            out.append({"type": "function", "function": {
                "name": name.strip(), "arguments": args if isinstance(args, str) else json.dumps(args or {})}})
    return out or None


def extract_tool_calls_from_text(text: str) -> list[dict[str, Any]] | None:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    dec = json.JSONDecoder()
    for i, ch in enumerate(text):  # first JSON list/object in the text, as AgentTune does
        if ch in "[{":
            try:
                obj, _ = dec.raw_decode(text[i:])
            except json.JSONDecodeError:
                continue
            calls = _calls(obj)
            if calls:
                return calls
    return None
