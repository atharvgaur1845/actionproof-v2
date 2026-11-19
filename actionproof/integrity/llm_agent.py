"""A model agent for the gateway: any OpenAI-compatible chat endpoint (vLLM on the cluster, or an HF Inference
Endpoint serving Aya). Same prompt format, tool schemas and tool-call parser as training and evaluation.

  AP_LLM_URL   e.g. http://127.0.0.1:8000/v1   (required)
  AP_LLM_MODEL e.g. CohereLabs/aya-expanse-32b (default: whatever the server serves first)
  AP_LLM_KEY   bearer token if the endpoint needs one
The system prompt is rendered once (`data/bench/proc/system_prompt.txt`, written by `scripts/render_proc_prompt.py`
on a machine with transformers) so the demo machine needs no ML stack.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx

from actionproof.bench.prompts import TOOL_RESULT_FORMAT
from actionproof.rollout.lite_format import extract_tool_calls_from_text

PROMPT_FILE = Path(__file__).resolve().parents[2] / "data" / "bench" / "proc" / "system_prompt.txt"


def system_prompt(env: Any) -> str:
    if PROMPT_FILE.exists():
        return PROMPT_FILE.read_text()
    from actionproof.rollout.episode import system_prompt as sp
    return sp(env, "")


def run_llm_agent(gw: Any, task: Any, max_turns: int = 16, max_tokens: int = 256) -> list[dict[str, str]]:
    url = os.environ["AP_LLM_URL"].rstrip("/")
    headers = {"Authorization": f"Bearer {os.environ['AP_LLM_KEY']}"} if os.environ.get("AP_LLM_KEY") else {}
    client = httpx.Client(timeout=120, headers=headers)
    model = os.environ.get("AP_LLM_MODEL") or client.get(f"{url}/models").json()["data"][0]["id"]
    conv = [{"role": "system", "content": system_prompt(gw)}, {"role": "user", "content": task.prompt}]
    for _ in range(max_turns):
        r = client.post(f"{url}/chat/completions", json={"model": model, "messages": conv, "temperature": 0,
                                                        "max_tokens": max_tokens})
        text = r.raise_for_status().json()["choices"][0]["message"]["content"] or ""
        conv.append({"role": "assistant", "content": text})
        calls = extract_tool_calls_from_text(text) or []
        if not calls:
            break
        results = []
        for c in calls:
            try:
                args = json.loads(c["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                args = None
            obs = gw.call(c["function"]["name"], args) if isinstance(args, dict) else \
                {"status": "error", "error": "invalid_arguments"}
            results.append(TOOL_RESULT_FORMAT.format(name=c["function"]["name"], content=json.dumps(obs)))
        conv.append({"role": "user", "content": "\n\n".join(results)})
        if gw.done:
            break
    return conv
