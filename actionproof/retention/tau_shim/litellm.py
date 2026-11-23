"""Minimal stand-in for `litellm.completion`, used only by τ-bench inside R1-T (PREREGISTRATION_R1_TAU.md §3).

τ-bench calls `completion(model=..., custom_llm_provider=..., messages=..., temperature=...)` in its agents and
its user simulator. Here each provider name maps to a local OpenAI-compatible vLLM server
(`AP_TAU_<PROVIDER>_BASE`, e.g. AP_TAU_AGENT_BASE). The request is sent unchanged. Sampling parameters the
caller does not set are left to the server.

Two differences from litellm:
- `response_cost` is 0: these are local servers;
- a context-length rejection raises `ContextOverflow`, which the R1-T runner scores as a failed episode.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from types import SimpleNamespace


class ContextOverflow(Exception):
    pass


class _Msg(SimpleNamespace):
    def model_dump(self) -> dict:
        return {"role": self.role, "content": self.content}


def _post(base: str, body: dict) -> dict:
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def completion(model: str, messages: list, custom_llm_provider: str | None = None, temperature: float | None = None,
               **kw) -> SimpleNamespace:
    base = os.environ[f"AP_TAU_{(custom_llm_provider or 'agent').upper()}_BASE"]
    body = {"model": model, "messages": [{"role": m["role"], "content": m["content"]} for m in messages]}
    if temperature is not None:
        body["temperature"] = temperature
    if os.environ.get("AP_TAU_MAX_TOKENS") and (custom_llm_provider or "agent") == "agent":
        body["max_tokens"] = int(os.environ["AP_TAU_MAX_TOKENS"])
    try:
        d = _post(base, body)
    except urllib.error.HTTPError as e:
        msg = e.read().decode()
        if e.code == 400 and "max_tokens" in body and ("context" in msg.lower() or "maximum" in msg.lower()):
            body.pop("max_tokens")  # the prompt fits but prompt + budget does not: use the remaining room
            try:
                d = _post(base, body)
            except urllib.error.HTTPError as e2:
                raise ContextOverflow(e2.read().decode()[:300]) from e2
        elif e.code == 400 and ("context" in msg.lower() or "maximum" in msg.lower()):
            raise ContextOverflow(msg[:300]) from e
        else:
            raise
    ch = d["choices"][0]["message"]
    out = SimpleNamespace(choices=[SimpleNamespace(message=_Msg(role=ch.get("role", "assistant"),
                                                                 content=ch.get("content") or ""))],
                          usage=d.get("usage", {}))
    out._hidden_params = {"response_cost": 0.0}
    return out
