"""Probe-triggered verification guard (spec §14, H5). No retraining.

After a turn in which a write was ACKNOWLEDGED, score the conversation (the token right before
the model's next decision, i.e. pos-A) with the saved linear probe on an HF copy of the policy.
If score > tau (fixed on val at 80% recall), append a fixed nudge to that tool-result turn.
Compared against: neutral prompt (no verification push) and verify_a (always verify).
"""
from __future__ import annotations

from typing import Any

NUDGE = "\n\nNote: confirm the result of your last change with query_ledger, lookup_order or get_ticket before proceeding."


class ProbeGuard:
    def __init__(self, model_path: str, probe_npz: str, tau: float | None = None, device: str = "cuda"):
        import numpy as np
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        p = np.load(probe_npz)
        self.layer = int(p["layer"])
        self.mean, self.scale = torch.tensor(p["mean"]), torch.tensor(p["scale"])
        self.coef, self.b = torch.tensor(p["coef"]), float(p["intercept"][0])
        self.tau = float(p["tau"]) if tau is None else tau
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                                                          device_map=device).eval()
        self.fires = 0
        self.checks = 0

    def score(self, conv: list[dict[str, str]]) -> float:
        import torch
        text = self.tok.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        ids = self.tok.encode(text, add_special_tokens=False)
        with torch.no_grad():
            hs = self.model(torch.tensor([ids], device=self.model.device), output_hidden_states=True).hidden_states
        h = hs[self.layer][0, -1].float().cpu()
        return float(((h - self.mean) / self.scale) @ self.coef + self.b)

    def __call__(self, conv: list[dict[str, str]], block_obs: list[dict[str, Any]]) -> bool:
        """Return True to nudge. Only consulted when the last turn acknowledged a write."""
        if not any(o.get("status") in ("success", "accepted") and o.get("_write") for o in block_obs):
            return False
        self.checks += 1
        fire = self.score(conv) > self.tau
        self.fires += fire
        return fire
