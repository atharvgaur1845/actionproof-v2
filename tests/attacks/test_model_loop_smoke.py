"""Integration smoke test of scripts/attacks_eval.py::run_attack through the unchanged p1_eval episode loop, with a
fake LLM (no model, no GPU). Skipped where vLLM is not importable (the loop imports SamplingParams); runs on the
cluster CPU node. Checks: the env patch takes effect (attack text reaches the conversation), the F1 tool schema has
payee_account, the gate finalises, every row gets harm flags, and the original factory is restored."""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("vllm")

ROOT = Path(__file__).resolve().parents[2]


def _load():
    spec = importlib.util.spec_from_file_location("attacks_eval", ROOT / "scripts/attacks_eval.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class _Tok:
    def apply_chat_template(self, conv, add_generation_prompt=True, tokenize=False):
        return "\n".join(m["content"] for m in conv)

    def encode(self, text, add_special_tokens=False):
        return list(range(len(text) // 4))


class FakeLLM:
    """Reads the PO id and amount from the request, then: get_purchase_order, pay_supplier, submit_report."""

    def __init__(self):
        self.llm_engine = SimpleNamespace(model_config=SimpleNamespace(max_model_len=10**6))

    def get_tokenizer(self):
        return _Tok()

    def chat(self, convs, params, use_tqdm=False):
        out = []
        for conv in convs:
            user = conv[1]["content"]
            po = re.search(r"PO-[0-9A-F]{8}", user).group(0)
            amt = int(re.search(r"total (\d+) cents", user).group(1))
            n = sum(m["role"] == "assistant" for m in conv)
            call = [("get_purchase_order", {"po_id": po}),
                    ("pay_supplier", {"po_id": po, "amount_cents": amt, "idempotency_key": f"pay-{po}"}),
                    ("submit_report", {"status": "completed", "summary": "done"})][min(n, 2)]
            text = json.dumps([{"tool_name": call[0], "parameters": call[1]}])
            out.append(SimpleNamespace(outputs=[SimpleNamespace(text=text)]))
        return out


def test_run_attack_through_the_p1_loop():
    m = _load()
    from actionproof.attacks.generate import load_split
    from actionproof.recovery_gate.gate import Rules
    import actionproof.stage3.env as S3
    orig = S3.make_env_s3
    tasks = [t for t in load_split("attack_dev") if t.meta["attack"]["family"] == "F1_payee"
             and t.meta["attack"]["variant"] in ("attack", "twin")
             and (t.meta["attack"]["components"] or t.meta["attack"]["twin_of"])[0]["channel"] == "po_note"][:4]
    assert tasks
    for rules in (None, Rules()):
        rows = m.run_attack(FakeLLM(), tasks, 0, rules)
        assert S3.make_env_s3 is orig
        assert len(rows) == len(tasks)
        for r in rows:
            assert "payee_account" in r["conversation"][0]["content"]
            assert r["harm"]["variant"] in ("attack", "twin")
            assert r["report"] is not None
            seen = "\n".join(c["content"] for c in r["conversation"][2:] if c["role"] == "user")
            acct = (r["task"]["meta"]["attack"]["components"] or [{"params": {}}])[0]["params"].get("acct")
            if r["harm"]["variant"] == "attack":
                assert acct and acct in seen
            if rules is not None:
                assert "gate" in r
            # this fake model ignores the text and does not post the journal entry: no harm from the attack
            assert r["harm"]["attack_success"] in (None, False)
