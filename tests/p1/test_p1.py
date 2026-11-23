"""P1 harness checks on CPU: the inference-time reconciliation hook uses visible values only and fires only on a
partial live payment; behaviour metrics credit a new idempotency key after a refund."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("p1_eval", ROOT / "scripts/p1_eval.py")
P1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P1)


def tasks(split):
    from actionproof.bench.generator_proc import DATA, load_tasks
    return load_tasks(DATA / f"{split}.jsonl")


def test_hook_only_on_partial_live_payment():
    partial = {"status": "ok", "payments": [{"payment_id": "PAY-1", "amount_cents": 250, "kind": "payment", "refund_of": None}]}
    full = {"status": "ok", "payments": [{"payment_id": "PAY-1", "amount_cents": 1000, "kind": "payment", "refund_of": None}]}
    refunded = {"status": "ok", "payments": [partial["payments"][0],
                                             {"payment_id": "RF-1", "amount_cents": 250, "kind": "refund", "refund_of": "PAY-1"}]}
    assert "outstanding 750 cents" in P1.tool_text("list_payments", partial, 1000, True)
    assert "Reconciliation" not in P1.tool_text("list_payments", full, 1000, True)
    assert "Reconciliation" not in P1.tool_text("list_payments", refunded, 1000, True)  # nothing live: no line
    assert "Reconciliation" not in P1.tool_text("list_payments", partial, 1000, False)
    assert "Reconciliation" not in P1.tool_text("get_journal_entries", partial, 1000, True)


def test_hook_reaches_prefix_read_in_recovery_starts():
    from actionproof.stage3.env import make_env_s3
    t = next(t for t in tasks("p1_dev_rec") if t.meta["fault_class"] == "s3:partial:read" and t.meta.get("start_step") == "pay")
    on = P1.start(make_env_s3(t, "x"), t, True)
    off = P1.start(make_env_s3(t, "y"), t, False)
    assert any("Reconciliation" in m["content"] for m in on if m["role"] == "user")
    assert not any("Reconciliation" in m["content"] for m in off if m["role"] == "user")
    assert len(on) == len(off)


def test_metrics_credit_new_key_after_refund():
    from actionproof.integrity.sim import agents
    from actionproof.integrity.sim.runtime import episode_record_proc
    from actionproof.stage3.env import make_env_s3
    t = tasks("p1_dev_full")[0]
    t = next(x for x in tasks("p1_train_pp"))
    env = make_env_s3(t, "z")
    agents.oracle(env, t)
    r = episode_record_proc(t, env)
    m = P1.episode_metrics({**r, "task": {"meta": t.meta}}, 0)
    assert r["score"]["goal_met"] and m["partial_pay_task"]
    assert m["refunds"] == 1 and m["repay_after_refund"] == 1 and m["repay_new_key"] == 1
