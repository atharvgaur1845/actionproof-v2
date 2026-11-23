"""CPU tests for actionproof/stage4/progress.py (C5 verified-progress credit, G realizable hint)."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from actionproof.stage4.progress import c5_advantages, hinted_result, phi, recon_line, request_total

ROOT = Path(__file__).resolve().parents[2]


def _task(wf: str = "pay_post") -> dict:
    for line in (ROOT / "data/bench/proc/e2_pp.jsonl").open():
        t = json.loads(line)
        if t["workflow"] == wf:
            return t
    raise AssertionError


def _pay(t, amt, pid="P1", kind="payment", refund_of=None):
    po = next(p for p in t["initial_state"]["pos"] if p["po_id"] == t["meta"]["po_id"])
    sup = next(s for s in t["initial_state"]["suppliers"] if s["supplier_id"] == po["supplier_id"])
    return {"payment_id": pid, "po_id": po["po_id"], "amount_cents": amt, "payee_account": sup["payee_account"],
            "kind": kind, "refund_of": refund_of}, po


def test_phi_orders_the_repair_chain():
    t = _task("pay_post")
    s0 = copy.deepcopy(t["initial_state"])
    n = len(t["meta"]["requested"])
    assert abs(phi(t, s0) - 0.5) < 1e-9                      # every step ABSENT: cost n of 2n
    p, po = _pay(t, 0)
    partial = dict(p, amount_cents=po["amount_cents"] // 4)
    s1 = dict(s0, payments=[partial])
    s2 = dict(s0, payments=[partial, dict(partial, payment_id="R1", kind="refund", refund_of="P1")])
    s3 = dict(s0, payments=s2["payments"] + [dict(partial, payment_id="P2", amount_cents=po["amount_cents"])])
    v = [phi(t, s) for s in (s1, s2, s3)]
    # a partial payment is worse than none; the refund recovers to "absent"; the full re-pay verifies the step
    assert v[0] < v[1] == phi(t, s0) < v[2], v
    assert abs(v[2] - (1 - (n - 1) / (2 * n))) < 1e-9


def test_c5_credits_a_correct_first_step_in_a_failed_episode():
    # two rollouts of one prompt: A refunds (+progress) then stops (fails); B posts (no progress) and fails too
    meta = [{"group": "g", "turns": [(0, 2, 1 / 6), (2, 4, 0.0)]}, {"group": "g", "turns": [(0, 2, 0.0), (2, 4, 0.0)]}]
    adv, diag = c5_advantages([0.0, 0.0], meta, 5)
    assert adv[0][0] > 0 and adv[0][0] > adv[1][0] and adv[0][4] == 0.0 and diag[0]["any_progress"]


def test_c5_is_plain_grpo_without_progress():
    meta = [{"group": "g", "turns": [(0, 3, 0.0)]}, {"group": "g", "turns": [(0, 3, 0.0)]}]
    adv, _ = c5_advantages([1.0, -1.0], meta, 3)
    assert adv == [[1.0] * 3, [-1.0] * 3]


def test_hint_is_computed_from_visible_values_only():
    t = _task("receive_pay")
    total = request_total(t["prompt"])
    p, po = _pay(t, 0)
    assert total == po["amount_cents"]
    assert recon_line({"payments": [dict(p, amount_cents=total // 2)]}, total).endswith(f"outstanding {total - total // 2} cents.")
    assert recon_line({"payments": [dict(p, amount_cents=total)]}, total) is None
    assert recon_line({"payments": []}, total) is None
    r = "Tool result for list_payments: {}\nContinue the task. Call submit_report when you are finished."
    assert hinted_result(r, "X").count("X\nContinue the task.") == 1


def test_every_training_prompt_states_its_total():
    for f in ("s3_trainb_half", "s3_rec_train_half"):
        for line in (ROOT / f"data/bench/proc/{f}.jsonl").open():
            t = json.loads(line)
            po = next(p for p in t["initial_state"]["pos"] if p["po_id"] == t["meta"]["po_id"])
            assert request_total(t["prompt"]) == po["amount_cents"], t["task_id"]
