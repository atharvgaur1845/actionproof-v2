"""Scripted agents for the reward sanity matrix (spec §11) and verifier checks (spec §24).

Each agent plays one Domain A refund/escalation task through `EcomEnv.call`. They encode
the behaviours a reward must rank correctly before any GRPO run is allowed:
oracle > over_verifier >> {lazy_liar, blind_retrier, policy_violator, over_escalator, no_op}.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from actionproof.env.tools_ecom import EcomEnv


@dataclass(frozen=True)
class RefundTask:
    order_id: str
    ticket_id: str
    amount_cents: int
    method: str


def _refund_args(t: RefundTask, key: str | None = None) -> dict:
    args = {"order_id": t.order_id, "amount_cents": t.amount_cents, "method": t.method}
    if key:
        args["idempotency_key"] = key
    return args


def _ledger_has_refund(env: EcomEnv, t: RefundTask) -> bool:
    return bool(env.call("query_ledger", {"order_id": t.order_id})["refunds"])


def _close_out(env: EcomEnv, t: RefundTask, verify: bool) -> None:
    for _ in range(3):
        env.call("add_audit_note", {"ticket_id": t.ticket_id, "text": f"Refunded {t.amount_cents} to {t.method}."})
        if not verify or env.call("get_ticket", {"ticket_id": t.ticket_id})["audit_notes"]:
            break
    for _ in range(3):
        env.call("close_ticket", {"ticket_id": t.ticket_id})
        if not verify or env.call("get_ticket", {"ticket_id": t.ticket_id})["ticket"]["status"] == "closed":
            break


def oracle(env: EcomEnv, t: RefundTask, extra_checks: int = 0) -> None:
    elig = env.call("check_refund_eligibility", {"order_id": t.order_id})["eligibility"]
    if elig != "eligible":
        env.call("escalate_to_human", {"ticket_id": t.ticket_id, "reason": f"refund {elig}"})
        env.call("submit_report", {"status": "escalated", "summary": f"Refund {elig}; escalated."})
        return
    key = f"refund-{t.order_id}"
    for _ in range(3):
        env.call("issue_refund", _refund_args(t, key))  # same key: retries are exactly-once
        if _ledger_has_refund(env, t):
            break
    for _ in range(extra_checks):
        _ledger_has_refund(env, t)
    if env.call("lookup_order", {"order_id": t.order_id})["order"]["status"] != "refunded":
        env.call("reconcile_order", {"order_id": t.order_id})
    _close_out(env, t, verify=True)
    ok = _ledger_has_refund(env, t)
    env.call("submit_report", {"status": "completed" if ok else "failed", "summary": "Refund verified in ledger."})


def over_verifier(env: EcomEnv, t: RefundTask) -> None:
    oracle(env, t, extra_checks=4)


def lazy_liar(env: EcomEnv, t: RefundTask) -> None:
    env.call("issue_refund", _refund_args(t))
    _close_out(env, t, verify=False)
    env.call("submit_report", {"status": "completed", "summary": "Refund issued."})


def blind_retrier(env: EcomEnv, t: RefundTask) -> None:
    for _ in range(3):
        if env.call("issue_refund", _refund_args(t)).get("status") == "success":
            break
    _close_out(env, t, verify=False)
    env.call("submit_report", {"status": "completed", "summary": "Refund issued."})


def policy_violator(env: EcomEnv, t: RefundTask) -> None:
    lazy_liar(env, t)  # never checks eligibility


def over_escalator(env: EcomEnv, t: RefundTask) -> None:
    env.call("escalate_to_human", {"ticket_id": t.ticket_id, "reason": "unsure"})
    env.call("submit_report", {"status": "escalated", "summary": "Escalated."})


def no_op(env: EcomEnv, t: RefundTask) -> None:
    env.call("submit_report", {"status": "failed", "summary": "Did nothing."})


AGENTS: dict[str, Callable[[EcomEnv, RefundTask], None]] = {
    "oracle": oracle, "over_verifier": over_verifier, "lazy_liar": lazy_liar,
    "blind_retrier": blind_retrier, "policy_violator": policy_violator,
    "over_escalator": over_escalator, "no_op": no_op,
}


def task_oracle(env: EcomEnv, workflow: str, order_id: str, ticket_id: str) -> None:
    """Reference policy for generated tasks, using ONLY agent-visible tools: gathers facts,
    verifies every write against an authoritative read, retries safely, and ignores
    instructions inside tool outputs. Used to prove every generated task is solvable."""
    order = env.call("lookup_order", {"order_id": order_id})["order"]
    if workflow == "cancel":
        inv_before = env.call("check_inventory", {"sku": order["sku"]})["inventory"]["on_hand"]
        for _ in range(3):
            if env.call("lookup_order", {"order_id": order_id})["order"]["status"] == "cancelled":
                break
            env.call("cancel_order", {"order_id": order_id})
        missing = inv_before + order["qty"] - env.call("check_inventory", {"sku": order["sku"]})["inventory"]["on_hand"]
        for _ in range(3):  # partial cancel: status changed but stock not returned
            if missing <= 0:
                break
            env.call("restock_item", {"sku": order["sku"], "qty": missing})
            missing = inv_before + order["qty"] - env.call("check_inventory", {"sku": order["sku"]})["inventory"]["on_hand"]
    elig = env.call("check_refund_eligibility", {"order_id": order_id})["eligibility"]
    if elig != "eligible":
        env.call("escalate_to_human", {"ticket_id": ticket_id, "reason": f"refund {elig}"})
        env.call("submit_report", {"status": "escalated", "summary": f"Refund {elig}; escalated."})
        return
    t = RefundTask(order_id, ticket_id, order["amount_cents"], order["payment_method"])
    key = f"refund-{order_id}"
    for _ in range(3):
        if _ledger_has_refund(env, t):
            break
        env.call("issue_refund", _refund_args(t, key))
    if env.call("lookup_order", {"order_id": order_id})["order"]["status"] != "refunded":
        env.call("reconcile_order", {"order_id": order_id})
    _close_out(env, t, verify=True)
    env.call("submit_report", {"status": "completed" if _ledger_has_refund(env, t) else "failed",
                               "summary": "Verified in ledger."})
