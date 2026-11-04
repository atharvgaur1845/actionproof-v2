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


def lean_expert(env: EcomEnv, order_id: str, ticket_id: str) -> None:
    """Refund-workflow expert that fits the 10-turn policy budget (RFT demonstrations).

    Unlike `task_oracle` (a solvability proof that verifies everything), it verifies the way
    we want a policy to: one cheap read after the risky write (payments), and the
    authoritative ledger only when that read disagrees with the ack. Low-fault ticketing
    writes are trusted. Worst case (phantom or partial refund) is exactly 10 calls.
    """
    order = env.call("lookup_order", {"order_id": order_id})["order"]
    elig = env.call("check_refund_eligibility", {"order_id": order_id})["eligibility"]
    if elig != "eligible":
        env.call("escalate_to_human", {"ticket_id": ticket_id, "reason": f"refund {elig}"})
        env.call("submit_report", {"status": "escalated", "summary": f"Refund is {elig}; escalated to a human."})
        return
    t = RefundTask(order_id, ticket_id, order["amount_cents"], order["payment_method"])
    args = _refund_args(t, f"refund-{order_id}")
    obs = env.call("issue_refund", args)
    if obs.get("status") == "success" and \
            env.call("lookup_order", {"order_id": order_id})["order"]["status"] == "refunded":
        pass  # ack confirmed by an independent read
    elif _ledger_has_refund(env, t):  # refund landed; order not updated (partial) or ack lost (timeout)
        env.call("reconcile_order", {"order_id": order_id})
    else:  # ack was false: nothing landed. Same idempotency key, so a retry cannot duplicate.
        env.call("issue_refund", args)
        if env.call("lookup_order", {"order_id": order_id})["order"]["status"] != "refunded":
            env.call("submit_report", {"status": "failed", "summary": "Refund did not land after a retry."})
            return
    env.call("add_audit_note", {"ticket_id": ticket_id,
                                "text": f"Refunded {t.amount_cents} cents to {t.method}; confirmed on the order."})
    env.call("close_ticket", {"ticket_id": ticket_id})
    env.call("submit_report", {"status": "completed",
                               "summary": f"Refund of {t.amount_cents} cents to {t.method} confirmed; ticket closed."})


def bank_oracle(env, ids: dict) -> None:
    """Reference policy for Domain B (solvability proof), agent-visible tools only: checks policy from
    reads, verifies every write against the ledger / account / dispute, retries with one key."""
    acct, txn, disp = ids["account"], ids["txn"], ids["dispute"]
    a = env.call("query_account", {"account_id": acct})["account"]
    t = next(x for x in env.call("query_txn_ledger", {"account_id": acct})["transactions"] if x["txn_id"] == txn)
    if a["kyc_state"] == "pending":
        for _ in range(3):
            env.call("set_kyc_state", {"account_id": acct, "state": "verified", "document_id": ids["doc"]})
            if env.call("query_account", {"account_id": acct})["account"]["kyc_state"] == "verified":
                break
        a = env.call("query_account", {"account_id": acct})["account"]
    from actionproof.env.tools_bank import DISPUTE_WINDOW_DAYS, REVERSAL_LIMIT_CENTS
    reason = ("kyc_unverified" if a["kyc_state"] != "verified" else "outside_dispute_window"
              if t["days_ago"] > DISPUTE_WINDOW_DAYS else "requires_approval" if t["amount_cents"] > REVERSAL_LIMIT_CENTS else None)
    if reason:
        env.call("escalate_case", {"dispute_id": disp, "reason": reason})
        env.call("submit_report", {"status": "escalated", "summary": f"Reversal not allowed ({reason}); escalated."})
        return

    def reversals():
        return [x for x in env.call("query_txn_ledger", {"account_id": acct})["transactions"]
                if x["kind"] == "reversal" and x["ref_txn"] == txn]

    for _ in range(3):
        if reversals():
            break
        env.call("post_reversal", {"account_id": acct, "txn_id": txn, "amount_cents": t["amount_cents"],
                                   "idempotency_key": f"rev-{txn}"})
    if any(not r["balance_applied"] for r in reversals()):
        env.call("sync_balance", {"account_id": acct})
    for _ in range(3):
        env.call("close_dispute", {"dispute_id": disp, "resolution": f"Reversed {t['amount_cents']} cents of {txn}."})
        if env.call("get_dispute", {"dispute_id": disp})["dispute"]["status"] == "resolved":
            break
    env.call("submit_report", {"status": "completed" if reversals() else "failed", "summary": "Verified in ledger."})
