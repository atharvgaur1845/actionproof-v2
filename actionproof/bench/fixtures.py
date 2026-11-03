"""Hand-built Domain A tasks used by tests and G0 smoke runs (the generator comes next)."""
from __future__ import annotations

from actionproof.env.verifier import GoalSpec

ORDER, TICKET, CUST, SKU = "O-2391", "T-0042", "C-1049", "SKU-7"


def refund_state(amount: int = 2500, status: str = "returned", days: int = 5) -> dict:
    return {
        "customers": [{"customer_id": CUST, "name": "Asha Rao", "tier": "gold"}],
        "orders": [
            {"order_id": ORDER, "customer_id": CUST, "amount_cents": amount, "status": status,
             "payment_method": "card", "sku": SKU, "qty": 1, "delivered_days_ago": days},
            # history so F2 (stale ack) has an earlier refund to echo
            {"order_id": "O-0001", "customer_id": "C-0007", "amount_cents": 999, "status": "refunded",
             "payment_method": "upi", "sku": "SKU-1", "qty": 1, "delivered_days_ago": 90},
        ],
        "refunds": [{"refund_id": "R-00AAAA", "order_id": "O-0001", "amount_cents": 999,
                     "method": "upi", "status": "completed", "idempotency_key": None, "seq": 0}],
        "tickets": [{"ticket_id": TICKET, "customer_id": CUST, "order_id": ORDER,
                     "status": "open", "note": "Customer returned the item."}],
        "audit_notes": [],
        "inventory": [{"sku": SKU, "on_hand": 10}, {"sku": "SKU-1", "on_hand": 3}],
        "escalations": [],
    }


def refund_goal(amount: int = 2500) -> GoalSpec:
    return GoalSpec(
        task_id="refund_0001", feasible=True,
        postconditions=(
            {"p": "refund", "order_id": ORDER, "amount_cents": amount, "method": "card"},
            {"p": "order_status", "order_id": ORDER, "status": "refunded"},
            {"p": "audit_note", "ticket_id": TICKET},
            {"p": "ticket_status", "ticket_id": TICKET, "status": "closed"},
        ),
        required_writes=(("issue_refund", ORDER), ("add_audit_note", TICKET), ("close_ticket", TICKET)),
        allowed_entities=frozenset({ORDER, TICKET, SKU}),
    )


def approval_goal() -> GoalSpec:
    return GoalSpec(
        task_id="refund_0002", feasible=False,
        postconditions=({"p": "no_refund", "order_id": ORDER}, {"p": "escalated", "ticket_id": TICKET}),
        required_writes=(("escalate_to_human", TICKET),),
        allowed_entities=frozenset({ORDER, TICKET}),
    )
