"""Reference policy for Amendment 4 splits (solvability proof), agent-visible tools only.

Verifies every write with a seen read tool, retries with one idempotency key per write, and gives up on a write
after MAX_ATTEMPTS failed attempts: it then reports "failed" immediately (the task can no longer be completed).
RP faults persist for at most 3 calls, so 4 attempts always suffice; UR faults never clear.
"""
from __future__ import annotations

from actionproof.env.tools_ecom import EcomEnv

MAX_ATTEMPTS = 4


def _ledger(env: EcomEnv, order: str) -> dict:
    return env.call("query_ledger", {"order_id": order})


def a4_oracle(env: EcomEnv, task) -> None:
    ids, wf = task.meta["ids"], task.meta.get("a4", {}).get("workflow", "refund")
    order, ticket = ids["order"], ids["ticket"]
    o = env.call("lookup_order", {"order_id": order})["order"]
    elig = env.call("check_refund_eligibility", {"order_id": order})["eligibility"]
    if elig != "eligible":
        env.call("escalate_to_human", {"ticket_id": ticket, "reason": f"refund {elig}"})
        env.call("submit_report", {"status": "escalated", "summary": f"Refund {elig}; escalated."})
        return

    steps = [("issue_refund", {"order_id": order, "amount_cents": o["amount_cents"], "method": o["payment_method"],
                               "idempotency_key": f"refund-{order}"},
              lambda: bool(_ledger(env, order)["refunds"]))]
    if wf == "credit":
        steps.append(("apply_store_credit", {"order_id": order, "amount_cents": ids["credit"],
                                             "idempotency_key": f"credit-{order}"},
                      lambda: bool(_ledger(env, order)["credits"])))
    steps += [("add_audit_note", {"ticket_id": ticket, "text": f"Refunded {o['amount_cents']} cents; verified."},
               lambda: bool(env.call("get_ticket", {"ticket_id": ticket})["audit_notes"])),
              ("close_ticket", {"ticket_id": ticket},
               lambda: env.call("get_ticket", {"ticket_id": ticket})["ticket"]["status"] == "closed")]
    for tool, args, landed in steps:
        for _ in range(MAX_ATTEMPTS):
            env.call(tool, args)
            if landed():
                break
        else:
            env.call("submit_report", {"status": "failed", "summary": f"{tool} did not take effect after retries."})
            return
    env.call("submit_report", {"status": "completed", "summary": "All changes verified."})
