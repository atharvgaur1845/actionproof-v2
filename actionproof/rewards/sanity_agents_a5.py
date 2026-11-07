"""Reference policies for the A5 slice. Agent-visible tools only; they never read env internals.

`a5_oracle`   : evidence-aware. Checks provenance before writing, uses one idempotency key, resolves outcomes
                with the operations service, waits only when a settle bound is documented, reconciles partial
                commits, and reports `uncertain` with the unresolved op when the evidence cannot decide.
`retry_blind` : the naive baseline. Retries on any error without a key, trusts success acks, reports completed.
"""
from __future__ import annotations


def _make_env(task, rid: str = "o"):
    from actionproof.env.tools_a5 import EcomEnvA5
    return EcomEnvA5(task.initial_state, task.schedule, rid, a5=json_copy(task.meta["a5"]))


def json_copy(x):
    import json
    return json.loads(json.dumps(x))


def a5_oracle(env, task) -> None:
    ids = task.meta["ids"]
    order, ticket = ids["order"], ids["ticket"]
    o = env.call("lookup_order", {"order_id": order})["order"]
    env.call("check_refund_eligibility", {"order_id": order})
    idx = len(env.trace)
    existing = [x for x in env.call("list_operations", {"entity_id": order})["operations"]
                if x["tool"] == "issue_refund" and x["status"] == "committed"]
    effect, evidence, unresolved = None, [], []
    if existing:  # goal already satisfied by someone: do not write, attribute correctly
        effect, evidence = existing[0], [idx]
    else:
        args = {"order_id": order, "amount_cents": o["amount_cents"], "method": o["payment_method"],
                "idempotency_key": f"refund-{order}"}
        for _ in range(3):
            obs = env.call("issue_refund", args)
            op_id = obs.get("op_id") or obs.get("request_id")
            i = len(env.trace)
            op = env.call("get_operation", {"op_id": op_id})["operation"]
            if op["status"] == "pending" and op.get("settle_bound_s"):
                env.call("wait", {"seconds": op["settle_bound_s"]})
                i = len(env.trace)
                op = env.call("get_operation", {"op_id": op_id})["operation"]
            if op["status"] == "committed":
                effect, evidence = op, [i]
                break
            if op["status"] == "pending":  # no bound: the outcome cannot be determined; a retry could duplicate
                unresolved = [op_id]
                break
            # failed: demonstrably not executed -> safe to retry (failed ops do not hold the key)
    if effect and env.call("lookup_order", {"order_id": order})["order"]["status"] != "refunded":
        env.call("reconcile_order", {"order_id": order})  # partial commit: finish it, do not re-refund
    if unresolved:
        env.call("add_audit_note", {"ticket_id": ticket, "text": f"Refund request {unresolved[0]} still in flight."})
        env.call("submit_report", {"status": "uncertain", "summary": "Refund outcome cannot be determined yet.",
                                   "certificate": {"effects": [], "evidence": [], "unresolved": unresolved}})
        return
    if not effect:
        env.call("submit_report", {"status": "failed", "summary": "Refund did not execute after retries.",
                                   "certificate": {"effects": [], "evidence": [], "unresolved": []}})
        return
    env.call("add_audit_note", {"ticket_id": ticket, "text": f"Refund {effect['op_id']} by {effect['actor']} verified."})
    env.call("close_ticket", {"ticket_id": ticket})
    env.call("submit_report", {"status": "completed", "summary": "Refund verified in the operations log.",
                               "certificate": {"effects": [{"op_id": effect["op_id"], "tool": "issue_refund",
                                                            "entity": order, "actor": effect["actor"],
                                                            "state": "committed"}],
                                               "evidence": evidence, "unresolved": []}})


def retry_blind(env, task) -> None:
    ids = task.meta["ids"]
    order, ticket = ids["order"], ids["ticket"]
    o = env.call("lookup_order", {"order_id": order})["order"]
    op_id = None
    for _ in range(3):
        obs = env.call("issue_refund", {"order_id": order, "amount_cents": o["amount_cents"], "method": o["payment_method"]})
        op_id = obs.get("op_id") or obs.get("request_id")
        if obs.get("status") == "success":
            break
    env.call("add_audit_note", {"ticket_id": ticket, "text": "Refunded."})
    env.call("close_ticket", {"ticket_id": ticket})
    env.call("submit_report", {"status": "completed", "summary": "Refund issued.",
                               "certificate": {"effects": [{"op_id": op_id, "tool": "issue_refund", "entity": order,
                                                            "actor": "agent", "state": "committed"}],
                                               "evidence": [], "unresolved": []}})
