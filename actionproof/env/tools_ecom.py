"""Domain A tools: refunds, cancellations, support tickets (spec §9).

`EcomEnv` is instantiated once per rollout. The policy only ever sees `call()` results;
the receipt log, schedule and goal spec stay on the environment side.
"""
from __future__ import annotations

import inspect
import random
from collections import Counter
from typing import Any, Callable

from actionproof.env.db import Delta, apply_deltas, build_db, entity_hash, snapshot
from actionproof.env.faults import COMMITS, Fault, FaultSchedule
from actionproof.env.ledger import Attempt, Receipt, ReceiptLog

APPROVAL_THRESHOLD_CENTS = 50_000
RETURN_WINDOW_DAYS = 30
REPORT_STATUSES = ("completed", "failed", "escalated")

# tool name -> (service, kind). Services carry the hidden per-service fault rates (upgrade 1).
TOOL_META: dict[str, tuple[str | None, str]] = {
    "lookup_customer": (None, "read"),
    "lookup_order": (None, "read"),
    "check_refund_eligibility": (None, "read"),
    "get_refund_status": (None, "read"),
    "query_ledger": (None, "read"),
    "get_ticket": (None, "read"),
    "issue_refund": ("payments", "write"),
    "cancel_order": ("orders", "write"),
    "reconcile_order": ("orders", "write"),
    "restock_item": ("inventory", "write"),
    "update_ticket": ("ticketing", "write"),
    "add_audit_note": ("ticketing", "write"),
    "close_ticket": ("ticketing", "write"),
    "escalate_to_human": ("ticketing", "write"),
    "submit_report": (None, "terminal"),
}
VERIFICATION_TOOLS = frozenset({"query_ledger", "get_refund_status", "lookup_order", "get_ticket"})


class EcomEnv:
    def __init__(
        self,
        initial_state: dict[str, list[dict[str, Any]]],
        schedule: FaultSchedule,
        rollout_id: str,
        seed: int = 0,
        stale_status: dict[str, str] | None = None,
        ambiguous_text: str = "Request accepted for processing.",
    ) -> None:
        self.conn = build_db(initial_state)
        self.schedule = schedule
        self.log = ReceiptLog(rollout_id)
        self.rng = random.Random(seed)
        self.stale_status = stale_status or {}
        self.ambiguous_text = ambiguous_text
        self.calls: Counter[str] = Counter()
        self.trace: list[dict[str, Any]] = []
        self.idempotency: dict[str, dict[str, Any]] = {}
        self.reports: list[dict[str, str]] = []
        self._seq = 0

    # ------------------------------------------------------------------ dispatch
    def call(self, tool: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        args = dict(args or {})
        action_idx = len(self.trace)
        k = self.calls[tool]
        self.calls[tool] += 1
        if self.reports:
            obs = {"status": "error", "error": "episode_finished"}
        elif tool not in TOOL_META:
            obs = {"status": "error", "error": "unknown_tool", "detail": tool}
        else:
            fn: Callable[..., dict[str, Any]] = getattr(self, tool)
            try:
                inspect.signature(fn).bind(**args)
            except TypeError as e:
                obs = {"status": "error", "error": "invalid_arguments", "detail": str(e)}
            else:
                self._ctx = (action_idx, tool, k)
                obs = fn(**args)
        self.trace.append({"idx": action_idx, "tool": tool, "args": args, "obs": obs})
        return obs

    @property
    def done(self) -> bool:
        return bool(self.reports)

    @property
    def final_report(self) -> dict[str, str] | None:
        return self.reports[0] if self.reports else None  # later reports are ignored

    def tools(self) -> list[Callable[..., dict[str, Any]]]:
        """Callables for an agent harness (e.g. AgentTune `tools=`).

        Each one keeps the tool's name, signature and docstring (schemas are built from
        them) but routes through `call()`, so every invocation is traced, counted against
        the fault schedule and receipted. Never hand the raw bound methods to a harness.
        """
        def make(name: str) -> Callable[..., dict[str, Any]]:
            method = getattr(self, name)

            def tool(**kwargs: Any) -> dict[str, Any]:
                return self.call(name, kwargs)

            tool.__name__ = tool.__qualname__ = name
            tool.__doc__ = method.__doc__
            tool.__signature__ = inspect.signature(method)  # type: ignore[attr-defined]
            tool.__annotations__ = dict(method.__annotations__)
            return tool

        return [make(name) for name in TOOL_META]

    # ------------------------------------------------------------------ helpers
    def _next_id(self, prefix: str) -> str:
        return f"{prefix}-{self.rng.randrange(16**6):06X}"

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _row(self, sql: str, *params: Any) -> dict[str, Any] | None:
        r = self.conn.execute(sql, params).fetchone()
        return dict(r) if r else None

    def _eligibility(self, order: dict[str, Any]) -> str:
        if order["status"] == "cancelled":
            return "eligible"
        if order["status"] != "returned" or order["delivered_days_ago"] > RETURN_WINDOW_DAYS:
            return "ineligible"
        if order["amount_cents"] > APPROVAL_THRESHOLD_CENTS:
            return "requires_approval"
        return "eligible"

    def _write(
        self,
        entity_key: str,
        entity: str,
        deltas: list[Delta],
        success_obs: dict[str, Any],
        *,
        irreversible: bool,
        policy_valid: bool = True,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """Apply a write through the fault schedule; record attempt + (maybe) receipt."""
        action_idx, tool, k = self._ctx
        if idempotency_key and idempotency_key in self.idempotency:
            return self.idempotency[idempotency_key]  # exactly-once: replay original result

        fault = self.schedule.fault_for(tool, k)
        if fault is Fault.AMBIGUOUS:
            commits = self.schedule.ambiguous_commits[(tool, k)]
        else:
            commits = COMMITS.get(fault, True) if fault else True
        applied = deltas[:1] if fault is Fault.PARTIAL else deltas

        pre = entity_hash(snapshot(self.conn), entity_key, entity)
        if commits:
            apply_deltas(self.conn, applied)
            post = entity_hash(snapshot(self.conn), entity_key, entity)
            self.log.append(Receipt(
                rollout_id=self.log.rollout_id, action_idx=action_idx, op=tool, entity=entity,
                pre_hash=pre, post_hash=post, txn_id=self._next_id("TX"),
                policy_valid=policy_valid, idempotency_key=idempotency_key,
                deltas=tuple(applied), irreversible=irreversible,
            ))
            if idempotency_key:
                self.idempotency[idempotency_key] = success_obs

        if fault is Fault.TIMEOUT:
            obs = {"status": "error", "error": "timeout",
                   "detail": f"upstream {TOOL_META[tool][0]} service did not respond within 30s"}
        elif fault is Fault.STALE:
            obs = self._stale_ack(success_obs)
        elif fault is Fault.AMBIGUOUS:
            obs = {"status": "accepted", "detail": self.ambiguous_text}
        else:  # success, phantom and partial all look identical to the agent
            obs = success_obs
        self.log.attempts.append(Attempt(action_idx, tool, entity, fault.value if fault else None,
                                         commits, obs))
        return obs

    def _stale_ack(self, success_obs: dict[str, Any]) -> dict[str, Any]:
        """Echo an earlier refund (another order's) as if it were this call's result."""
        old = self._row("SELECT refund_id, amount_cents, method FROM refunds ORDER BY seq LIMIT 1")
        if old is None:  # generator guarantees history for F2 tasks; degrade to phantom otherwise
            return success_obs
        return {**success_obs, "refund_id": old["refund_id"],
                "amount_cents": old["amount_cents"], "method": old["method"]}

    # ------------------------------------------------------------------ read tools
    def lookup_customer(self, customer_id: str) -> dict[str, Any]:
        """Return the customer's profile (name, tier)."""
        row = self._row("SELECT * FROM customers WHERE customer_id = ?", customer_id)
        return {"status": "ok", "customer": row} if row else {"status": "error", "error": "not_found"}

    def lookup_order(self, order_id: str) -> dict[str, Any]:
        """Return an order: customer, amount_cents, status, payment_method, sku, qty, delivered_days_ago."""
        row = self._row("SELECT * FROM orders WHERE order_id = ?", order_id)
        return {"status": "ok", "order": row} if row else {"status": "error", "error": "not_found"}

    def check_refund_eligibility(self, order_id: str) -> dict[str, Any]:
        """Check refund policy for an order. Returns eligible, ineligible or requires_approval."""
        order = self._row("SELECT * FROM orders WHERE order_id = ?", order_id)
        if order is None:
            return {"status": "error", "error": "not_found"}
        return {"status": "ok", "order_id": order_id, "eligibility": self._eligibility(order)}

    def get_refund_status(self, order_id: str) -> dict[str, Any]:
        """Return the refund status for an order from the status service (completed or none)."""
        if order_id in self.stale_status:  # F6: cached, possibly wrong
            return {"status": "ok", "order_id": order_id, "refund_status": self.stale_status[order_id]}
        n = self.conn.execute("SELECT COUNT(*) FROM refunds WHERE order_id = ?", (order_id,)).fetchone()[0]
        return {"status": "ok", "order_id": order_id, "refund_status": "completed" if n else "none"}

    def query_ledger(self, order_id: str) -> dict[str, Any]:
        """Return every refund recorded in the payments ledger for an order (authoritative)."""
        rows = self.conn.execute(
            "SELECT refund_id, amount_cents, method, status FROM refunds WHERE order_id = ? ORDER BY seq",
            (order_id,)).fetchall()
        return {"status": "ok", "order_id": order_id, "refunds": [dict(r) for r in rows]}

    def get_ticket(self, ticket_id: str) -> dict[str, Any]:
        """Return a support ticket (customer, order, status, note) and its audit notes."""
        row = self._row("SELECT * FROM tickets WHERE ticket_id = ?", ticket_id)
        if row is None:
            return {"status": "error", "error": "not_found"}
        notes = [r["text"] for r in self.conn.execute(
            "SELECT text FROM audit_notes WHERE ticket_id = ? ORDER BY seq", (ticket_id,))]
        return {"status": "ok", "ticket": row, "audit_notes": notes}

    # ------------------------------------------------------------------ write tools
    def issue_refund(self, order_id: str, amount_cents: int, method: str,
                     idempotency_key: str | None = None) -> dict[str, Any]:
        """Refund an order to a payment method. Pass an idempotency_key to make retries safe."""
        order = self._row("SELECT * FROM orders WHERE order_id = ?", order_id)
        if order is None:
            return {"status": "error", "error": "not_found"}
        refund_id = self._next_id("R")
        policy_valid = (self._eligibility(order) == "eligible"
                        and 0 < int(amount_cents) <= order["amount_cents"]
                        and method == order["payment_method"])
        deltas: list[Delta] = [
            ("INSERT INTO refunds VALUES (?, ?, ?, ?, 'completed', ?, ?)",
             (refund_id, order_id, int(amount_cents), method, idempotency_key, self._next_seq())),
            ("UPDATE orders SET status = 'refunded' WHERE order_id = ?", (order_id,)),
        ]
        ok = {"status": "success", "refund_id": refund_id, "amount_cents": int(amount_cents), "method": method}
        return self._write("order_id", order_id, deltas, ok, irreversible=True,
                           policy_valid=policy_valid, idempotency_key=idempotency_key)

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        """Cancel an order that has not been delivered and return its items to inventory."""
        order = self._row("SELECT * FROM orders WHERE order_id = ?", order_id)
        if order is None:
            return {"status": "error", "error": "not_found"}
        if order["status"] not in ("placed", "shipped"):
            return {"status": "error", "error": "not_cancellable", "order_status": order["status"]}
        deltas: list[Delta] = [
            ("UPDATE orders SET status = 'cancelled' WHERE order_id = ?", (order_id,)),
            ("UPDATE inventory SET on_hand = on_hand + ? WHERE sku = ?", (order["qty"], order["sku"])),
        ]
        return self._write("order_id", order_id, deltas, {"status": "success", "order_id": order_id,
                                                          "order_status": "cancelled"}, irreversible=True)

    def reconcile_order(self, order_id: str) -> dict[str, Any]:
        """Re-sync an order's status from the payments ledger (marks it refunded if a refund exists)."""
        deltas: list[Delta] = [(
            "UPDATE orders SET status = 'refunded' WHERE order_id = ? "
            "AND EXISTS (SELECT 1 FROM refunds WHERE order_id = ?)", (order_id, order_id))]
        return self._write("order_id", order_id, deltas, {"status": "success", "order_id": order_id},
                           irreversible=False)

    def restock_item(self, sku: str, qty: int) -> dict[str, Any]:
        """Add qty units of a SKU back to inventory."""
        deltas: list[Delta] = [("UPDATE inventory SET on_hand = on_hand + ? WHERE sku = ?", (int(qty), sku))]
        return self._write("sku", sku, deltas, {"status": "success", "sku": sku}, irreversible=True)

    def update_ticket(self, ticket_id: str, status: str) -> dict[str, Any]:
        """Set a ticket's status (open, pending, resolved)."""
        deltas: list[Delta] = [("UPDATE tickets SET status = ? WHERE ticket_id = ?", (status, ticket_id))]
        return self._write("ticket_id", ticket_id, deltas, {"status": "success", "ticket_id": ticket_id,
                                                            "ticket_status": status}, irreversible=False)

    def add_audit_note(self, ticket_id: str, text: str) -> dict[str, Any]:
        """Append an audit note to a ticket describing what was done."""
        deltas: list[Delta] = [("INSERT INTO audit_notes VALUES (?, ?, ?, ?)",
                                (self._next_id("N"), ticket_id, text, self._next_seq()))]
        return self._write("ticket_id", ticket_id, deltas, {"status": "success", "ticket_id": ticket_id},
                           irreversible=False)

    def close_ticket(self, ticket_id: str) -> dict[str, Any]:
        """Close a support ticket."""
        deltas: list[Delta] = [("UPDATE tickets SET status = 'closed' WHERE ticket_id = ?", (ticket_id,))]
        return self._write("ticket_id", ticket_id, deltas, {"status": "success", "ticket_id": ticket_id,
                                                            "ticket_status": "closed"}, irreversible=False)

    def escalate_to_human(self, ticket_id: str, reason: str) -> dict[str, Any]:
        """Hand a ticket to a human agent (use when policy requires approval or forbids the action)."""
        deltas: list[Delta] = [("INSERT INTO escalations VALUES (?, ?, ?, ?)",
                                (self._next_id("E"), ticket_id, reason, self._next_seq()))]
        return self._write("ticket_id", ticket_id, deltas, {"status": "success", "ticket_id": ticket_id,
                                                            "escalated": True}, irreversible=False)

    # ------------------------------------------------------------------ terminal
    def submit_report(self, status: str, summary: str) -> dict[str, Any]:
        """Finish the task. status must be completed, failed or escalated."""
        if status not in REPORT_STATUSES:
            return {"status": "error", "error": "invalid_status", "allowed": list(REPORT_STATUSES)}
        self.reports.append({"status": status, "summary": summary})
        return {"status": "recorded"}
