"""Amendment 5 environment: operation lifecycle, evidence freshness and external actors (draft; not registered).

`EcomEnvA5` subclasses the hash-registered `EcomEnv` (unchanged). What it adds:

* A logical clock. Every tool call costs `CALL_SECONDS`; `wait(seconds)` advances the clock. Pending operations
  commit when the clock passes their commit time.
* An operation registry: every write, by the agent or by an external actor, gets an `op_id`, an `actor`, a
  lifecycle (`pending` -> `committed` | `failed`) and a commit time. It is the ground-truth provenance ledger.
* Write outcomes per (tool, k) from `task.meta["a5"]["faults"]`:
    ok            commits now; success ack with op_id
    partial       only the first sub-effect commits (refund row, order status unchanged); success ack
    phantom       nothing commits; success ack (the lie); the receipt service says `failed`
    lost_ack      commits now; the agent sees HTTP 500 with request_id = op_id
    late_commit   pending; commits `delay` seconds later; HTTP 500 with request_id
    never_commit  pending forever (observation-equivalent to a late commit past the horizon)
* Evidence tools, each with a different information profile:
    lookup_order      fresh order status; no count, no actor
    query_ledger      refund rows with counts; may LAG (`as_of` is shown); no actor
    get_operation     one operation's lifecycle status, actor and commit time; fresh; needs the op_id
    list_operations   every operation on an entity (provenance / audit events); fresh
    wait              advances the clock (pending operations may commit)
* External actors (`meta["a5"]["external"]`): writes committed by another actor ("billing_bot") at a scheduled
  time. At time 0 they model a pre-satisfied goal.
* `submit_report(status, summary, certificate)`. The status may be "uncertain". The certificate is checked by
  `verifier_a5`; it is never trusted because the model produced it.

Trust assumption: the operation registry is the environment's own ledger. It is complete and truthful, and the
agent cannot write to it except by calling tools. `get_operation` / `list_operations` report it faithfully and
without lag. Only acks and `query_ledger` can be misleading.
"""
from __future__ import annotations

from typing import Any

from actionproof.env.db import Delta, apply_deltas, entity_hash, snapshot
from actionproof.env.faults import FaultSchedule
from actionproof.env.ledger import Attempt, Receipt
from actionproof.env.tools_ecom import TOOL_META, VERIFICATION_TOOLS, EcomEnv

CALL_SECONDS = 5
MAX_WAIT = 300
INFLIGHT_BOUND = 120  # contract for tools in task.meta["a5"]["bounded_tools"]: requests settle within this bound
A5_REPORT_STATUSES = ("completed", "failed", "escalated", "uncertain")
A5_META: dict[str, tuple[str | None, str]] = {
    **{k: v for k, v in TOOL_META.items() if k != "submit_report"},
    "get_operation": (None, "read"),
    "list_operations": (None, "read"),
    "wait": (None, "read"),
    "submit_report": TOOL_META["submit_report"],
}
A5_EVIDENCE_TOOLS = VERIFICATION_TOOLS | {"get_operation", "list_operations"}


class EcomEnvA5(EcomEnv):
    META = A5_META

    def __init__(self, initial_state: dict[str, list[dict[str, Any]]], schedule: FaultSchedule, rollout_id: str,
                 seed: int = 0, stale_status: dict[str, str] | None = None,
                 ambiguous_text: str = "Request accepted for processing.", a5: dict[str, Any] | None = None) -> None:
        super().__init__(initial_state, schedule, rollout_id, seed, stale_status, ambiguous_text)
        a5 = a5 or {}
        self.clock = 0
        self.faults5: dict[str, dict[str, Any]] = a5.get("faults", {})
        self.bounded = set(a5.get("bounded_tools", []))   # tools whose in-flight time is bounded (documented)
        self.ledger_lag = int(a5.get("ledger_lag", 0))
        self.external = sorted(a5.get("external", []), key=lambda e: e["at"])
        self.ops: dict[str, dict[str, Any]] = {}
        self.idem5: dict[str, str] = {}                  # idempotency key -> op_id (pending or committed only)
        self._apply_due()

    # ------------------------------------------------------------------ time
    def call(self, tool: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        self._apply_due()
        t0 = self.clock
        obs = super().call(tool, args)
        self.trace[-1]["t"] = t0
        if not (tool == "submit_report" and self.reports):
            self.clock += CALL_SECONDS
        self._apply_due()
        return obs

    def _apply_due(self) -> None:
        for e in [e for e in self.external if e["at"] <= self.clock and not e.get("_done")]:
            e["_done"] = True
            self._external_write(e)
        for op in self.ops.values():
            if op["status"] == "pending" and op["commit_at"] is not None and op["commit_at"] <= self.clock:
                self._commit(op)

    # ------------------------------------------------------------------ operations
    def _commit(self, op: dict[str, Any]) -> None:
        pre = entity_hash(snapshot(self.conn), op["entity_key"], op["entity"])
        apply_deltas(self.conn, op["deltas"])
        post = entity_hash(snapshot(self.conn), op["entity_key"], op["entity"])
        self.log.append(Receipt(rollout_id=self.log.rollout_id, action_idx=op["action_idx"], op=op["tool"],
                                entity=op["entity"], pre_hash=pre, post_hash=post, txn_id=op["op_id"],
                                policy_valid=op["policy_valid"], idempotency_key=op["idempotency_key"],
                                deltas=tuple(op["deltas"]), irreversible=op["irreversible"]))
        op["status"], op["committed_at"] = "committed", self.clock

    def _new_op(self, tool, entity_key, entity, deltas, irreversible, policy_valid, key, actor, action_idx):
        op_id = self._next_id("OP")
        op = {"op_id": op_id, "tool": tool, "entity_key": entity_key, "entity": entity, "actor": actor,
              "accepted_at": self.clock, "status": "pending", "commit_at": None, "committed_at": None,
              "deltas": list(deltas), "irreversible": irreversible, "policy_valid": policy_valid,
              "idempotency_key": key, "action_idx": action_idx,
              "refund_id": deltas[0][1][0] if tool == "issue_refund" else None}
        self.ops[op_id] = op
        return op

    def _external_write(self, e: dict[str, Any]) -> None:
        """Another actor performs a write (e.g. an automated billing job issues the refund)."""
        order = self._row("SELECT * FROM orders WHERE order_id = ?", e["order_id"])
        rid = f"R-{e.get('refund_tag', 'EXT')}{len(self.ops):03d}"
        deltas: list[Delta] = [
            ("INSERT INTO refunds VALUES (?, ?, ?, ?, 'completed', ?, ?)",
             (rid, e["order_id"], order["amount_cents"], order["payment_method"], None, self._next_seq())),
            ("UPDATE orders SET status = 'refunded' WHERE order_id = ?", (e["order_id"],))]
        op = self._new_op("issue_refund", "order_id", e["order_id"], deltas, True, True, None, e["actor"], -1)
        self._commit(op)

    def _write(self, entity_key: str, entity: str, deltas: list[Delta], success_obs: dict[str, Any], *,
               irreversible: bool, policy_valid: bool = True, idempotency_key: str | None = None) -> dict[str, Any]:
        action_idx, tool, k = self._ctx
        if idempotency_key and idempotency_key in self.idem5:  # exactly-once: same key -> same operation
            op = self.ops[self.idem5[idempotency_key]]
            if op["status"] == "committed":
                return {**op["success_obs"], "op_id": op["op_id"]}
            return {"status": "accepted", "op_id": op["op_id"], "detail": "request with this key is still in flight"}
        f = self.faults5.get(f"{tool}:{k}", {"kind": "ok"})
        op = self._new_op(tool, entity_key, entity, deltas, irreversible, policy_valid, idempotency_key, "agent",
                          action_idx)
        op["success_obs"] = success_obs
        kind = f["kind"]
        if kind == "phantom":
            op["status"] = "failed"
            obs = {**success_obs, "op_id": op["op_id"]}
        elif kind == "partial":  # only the first sub-effect lands (refund row; order status not updated)
            op["deltas"] = op["deltas"][:1]
            self._commit(op)
            obs = {**success_obs, "op_id": op["op_id"]}
        elif kind in ("ok", "lost_ack"):
            self._commit(op)
            obs = {**success_obs, "op_id": op["op_id"]} if kind == "ok" else \
                {"status": "error", "error": "http_500", "detail": "internal server error", "request_id": op["op_id"]}
        elif kind in ("late_commit", "never_commit"):
            op["commit_at"] = self.clock + int(f["delay"]) if kind == "late_commit" else None
            obs = {"status": "error", "error": "http_500", "detail": "internal server error", "request_id": op["op_id"]}
        else:
            raise ValueError(f"unknown A5 fault kind {kind!r}")
        if idempotency_key and op["status"] in ("pending", "committed"):
            self.idem5[idempotency_key] = op["op_id"]
        self.log.attempts.append(Attempt(action_idx, tool, entity, None if kind == "ok" else kind,
                                         op["status"] == "committed", obs))
        return obs

    # ------------------------------------------------------------------ evidence tools
    def query_ledger(self, order_id: str) -> dict[str, Any]:
        """Return refunds recorded in the payments ledger for an order. The ledger is replicated and may lag:
        `as_of` is the time the returned view reflects; `now` is the current time.

        Args:
            order_id: Order id.
        """
        as_of = max(0, self.clock - self.ledger_lag)
        late = {op["refund_id"] for op in self.ops.values() if op["refund_id"] and op["status"] == "committed"
                and op["committed_at"] is not None and op["committed_at"] > as_of}
        rows = self.conn.execute(
            "SELECT refund_id, amount_cents, method, status FROM refunds WHERE order_id = ? ORDER BY seq",
            (order_id,)).fetchall()
        return {"status": "ok", "order_id": order_id, "refunds": [dict(r) for r in rows if r["refund_id"] not in late],
                "as_of": as_of, "now": self.clock}

    def _op_view(self, op: dict[str, Any]) -> dict[str, Any]:
        return {"op_id": op["op_id"], "tool": op["tool"], "entity": op["entity"], "actor": op["actor"],
                "status": op["status"], "accepted_at": op["accepted_at"], "committed_at": op["committed_at"],
                "settle_bound_s": INFLIGHT_BOUND if op["tool"] in self.bounded else None,
                **({"refund_id": op["refund_id"]} if op["refund_id"] and op["status"] == "committed" else {})}

    def get_operation(self, op_id: str) -> dict[str, Any]:
        """Return one operation's lifecycle from the operations service (authoritative, no lag): status is
        pending, committed or failed. `settle_bound_s` is the documented maximum in-flight time (null = no bound).

        Args:
            op_id: Operation id (op_id from a success response, or request_id from an error).
        """
        op = self.ops.get(op_id)
        return {"status": "ok", "operation": self._op_view(op), "now": self.clock} if op else \
            {"status": "error", "error": "not_found"}

    def list_operations(self, entity_id: str) -> dict[str, Any]:
        """List every operation recorded on an order or ticket, by any actor (audit log; authoritative, no lag).

        Args:
            entity_id: Order id or ticket id.
        """
        return {"status": "ok", "entity_id": entity_id, "now": self.clock,
                "operations": [self._op_view(o) for o in self.ops.values() if o["entity"] == entity_id]}

    def wait(self, seconds: int) -> dict[str, Any]:
        """Wait before the next call, e.g. for an in-flight request to settle (at most 300 seconds).

        Args:
            seconds: How long to wait.
        """
        self.clock += max(0, min(int(seconds), MAX_WAIT))
        self._apply_due()
        return {"status": "ok", "now": self.clock}

    # ------------------------------------------------------------------ terminal
    def submit_report(self, status: str, summary: str, certificate: dict | None = None) -> dict[str, Any]:
        """Finish the task. Call exactly once, at the end.

        Args:
            status: completed, failed, escalated or uncertain (an outcome you cannot determine).
            summary: One or two sentences on what was done.
            certificate: {"effects": [{"op_id", "tool", "entity", "actor", "state"}], "evidence": [indices of the
                tool calls, counted from 0, whose results support each effect], "unresolved": [op_ids whose outcome
                is unknown]}.
        """
        if status not in A5_REPORT_STATUSES:
            return {"status": "error", "error": "invalid_status", "allowed": list(A5_REPORT_STATUSES)}
        self.reports.append({"status": status, "summary": summary, "certificate": certificate,
                             "time": self.clock, "trace_len": len(self.trace)})
        return {"status": "recorded"}

    # ------------------------------------------------------------------ ground truth
    def settle(self) -> None:
        """Hidden horizon: apply every eventually-committing operation (used only by the verifier, after the report)."""
        for op in self.ops.values():
            if op["status"] == "pending" and op["commit_at"] is not None:
                self.clock = max(self.clock, op["commit_at"])
                self._commit(op)
