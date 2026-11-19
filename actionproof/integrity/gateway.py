"""ActionProof execution gateway for ProcureSim: interception, fail-closed policy, verification against the system of
record, recovery, and an evidence ledger. Modes (identical in the product and in the experiments):

  passthrough      no gateway (experiment baseline only)
  observe          verifies and logs every write; never changes execution or what the agent sees
  enforce          + fail-closed grants and limits, commit points between steps, idempotent replay of verified steps,
                   recovery (safe retry, compensation, escalation); blocked calls get an in-schema refusal with no detail
  enforce_inform   + the agent's acknowledgement is replaced, in schema, by the gateway's verdict and evidence

The gateway's verifier reads the services' authoritative state through its own read path (`StateProbe`), never the
agent's acknowledgements or text, and shares no code with `sim/truth.py` (the scorer). It guarantees no false
finalization (it never finalizes success without fresh evidence for every requested step) and, under the stated
provider assumptions (keys scoped to (op, entity), retained for the transaction, no concurrent in-flight call with the
same key), at most one committed irreversible effect per (op, entity, transaction) that it caused itself. It gives no
exactly-once guarantee, no atomicity across services, and cannot detect a system of record that lies about itself.
"""
from __future__ import annotations

import copy
import uuid
from typing import Any

from actionproof.integrity.contracts import PREREQUISITES, STEP_WRITE, Contract
from actionproof.integrity.ledger import EvidenceLedger

MODES = ("passthrough", "observe", "enforce", "enforce_inform")
WRITE_STEP = {v: k for k, v in STEP_WRITE.items()}
STEP_SERVICE = {"order": "supplier", "pay": "payments", "receive": "inventory", "post": "accounting"}
REFUSED = {"status": "error", "error": "operation_not_permitted"}
IRREVERSIBLE = {"pay"}


class StateProbe:
    """The gateway's read path to the systems of record (ProcureSim adapter). Reads the primary copy, with the
    service version as the freshness token; it does not go through the agent's tools, counters or faults."""

    def __init__(self, env: Any) -> None:
        self.env = env

    def version(self, step: str) -> int:
        return self.env.version[STEP_SERVICE[step]]

    def read(self, step: str, exp: Any) -> dict[str, Any]:
        st, pid = self.env.state, exp.po_id
        if step == "order":
            rows = [dict(o) for o in st["supplier_orders"] if o["po_id"] == pid and o["status"] == "placed"]
            good = [o for o in rows if o["sku"] == exp.sku and o["qty"] == exp.qty]
            cls = ("absent" if not rows else "duplicate" if len(rows) > 1 else "held" if good
                   else "partial" if rows[0]["sku"] == exp.sku and 0 < rows[0]["qty"] < exp.qty else "wrong")
        elif step == "pay":
            refunded = {p["refund_of"] for p in st["payments"] if p["kind"] == "refund"}
            rows = [dict(p) for p in st["payments"]
                    if p["po_id"] == pid and p["kind"] == "payment" and p["payment_id"] not in refunded]
            cls = ("absent" if not rows else "duplicate" if len(rows) > 1 else
                   "held" if rows[0]["amount_cents"] == exp.amount_cents and rows[0]["payee_account"] == exp.payee_account
                   else "partial" if 0 < rows[0]["amount_cents"] < exp.amount_cents else "wrong")
        elif step == "receive":
            rows = [dict(r) for r in st["receipts"] if r["po_id"] == pid]
            net = sum(r["qty"] if r["kind"] == "receipt" else -r["qty"] for r in rows)
            cls = ("held" if net == exp.qty else "absent" if net == 0 else "partial" if 0 < net < exp.qty
                   else "duplicate" if net > exp.qty else "wrong")
        else:
            rows = [dict(j) for j in st["journal"] if j["po_id"] == pid]
            amt = sum(j["amount_cents"] if j["kind"] == "payment" else -j["amount_cents"] for j in rows)
            cnt = sum(1 if j["kind"] == "payment" else -1 for j in rows)
            cls = ("held" if amt == exp.amount_cents and cnt == 1 else "duplicate" if cnt >= 2 or amt > exp.amount_cents
                   else "absent" if amt == 0 else "partial" if 0 < amt < exp.amount_cents else "wrong")
        return {"class": cls, "rows": rows, "as_of_version": self.version(step)}


class ProcGateway:
    """Wraps a ProcEnv (duck-typed: call, tools, done, trace, final_report, state, attempts...)."""

    def __init__(self, env: Any, contract: Contract, mode: str = "enforce", ledger: EvidenceLedger | None = None,
                 tx_id: str | None = None) -> None:
        if mode not in MODES:
            raise ValueError(mode)
        self.env, self.contract, self.mode = env, contract, mode
        self.ledger = ledger or EvidenceLedger()
        self.tx = tx_id or f"tx-{uuid.uuid4().hex[:10]}"
        self.probe = StateProbe(env)
        self.step_state: dict[str, str] = {s: "pending" for s in contract.requested}
        self.verified_obs: dict[str, dict[str, Any]] = {}
        self.attempts: dict[str, int] = {s: 0 for s in contract.requested}
        self.escalated = False
        self.agent_view: list[dict[str, Any]] = []   # every agent call and the response the agent saw
        self.decisions: list[dict[str, Any]] = []
        self.ledger.append(self.tx, "contract_bound", "gateway",
                           {"contract_id": contract.contract_id, "content_hash": contract.content_hash(), "mode": mode,
                            "requested": list(contract.requested), "po_id": contract.expected.po_id})

    # ------------------------------------------------------------- environment interface (proxied)
    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def adapt_prompt(self, base: str) -> str:
        return self.env.adapt_prompt(base)

    @property
    def done(self) -> bool:
        return self.env.done

    def tools(self) -> list[Any]:
        import inspect
        tools = self.env.tools()  # same names, signatures and docstrings, routed through the gateway

        def make(src: Any) -> Any:
            def tool(**kwargs: Any) -> dict[str, Any]:
                return self.call(src.__name__, kwargs)
            tool.__name__ = tool.__qualname__ = src.__name__
            tool.__doc__ = src.__doc__
            tool.__signature__ = inspect.signature(src)  # type: ignore[attr-defined]
            tool.__annotations__ = dict(src.__annotations__)
            return tool
        return [make(t) for t in tools]

    # ------------------------------------------------------------- core
    def _log(self, kind: str, actor: str, body: dict[str, Any], op: str | None = None) -> None:
        self.ledger.append(self.tx, kind, actor, body, op_id=op)

    def _refuse(self, tool: str, args: dict[str, Any], reason: str) -> dict[str, Any]:
        self._log("policy_decision", "gateway", {"tool": tool, "args": args, "decision": "reject", "reason": reason})
        return dict(REFUSED)

    def _policy(self, tool: str, args: dict[str, Any]) -> str | None:
        """Fail-closed checks; returns a rejection reason or None."""
        c = self.contract
        if tool not in c.grants():
            return "tool_not_granted"
        po = args.get("po_id")
        if tool not in ("submit_report", "refund_payment") and po is not None and po != c.expected.po_id:
            return "entity_out_of_scope"
        amt = args.get("amount_cents")
        if isinstance(amt, int) and amt > c.limits.per_op_max_cents:
            return "amount_over_limit"
        step = WRITE_STEP.get(tool)
        if step:
            missing = [p for p in PREREQUISITES[step] if p in c.requested and self.step_state.get(p) != "verified"]
            if missing:
                return f"commit_point_unmet:{','.join(missing)}"
        if tool == "refund_payment":
            pay = next((p for p in self.env.state["payments"] if p["payment_id"] == args.get("payment_id")), None)
            if pay is None or pay["po_id"] != c.expected.po_id:
                return "entity_out_of_scope"
        return None

    def call(self, tool: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        args = dict(args or {})
        if self.mode == "passthrough":
            obs = self.env.call(tool, args)
            self.agent_view.append({"tool": tool, "args": args, "obs": copy.deepcopy(obs)})
            return obs
        self._log("call_intercepted", "agent", {"tool": tool, "args": args})
        enforce = self.mode in ("enforce", "enforce_inform")
        step = WRITE_STEP.get(tool)
        if enforce and not self.env.done:
            reason = self._policy(tool, args)
            if reason:
                obs = self._refuse(tool, args, reason)
                self.agent_view.append({"tool": tool, "args": args, "obs": obs, "refused": reason})
                return obs
            if step and self.step_state.get(step) == "verified":  # idempotent replay: never write a verified step again
                obs = copy.deepcopy(self.verified_obs[step])
                self._log("recovery_decision", "recovery", {"step": step, "action": "NOOP_ALREADY_COMMITTED"})
                self.agent_view.append({"tool": tool, "args": args, "obs": obs, "replayed": True})
                return self._inform(step, obs) if self.mode == "enforce_inform" else obs
        if tool == "submit_report":
            obs = self.env.call(tool, args)
            self._log("claim", "agent", {"report": args})
            self.finalize()
            self.agent_view.append({"tool": tool, "args": args, "obs": copy.deepcopy(obs)})
            return obs
        dispatch_v = self.probe.version(step) if step else None
        obs = self.env.call(tool, args)
        self.agent_view.append({"tool": tool, "args": args, "obs": copy.deepcopy(obs)})
        if not step or step not in self.contract.requested or obs.get("error") in ("invalid_arguments", "not_found",
                                                                                   "episode_finished", "unknown_tool"):
            return obs
        op = f"{self.tx}:{step}:{self.attempts[step]}"
        self.attempts[step] += 1
        self._log("dispatched", "gateway", {"step": step, "tool": tool, "args": args, "dispatch_version": dispatch_v}, op)
        self._log("ack_received", "external", {"step": step, "ack": obs, "treated_as": "claim"}, op)
        verdict = self._verify(step, op, dispatch_v)
        if enforce:
            verdict = self._recover(step, args, verdict)
            if verdict == "held":
                self.verified_obs[step] = obs if obs.get("status") == "ok" else {"status": "ok", "verified": True}
        return self._inform(step, obs) if self.mode == "enforce_inform" else obs

    def _verify(self, step: str, op: str | None, dispatch_v: int | None) -> str:
        ev = self.probe.read(step, self.contract.expected)
        fresh = dispatch_v is None or ev["as_of_version"] >= dispatch_v
        cls = ev["class"] if fresh else "stale"
        self._log("evidence", "verifier", {"step": step, "observed": ev["rows"], "as_of_version": ev["as_of_version"],
                                           "fresh": fresh}, op)
        state = {"held": "verified", "absent": "absent", "partial": "partial", "duplicate": "duplicate",
                 "wrong": "wrong", "stale": "in_doubt"}[cls]
        if self.step_state.get(step) != "escalated":
            self.step_state[step] = state
        self._log("verdict", "verifier", {"step": step, "state": state}, op)
        return cls

    def _harness(self, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        return self.env.call(tool, args, by="harness")

    def _decide(self, step: str, action: str, **kw: Any) -> None:
        d = {"step": step, "action": action, **kw}
        self.decisions.append(d)
        self._log("recovery_decision", "recovery", d)

    def _recover(self, step: str, args: dict[str, Any], cls: str) -> str:
        """Decision table (retry-safe and idempotent are different properties)."""
        c, exp = self.contract, self.contract.expected
        while cls != "held":
            if self.attempts[step] >= c.max_write_attempts and cls in ("absent", "in_doubt", "stale"):
                return self._escalate(step, "write never verified after retries")
            if cls == "absent":  # verified absent at a version after dispatch: a retry cannot duplicate
                key = args.get("idempotency_key") or f"ap-{self.tx}-{step}"
                retry = {k: v for k, v in args.items() if k != "idempotency_key"}
                if step != "receive":
                    retry["idempotency_key"] = key
                self._decide(step, "SAFE_RETRY", reuse_key=retry.get("idempotency_key"))
                dv = self.probe.version(step)
                self._harness(STEP_WRITE[step], retry)
                self.attempts[step] += 1
                cls = self._verify(step, None, dv)
            elif cls == "partial" and step == "pay" and c.allow_compensation:
                pay = self.probe.read("pay", exp)["rows"][0]
                self._decide(step, "COMPENSATE", refund_of=pay["payment_id"], amount_cents=pay["amount_cents"])
                r = self._harness("refund_payment", {"payment_id": pay["payment_id"]})
                if r.get("status") != "ok" or self.probe.read("pay", exp)["class"] != "absent":
                    return self._escalate(step, "compensation failed", residual_cents=pay["amount_cents"])
                args = {**args, "idempotency_key": f"ap-{self.tx}-pay-r{self.attempts[step]}"}
                cls = "absent"
            elif cls == "partial" and step == "receive":
                got = sum(r["qty"] if r["kind"] == "receipt" else -r["qty"]
                          for r in self.probe.read("receive", exp)["rows"])
                self._decide(step, "RECONCILE", record_missing_qty=exp.qty - got)
                dv = self.probe.version(step)
                self._harness("record_goods_receipt", {"po_id": exp.po_id, "qty": exp.qty - got})
                cls = self._verify(step, None, dv)
                if cls != "held":
                    return self._escalate(step, "reconciliation did not hold")
            elif cls == "duplicate" and step == "pay" and c.allow_compensation:
                extras = self.probe.read("pay", exp)["rows"][1:]
                for p in extras:
                    self._decide(step, "COMPENSATE", refund_of=p["payment_id"], amount_cents=p["amount_cents"])
                    self._harness("refund_payment", {"payment_id": p["payment_id"]})
                cls = self._verify(step, None, None)
                if cls != "held":
                    return self._escalate(step, "compensation failed",
                                          residual_cents=sum(p["amount_cents"] for p in extras))
            else:  # partial order/post, duplicate receipt/order/post, wrong: no safe automatic repair
                residual = sum(p["amount_cents"] for p in self.probe.read("pay", exp)["rows"]) if step == "pay" else 0
                return self._escalate(step, f"{cls} effect cannot be repaired automatically", residual_cents=residual)
        return "held"

    def _escalate(self, step: str, reason: str, residual_cents: int = 0) -> str:
        self._decide(step, "ESCALATE", reason=reason, residual_cents=residual_cents)
        if not self.escalated:
            self._harness("escalate_to_human", {"po_id": self.contract.expected.po_id, "reason": f"{step}: {reason}"})
        self.escalated = True
        self.step_state[step] = "escalated"
        return "escalated"

    def _inform(self, step: str, obs: dict[str, Any]) -> dict[str, Any]:
        """Enforce+Inform: the acknowledgement replaced in schema by the verdict and the authoritative rows."""
        ev = self.probe.read(step, self.contract.expected)
        st = self.step_state.get(step, "pending")
        if st == "verified":
            return {**{k: v for k, v in obs.items() if k != "error"}, "status": "ok", "verdict": "VERIFIED_SUCCESS",
                    "observed": ev["rows"]}
        return {"status": "error", "error": "not_committed" if st in ("absent", "escalated") else st,
                "verdict": st.upper(), "observed": ev["rows"]}

    # ------------------------------------------------------------- disposition
    def disposition(self) -> dict[str, Any]:
        for s in self.contract.requested:  # final check against the system of record
            if self.step_state.get(s) != "escalated":
                self._verify(s, None, None)
        all_ok = all(self.step_state[s] == "verified" for s in self.contract.requested)
        status = "completed" if all_ok else "escalated" if self.escalated else "failed"
        exp = self.contract.expected
        paid = self.probe.read("pay", exp)["rows"] if "pay" in self.contract.requested else []
        exposure = 0 if self.step_state.get("pay", "verified") == "verified" else sum(p["amount_cents"] for p in paid)
        agent = (self.env.final_report or {}).get("status")
        claim = ("SUPPORTED" if (agent == "completed") == all_ok else
                 "CONTRADICTED_CLAIM" if agent == "completed" else "UNDERCLAIM") if agent else "NO_CLAIM"
        return {"status": status, "steps": dict(self.step_state), "claim": claim, "exposure_cents": exposure,
                "escalated": self.escalated}

    def finalize(self) -> dict[str, Any]:
        d = self.disposition()
        self._log("finalized", "gateway", {**d, "ledger_head": self.ledger.head()[1]})
        return d
