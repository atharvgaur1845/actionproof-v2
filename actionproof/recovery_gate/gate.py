"""Recovery Gate (RG1): an evidence-only runtime controller between a tool-using agent and its tools.

Spec: docs/research/RECOVERY_GATE_PROTOCOL.md. The gate sees only what the agent sees:
- the task's stated intent (PO id, approved amount, quantity, requested steps; all written in the task prompt);
- each call's name, arguments and reply.
It never receives the environment. Calls go through an `execute(tool, args, tag)` callback, so simulator state,
fault plans, evaluator labels and trace metadata are out of its reach (tests/recovery_gate/test_leakage.py).

Rules (each can be switched off for the ablations):
  verify    A "completed" report needs, for every requested step, a read after the step's last write that shows the
            step in effect. The gate makes the missing reads. A read that does not show it rejects the report, with
            the record attached; after `max_rejections` rejections the gate hands the PO to a human.
  retry     A second write for a step is sent only after a read of that step's record since the last write; the gate
            makes that read in place of the write. A write for a step that the current record shows in effect is
            not sent.
  fresh     Reads are served from the primary copy (`consistent=true`), so stale replica views never count as
            evidence. Off: reads go as the agent sent them and the gate's own reads use the default (replica).
  repair    A partial receipt is topped up by the missing quantity only. A partial payment is refunded before a
            full payment is sent. A refund needs a current record showing that the payment is partial or an extra;
            the only correct live payment is never refunded. A key already acknowledged is not reused for a repay.
  escalate  Once a step was written and its record cannot be read, the step is not written again, and a report
            other than "escalated" is replaced by a hand-off that carries the evidence. An agent that stops without
            a report is finalised: verified completion if every step reads in effect, otherwise a hand-off.
Intent binding is always on: a write must name the task's PO, the approved amount and a requested step.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Any, Callable

from actionproof.integrity.sim.agents import _observed
from actionproof.integrity.sim.env import STEP_READ, STEP_TOOL, TOOL_STEP

READ_STEP = {v: k for k, v in STEP_READ.items()}
NOT_SENT = frozenset({"invalid_arguments", "unknown_tool", "not_found", "episode_finished"})
Execute = Callable[[str, dict[str, Any], str], dict[str, Any]]


@dataclass(frozen=True)
class Rules:
    verify: bool = True
    retry: bool = True
    fresh: bool = True
    repair: bool = True
    escalate: bool = True
    max_rejections: int = 2


ABLATIONS: dict[str, Rules] = {
    "full": Rules(),
    "no_verify": Rules(verify=False),
    "no_retry": Rules(retry=False),
    "no_fresh": Rules(fresh=False),
    "no_repair": Rules(repair=False),
    "no_escalate": Rules(escalate=False),
}


@dataclass(frozen=True)
class Intent:
    po_id: str
    steps: tuple[str, ...]
    amount_cents: int
    qty: int
    sku: str

    def po(self) -> dict[str, Any]:
        return {"po_id": self.po_id, "amount_cents": self.amount_cents, "qty": self.qty, "sku": self.sku}


def intent_of(task: Any) -> Intent:
    """The intent the task prompt states (tests check that each value appears in the prompt)."""
    m = task.meta
    po = next(p for p in task.initial_state["pos"] if p["po_id"] == m["po_id"])
    return Intent(po["po_id"], tuple(m["requested"]), int(po["amount_cents"]), int(po["qty"]), str(po["sku"]))


@dataclass
class StepView:
    writes: list[dict[str, Any]] = field(default_factory=list)   # {"n", "key", "ack"} for requests that were sent
    read: dict[str, Any] | None = None                           # latest read since the last write or refund
    refunded: bool = False                                       # a refund changed this record since the last write


def _ack(obs: dict[str, Any]) -> str:
    return "ok" if obs.get("status") == "ok" else str(obs.get("error") or obs.get("status"))


def _live(obs: dict[str, Any]) -> list[dict[str, Any]]:
    pays = obs.get("payments") or []
    refunded = {p.get("refund_of") for p in pays if p.get("kind") == "refund"}
    return [p for p in pays if p.get("kind") == "payment" and p.get("payment_id") not in refunded]


def _net_received(obs: dict[str, Any]) -> int:
    return sum(r["qty"] if r.get("kind") == "receipt" else -r["qty"] for r in obs.get("receipts") or [])


class Gate:
    def __init__(self, intent: Intent, rules: Rules = Rules(), name: Callable[[str], str] = lambda s: s) -> None:
        self.intent, self.rules, self.N = intent, rules, name
        self.steps = {s: StepView() for s in STEP_TOOL}
        self.n = 0
        self.rejections = 0
        self.agent_escalated = False
        self.finished = False
        self.log: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ evidence bookkeeping (visible data only)
    def observe(self, tool: str, args: dict[str, Any], obs: dict[str, Any]) -> None:
        self.n += 1
        if args.get("po_id") not in (None, self.intent.po_id):
            return
        if tool in TOOL_STEP:
            if obs.get("error") in NOT_SENT:
                return
            st = self.steps[TOOL_STEP[tool]]
            st.writes.append({"n": self.n, "key": args.get("idempotency_key"), "ack": _ack(obs)})
            st.read, st.refunded = None, False
        elif tool == "refund_payment":
            if obs.get("error") in NOT_SENT:
                return
            st = self.steps["pay"]
            st.read, st.refunded = None, True
        elif tool in READ_STEP:
            if obs.get("error") == "service_unavailable":
                seen = "unavailable"
            elif obs.get("status") == "ok":
                try:
                    seen = _observed(READ_STEP[tool], obs, self.intent.po())
                except (KeyError, TypeError):
                    return
            else:
                return
            self.steps[READ_STEP[tool]].read = {"n": self.n, "fresh": args.get("consistent") is True, "seen": seen,
                                                "obs": obs}
        elif tool == "escalate_to_human" and obs.get("status") == "ok":
            self.agent_escalated = True
        elif tool == "submit_report" and obs.get("status") == "ok":
            self.finished = True

    def _confirmed(self, step: str) -> bool:
        r = self.steps[step].read
        return r is not None and r["seen"] == "held"

    def _unresolvable(self, step: str) -> bool:
        st = self.steps[step]
        return bool(st.writes) and st.read is not None and st.read["seen"] == "unavailable"

    # ------------------------------------------------------------------ helpers
    def _run(self, execute: Execute, tool: str, args: dict[str, Any], tag: str) -> dict[str, Any]:
        obs = execute(tool, args, tag)
        self.observe(tool, args, obs)
        return obs

    def _read_args(self, step: str) -> dict[str, Any]:
        a: dict[str, Any] = {"po_id": self.intent.po_id}
        if self.rules.fresh:
            a["consistent"] = True
        return a

    def _note(self, rule: str, action: str, proposed: tuple[str, dict[str, Any]] | None, detail: str) -> None:
        self.log.append({"n": self.n, "rule": rule, "action": action, "detail": detail,
                         "proposed": None if proposed is None else {"tool": proposed[0], "args": proposed[1]}})

    def _block(self, rule: str, tool: str, args: dict[str, Any], detail: str) -> dict[str, Any]:
        self._note(rule, "block", (tool, args), detail)
        return {"status": "error", "error": "blocked_by_gate", "rule": rule, "detail": detail}

    def _bundle(self) -> dict[str, Any]:
        out = {}
        for s in self.intent.steps:
            st = self.steps[s]
            out[s] = {"requests": [f"{w['key'] or 'no key'}: {w['ack']}" for w in st.writes],
                      "record": st.read["seen"] if st.read else "not read"}
        return out

    def _escalate(self, execute: Execute, why: str, proposed: tuple[str, dict[str, Any]] | None) -> dict[str, Any]:
        self._note("escalate", "hand_off", proposed, why)
        reason = f"Recovery gate: {why} Evidence: {json.dumps(self._bundle(), sort_keys=True)}"
        if not self.agent_escalated:
            self._run(execute, "escalate_to_human", {"po_id": self.intent.po_id, "reason": reason}, "gate_escalate")
        self._run(execute, "submit_report", {"status": "escalated", "summary": f"{self.intent.po_id} handed to a "
                                             f"human operator: {why}"}, "gate_escalate")
        return {"status": "ok", "gate": {"rule": "escalate", "detail": f"The report was replaced by a hand-off: {why}"}}

    # ------------------------------------------------------------------ the decision on one agent call
    def handle(self, tool: str, args: dict[str, Any], execute: Execute) -> dict[str, Any]:
        """Process one call the agent proposed; returns the reply the agent sees."""
        if self.finished:
            return execute(tool, args, "agent")
        if tool in READ_STEP or tool == "get_purchase_order":
            if self.rules.fresh and args.get("consistent") is not True:
                args = {**args, "consistent": True}
            return self._run(execute, tool, args, "agent")
        if tool in TOOL_STEP:
            return self._write(tool, args, execute)
        if tool == "refund_payment":
            return self._refund(args, execute)
        if tool == "submit_report":
            return self._report(args, execute)
        return self._run(execute, tool, args, "agent")

    def _write(self, tool: str, args: dict[str, Any], execute: Execute) -> dict[str, Any]:
        I, R, N = self.intent, self.rules, self.N
        step = TOOL_STEP[tool]
        st = self.steps[step]
        rd = N(STEP_READ[step])
        if args.get("po_id") != I.po_id:
            return self._block("intent", tool, args, f"This task concerns {I.po_id} only; the request names "
                                                     f"{args.get('po_id')!r}. Nothing was sent.")
        if step not in I.steps:
            return self._block("intent", tool, args, f"The task does not ask for {N(tool)}. Nothing was sent.")
        if step in ("pay", "post") and args.get("amount_cents") != I.amount_cents:
            return self._block("intent", tool, args, f"The approved amount is {I.amount_cents} cents; the request "
                                                     f"says {args.get('amount_cents')!r}. Nothing was sent.")
        if R.escalate and self._unresolvable(step):
            return self._block("escalate", tool, args,
                               f"{N(tool)} was already sent (reply: {st.writes[-1]['ack']}) and {rd} is unavailable, "
                               f"so its effect cannot be confirmed. Do not send it again; hand the purchase order to "
                               f"a human with {N('escalate_to_human')}.")
        if R.retry and st.writes:
            if st.read is None:
                obs = self._run(execute, STEP_READ[step], self._read_args(step), "gate_read")
                detail = (f"{N(tool)} was already sent for {I.po_id} (last reply: {st.writes[-1]['ack']}), and its "
                          f"record was not read since. The gate read {rd} instead of sending the request again. "
                          f"Decide from this record.")
                self._note("retry", "read_instead", (tool, args), detail)
                return {"status": "error", "error": "not_sent_by_gate", "rule": "read_before_retry", "detail": detail,
                        "record": [{"tool": STEP_READ[step], "result": obs}]}
            if st.read["seen"] == "held":
                return self._block("retry", tool, args, f"{rd} (call {st.read['n']}) shows this step in effect. "
                                                        f"{N(tool)} was not sent again.")
        if R.repair:
            seen = st.read["seen"] if st.read else None
            if step == "receive":
                net = _net_received(st.read["obs"]) if st.read and seen in ("held", "absent", "partial", "other") else 0
                need = I.qty - net
                if need <= 0:
                    return self._block("repair", tool, args, f"{rd} shows {net} of {I.qty} received; nothing is "
                                                             f"missing. Nothing was sent.")
                if args.get("qty") != need:
                    return self._block("repair", tool, args, f"Net goods received so far: {net} of {I.qty}. A receipt "
                                                             f"must be for the missing {need}. Nothing was sent.")
            if step == "pay" and seen == "partial":
                p = _live(st.read["obs"])[0]
                return self._block("repair", tool, args,
                                   f"Payment {p.get('payment_id')} of {p.get('amount_cents')} cents is live and is not "
                                   f"the approved amount. Refund it with {N('refund_payment')} before paying the full "
                                   f"amount. Nothing was sent.")
            if step == "pay" and seen == "other":
                return self._block("repair", tool, args, f"{rd} shows {len(_live(st.read['obs']))} live payments. "
                                                         f"Refund the extra ones; do not pay again. Nothing was sent.")
            if step == "pay" and st.refunded and args.get("idempotency_key") is not None and any(
                    w["key"] == args.get("idempotency_key") and w["ack"] == "ok" for w in st.writes):
                return self._block("repair", tool, args,
                                   f"The key {args.get('idempotency_key')!r} was already acknowledged; a request with "
                                   f"the same key is answered from that record and pays nothing. Use a new key for "
                                   f"the replacement payment. Nothing was sent.")
        return self._run(execute, tool, args, "agent")

    def _refund(self, args: dict[str, Any], execute: Execute) -> dict[str, Any]:
        if not self.rules.repair:
            return self._run(execute, "refund_payment", args, "agent")
        st, N = self.steps["pay"], self.N
        if st.read is None or st.read["seen"] == "unavailable":
            obs = self._run(execute, "list_payments", self._read_args("pay"), "gate_read")
            if st.read is None or st.read["seen"] == "unavailable":
                return self._block("repair", "refund_payment", args, f"{N('list_payments')} could not be read, so "
                                                                     f"no refund is safe. Nothing was sent.")
            if obs.get("status") != "ok":
                return self._block("repair", "refund_payment", args, "The payment record could not be read.")
        live = _live(st.read["obs"])
        pid = args.get("payment_id")
        target = next((p for p in live if p.get("payment_id") == pid), None)
        if target is None:
            return self._block("repair", "refund_payment", args, f"{pid!r} is not a live payment of {self.intent.po_id} "
                                                                 f"in {N('list_payments')} (call {st.read['n']}). "
                                                                 f"Nothing was sent.")
        full = [p for p in live if p.get("amount_cents") == self.intent.amount_cents]
        if target in full and len(full) == 1:
            return self._block("repair", "refund_payment", args,
                               f"{pid} is the only live payment of the approved amount; refunding it would reverse a "
                               f"correct payment. Nothing was sent.")
        return self._run(execute, "refund_payment", args, "agent")

    def _verify(self, execute: Execute) -> dict[str, str]:
        """Reads every requested step that is not confirmed; returns {step: what its record shows} for the failures."""
        for s in self.intent.steps:
            if not self._confirmed(s):
                self._run(execute, STEP_READ[s], self._read_args(s), "gate_verify")
        return {s: (self.steps[s].read or {}).get("seen", "not read") for s in self.intent.steps if not self._confirmed(s)}

    def _report(self, args: dict[str, Any], execute: Execute) -> dict[str, Any]:
        R, N = self.rules, self.N
        status = args.get("status")
        stuck = [s for s in self.intent.steps if self._unresolvable(s)]
        if R.escalate and stuck and status != "escalated":
            return self._escalate(execute, f"the effect of {', '.join(N(STEP_TOOL[s]) for s in stuck)} cannot be "
                                           f"confirmed: {', '.join(N(STEP_READ[s]) for s in stuck)} is unavailable.",
                                  ("submit_report", args))
        if status == "failed" and R.escalate:
            # a failure claim about a request whose effect was never read: read it; unreadable means hand-off
            for s in self.intent.steps:
                if self.steps[s].writes and self.steps[s].read is None:
                    self._run(execute, STEP_READ[s], self._read_args(s), "gate_verify")
            stuck = [s for s in self.intent.steps if self._unresolvable(s)]
            if stuck:
                return self._escalate(execute, f"the effect of {', '.join(N(STEP_TOOL[s]) for s in stuck)} cannot "
                                               f"be confirmed: {', '.join(N(STEP_READ[s]) for s in stuck)} is "
                                               f"unavailable.", ("submit_report", args))
        if status != "completed" or not R.verify:
            return self._run(execute, "submit_report", args, "agent")
        bad = self._verify(execute)
        if not bad:
            return self._run(execute, "submit_report", args, "agent")
        what = "; ".join(f"{N(STEP_READ[s])} shows {v}" for s, v in bad.items())
        if R.escalate and any(v == "unavailable" for v in bad.values()):
            return self._escalate(execute, f"completion cannot be confirmed ({what}).", ("submit_report", args))
        self.rejections += 1
        if R.escalate and self.rejections > R.max_rejections:
            return self._escalate(execute, f"completion was not confirmed after {R.max_rejections} rejected reports "
                                           f"({what}).", ("submit_report", args))
        detail = (f"Report not accepted: the system of record does not show every requested step in effect ({what}). "
                  f"Repair the missing step, or escalate.")
        self._note("verify", "reject_report", ("submit_report", args), detail)
        return {"status": "error", "error": "report_rejected_by_gate", "rule": "verify", "detail": detail,
                "record": [{"tool": STEP_READ[s], "result": self.steps[s].read["obs"]} for s in bad
                           if self.steps[s].read is not None]}

    # ------------------------------------------------------------------ the agent stopped without a report
    def finalize(self, execute: Execute) -> None:
        if self.finished:
            return
        R = self.rules
        if R.verify:
            bad = self._verify(execute)
            if not bad:
                self._note("verify", "finalize_completed", None, "the agent stopped without a report; every "
                                                                 "requested step reads in effect")
                self._run(execute, "submit_report", {"status": "completed", "summary": f"{self.intent.po_id}: every "
                          "requested step confirmed in its system of record (finalised by the recovery gate)"},
                          "gate_finalize")
                return
        if R.escalate:
            self._escalate(execute, "the agent stopped without a report and completion is not confirmed.", None)


def rules_of(name: str) -> Rules:
    if name not in ABLATIONS:
        raise SystemExit(f"unknown gate rules {name!r}; one of {sorted(ABLATIONS)}")
    return ABLATIONS[name]


__all__ = ["Gate", "Rules", "Intent", "ABLATIONS", "intent_of", "rules_of", "replace"]
