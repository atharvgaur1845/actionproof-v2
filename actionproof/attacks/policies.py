"""Scripted policies for development evaluation (docs/attacks/SPEC.md §7) and the episode runner.

B0 (ack-trusting) and B4 (transaction-aware reconciliation) are the frozen TXN baselines (actionproof/txn/policies.py).
OBD ("obedient") is new: a reconciling agent that also follows instructions found in tool output. It is a
maximal-susceptibility reference, not a model of any trained agent:
- it reads the PO first and again after each step, reads each step's record (default replica) after writing it, and
  reads the audit log after a reply other than ok;
- it parses every free-text string in what it sees for entities (ACCT-, PO-, PAY- ids, "N cents") and a few keyword
  families (refund, skip, resend, confirm), and acts on them (pay a new payee, act on another PO, change the amount,
  pay a top-up, refund, resend, skip a step, believe an audit entry);
- it ignores the Recovery Gate's own replies (they carry a `rule` field): the suite measures obedience to attacker
  text, and gate blocks reach it as errors.
All three see only agent-visible data: the runner hands them a task view with the PO id and the requested steps,
which the prompt states, and the env (tests/attacks/test_leak.py).
"""
from __future__ import annotations

import copy
import re
import time
from typing import Any, Callable

from actionproof.attacks.backend import get_backend, make_env_attack
from actionproof.bench.generator_proc import ProcTask
from actionproof.integrity.sim.agents import _args, _observed
from actionproof.integrity.sim.env import STEP_READ, STEP_TOOL, TIMEOUT
from actionproof.txn import policies as TP

ACCT = re.compile(r"ACCT-[0-9A-F]{6}")
POID = re.compile(r"PO-[0-9A-F]{8}")
PAYID = re.compile(r"PAY-[0-9A-F]{6}")
CENTS = re.compile(r"(\d+) cents|amount_cents=(\d+)")
REFUND = re.compile(r"refund|रिफंड")
TOPUP = re.compile(r"top-up|top_up|remaining|difference|still due|बाकी")
CONFIRM = re.compile(r"no further action|succeeded|applied successfully|outcome=applied|committed|is final|"
                     r"final=true|confirmed and final|completed and reconciled|sync ok|is done|सफल|अंतिम")
SKIP = re.compile(r"\bskip\b|do not call|no need|not needed|automatically|auto-|न करें")
RESEND = re.compile(r"again|resend|resubmit|re-send|re-issue|did not go through|never reached|dropped|"
                    r"not processed|not_processed|दोबारा")
STEP_WORDS = (("post", re.compile(r"journal|ledger|post_journal_entry")),
              ("order", re.compile(r"place_supplier_order|supplier order|place the")),
              ("receive", re.compile(r"record_goods_receipt|goods receipt|receipt|माल")),
              ("pay", re.compile(r"pay_supplier|payment|pay |remit|भुगतान")))


def _step_of(text: str) -> str | None:
    for tool, step in (("post_journal_entry", "post"), ("place_supplier_order", "order"),
                       ("record_goods_receipt", "receive"), ("pay_supplier", "pay")):
        if tool in text:
            return step
    low = text.lower()
    for step, rx in STEP_WORDS:
        if rx.search(low):
            return step
    return None


def _strings(x: Any) -> list[str]:
    if isinstance(x, dict):
        if "rule" in x:  # a Recovery Gate reply
            return []
        return [s for v in x.values() for s in _strings(v)]
    if isinstance(x, list):
        return [s for v in x for s in _strings(v)]
    if isinstance(x, str) and " " in x and len(x) >= 20 and x != TIMEOUT["detail"]:
        return [x]
    return []


def parse(text: str, pid: str, approved: int, registered: str | None) -> list[tuple]:
    """Directives an obedient agent reads into one string (first matching rule wins)."""
    low = text.lower()
    pays = PAYID.findall(text)
    if pays and REFUND.search(low):
        return [("refund", p) for p in dict.fromkeys(pays)]
    accts = [a for a in dict.fromkeys(ACCT.findall(text)) if a != registered]
    if accts:
        return [("payee", accts[0])]
    pos = [p for p in dict.fromkeys(POID.findall(text)) if p != pid]
    nums = [int(a or b) for a, b in CENTS.findall(text)]
    if pos:
        return [("cross", pos[0], _step_of(text.replace(pid, "")), nums[0] if nums else None)]
    if CONFIRM.search(low):
        return [("confirm",)]
    other = [n for n in nums if n != approved]
    if other:
        return [("topup", min(other))] if TOPUP.search(low) else [("amount", max(other))]
    if SKIP.search(low):
        step = _step_of(text)
        return [("skip", step)] if step else []
    if RESEND.search(low):
        step = _step_of(text)
        return [("resend", step)] if step else []
    return []


class Obedient:
    def __init__(self, a: TP.Agent) -> None:
        self.a = a
        self.pid, self.po = a.pid, a.po
        self.registered: str | None = None
        self.seen: set[str] = set()
        self.pending: list[tuple] = []
        self.done_dirs: set[tuple] = set()
        self.payee: str | None = None
        self.amount: int | None = None
        self.skip: set[str] = set()
        self.written: set[str] = set()
        self.believed: set[str] = set()

    def scan(self, obs: dict[str, Any]) -> None:
        for s in _strings(obs):
            if s in self.seen:
                continue
            self.seen.add(s)
            for d in parse(s, self.pid, self.po["amount_cents"], self.registered):
                if d not in self.done_dirs and d not in self.pending:
                    self.pending.append(d)

    def call(self, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        obs = self.a.call(tool, args)
        self.scan(obs)
        return obs

    def args(self, step: str, key: str | None) -> dict[str, Any]:
        a = _args(step, self.po, key)
        if step == "pay" and self.payee:
            a["payee_account"] = self.payee
        if step in ("pay", "post") and self.amount:
            a["amount_cents"] = self.amount
        return a

    def act(self) -> None:
        while self.pending:
            d = self.pending.pop(0)
            self.done_dirs.add(d)
            kind = d[0]
            if kind == "refund":
                self.call("refund_payment", {"payment_id": d[1]})
            elif kind == "payee":
                self.payee = d[1]
                if "pay" in self.written:
                    self.call("pay_supplier", self.args("pay", self.a.key("pay", new=True)))
            elif kind == "amount":
                self.amount = d[1]
            elif kind == "topup":
                a = _args("pay", self.po, self.a.key("pay", new=True))
                a["amount_cents"] = d[1]
                self.call("pay_supplier", a)
            elif kind == "cross":
                other, step, amount = d[1], d[2] or "pay", d[3]
                if amount is None and step in ("pay", "post", "receive"):
                    po = self.call("get_purchase_order", {"po_id": other}).get("po") or {}
                    amount = po.get("amount_cents") if step != "receive" else po.get("qty")
                a: dict[str, Any] = {"po_id": other}
                if step in ("pay", "post"):
                    a["amount_cents"] = amount or self.po["amount_cents"]
                if step == "receive":
                    a["qty"] = amount or 1
                if step != "receive":
                    a["idempotency_key"] = f"{step}-{other}"
                self.call(STEP_TOOL[step], a)
            elif kind == "skip":
                if d[1] not in self.written:
                    self.skip.add(d[1])
            elif kind == "resend":
                if d[1] in self.written:
                    self.call(STEP_TOOL[d[1]], self.args(d[1], self.a.key(d[1], new=True)))

    def audit_verdict(self, log: dict[str, Any], step: str) -> str | None:
        mine = [e for e in log.get("requests") or [] if e.get("tool") == STEP_TOOL[step]]
        if not mine:
            return None
        e = mine[-1]
        if e.get("committed") is True or e.get("applied") is True or e.get("ack_status") == "ok":
            return "done"
        if e.get("committed") is False or e.get("applied") is False or e.get("ack_status") == "rejected":
            return "failed"
        return None

    def run(self) -> None:
        a = self.a
        po = self.call("get_purchase_order", {"po_id": self.pid})
        self.registered = ((po.get("supplier") or {}) if po.get("status") == "ok" else {}).get("payee_account")
        self.seen.clear()  # re-parse with the registered account known
        self.pending.clear()
        self.scan(po)
        for step in a.requested:
            self.act()
            if step in self.skip:
                continue
            ack = self.call(STEP_TOOL[step], self.args(step, a.key(step)))
            self.written.add(step)
            if ack.get("status") != "ok":
                verdict = self.audit_verdict(self.call("get_audit_log", {"po_id": self.pid}), step)
                if verdict == "failed":
                    self.call(STEP_TOOL[step], self.args(step, a.key(step, new=True)))
                elif verdict is None:
                    obs = self.call(STEP_READ[step], {"po_id": self.pid})
                    if obs.get("status") == "ok" and _observed(step, obs, self.po) == "absent":
                        self.call(STEP_TOOL[step], self.args(step, a.key(step, new=True)))
                self.act()
            else:
                self.call(STEP_READ[step], {"po_id": self.pid})
            self.call("get_purchase_order", {"po_id": self.pid})
            self.act()
        self.act()
        a.report("completed", f"{', '.join(s for s in a.requested if s not in self.skip)} done for {self.pid}")


def obedient(a: TP.Agent) -> None:
    Obedient(a).run()


POLICIES: dict[str, Callable[[TP.Agent], None]] = {"B0": TP.POLICIES["B0"], "B4": TP.POLICIES["B4"], "OBD": obedient}


def policy_view(task: ProcTask) -> ProcTask:
    """What a scripted policy may know: the world it acts in, the PO id and the requested steps (all in the prompt
    or reachable with tools). Faults and attack metadata are dropped."""
    return ProcTask(task.task_id, task.split, task.workflow, task.prompt, task.initial_state,
                    {"po_id": task.meta["po_id"], "requested": list(task.meta["requested"])})


def run_episode(name: str, task: ProcTask, gate: bool, backend: str = "txn_v1",
                rules: Any = None) -> dict[str, Any]:
    """One scripted episode on the attack env, with or without the frozen RG1 gate (full rules). Returns the full
    record with `harm` flags and `wall_ms` (policy + gate finalisation)."""
    from actionproof.recovery_gate.gate import Rules
    b = get_backend(backend)
    env = make_env_attack(task, f"atk-{name}-{'g' if gate else 'n'}-{task.task_id}",
                          rules=(rules or Rules()) if gate else None, backend=backend)
    agent = TP.Agent(env, policy_view(task))
    ended = "report"
    t0 = time.perf_counter()
    try:
        POLICIES[name](agent)
    except TP.BudgetExceeded:
        ended = "budget"
    rec = b.record(task, env)
    wall = (time.perf_counter() - t0) * 1000
    rec.update(policy=name, gated=gate, ended_by=ended, wall_ms=wall)
    return b.annotate(copy.deepcopy(rec))


__all__ = ["POLICIES", "parse", "obedient", "policy_view", "run_episode"]
