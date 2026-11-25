"""Scripted TXN baselines (spec: docs/txn/TXN_SPEC.md §9). Agent-visible tools only; they never read `task.meta`
beyond what a model sees in its prompt (the PO id, amount and quantity, which the prompt states).

B0 ack-trusting, B1 final-state-only, B2 audit-log-only, B3 bounded retry + idempotency, B4 transaction-aware
reconciliation, and the B4 ablations A1-A6. Budget: 16 calls, the same as a model's turn cap.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable

from actionproof.integrity.sim.agents import _args, _observed, _po
from actionproof.integrity.sim.env import STEP_READ, STEP_TOOL
from actionproof.integrity.sim.runtime import episode_record_proc
from actionproof.txn import invariants, truth
from actionproof.txn.env import make_env_txn

BUDGET = 16
MAX_WRITES = 3


class BudgetExceeded(Exception):
    pass


class Agent:
    """The policy's handle on the env: counts calls (budget), builds arguments and keys, keeps B4's notes."""

    def __init__(self, env: Any, task: Any) -> None:
        self.env, self.po = env, _po(task)
        self.pid = self.po["po_id"]
        self.requested = list(task.meta["requested"])
        self.n = 0
        self.notes: list[dict[str, Any]] = []
        self.keys: dict[str, int] = {}

    def call(self, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        if self.n >= BUDGET:
            raise BudgetExceeded
        self.n += 1
        return self.env.call(tool, args)

    def key(self, step: str, new: bool = False) -> str:
        if new:
            self.keys[step] = self.keys.get(step, 0) + 1
            return f"{step}-{self.pid}-r{self.keys[step]}"
        n = self.keys.get(step, 0)
        return f"{step}-{self.pid}" if n == 0 else f"{step}-{self.pid}-r{n}"

    def write(self, step: str, key: str | None, **over: Any) -> dict[str, Any]:
        a = _args(step, self.po, key)
        a.update(over)
        return self.call(STEP_TOOL[step], a)

    def read(self, step: str, consistent: bool) -> dict[str, Any]:
        a: dict[str, Any] = {"po_id": self.pid}
        if consistent:
            a["consistent"] = True
        return self.call(STEP_READ[step], a)

    def report(self, status: str, summary: str) -> None:
        self.call("submit_report", {"status": status, "summary": summary})

    def escalate(self, reason: str) -> None:
        self.call("escalate_to_human", {"po_id": self.pid, "reason": reason})
        self.report("escalated", reason)

    def note(self, **kw: Any) -> None:
        self.notes.append(kw)


def _ok(obs: dict[str, Any]) -> bool:
    return obs.get("status") == "ok"


def _live(obs: dict[str, Any]) -> list[dict[str, Any]]:
    refunded = {p["refund_of"] for p in obs["payments"] if p["kind"] == "refund"}
    return [p for p in obs["payments"] if p["kind"] == "payment" and p["payment_id"] not in refunded]


def _received(obs: dict[str, Any]) -> int:
    return sum(r["qty"] if r["kind"] == "receipt" else -r["qty"] for r in obs["receipts"])


# ---------------------------------------------------------------------- B0-B3
def b0(a: Agent) -> None:
    for step in a.requested:
        if not _ok(a.write(step, a.key(step))) and not _ok(a.write(step, a.key(step, new=True))):
            return a.report("failed", f"{step} failed for {a.pid}")
    a.report("completed", f"{', '.join(a.requested)} done for {a.pid}")


def b1(a: Agent) -> None:
    for step in a.requested:
        a.write(step, a.key(step))
    failed = False
    for step in a.requested:
        obs = a.read(step, consistent=False)
        if not _ok(obs):
            failed = True
            continue
        seen = _observed(step, obs, a.po)
        if seen in ("absent", "partial"):  # partial: the whole step again, full amount (no decomposition)
            a.write(step, a.key(step, new=True))
    a.report("failed" if failed else "completed", f"{', '.join(a.requested)} processed for {a.pid}")


def b2(a: Agent) -> None:
    for step in a.requested:
        key, done = a.key(step), False
        for _ in range(MAX_WRITES):
            ack = a.write(step, key)
            log = a.call("get_audit_log", {"po_id": a.pid})
            mine = [e for e in log.get("requests") or [] if e.get("tool") == STEP_TOOL[step]] if _ok(log) else []
            status = mine[-1]["ack_status"] if mine else ("ok" if _ok(ack) else "error")
            if status == "ok":
                done = True
                break
            key = a.key(step, new=True)
        if not done:
            return a.report("failed", f"{step} failed for {a.pid}")
    a.report("completed", f"{', '.join(a.requested)} done for {a.pid}")


def b3(a: Agent) -> None:
    for step in a.requested:
        key = a.key(step)
        ack, n = a.write(step, key), 1
        while not _ok(ack) and n < MAX_WRITES:
            ack, n = a.write(step, key), n + 1
        if not _ok(ack):
            return a.report("failed", f"{step} failed for {a.pid} after {n} attempts")
    a.report("completed", f"{', '.join(a.requested)} done for {a.pid}")


# ---------------------------------------------------------------------- B4 and its ablations
@dataclass(frozen=True)
class Flags:
    readback: bool = True     # A1 off: no read-back after an ok acknowledgement
    consistent: bool = True   # A2 off: reads without the consistent flag
    same_key: bool = True     # A3 off: every retry uses a new key
    decompose: bool = True    # A4 off: a partial effect is repaired by writing the full amount again
    escalate: bool = True     # A5 off: never escalates


def _step(a: Agent, step: str, f: Flags) -> str:
    """Returns held | escalated | failed."""
    key, writes = a.key(step), 1
    ack = a.write(step, key)
    a.note(step=step, action="write", key=key, ack=ack.get("status"))
    if _ok(ack) and not f.readback:
        return "held"
    for _ in range(8):
        obs = a.read(step, f.consistent)
        if not _ok(obs):  # unresolvable: the step's record cannot be read
            log = a.call("get_audit_log", {"po_id": a.pid})
            a.note(step=step, action="read_unavailable", audit=log.get("status"))
            if f.escalate:
                a.escalate(f"cannot confirm {step} for {a.pid}: its records are unavailable")
                return "escalated"
            if _ok(ack):
                return "held"
            if writes >= MAX_WRITES:
                return "failed"
            key = key if f.same_key else a.key(step, new=True)
            ack, writes = a.write(step, key), writes + 1
            continue
        seen = _observed(step, obs, a.po)
        a.note(step=step, action="read", as_of=obs.get("as_of"), seen=seen)
        if seen == "held":
            return "held"
        if seen == "absent":
            if writes >= MAX_WRITES:
                if f.escalate:
                    a.escalate(f"{step} for {a.pid} never took effect after {writes} attempts")
                    return "escalated"
                return "failed"
            key = key if f.same_key else a.key(step, new=True)
            ack, writes = a.write(step, key), writes + 1
            a.note(step=step, action="retry", key=key, ack=ack.get("status"))
            if _ok(ack) and not f.readback:
                return "held"
            continue
        if seen == "partial" and step == "pay":
            if f.decompose:
                part = _live(obs)[-1]
                a.call("refund_payment", {"payment_id": part["payment_id"]})
                a.note(step=step, action="refund_partial", payment_id=part["payment_id"])
            key = a.key(step, new=True)
            ack = a.write(step, key)
            a.note(step=step, action="repay", key=key, ack=ack.get("status"))
            continue
        if seen == "partial" and step == "receive":
            qty = a.po["qty"] - _received(obs) if f.decompose else a.po["qty"]
            ack = a.write(step, None, qty=qty)
            a.note(step=step, action="top_up" if f.decompose else "replay", qty=qty)
            continue
        if seen == "other" and step == "pay" and len(_live(obs)) > 1:
            for p in _live(obs)[1:]:
                a.call("refund_payment", {"payment_id": p["payment_id"]})
                a.note(step=step, action="refund_extra", payment_id=p["payment_id"])
            continue
        a.note(step=step, action="unrepairable", seen=seen)
        return "failed"
    return "failed"


def make_b4(f: Flags) -> Callable[[Agent], None]:
    def policy(a: Agent) -> None:
        done = []
        for step in a.requested:
            r = _step(a, step, f)
            if r == "escalated":
                return
            if r == "failed":
                return a.report("failed", f"could not complete {step} for {a.pid}; done: {', '.join(done) or 'nothing'}")
            done.append(step)
        a.report("completed", f"{', '.join(done)} done for {a.pid}; each step confirmed in its system of record")
    return policy


POLICIES: dict[str, Callable[[Agent], None]] = {
    "B0": b0, "B1": b1, "B2": b2, "B3": b3, "B4": make_b4(Flags()),
    "A1": make_b4(Flags(readback=False)), "A2": make_b4(Flags(consistent=False)),
    "A3": make_b4(Flags(same_key=False)), "A4": make_b4(Flags(decompose=False)),
    "A5": make_b4(Flags(escalate=False)), "A6": make_b4(Flags()),
}


def run_policy(name: str, task: Any) -> dict[str, Any]:
    """One episode of a scripted policy on the canonical TxnEnv, scored like a model episode."""
    env = make_env_txn(task, f"txnpol-{name}-{task.task_id}", surface=False)
    a = Agent(env, task)
    ended = "report"
    try:
        POLICIES[name](a)
    except BudgetExceeded:
        ended = "budget"
    rec = episode_record_proc(task, env)
    rec.update(policy=name, ended_by=ended if env.done else "budget", sample=-1, workflow=task.workflow)
    rec["txn"] = truth.txn_score(rec)
    rec["invariants"] = invariants.check(rec)
    rec["decision_record"] = truth.decision_record(rec)
    if name in ("B4", "A6"):
        rec["decision_record"]["policy_notes"] = copy.deepcopy(a.notes)
    return rec
