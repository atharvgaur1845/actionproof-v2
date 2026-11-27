"""SEL1 scripted policies on the SEL1 env.

B5 (selective reconciliation) is TXN's B4 (`actionproof/txn/policies.py::_step`, frozen) with one change: when the
step's record cannot be read, B5 reads it again, up to READS_BEFORE_ESCALATING reads in all, before it consults the
audit log and escalates. Under a transient outage one of those reads is served and B5 continues as B4 would; under a
permanent outage it escalates after the last failed read. This is the rule in payment-API practice: reconcile an
unknown outcome before acting again, and hand off when it cannot be established (Stripe, "Advanced error handling").

B4 itself escalates on the first unavailable read, so on SEL1's transient tasks it is the "escalate on any outage"
reference; A5 (B4 that never escalates) is the "never escalate" reference. ALWAYS (escalate at once) is the
always-escalate reference.

Policies see agent-visible observations only and never read `task.meta` beyond the prompt facts (as in TXN §9).
"""
from __future__ import annotations

import copy
from typing import Any, Callable

from actionproof.integrity.sim.agents import _observed
from actionproof.integrity.sim.runtime import episode_record_proc
from actionproof.txn import invariants, truth
from actionproof.txn.policies import MAX_WRITES, POLICIES, Agent, BudgetExceeded, _live, _ok, _received
from actionproof.sel1.env import make_env_sel

READS_BEFORE_ESCALATING = 3


def _step_b5(a: Agent, step: str) -> str:
    """B4's `_step` with Flags() defaults, except the unavailable branch. Returns held | escalated | failed."""
    key, writes = a.key(step), 1
    ack = a.write(step, key)
    a.note(step=step, action="write", key=key, ack=ack.get("status"))
    unavailable = 0
    for _ in range(10):
        obs = a.read(step, True)
        if not _ok(obs):
            unavailable += 1
            a.note(step=step, action="read_unavailable", n=unavailable)
            if unavailable < READS_BEFORE_ESCALATING:
                continue
            log = a.call("get_audit_log", {"po_id": a.pid})
            a.note(step=step, action="audit_log", audit=log.get("status"))
            a.escalate(f"cannot confirm {step} for {a.pid}: its records stayed unavailable after "
                       f"{READS_BEFORE_ESCALATING} reads")
            return "escalated"
        unavailable = 0
        seen = _observed(step, obs, a.po)
        a.note(step=step, action="read", as_of=obs.get("as_of"), seen=seen)
        if seen == "held":
            return "held"
        if seen == "absent":
            if writes >= MAX_WRITES:
                a.escalate(f"{step} for {a.pid} never took effect after {writes} attempts")
                return "escalated"
            ack, writes = a.write(step, key), writes + 1
            a.note(step=step, action="retry", key=key, ack=ack.get("status"))
            continue
        if seen == "partial" and step == "pay":
            part = _live(obs)[-1]
            a.call("refund_payment", {"payment_id": part["payment_id"]})
            a.note(step=step, action="refund_partial", payment_id=part["payment_id"])
            key = a.key(step, new=True)
            ack = a.write(step, key)
            a.note(step=step, action="repay", key=key, ack=ack.get("status"))
            continue
        if seen == "partial" and step == "receive":
            qty = a.po["qty"] - _received(obs)
            ack = a.write(step, None, qty=qty)
            a.note(step=step, action="top_up", qty=qty)
            continue
        if seen == "other" and step == "pay" and len(_live(obs)) > 1:
            for p in _live(obs)[1:]:
                a.call("refund_payment", {"payment_id": p["payment_id"]})
                a.note(step=step, action="refund_extra", payment_id=p["payment_id"])
            continue
        a.note(step=step, action="unrepairable", seen=seen)
        return "failed"
    return "failed"


def b5(a: Agent) -> None:
    done: list[str] = []
    for step in a.requested:
        r = _step_b5(a, step)
        if r == "escalated":
            return
        if r == "failed":
            return a.report("failed", f"could not complete {step} for {a.pid}; done: {', '.join(done) or 'nothing'}")
        done.append(step)
    a.report("completed", f"{', '.join(done)} done for {a.pid}; each step confirmed in its system of record")


def always(a: Agent) -> None:
    a.escalate(f"handing {a.pid} to a reviewer before any write")


SEL_POLICIES: dict[str, Callable[[Agent], None]] = {"B5": b5, "B4": POLICIES["B4"], "A5": POLICIES["A5"],
                                                     "B3": POLICIES["B3"], "B0": POLICIES["B0"], "ALWAYS": always}


def run_policy(name: str, task: Any, txn_env: bool = False) -> tuple[dict[str, Any], Any]:
    """One episode of a scripted policy, scored like a model episode (TXN's run_policy). `txn_env=True` uses TXN's
    canonical env (for txn_test, as TXN's own policy runs do); otherwise the SEL1 env."""
    from actionproof.txn.env import make_env_txn
    rid = f"sel1pol-{name}-{task.task_id}"
    env = make_env_txn(task, rid, surface=False) if txn_env else make_env_sel(task, rid)
    a = Agent(env, task)
    ended = "report"
    try:
        SEL_POLICIES[name](a)
    except BudgetExceeded:
        ended = "budget"
    rec = episode_record_proc(task, env)
    rec.update(policy=name, ended_by=ended, sample=-1, workflow=task.workflow)
    rec["txn"] = truth.txn_score(rec)
    rec["invariants"] = invariants.check(rec)
    rec["decision_record"] = truth.decision_record(rec)
    if name == "B5":
        rec["decision_record"]["policy_notes"] = copy.deepcopy(a.notes)
    return rec, env
