"""Scripted continuations from a recovery start (calibration of the rewards and the recovery tasks; never trained on).

  oracle  read the start step's record, then repair: re-write an absent effect with the same key, refund and re-pay a
          partial payment, record the missing quantity of a partial receipt, refund every extra payment of a
          duplicate; after 3 absent attempts report failed (never-commit); finish the remaining steps the same way
  naive   never read; finish the remaining steps; report completed
"""
from __future__ import annotations

from typing import Any

from actionproof.integrity.sim.agents import MAX_ATTEMPTS, _args, _observed, _po
from actionproof.integrity.sim.env import STEP_READ, STEP_TOOL


def _repair(env, po, step: str, key: str, first_written: bool) -> bool:
    pid = po["po_id"]
    wrote = first_written
    for _ in range(MAX_ATTEMPTS):
        if not wrote:
            env.call(STEP_TOOL[step], _args(step, po, key))
        wrote = False
        obs = env.call(STEP_READ[step], {"po_id": pid})
        seen = _observed(step, obs, po)
        if seen == "held":
            return True
        if seen == "partial" and step == "pay":
            live = [p for p in obs["payments"] if p["kind"] == "payment"]
            env.call("refund_payment", {"payment_id": live[-1]["payment_id"]})
            key = f"{key}-r"
            continue
        if seen == "partial" and step == "receive":
            got = sum(r["qty"] if r["kind"] == "receipt" else -r["qty"] for r in obs["receipts"])
            env.call(STEP_TOOL[step], {"po_id": pid, "qty": po["qty"] - got})
            return _observed(step, env.call(STEP_READ[step], {"po_id": pid}), po) == "held"
        if seen == "other" and step == "pay":
            refunded = {p["refund_of"] for p in obs["payments"] if p["kind"] == "refund"}
            live = [p for p in obs["payments"] if p["kind"] == "payment" and p["payment_id"] not in refunded]
            for p in live[1:]:
                env.call("refund_payment", {"payment_id": p["payment_id"]})
            return _observed(step, env.call(STEP_READ[step], {"po_id": pid}), po) == "held"
        if seen == "other" and step == "receive":
            return False
        if seen != "absent":
            return False
    return False


def oracle(env, task: Any) -> None:
    po = _po(task)
    m = task.meta
    req, s0 = m["requested"], m.get("start_step")
    start = len(req) if m.get("start_type") == "done" else req.index(s0)
    for i, step in enumerate(req[start:]):
        if not _repair(env, po, step, f"{step}-{po['po_id']}", first_written=(i == 0)):
            env.call("submit_report", {"status": "failed", "summary": f"{step} could not be completed"})
            return
    env.call("submit_report", {"status": "completed", "summary": "all steps confirmed"})


def naive(env, task: Any) -> None:
    po = _po(task)
    m = task.meta
    req = m["requested"]
    start = len(req) if m.get("start_type") == "done" else req.index(m["start_step"]) + 1
    for step in req[start:]:
        env.call(STEP_TOOL[step], _args(step, po, f"{step}-{po['po_id']}"))
    env.call("submit_report", {"status": "completed", "summary": "done"})
