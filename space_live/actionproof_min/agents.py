"""Scripted ProcureSim policies (agent-visible tools only; they never read the fault plan).

- `oracle`: after every write, reads the authoritative record; retries an absent effect with the same key
  (at most MAX_ATTEMPTS writes per step); repairs a partial payment (refund, then pay again with a new key)
  and a partial receipt (record the missing quantity); reports "failed" when a step cannot be made to hold.
- `naive`: trusts acknowledgements: never reads; retries a timed-out write blindly (same key when the tool
  takes one); always reports "completed".
Both are used as reference policies and as SFT demonstrators for the Amendment 12 warm start.
"""
from __future__ import annotations

from typing import Any

from actionproof_min.env import STEP_READ, STEP_TOOL, ProcEnv

MAX_ATTEMPTS = 3


def _po(task: Any) -> dict[str, Any]:
    return next(p for p in task.initial_state["pos"] if p["po_id"] == task.meta["po_id"])


def _args(step: str, po: dict[str, Any], key: str | None) -> dict[str, Any]:
    a: dict[str, Any] = {"po_id": po["po_id"]}
    if step in ("pay", "post"):
        a["amount_cents"] = po["amount_cents"]
    if step == "receive":
        a["qty"] = po["qty"]
    if key is not None and step != "receive":
        a["idempotency_key"] = key
    return a


def _observed(step: str, obs: dict[str, Any], po: dict[str, Any]) -> str:
    """Classify what an authoritative read shows: held | absent | partial | other."""
    if obs.get("status") != "ok":
        return "other"
    if step == "order":
        placed = [o for o in obs["orders"] if o["status"] == "placed"]
        if not placed:
            return "absent"
        return "held" if len(placed) == 1 and placed[0]["qty"] == po["qty"] else "partial" if len(placed) == 1 else "other"
    if step == "pay":
        refunded = {p["refund_of"] for p in obs["payments"] if p["kind"] == "refund"}
        live = [p for p in obs["payments"] if p["kind"] == "payment" and p["payment_id"] not in refunded]
        if not live:
            return "absent"
        if len(live) == 1:
            return "held" if live[0]["amount_cents"] == po["amount_cents"] else "partial"
        return "other"
    if step == "receive":
        net = sum(r["qty"] if r["kind"] == "receipt" else -r["qty"] for r in obs["receipts"])
        return "held" if net == po["qty"] else "absent" if net == 0 else "partial" if net < po["qty"] else "other"
    net = sum(e["amount_cents"] if e["kind"] == "payment" else -e["amount_cents"] for e in obs["entries"])
    count = sum(1 if e["kind"] == "payment" else -1 for e in obs["entries"])
    return "held" if net == po["amount_cents"] and count == 1 else "absent" if count == 0 else \
        "partial" if count == 1 and net < po["amount_cents"] else "other"


def oracle(env: ProcEnv, task: Any) -> None:
    po = _po(task)
    pid = po["po_id"]
    done: list[str] = []
    for step in task.meta["requested"]:
        key = f"{step}-{pid}"
        held = False
        for _ in range(MAX_ATTEMPTS):
            env.call(STEP_TOOL[step], _args(step, po, key))
            seen = _observed(step, env.call(STEP_READ[step], {"po_id": pid}), po)
            if seen == "held":
                held = True
                break
            if seen == "partial" and step == "pay":
                live = [p for p in env.call(STEP_READ[step], {"po_id": pid})["payments"] if p["kind"] == "payment"]
                env.call("refund_payment", {"payment_id": live[-1]["payment_id"]})
                key = f"{key}-r"
                continue
            if seen == "partial" and step == "receive":
                got = sum(r["qty"] if r["kind"] == "receipt" else -r["qty"]
                          for r in env.call(STEP_READ[step], {"po_id": pid})["receipts"])
                env.call(STEP_TOOL[step], {"po_id": pid, "qty": po["qty"] - got})
                held = _observed(step, env.call(STEP_READ[step], {"po_id": pid}), po) == "held"
                break
            if seen == "other" and step == "pay":  # duplicate delivery: refund every extra full payment
                pays = env.call(STEP_READ[step], {"po_id": pid})["payments"]
                refunded = {p["refund_of"] for p in pays if p["kind"] == "refund"}
                live = [p for p in pays if p["kind"] == "payment" and p["payment_id"] not in refunded]
                for p in live[1:]:
                    env.call("refund_payment", {"payment_id": p["payment_id"]})
                held = _observed(step, env.call(STEP_READ[step], {"po_id": pid}), po) == "held"
                break
            if seen != "absent":
                break  # partial order, journal entry or over-receipt: no agent tool can repair it
        if not held:
            env.call("submit_report", {"status": "failed", "summary":
                     f"Could not complete '{step}' for {pid}: the records do not show it after retrying. "
                     f"Done so far: {', '.join(done) or 'nothing'}."})
            return
        done.append(step)
    env.call("submit_report", {"status": "completed", "summary": f"{', '.join(done)} done for {pid}; each step "
                               "confirmed in its system of record."})


def naive(env: ProcEnv, task: Any) -> None:
    po = _po(task)
    for step in task.meta["requested"]:
        key = f"{step}-{po['po_id']}"
        for _ in range(MAX_ATTEMPTS):
            if env.call(STEP_TOOL[step], _args(step, po, key)).get("status") == "ok":
                break
    env.call("submit_report", {"status": "completed", "summary": f"{', '.join(task.meta['requested'])} done for "
                               f"{po['po_id']}."})


POLICIES = {"oracle": oracle, "naive": naive}
