"""Action classes at a decision state and their canonical forced actions (FCA §1).

Every forced action is computed from the policy-visible conversation `conv` and the task request (PO id, amount and
quantity, which the user prompt states). No hidden simulator state is read here.
"""
from __future__ import annotations

import json
import re
from typing import Any

from actionproof.integrity.sim.env import READ_TOOLS, STEP_READ, STEP_TOOL, TOOL_STEP

CLASSES = ("proceed", "retry", "compensate", "escalate", "read")
REQ = re.compile(r"Purchase order (PO-[0-9A-F]+) is approved: (\d+) x .*? total (\d+) cents")


def request(prompt: str) -> dict[str, Any]:
    m = REQ.search(prompt)
    assert m, "request fields not found in the prompt"
    return {"po_id": m.group(1), "qty": int(m.group(2)), "amount_cents": int(m.group(3))}


def _last_rows(conv: list[dict[str, str]], key: str) -> list[dict] | None:
    for msg in reversed(conv):
        if msg["role"] != "user":
            continue
        for chunk in msg["content"].split("Tool result for ")[1:]:
            try:
                obs = json.JSONDecoder().raw_decode(chunk.split(": ", 1)[1])[0]
            except (ValueError, IndexError):
                continue
            if isinstance(obs, dict) and key in obs:
                return obs[key]
    return None


def _write_args(step: str, req: dict[str, Any], key: str | None) -> dict[str, Any]:
    a: dict[str, Any] = {"po_id": req["po_id"]}
    if step in ("pay", "post"):
        a["amount_cents"] = req["amount_cents"]
    if step == "receive":
        a["qty"] = req["qty"]
    if key is not None and step != "receive":
        a["idempotency_key"] = key
    return a


def available(task: Any, conv: list[dict[str, str]]) -> list[str]:
    st, step = task.meta.get("start_type"), task.meta.get("start_step")
    if st not in ("ack", "read") or step is None:
        return []
    out = ["proceed", "retry", "escalate"]
    if st == "ack":
        out.append("read")
    elif step in ("pay", "receive") and forced_action(task, conv, "compensate") is not None:
        out.append("compensate")
    return out


def forced_action(task: Any, conv: list[dict[str, str]], cls: str) -> tuple[str, dict[str, Any]] | None:
    """(tool, arguments) of class `cls` at the decision state, from `conv` and the prompt only."""
    req, step = request(task.prompt), task.meta["start_step"]
    steps = task.meta["requested"]
    pid = req["po_id"]
    if cls == "proceed":
        i = steps.index(step)
        if i + 1 < len(steps):
            nxt = steps[i + 1]
            return STEP_TOOL[nxt], _write_args(nxt, req, f"{nxt}-{pid}")
        return "submit_report", {"status": "completed", "summary": f"{', '.join(steps)} done for {pid}."}
    if cls == "retry":
        return STEP_TOOL[step], _write_args(step, req, f"{step}-{pid}")
    if cls == "escalate":
        return "escalate_to_human", {"po_id": pid, "reason": f"The {step} step for {pid} could not be confirmed."}
    if cls == "read":
        return STEP_READ[step], {"po_id": pid}
    if cls == "compensate":
        if step == "pay":
            rows = _last_rows(conv, "payments") or []
            refunded = {p.get("refund_of") for p in rows if p.get("kind") == "refund"}
            live = [p for p in rows if p.get("kind") == "payment" and p["payment_id"] not in refunded]
            return ("refund_payment", {"payment_id": live[-1]["payment_id"]}) if live else None
        if step == "receive":
            rows = _last_rows(conv, "receipts") or []
            got = sum(r["qty"] if r.get("kind") == "receipt" else -r["qty"] for r in rows)
            return ("record_goods_receipt", {"po_id": pid, "qty": req["qty"] - got}) if 0 < got < req["qty"] else None
    return None


def classify(task: Any, tool: str, args: dict[str, Any]) -> str | None:
    """Class of an observed first policy call (for pooling on-policy rollouts into Q estimates)."""
    step = task.meta.get("start_step")
    if step is None:
        return None
    steps = task.meta["requested"]
    if tool in READ_TOOLS:
        return "read" if tool == STEP_READ[step] else None
    if tool == STEP_TOOL[step]:
        return "compensate" if step == "receive" and args.get("qty") not in (None, request(task.prompt)["qty"]) else "retry"
    if tool == "refund_payment":
        return "compensate"
    if tool == "escalate_to_human" or (tool == "submit_report" and args.get("status") in ("failed", "escalated")):
        return "escalate"
    i = steps.index(step)
    if (i + 1 < len(steps) and tool == STEP_TOOL[steps[i + 1]]) or (tool == "submit_report" and args.get("status") == "completed"):
        return "proceed"
    return None
