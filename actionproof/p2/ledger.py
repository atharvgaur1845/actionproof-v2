"""Observable-state ledger (PREREGISTRATION_P2.md §3).

A pure function of the conversation text the policy has already seen: the request, the policy's own tool calls,
and the tool results. It never reads the simulator, its database, its fault log or the task metadata, so it
cannot leak hidden state.

**What it records.**
- Purchase-order facts: the total and quantity stated in the request, or from a `get_purchase_order` result.
- The last read of each record type: payments, goods receipts, journal entries, supplier orders.
- Whether that read is **stale**, i.e. a write to the same records was acknowledged or attempted after it.
- Idempotency keys already used, per write tool, with the payment id each acknowledgement returned. A key whose
  payment was later seen refunded is marked as such.
- Writes not read back since they were sent.

Anything never read is printed as "unknown"; nothing is guessed. It states facts and no recommended action, because
P1's development data showed that an "outstanding" figure alone invited a forbidden top-up
(`docs/research/P1_DEV_DIAGNOSIS.md`).

`ledger_text(conv)` → the block. `with_ledger(result_text, conv)` → the tool-result text with the block inserted
before the final "Continue the task." line.
"""
from __future__ import annotations

import json
import re
from typing import Any

READS = {"list_payments": "payments", "list_goods_receipts": "receipts", "get_journal_entries": "journal",
         "find_supplier_order": "orders"}
WRITES = {"pay_supplier": "payments", "refund_payment": "payments", "record_goods_receipt": "receipts",
          "post_journal_entry": "journal", "place_supplier_order": "orders"}
RESULT = re.compile(r"Tool result for ([A-Za-z_]+): (.*?)\nContinue the task\.", re.S)
TAIL = "\nContinue the task."


def _calls(text: str) -> list[tuple[str, dict]]:
    """The tool calls in one assistant message (the policy's JSON format); unparseable text gives none."""
    try:
        v = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        m = re.search(r"\[.*\]", text or "", re.S)
        try:
            v = json.loads(m.group(0)) if m else []
        except json.JSONDecodeError:
            return []
    if isinstance(v, dict):
        v = [v]
    out = []
    for c in v if isinstance(v, list) else []:
        if isinstance(c, dict) and isinstance(c.get("tool_name"), str):
            out.append((c["tool_name"], c.get("parameters") if isinstance(c.get("parameters"), dict) else {}))
    return out


def _results(text: str) -> list[tuple[str, Any]]:
    out = []
    for name, body in RESULT.findall(text or ""):
        try:  # the leading JSON value only: a ledger block may follow it inside the same result
            out.append((name, json.JSONDecoder().raw_decode(body.lstrip())[0]))
        except json.JSONDecodeError:
            out.append((name, None))
    return out


def _request_facts(req: str) -> dict:
    f = {}
    m = re.search(r"total (\d+) cents", req) or re.search(r"कुल (\d+) सेंट", req) or re.search(r"(\d+) cents in total", req)
    if m:
        f["total"] = int(m.group(1))
    m = re.search(r"(\d+) x ", req) or re.search(r"(\d+) units of", req)
    if m:
        f["qty"] = int(m.group(1))
    m = re.search(r"\b(PO-[0-9A-F]+)\b", req)
    if m:
        f["po"] = m.group(1)
    return f


def state(conv: list[dict[str, str]]) -> dict:
    """Replay the visible conversation into ledger facts. conv[0] is the system prompt, conv[1] the request."""
    req = conv[1]["content"] if len(conv) > 1 else ""
    st: dict[str, Any] = {"po": _request_facts(req), "last": {}, "stale": {}, "keys": [], "unread": [], "n": 0}
    pending: list[tuple[str, dict]] = []
    for m in conv[2:]:
        if m["role"] == "assistant":
            pending = _calls(m["content"])
            continue
        res = _results(m["content"])
        for i, (name, obs) in enumerate(res):
            args = pending[i][1] if i < len(pending) and pending[i][0] == name else {}
            st["n"] += 1
            ok = isinstance(obs, dict) and obs.get("status") == "ok"
            if name == "get_purchase_order" and ok and isinstance(obs.get("po"), dict):
                po = obs["po"]
                st["po"].setdefault("total", po.get("amount_cents"))
                st["po"].setdefault("qty", po.get("qty"))
            elif name in READS and ok:
                kind = READS[name]
                st["last"][kind] = {"obs": obs, "call": st["n"]}
                st["stale"][kind] = False
                st["unread"] = [u for u in st["unread"] if WRITES[u[0]] != kind]
            elif name in WRITES:
                kind = WRITES[name]
                st["stale"][kind] = kind in st["last"]
                ack = (obs or {}).get("status") if isinstance(obs, dict) else "unparsed"
                st["unread"].append((name, st["n"], ack if ack == "ok" else f"{ack}: {(obs or {}).get('error', '')}"))
                if "idempotency_key" in args:
                    st["keys"].append({"tool": name, "key": args["idempotency_key"],
                                       "payment_id": (obs or {}).get("payment_id") if isinstance(obs, dict) else None})
    return st


def _cents(v: Any) -> str:
    return f"{v} cents" if isinstance(v, int) else "unknown"


def _payments(st: dict) -> tuple[str, set]:
    last = st["last"].get("payments")
    if not last:
        return "Payments: unknown (not read).", set()
    pays = last["obs"].get("payments", [])
    refunded = {p.get("refund_of") for p in pays if p.get("kind") == "refund"}
    live = [p for p in pays if p.get("kind") == "payment" and p.get("payment_id") not in refunded]
    total = sum(p.get("amount_cents", 0) for p in live)
    po_total = st["po"].get("total")
    cmp = "" if not isinstance(po_total, int) else (" (equal to the PO total)" if total == po_total else
                                                    f" ({'below' if total < po_total else 'above'} the PO total by {abs(po_total - total)} cents)")
    ids = ", ".join(f"{p['payment_id']} {p.get('amount_cents')}" for p in live) or "none"
    txt = (f"Payments (read at call {last['call']}{', stale: written since' if st['stale'].get('payments') else ''}): "
           f"live {ids}; refunded {', '.join(sorted(x for x in refunded if x)) or 'none'}; live total {total} cents{cmp}.")
    return txt, {x for x in refunded if x}


def _simple(st: dict, kind: str, label: str, field: str) -> str:
    last = st["last"].get(kind)
    if not last:
        return f"{label}: unknown (not read)."
    rows = last["obs"].get({"receipts": "receipts", "journal": "entries", "orders": "orders"}[kind], [])
    tot = sum(r.get(field, 0) or 0 for r in rows)
    ref = st["po"].get("total" if field == "amount_cents" else "qty")
    unit = " cents" if field == "amount_cents" else ""
    cmp = "" if not isinstance(ref, int) else (" (equal to the PO)" if tot == ref else f" (PO: {ref}{unit})")
    return (f"{label} (read at call {last['call']}{', stale: written since' if st['stale'].get(kind) else ''}): "
            f"{len(rows)} record(s), total {tot}{unit}{cmp}.")


def ledger_text(conv: list[dict[str, str]]) -> str:
    st = state(conv)
    pay, refunded = _payments(st)
    keys, seen = [], {}
    for k in st["keys"]:
        tag = (k["tool"], k["key"])
        if tag in seen:  # a repeat of a key: say whether the reply named the earlier record (no new write)
            j, first_pid = seen[tag]
            if k.get("payment_id") and k["payment_id"] == first_pid and not keys[j].endswith("]"):
                keys[j] += " [reused; reply named the same payment]"
            continue
        seen[tag] = (len(keys), k.get("payment_id"))
        note = f" -> {k['payment_id']}" if k.get("payment_id") else ""
        if k.get("payment_id") in refunded:
            note += " (refunded)"
        keys.append(f"{k['tool']} {k['key']}{note}")
    unread = "; ".join(f"{n} (call {i}, reply {a})" for n, i, a in st["unread"]) or "none"
    lines = ["Ledger (request and tool results above only; unknown = not read):",
             f"- PO {st['po'].get('po', 'unknown')}: total {_cents(st['po'].get('total'))}, quantity {st['po'].get('qty', 'unknown')}.",
             f"- {pay}",
             f"- {_simple(st, 'receipts', 'Goods receipts', 'qty')}",
             f"- {_simple(st, 'journal', 'Journal entries', 'amount_cents')}",
             f"- {_simple(st, 'orders', 'Supplier orders', 'qty')}",
             f"- Keys used: {', '.join(keys) or 'none'}.",
             f"- Not read back since written: {unread}."]
    return "\n".join(lines)


def with_ledger(result_text: str, conv_before: list[dict[str, str]]) -> str:
    """Insert the ledger for conv_before + [this tool-result message] before the final 'Continue the task.'."""
    block = ledger_text(conv_before + [{"role": "user", "content": result_text}])
    i = result_text.rfind(TAIL)
    if i < 0:
        return result_text + "\n" + block
    return result_text[:i] + "\n" + block + result_text[i:]
