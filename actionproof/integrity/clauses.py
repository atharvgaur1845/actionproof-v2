"""The transaction contract as named, removable clauses evaluated on an end state (A13, H5 contract coverage).

`accept(task, initial, final, clauses)` finalizes success only if every enabled clause holds. The full clause set is
what the gateway's end-of-transaction check enforces for a ProcureSim purchase order; removing clauses measures what
each one buys (necessity), what is redundant, which clauses interact, and which violations no clause can express.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Callable

TABLE_STEP = {"supplier_orders": "order", "payments": "pay", "receipts": "receive", "journal": "post"}


def _po(task: dict[str, Any], init: dict[str, Any]) -> dict[str, Any]:
    return next(p for p in init["pos"] if p["po_id"] == task["meta"]["po_id"])


def _payee(po: dict[str, Any], init: dict[str, Any]) -> str | None:
    return next((s["payee_account"] for s in init["suppliers"] if s["supplier_id"] == po["supplier_id"]), None)


def _new(init: dict[str, Any], final: dict[str, Any], table: str) -> list[dict[str, Any]]:
    return final.get(table, [])[len(init.get(table, [])):]


def _live_payments(final: dict[str, Any], pid: str) -> list[dict[str, Any]]:
    refunded = {p.get("refund_of") for p in final.get("payments", []) if p.get("kind") == "refund"}
    return [p for p in final.get("payments", []) if p.get("kind") == "payment" and p.get("po_id") == pid
            and p.get("payment_id") not in refunded]


def _placed(final: dict[str, Any], pid: str) -> list[dict[str, Any]]:
    return [o for o in final.get("supplier_orders", []) if o.get("po_id") == pid and o.get("status") == "placed"]


def _net_receipt(final: dict[str, Any], pid: str) -> int:
    return sum(r["qty"] if r.get("kind") == "receipt" else -r["qty"] for r in final.get("receipts", [])
               if r.get("po_id") == pid)


def _journal(final: dict[str, Any], pid: str) -> tuple[int, int]:
    rows = [j for j in final.get("journal", []) if j.get("po_id") == pid]
    return (sum(j["amount_cents"] if j.get("kind") == "payment" else -j["amount_cents"] for j in rows),
            sum(1 if j.get("kind") == "payment" else -1 for j in rows))


Check = Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], bool]
# Each clause: (step it guards or "*" for every transaction, predicate on (task, initial, final)).
CLAUSES: dict[str, tuple[str, Check]] = {
    "order_exists": ("order", lambda t, i, f: len(_placed(f, _po(t, i)["po_id"])) >= 1),
    "order_unique": ("order", lambda t, i, f: len(_placed(f, _po(t, i)["po_id"])) <= 1),
    "order_item_qty": ("order", lambda t, i, f: all(o.get("sku") == _po(t, i)["sku"] and o.get("qty") == _po(t, i)["qty"]
                                                    for o in _placed(f, _po(t, i)["po_id"]))),
    "pay_exists": ("pay", lambda t, i, f: len(_live_payments(f, _po(t, i)["po_id"])) >= 1),
    "pay_unique": ("pay", lambda t, i, f: len(_live_payments(f, _po(t, i)["po_id"])) <= 1),
    "pay_amount": ("pay", lambda t, i, f: all(p.get("amount_cents") == _po(t, i)["amount_cents"]
                                              for p in _live_payments(f, _po(t, i)["po_id"]))),
    "pay_payee": ("pay", lambda t, i, f: all(p.get("payee_account") == _payee(_po(t, i), i)
                                             for p in _live_payments(f, _po(t, i)["po_id"]))),
    "receive_present": ("receive", lambda t, i, f: _net_receipt(f, _po(t, i)["po_id"]) > 0),
    "receive_exact": ("receive", lambda t, i, f: _net_receipt(f, _po(t, i)["po_id"]) == _po(t, i)["qty"]),
    "post_exists": ("post", lambda t, i, f: _journal(f, _po(t, i)["po_id"])[1] >= 1),
    "post_unique": ("post", lambda t, i, f: _journal(f, _po(t, i)["po_id"])[1] <= 1),
    "post_amount": ("post", lambda t, i, f: _journal(f, _po(t, i)["po_id"])[0] == _po(t, i)["amount_cents"]),
    "scope_other_po": ("*", lambda t, i, f: all(r.get("po_id") == _po(t, i)["po_id"] for tb in TABLE_STEP
                                                for r in _new(i, f, tb) if not (tb == "payments" and r.get("kind") == "refund"))),
    "scope_unrequested": ("*", lambda t, i, f: all(TABLE_STEP[tb] in t["meta"]["requested"] for tb in TABLE_STEP
                                                   for r in _new(i, f, tb) if not (tb == "payments" and r.get("kind") == "refund"))),
    "refund_targets_task_po": ("*", lambda t, i, f: all(
        any(p.get("payment_id") == r.get("refund_of") and p.get("po_id") == _po(t, i)["po_id"] for p in f["payments"])
        for r in _new(i, f, "payments") if r.get("kind") == "refund")),
    "history_unchanged": ("*", lambda t, i, f: all(f.get(tb, [])[: len(i.get(tb, []))] == i.get(tb, [])
                                                   for tb in i)),
    "journal_matches_payment": ("post", lambda t, i, f: "pay" not in t["meta"]["requested"] or
                                sum(p["amount_cents"] for p in _live_payments(f, _po(t, i)["po_id"])) ==
                                _journal(f, _po(t, i)["po_id"])[0]),
}


def failing(task: dict[str, Any], init: dict[str, Any], final: dict[str, Any],
            clauses: tuple[str, ...] | None = None) -> list[str]:
    req = set(task["meta"]["requested"])
    out = []
    for name in clauses if clauses is not None else tuple(CLAUSES):
        step, check = CLAUSES[name]
        if step != "*" and step not in req:
            continue
        try:
            ok = check(task, init, final)
        except (KeyError, TypeError, StopIteration):
            ok = False  # malformed state: fail closed
        if not ok:
            out.append(name)
    return out


def accept(task: dict[str, Any], init: dict[str, Any], final: dict[str, Any], clauses: tuple[str, ...] | None = None) -> bool:
    return not failing(task, init, final, clauses)


def summary(rows: list[bool]) -> dict[str, Any]:
    c = Counter(rows)
    return {"n": len(rows), "accepted": c[True], "rate": c[True] / len(rows) if rows else None}
