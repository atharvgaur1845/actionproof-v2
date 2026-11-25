"""TxnEnv state transitions not covered by the goldens: audit log fields, read precedence, regimes, surfaces."""
from __future__ import annotations

import inspect

from actionproof.txn.env import (ALIAS_TOOLS, SCHEMA_ARGS, TXN_AGENT_TOOLS, UNAVAILABLE, TxnEnv, TxnSurfaceEnv,
                                 held_out_names)

from tests.txn.golden_cases import AMT, P, QTY, TASK_STATE


def env(faults=(), **txn):
    return TxnEnv(TASK_STATE, list(faults), "r", seed=7, txn=txn)


def test_tool_list_has_audit_log_and_consistent_flag():
    e = env()
    names = [t.__name__ for t in e.tools()]
    assert names == list(TXN_AGENT_TOOLS) and "get_audit_log" in names
    lp = next(t for t in e.tools() if t.__name__ == "list_payments")
    p = inspect.signature(lp).parameters["consistent"]
    assert p.default is False and "consistent:" in lp.__doc__


def test_audit_log_entry_fields_and_order():
    e = env()
    e.call("pay_supplier", {"po_id": P, "amount_cents": AMT, "idempotency_key": "K"})
    e.call("record_goods_receipt", {"po_id": P, "qty": QTY})
    e.call("place_supplier_order", {"po_id": P})
    e.call("pay_supplier", {"po_id": P, "amount_cents": AMT, "idempotency_key": "K"})  # store reply, still logged
    e.call("pay_supplier", {"po_id": "PO-404", "amount_cents": AMT})  # not found: not logged
    log = e.call("get_audit_log", {"po_id": P})
    assert log["as_of"] == 4
    r = log["requests"]
    assert list(r[0]) == ["request_id", "tool", "idempotency_key", "amount_cents", "ack_status", "by"]
    assert [x["request_id"] for x in r] == ["REQ-0001", "REQ-0002", "REQ-0003", "REQ-0004"]
    assert r[1]["qty"] == QTY and "amount_cents" not in r[2] and r[3]["ack_status"] == "ok"
    assert e.call("get_audit_log", {"po_id": "PO-404"}) == {"status": "error", "error": "not_found"}


def test_unavailable_precedes_not_found_and_covers_audit():
    e = env(unavailable_reads=["list_payments", "get_audit_log"])
    assert e.call("list_payments", {"po_id": "PO-404"}) == UNAVAILABLE
    assert e.call("get_audit_log", {"po_id": P}) == UNAVAILABLE
    assert e.call("list_goods_receipts", {"po_id": P})["status"] == "ok"


def test_consistent_must_be_bool():
    e = env()
    assert e.call("list_payments", {"po_id": P, "consistent": "yes"})["error"] == "invalid_arguments"


def test_consistent_read_bypasses_replica_lag_and_window():
    e = env([{"tool": "list_payments", "k": 1, "kind": "replica_lag"}])
    e.call("pay_supplier", {"po_id": P, "amount_cents": AMT, "idempotency_key": "K"})
    assert len(e.call("list_payments", {"po_id": P})["payments"]) == 1          # k=0 primary
    assert len(e.call("list_payments", {"po_id": P})["payments"]) == 0          # k=1 replica_lag
    assert len(e.call("list_payments", {"po_id": P, "consistent": True})["payments"]) == 1


def test_delayed_window_applies_per_service_and_resets_on_commit():
    e = env(regime="delayed")
    e.call("pay_supplier", {"po_id": P, "amount_cents": AMT, "idempotency_key": "K"})
    assert e.call("get_journal_entries", {"po_id": P})["as_of"] == 0  # other service: unaffected
    assert e.trace[-1]["txn"]["stale"] is False
    assert e.call("list_payments", {"po_id": P})["payments"] == []
    e.call("pay_supplier", {"po_id": P, "amount_cents": AMT, "idempotency_key": "K2"})  # new commit: new window
    a, b, c = (len(e.call("list_payments", {"po_id": P})["payments"]) for _ in range(3))
    assert (a, b, c) == (1, 1, 2)


def test_concurrent_colleague_first_and_counter_untouched():
    e = env([{"tool": "pay_supplier", "k": 0, "kind": "concurrent"}])
    obs = e.call("pay_supplier", {"po_id": P, "amount_cents": AMT, "idempotency_key": "K"})
    assert obs["status"] == "ok" and e.calls["pay_supplier"] == 1
    assert [s["by"] for s in e.trace] == ["colleague", "agent"] and [s["idx"] for s in e.trace] == [0, 1]
    pays = e.call("list_payments", {"po_id": P})["payments"]
    assert pays[0]["by"] == "colleague" and "by" not in pays[1] and pays[1]["payment_id"] == obs["payment_id"]
    assert e.trace[0]["args"]["idempotency_key"] == f"COL-{P}"
    assert e.attempts[0]["by"] == "colleague" and e.attempts[1]["fault"] == "concurrent"
    # a second agent pay with the same key is a store reply: no third row
    e.call("pay_supplier", {"po_id": P, "amount_cents": AMT, "idempotency_key": "K"})
    assert len([r for r in e.state["payments"] if r["po_id"] == P]) == 2


def test_late_commit_ignores_harness_calls_and_amount_not_in_scope():
    e = env([{"tool": "pay_supplier", "k": 0, "kind": "late_commit"}])
    e.call("pay_supplier", {"po_id": P, "amount_cents": AMT, "idempotency_key": "K"})
    e.call("get_purchase_order", {"po_id": P}, by="harness")
    e.call("get_purchase_order", {"po_id": P}, by="harness")
    assert not [r for r in e.state["payments"] if r["po_id"] == P]
    e.call("get_purchase_order", {"po_id": P})
    e.call("bogus_tool", {})  # invalid calls count
    assert len([r for r in e.state["payments"] if r["po_id"] == P]) == 1
    # same key, different amount: the store answers, nothing new (amount is not part of the scope)
    o = e.call("pay_supplier", {"po_id": P, "amount_cents": 5, "idempotency_key": "K"})
    assert o["status"] == "ok" and len([r for r in e.state["payments"] if r["po_id"] == P]) == 1


def test_late_commit_unkeyed_write_always_lands():
    e = env([{"tool": "record_goods_receipt", "k": 0, "kind": "late_commit"}])
    e.call("record_goods_receipt", {"po_id": P, "qty": QTY})
    e.call("record_goods_receipt", {"po_id": P, "qty": QTY})
    e.call("get_purchase_order", {"po_id": P})
    assert sum(r["qty"] for r in e.state["receipts"] if r["po_id"] == P) == 2 * QTY


def test_log_conflict_store_reply_after():
    e = env([{"tool": "pay_supplier", "k": 0, "kind": "log_conflict"}])
    o1 = e.call("pay_supplier", {"po_id": P, "amount_cents": AMT, "idempotency_key": "K"})
    o2 = e.call("pay_supplier", {"po_id": P, "amount_cents": AMT, "idempotency_key": "K"})
    assert o1["error"] == "timeout" and o2["status"] == "ok"
    assert o2["payment_id"] == [r for r in e.state["payments"] if r["po_id"] == P][0]["payment_id"]


def test_ack_text_identical_across_hidden_realities():
    args = {"po_id": P, "amount_cents": AMT, "idempotency_key": "K"}
    oks = [env([{"tool": "pay_supplier", "k": 0, "kind": k}] if k else []).call("pay_supplier", args)
           for k in (None, "phantom", "concurrent")]
    tos = [env([{"tool": "pay_supplier", "k": 0, "kind": k}]).call("pay_supplier", args)
           for k in ("lost_ack", "precommit_timeout", "log_conflict", "late_commit", "visibility_lag")]
    assert oks[0] == oks[1] and oks[0]["status"] == "ok"
    assert all(t == tos[0] for t in tos)


def test_reorder_and_lag_first_reads():
    e = env(reorder_fields=True, lag_first_reads=2)
    e.call("pay_supplier", {"po_id": P, "amount_cents": AMT, "idempotency_key": "K"})
    r1 = e.call("list_payments", {"po_id": P})
    r2 = e.call("list_payments", {"po_id": P})
    r3 = e.call("list_payments", {"po_id": P})
    assert list(r3) == ["as_of", "payments", "status"] and list(r3["payments"][0])[0] == "refund_of"
    assert (len(r1["payments"]), len(r2["payments"]), len(r3["payments"])) == (0, 0, 1)


def test_surfaces_disjoint_from_sr1_s4_and_canonical():
    bad = held_out_names()
    assert not set(ALIAS_TOOLS.values()) & bad and not set(SCHEMA_ARGS.values()) & bad
    assert set(ALIAS_TOOLS) == set(TXN_AGENT_TOOLS)


def _senv(surface, faults=()):
    return TxnSurfaceEnv(TASK_STATE, list(faults), "r", seed=7, txn={}, surface=surface)


def test_alias_surface_round_trip_keeps_canonical_trace():
    e = _senv("alias", [{"tool": "pay_supplier", "k": 0, "kind": "concurrent"}])
    names = [t.__name__ for t in e.tools()]
    assert set(names) == set(ALIAS_TOOLS.values())
    assert e.call("pay_supplier", {"po_id": P})["error"] == "unknown_tool"
    o = e.call("execute_vendor_payment", {"po_id": P, "amount_cents": AMT, "idempotency_key": "K"})
    assert o["status"] == "ok"
    log = e.call("request_audit_trail", {"po_id": P})
    assert {x["tool"] for x in log["requests"]} == {"execute_vendor_payment"}
    assert [s["tool"] for s in e.trace if s["by"] == "agent"][1:] == ["pay_supplier", "get_audit_log"]
    assert "file_final_report" in e.adapt_prompt("x") and "submit_report" not in e.adapt_prompt("x")


def test_schema_surface_renames_arguments_only():
    e = _senv("schema")
    lp = next(t for t in e.tools() if t.__name__ == "list_payments")
    assert list(inspect.signature(lp).parameters) == ["purchase_ref", "read_primary"]
    assert "purchase_ref" in lp.__doc__ and "po_id" not in lp.__doc__
    o = e.call("pay_supplier", {"purchase_ref": P, "amount_in_cents": AMT, "dedup_token": "K"})
    assert o["status"] == "ok"
    r = e.call("list_payments", {"purchase_ref": P, "read_primary": True})
    assert r["payments"][0]["amount_in_cents"] == AMT and "po_id" not in str(r)
    e.call("submit_report", {"report_status": "completed", "report_summary": "x"})
    assert e.final_report["status"] == "completed"
    assert 'report_status "completed"' in e.adapt_prompt("x")
