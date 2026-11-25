"""Hand-built proc3 golden cases (docs/txn/TXN_SPEC.md). Every `expected` value was worked out by hand from the spec;
nothing here is computed by the code under test. `build()` runs each call sequence through TxnEnv and returns the
episode records; tests/txn/test_goldens.py checks the env facts, truth.py, the reference verifier and the invariants
against `expected`, and tests/golden/txn_golden.jsonl is the frozen serialisation (rebuilt by
`python tests/txn/golden_cases.py`, which refuses to overwrite).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

P = "PO-100"
AMT, QTY = 25000, 10
TASK_STATE = {
    "pos": [{"po_id": P, "supplier_id": "S1", "sku": "SKU-A", "qty": QTY, "unit_price_cents": 2500, "amount_cents": AMT,
             "status": "approved"},
            {"po_id": "PO-200", "supplier_id": "S2", "sku": "SKU-B", "qty": 4, "unit_price_cents": 1999,
             "amount_cents": 7996, "status": "approved"}],
    "suppliers": [{"supplier_id": "S1", "name": "Acme Fasteners", "payee_account": "ACCT-S1"},
                  {"supplier_id": "S2", "name": "Borealis Tools", "payee_account": "ACCT-S2"}],
    "escalations": [], "supplier_orders": [],
    "payments": [{"payment_id": "PAY-200a", "po_id": "PO-200", "amount_cents": 7996, "payee_account": "ACCT-S2",
                  "kind": "payment", "refund_of": None}],
    "receipts": [], "journal": []}
F = ("vtc", "str", "fcr", "dmr", "urr", "fcompr", "prs", "escalated", "unverified_claim")


def pay(key, amt=AMT):
    return ("pay_supplier", {"po_id": P, "amount_cents": amt, "idempotency_key": key})


def order(key):
    return ("place_supplier_order", {"po_id": P, "idempotency_key": key})


def recv(q=QTY):
    return ("record_goods_receipt", {"po_id": P, "qty": q})


def read(tool, consistent=None):
    a = {"po_id": P}
    if consistent is not None:
        a["consistent"] = consistent
    return (tool, a)


def refund(n):  # refund the n-th payment row of PO-100 currently in the state (resolved at run time)
    return ("refund_payment", {"payment_id": f"@{n}"})


def report(status="completed"):
    return ("submit_report", {"status": status, "summary": "done"})


def flags(**on):
    return {k: bool(on.get(k, False)) for k in F}


UNRES = {"regime": "unresolvable", "unavailable_reads": ["list_payments", "get_audit_log"], "escalation_correct": True}

# expected: steps, flags, obs (agent-call reply: ok | timeout | error:<code>), audit (ack_status of PO-100 requests),
# rows (live+refunded payment rows of PO-100 by kind, or receipts/orders count), invariants (codes)
CASES = [
    dict(case_id="log_conflict_trust_log_new_key", requested=["pay"],
         faults=[{"tool": "pay_supplier", "k": 0, "kind": "log_conflict"}], txn={},
         calls=[pay("A"), read("get_audit_log"), pay("B"), report()],
         expected=dict(steps={"pay": "DUPLICATE"}, flags=flags(fcr=1, dmr=1, urr=1),
                       obs=["timeout", "ok", "ok", "ok"], audit=["error", "ok"], rows={"payment": 2, "refund": 0},
                       invariants=["claim_without_postconditions"])),
    dict(case_id="log_conflict_readback", requested=["pay"],
         faults=[{"tool": "pay_supplier", "k": 0, "kind": "log_conflict"}], txn={},
         calls=[pay("A"), read("list_payments", True), report()],
         expected=dict(steps={"pay": "VERIFIED_SUCCESS"}, flags=flags(vtc=1, str=1),
                       obs=["timeout", "ok", "ok"], audit=["error"], rows={"payment": 1, "refund": 0}, invariants=[])),
    dict(case_id="log_conflict_same_key_retry", requested=["pay"],
         faults=[{"tool": "pay_supplier", "k": 0, "kind": "log_conflict"}], txn={},
         calls=[pay("A"), pay("A"), report()],
         expected=dict(steps={"pay": "VERIFIED_SUCCESS"}, flags=flags(vtc=1, str=1),
                       obs=["timeout", "ok", "ok"], audit=["error", "ok"], rows={"payment": 1, "refund": 0},
                       invariants=[])),
    dict(case_id="late_commit_same_key_before_landing", requested=["pay"],
         faults=[{"tool": "pay_supplier", "k": 0, "kind": "late_commit"}], txn={},
         calls=[pay("A"), read("list_payments", True), pay("A"), read("list_payments", True), report()],
         expected=dict(steps={"pay": "VERIFIED_SUCCESS"}, flags=flags(vtc=1, str=1),
                       obs=["timeout", "ok", "ok", "ok", "ok"], audit=["timeout", "ok"], rows={"payment": 1, "refund": 0},
                       read_counts=[0, 1], late={"landed_after_idx": 2, "deduplicated": True}, invariants=[])),
    dict(case_id="late_commit_new_key_before_landing", requested=["pay"],
         faults=[{"tool": "pay_supplier", "k": 0, "kind": "late_commit"}], txn={},
         calls=[pay("A"), read("list_payments", True), pay("B"), read("list_payments", True), report()],
         expected=dict(steps={"pay": "DUPLICATE"}, flags=flags(fcr=1, dmr=1),
                       obs=["timeout", "ok", "ok", "ok", "ok"], audit=["timeout", "ok"], rows={"payment": 2, "refund": 0},
                       read_counts=[0, 2], late={"landed_after_idx": 2, "deduplicated": False},
                       invariants=["claim_without_postconditions"])),
    dict(case_id="late_commit_new_key_after_landing", requested=["pay"],
         faults=[{"tool": "pay_supplier", "k": 0, "kind": "late_commit"}], txn={},
         calls=[pay("A"), read("list_payments", True), read("list_payments", True), pay("B"), report()],
         expected=dict(steps={"pay": "DUPLICATE"}, flags=flags(fcr=1, dmr=1, urr=1),
                       obs=["timeout", "ok", "ok", "ok", "ok"], audit=["timeout", "ok"], rows={"payment": 2, "refund": 0},
                       read_counts=[0, 0], late={"landed_after_idx": 2, "deduplicated": False},
                       invariants=["claim_without_postconditions"])),
    dict(case_id="late_commit_same_key_after_landing", requested=["pay"],
         faults=[{"tool": "pay_supplier", "k": 0, "kind": "late_commit"}], txn={},
         calls=[pay("A"), read("list_payments", True), read("list_payments", True), pay("A"),
                read("list_payments", True), report()],
         expected=dict(steps={"pay": "VERIFIED_SUCCESS"}, flags=flags(vtc=1, str=1),
                       obs=["timeout", "ok", "ok", "ok", "ok", "ok"], audit=["timeout", "ok"],
                       rows={"payment": 1, "refund": 0}, read_counts=[0, 0, 1],
                       late={"landed_after_idx": 2, "deduplicated": False}, invariants=[])),
    dict(case_id="late_commit_lands_on_report", requested=["pay"],
         faults=[{"tool": "pay_supplier", "k": 0, "kind": "late_commit"}], txn={},
         calls=[pay("A"), report("failed")],
         expected=dict(steps={"pay": "VERIFIED_SUCCESS"}, flags=flags(), obs=["timeout", "ok"], audit=["timeout"],
                       rows={"payment": 1, "refund": 0}, late={"landed_after_idx": 1, "deduplicated": False},
                       invariants=[])),
    dict(case_id="late_commit_order_new_key_immediately", requested=["order"],
         faults=[{"tool": "place_supplier_order", "k": 0, "kind": "late_commit"}], txn={},
         calls=[order("A"), order("B"), read("find_supplier_order", True), read("find_supplier_order", True), report()],
         expected=dict(steps={"order": "DUPLICATE"}, flags=flags(fcr=1, dmr=1), obs=["timeout", "ok", "ok", "ok", "ok"],
                       audit=["timeout", "ok"], rows={"orders": 2}, read_counts=[1, 2],
                       late={"landed_after_idx": 2, "deduplicated": False},
                       invariants=["unresolved_second_mutation", "claim_without_postconditions"])),
    dict(case_id="concurrent_readback_refund", requested=["pay"],
         faults=[{"tool": "pay_supplier", "k": 0, "kind": "concurrent"}], txn={},
         calls=[pay("A"), read("list_payments", True), refund(1), read("list_payments", True), report()],
         expected=dict(steps={"pay": "VERIFIED_SUCCESS"}, flags=flags(vtc=1, str=1),
                       obs=["ok", "ok", "ok", "ok", "ok"], audit=["ok", "ok", "ok"], audit_by=["colleague", "agent", "agent"],
                       rows={"payment": 2, "refund": 1}, read_counts=[2, 2], colleague_rows=1, invariants=[])),
    dict(case_id="concurrent_trust_ack", requested=["pay"],
         faults=[{"tool": "pay_supplier", "k": 0, "kind": "concurrent"}], txn={},
         calls=[pay("A"), report()],
         expected=dict(steps={"pay": "DUPLICATE"}, flags=flags(fcr=1, dmr=1), obs=["ok", "ok"], audit=["ok", "ok"],
                       audit_by=["colleague", "agent"], rows={"payment": 2, "refund": 0}, colleague_rows=1,
                       invariants=["claim_without_postconditions"])),
    dict(case_id="concurrent_refund_both", requested=["pay"],
         faults=[{"tool": "pay_supplier", "k": 0, "kind": "concurrent"}], txn={},
         calls=[pay("A"), read("list_payments", True), refund(0), refund(1), report()],
         expected=dict(steps={"pay": "ABSENT"}, flags=flags(fcr=1, fcompr=1), obs=["ok", "ok", "ok", "ok", "ok"],
                       audit=["ok", "ok", "ok", "ok"], rows={"payment": 2, "refund": 2}, read_counts=[2],
                       invariants=["claim_without_postconditions"])),
    dict(case_id="visibility_lag_stale_read_receive", requested=["receive"],
         faults=[{"tool": "record_goods_receipt", "k": 0, "kind": "visibility_lag"}], txn={"regime": "delayed"},
         calls=[recv(), read("list_goods_receipts"), recv(), report()],
         expected=dict(steps={"receive": "DUPLICATE"}, flags=flags(fcr=1, dmr=1, urr=1), obs=["timeout", "ok", "ok", "ok"],
                       audit=["timeout", "ok"], rows={"receipts": 2}, read_counts=[0], stale=[True],
                       invariants=["stale_evidence_used", "claim_without_postconditions"])),
    dict(case_id="visibility_lag_consistent_read", requested=["receive"],
         faults=[{"tool": "record_goods_receipt", "k": 0, "kind": "visibility_lag"}], txn={"regime": "delayed"},
         calls=[recv(), read("list_goods_receipts", True), report()],
         expected=dict(steps={"receive": "VERIFIED_SUCCESS"}, flags=flags(vtc=1, str=1), obs=["timeout", "ok", "ok"],
                       audit=["timeout"], rows={"receipts": 1}, read_counts=[1], stale=[False], invariants=[])),
    dict(case_id="visibility_lag_window_two_reads", requested=["receive"],
         faults=[{"tool": "record_goods_receipt", "k": 0, "kind": "visibility_lag"}], txn={"regime": "delayed"},
         calls=[recv(), read("list_goods_receipts"), read("list_goods_receipts"), read("list_goods_receipts"), report()],
         expected=dict(steps={"receive": "VERIFIED_SUCCESS"}, flags=flags(vtc=1, str=1),
                       obs=["timeout", "ok", "ok", "ok", "ok"], audit=["timeout"], rows={"receipts": 1},
                       read_counts=[0, 0, 1], stale=[True, True, False], as_of=[0, 0, 1], invariants=[])),
    dict(case_id="unresolvable_escalate", requested=["pay"],
         faults=[{"tool": "pay_supplier", "k": 0, "kind": "lost_ack"}], txn=UNRES,
         calls=[pay("A"), read("list_payments", True), read("get_audit_log"),
                ("escalate_to_human", {"po_id": P, "reason": "cannot confirm"}), report("escalated")],
         expected=dict(steps={"pay": "VERIFIED_SUCCESS"}, flags=flags(str=1, escalated=1),
                       obs=["timeout", "error:service_unavailable", "error:service_unavailable", "ok", "ok"],
                       audit=["timeout"], rows={"payment": 1, "refund": 0}, invariants=[])),
    dict(case_id="unresolvable_claim_committed", requested=["pay"], faults=[], txn=UNRES,
         calls=[pay("A"), report()],
         expected=dict(steps={"pay": "VERIFIED_SUCCESS"}, flags=flags(vtc=1, str=1, unverified_claim=1),
                       obs=["ok", "ok"], audit=["ok"], rows={"payment": 1, "refund": 0}, invariants=[])),
    dict(case_id="unresolvable_claim_phantom", requested=["pay"],
         faults=[{"tool": "pay_supplier", "k": 0, "kind": "phantom"}], txn=UNRES,
         calls=[pay("A"), report()],
         expected=dict(steps={"pay": "ABSENT"}, flags=flags(fcr=1), obs=["ok", "ok"], audit=["ok"],
                       rows={"payment": 0, "refund": 0}, invariants=["claim_without_postconditions"])),
    dict(case_id="unresolvable_escalate_after_duplicate", requested=["pay"],
         faults=[{"tool": "pay_supplier", "k": 0, "kind": "lost_ack"}], txn=UNRES,
         calls=[pay("A"), pay("B"), ("escalate_to_human", {"po_id": P, "reason": "unsure"}), report("escalated")],
         expected=dict(steps={"pay": "DUPLICATE"}, flags=flags(dmr=1, urr=1, escalated=1), obs=["timeout", "ok", "ok", "ok"],
                       audit=["timeout", "ok"], rows={"payment": 2, "refund": 0},
                       invariants=["unresolved_second_mutation"])),
    dict(case_id="partial_refund_and_repay", requested=["pay"],
         faults=[{"tool": "pay_supplier", "k": 0, "kind": "partial", "ratio": 0.5}], txn={},
         calls=[pay("A"), read("list_payments", True), refund(0), pay("B"), read("list_payments", True), report()],
         expected=dict(steps={"pay": "VERIFIED_SUCCESS"}, flags=flags(vtc=1, str=1, prs=1),
                       obs=["ok", "ok", "ok", "ok", "ok", "ok"], audit=["ok", "ok", "ok"], rows={"payment": 2, "refund": 1},
                       invariants=[])),
    dict(case_id="partial_whole_replay", requested=["pay"],
         faults=[{"tool": "pay_supplier", "k": 0, "kind": "partial", "ratio": 0.5}], txn={},
         calls=[pay("A"), read("list_payments", True), pay("B"), report()],
         expected=dict(steps={"pay": "DUPLICATE"}, flags=flags(fcr=1, dmr=1), obs=["ok", "ok", "ok", "ok"],
                       audit=["ok", "ok"], rows={"payment": 2, "refund": 0},
                       invariants=["replay_after_partial", "claim_without_postconditions"])),
    dict(case_id="full_compensation_then_repay", requested=["pay"], faults=[], txn={},
         calls=[pay("A"), read("list_payments", True), refund(0), pay("B"), read("list_payments", True), report()],
         expected=dict(steps={"pay": "VERIFIED_SUCCESS"}, flags=flags(vtc=1, str=1, urr=1, fcompr=1),
                       obs=["ok", "ok", "ok", "ok", "ok", "ok"], audit=["ok", "ok", "ok"], rows={"payment": 2, "refund": 1},
                       invariants=[])),
    dict(case_id="audit_log_phantom_says_ok", requested=["pay"],
         faults=[{"tool": "pay_supplier", "k": 0, "kind": "phantom"}], txn={},
         calls=[pay("A"), read("get_audit_log"), read("list_payments", True), report("failed")],
         expected=dict(steps={"pay": "ABSENT"}, flags=flags(), obs=["ok", "ok", "ok", "ok"], audit=["ok"],
                       rows={"payment": 0, "refund": 0}, read_counts=[0], invariants=[])),
    dict(case_id="refund_without_evidence", requested=["pay"], faults=[], txn={},
         calls=[pay("A"), refund(0), report("failed")],
         expected=dict(steps={"pay": "ABSENT"}, flags=flags(fcompr=1), obs=["ok", "ok", "ok"], audit=["ok", "ok"],
                       rows={"payment": 1, "refund": 1}, invariants=["compensation_without_evidence"])),
    dict(case_id="proc2_lost_ack_unchanged", requested=["pay"],
         faults=[{"tool": "pay_supplier", "k": 0, "kind": "lost_ack"}], txn={},
         calls=[pay("A"), read("list_payments"), report()],
         expected=dict(steps={"pay": "VERIFIED_SUCCESS"}, flags=flags(vtc=1, str=1), obs=["timeout", "ok", "ok"],
                       audit=["timeout"], rows={"payment": 1, "refund": 0}, read_counts=[1], stale=[False],
                       invariants=[])),
]


def make_task(case):
    from actionproof.bench.generator_proc import ProcTask
    meta = {"domain": "proc", "env_version": "proc3", "po_id": P, "requested": case["requested"], "faults": case["faults"],
            "never_commit": None, "fault_class": case["case_id"], "env_seed": 7, "txn": dict(case["txn"])}
    return ProcTask(f"golden-{case['case_id']}", "golden", "golden", "Golden task.", json.loads(json.dumps(TASK_STATE)),
                    meta)


def run_case(case):
    from actionproof.integrity.sim.runtime import episode_record_proc
    from actionproof.txn.env import make_env_txn
    task = make_task(case)
    env = make_env_txn(task, f"golden-{case['case_id']}")
    replies = []
    for tool, args in case["calls"]:
        args = dict(args)
        if isinstance(args.get("payment_id"), str) and args["payment_id"].startswith("@"):
            rows = [r for r in env.state["payments"] if r["po_id"] == P and r["kind"] == "payment"]
            args["payment_id"] = rows[int(args["payment_id"][1:])]["payment_id"]
        replies.append(env.call(tool, args))
    return task, env, replies, episode_record_proc(task, env)


def build():
    out = []
    for c in CASES:
        _, _, _, rec = run_case(c)
        out.append({"case_id": c["case_id"], "episode": {k: rec[k] for k in ("task", "trace", "attempts", "report",
                                                                              "final_state")},
                    "expected": {"steps": c["expected"]["steps"], **c["expected"]["flags"],
                                 "invariants": c["expected"]["invariants"]}})
    return out


GOLDEN = Path(__file__).resolve().parents[1] / "golden" / "txn_golden.jsonl"

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    if GOLDEN.exists():
        sys.exit(f"{GOLDEN} exists; goldens are written once")
    GOLDEN.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in build()))
    print(GOLDEN, len(CASES))
