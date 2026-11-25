"""proc3 goldens: hand-built expectations (tests/txn/golden_cases.py) vs TxnEnv, truth.py, the reference verifier and
the invariants; and the frozen serialisation tests/golden/txn_golden.jsonl vs the verifiers alone."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from actionproof.integrity.sim.env import STEP_READ
from actionproof.txn import invariants, truth

from tests.txn.golden_cases import CASES, F, GOLDEN, P, run_case

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("reference_verifier_txn", ROOT / "scripts/reference_verifier_txn.py")
REF = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(REF)


def _status(o):
    if o.get("status") == "ok":
        return "ok"
    return "timeout" if o.get("error") == "timeout" else f"error:{o.get('error')}"


def _count(tool, obs):
    if tool == "list_payments":
        return sum(p["kind"] == "payment" for p in obs["payments"])
    return len(obs[{"find_supplier_order": "orders", "list_goods_receipts": "receipts",
                    "get_journal_entries": "entries"}[tool]])


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["case_id"])
def test_env_transitions(case):
    task, env, replies, rec = run_case(case)
    e = case["expected"]
    assert [_status(o) for o in replies] == e["obs"]
    audit = [a["entry"] for a in env.audit if a["po_id"] == P]
    assert [x["ack_status"] for x in audit] == e["audit"]
    if "audit_by" in e:
        assert [x["by"] for x in audit] == e["audit_by"]
    st = env.state
    rows = e["rows"]
    if "payment" in rows:
        mine = [r for r in st["payments"] if r["po_id"] == P]
        assert sum(r["kind"] == "payment" for r in mine) == rows["payment"]
        assert sum(r["kind"] == "refund" for r in mine) == rows["refund"]
    if "receipts" in rows:
        assert len([r for r in st["receipts"] if r["po_id"] == P]) == rows["receipts"]
    if "orders" in rows:
        assert len([r for r in st["supplier_orders"] if r["po_id"] == P]) == rows["orders"]
    step_reads = set(STEP_READ.values())
    agent = [s for s in env.trace if s["by"] == "agent"]
    reads = [s for s in agent if s["tool"] in step_reads and s["obs"].get("status") == "ok"]
    if "read_counts" in e:
        assert [_count(s["tool"], s["obs"]) for s in reads] == e["read_counts"]
    if "stale" in e:
        assert [s["txn"]["stale"] for s in reads] == e["stale"]
    if "as_of" in e:
        assert [s["obs"]["as_of"] for s in reads] == e["as_of"]
    if "late" in e:
        late = [a for a in env.attempts if a.get("fault") == "late_commit"]
        assert len(late) == 1 and late[0]["late"] == e["late"]
    if "colleague_rows" in e:
        assert sum(r.get("by") == "colleague" for r in st["payments"]) == e["colleague_rows"]
        assert sum(s["by"] == "colleague" for s in env.trace) == 1


def _check(rec, exp):
    got = truth.txn_score(rec)
    ref = REF.verify(rec)
    assert truth.score(rec)["steps"] == exp["steps"]
    assert ref["steps"] == exp["steps"]
    for k in F:
        assert bool(got[k]) == exp[k], ("truth", k)
        assert bool(ref[k]) == exp[k], ("reference", k)
    assert sorted(v["code"] for v in invariants.check(rec)) == sorted(exp["invariants"])


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["case_id"])
def test_verifiers_match_hand_expectations(case):
    _, _, _, rec = run_case(case)
    exp = {"steps": case["expected"]["steps"], **case["expected"]["flags"], "invariants": case["expected"]["invariants"]}
    _check(rec, exp)


def _golden():
    return [json.loads(line) for line in GOLDEN.read_text().splitlines() if line.strip()]


@pytest.mark.parametrize("g", _golden(), ids=lambda g: g["case_id"])
def test_frozen_golden_file(g):
    _check(g["episode"], g["expected"])


def test_golden_file_covers_every_case_and_new_fault():
    ids = {g["case_id"] for g in _golden()}
    assert ids == {c["case_id"] for c in CASES}
    kinds = {f["kind"] for c in CASES for f in c["faults"]}
    assert {"log_conflict", "late_commit", "concurrent", "visibility_lag"} <= kinds
    assert any(c["txn"].get("regime") == "unresolvable" for c in CASES)


def test_money_conserved_on_every_golden():
    for c in CASES:
        _, _, _, rec = run_case(c)
        assert not invariants.money_conservation(rec), c["case_id"]
