"""Scripted baselines on txn_dev: B4 sanity, B0 false completions, budget, verifier agreement, decision records."""
from __future__ import annotations

import importlib.util
import json
from functools import lru_cache
from pathlib import Path

import pytest

from actionproof.bench.generator_proc import DATA, load_tasks
from actionproof.txn import truth
from actionproof.txn.policies import BUDGET, POLICIES, run_policy

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("reference_verifier_txn", ROOT / "scripts/reference_verifier_txn.py")
REF = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(REF)


@lru_cache(maxsize=None)
def dev():
    return load_tasks(DATA / "txn_dev.jsonl")


@lru_cache(maxsize=None)
def runs(name):
    return [run_policy(name, t) for t in dev()]


def test_b4_reaches_str_1_on_every_dev_class_without_invariant_violations():
    recs = runs("B4")
    classes = {t.meta["fault_class"] for t in dev()}
    assert len(classes) == 12  # 8 core classes + 4 quadruple realities
    for t, r in zip(dev(), recs):
        assert r["txn"]["str"], (t.task_id, t.meta["fault_class"], r["score"]["steps"], r["report"])
        assert not r["invariants"], (t.task_id, r["invariants"])
    unres = [r for r in recs if r["txn"]["unresolvable"]]
    assert unres and all(r["txn"]["escalated"] for r in unres)


def test_b0_false_completions_on_phantom():
    ph = [r for t, r in zip(dev(), runs("B0")) if t.meta["fault_class"] == "phantom"]
    assert ph and all(r["txn"]["fcr"] for r in ph)


def test_ablations_differ_from_b4_where_designed():
    s = {n: truth.summarise([r["txn"] for r in runs(n)]) for n in ("B4", "A2", "A3", "A4", "A5")}
    assert s["A4"]["PRS"]["rate"] < s["B4"]["PRS"]["rate"] == 1.0
    assert s["A3"]["DMR_all"]["num"] > 0 and s["A2"]["DMR_all"]["num"] > 0
    assert s["A5"]["CER"]["rate"] == 0.0 and s["B4"]["CER"]["rate"] == 1.0
    assert [r["txn"]["str"] for r in runs("A6")] == [r["txn"]["str"] for r in runs("B4")]


@pytest.mark.parametrize("name", sorted(POLICIES))
def test_budget_report_and_reference_agreement(name):
    for r in runs(name):
        agent = [s for s in r["trace"] if s["by"] == "agent"]
        assert len(agent) <= BUDGET
        assert r["report"] is not None and agent[-1]["tool"] == "submit_report"
        ref = REF.verify(r)
        for k in REF.KEYS:
            assert bool(r["txn"][k]) == bool(ref[k]), (r["task"]["task_id"], k)
        assert ref["steps"] == r["score"]["steps"]


def test_decision_record_fields_and_wording():
    r = runs("B4")[0]
    d = r["decision_record"]
    for k in ("episode_id", "operations", "intent", "postconditions", "evidence", "conflicts", "disposition",
              "recovery_actions", "post_recovery_verification", "policy_notes"):
        assert k in d, k
    assert d["operations"][0]["idempotency_key"] and "ack" in d["operations"][0]
    text = json.dumps([x["decision_record"] for x in runs("B4")]).lower()
    assert "proof" not in text and "certificate" not in text


def test_decision_record_sees_conflicts():
    kinds = {}
    for t, r in zip(dev(), runs("B4")):
        for c in r["decision_record"]["conflicts"]:
            kinds.setdefault(c["kind"], set()).add(t.meta["fault_class"])
    assert "concurrent" in kinds["extra_effect"] and "log_conflict" in kinds["ack_vs_ledger"]
    b2 = {t.meta["fault_class"] for t, r in zip(dev(), runs("B2"))
          for c in r["decision_record"]["conflicts"] if c["kind"] == "audit_vs_ledger"}
    assert not b2  # B2 never reads the ledger, so it can see no audit/ledger conflict
