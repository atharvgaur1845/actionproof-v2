"""Gate v2 (RG2): bounded re-read under an outage; no hidden-state access; v1 unchanged."""
from __future__ import annotations

import ast
from pathlib import Path

from actionproof.bench.generator_proc import DATA, load_tasks
from actionproof.recovery_gate.rg2_env import make_env_rg2, run_policy_rg2

ROOT = Path(__file__).resolve().parents[2]
TRAIN = load_tasks(DATA / "sel1_train.jsonl")


def _find(regime, step="pay"):
    return next(t for t in TRAIN if t.meta["txn"].get("regime") == regime and t.meta["txn"].get("quad_step") == step)


def test_v2_reads_through_a_transient_outage_and_v1_hands_off():
    t = _find("transient")
    assert run_policy_rg2("B4", t, "v1")["txn"]["escalated"]
    r = run_policy_rg2("B4", t, "v2")
    assert not r["txn"]["escalated"] and r["txn"]["str"]
    assert any(s.get("gate") == "gate_reread" for s in r["trace"])


def test_v2_still_hands_off_under_a_permanent_outage_after_at_most_three_reads():
    t = _find("unresolvable")
    r = run_policy_rg2("B4", t, "v2")
    assert r["txn"]["escalated"] and r["txn"]["str"]
    reads = [s for s in r["trace"] if s["tool"] == "list_payments" and s.get("by", "agent") == "agent"]
    assert sum(s["obs"].get("error") == "service_unavailable" for s in reads) <= 3 + 1


def test_v2_module_reads_no_hidden_state():
    tree = ast.parse((ROOT / "actionproof/recovery_gate/gate_v2.py").read_text())
    bad = {"state", "attempts", "fault_plan", "txn_plan", "trace", "idem", "faults", "regime", "unavailable",
           "transient_left", "txn_cfg", "escalation_correct", "reports"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            assert node.attr not in bad, node.attr
    mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert mods <= {"__future__", "typing", "actionproof.integrity.sim.env", "actionproof.recovery_gate.gate"}, mods


def test_no_gate_rg2_env_matches_sel1_env_for_b4():
    from actionproof.sel1.policy import run_policy
    for t in TRAIN[:40]:
        ref, _ = run_policy("B4", t)
        r = run_policy_rg2("B4", t, None)
        assert r["trace"] == ref["trace"] and r["txn"] == ref["txn"]
