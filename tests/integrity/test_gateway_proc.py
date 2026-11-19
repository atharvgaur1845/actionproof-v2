"""ProcureSim gateway and evidence ledger: tamper detection, non-interference, no false finalization, recovery."""
from __future__ import annotations

import copy
import random

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from actionproof.bench.generator_proc import make_split
from actionproof.integrity.contracts import Contract
from actionproof.integrity.gateway import MODES, ProcGateway
from actionproof.integrity.ledger import EvidenceLedger
from actionproof.integrity.sim.agents import naive, oracle
from actionproof.integrity.sim.env import STEP_TOOL
from actionproof.integrity.sim.runtime import episode_record_proc, make_env_proc

bodies = st.dictionaries(st.text(min_size=1, max_size=5), st.integers() | st.text(max_size=8), max_size=4)


@settings(max_examples=60, deadline=None)
@given(st.lists(bodies, min_size=2, max_size=25), st.data())
def test_ledger_detects_any_single_modification(evs, data):
    led = EvidenceLedger()
    for b in evs:
        led.append("tx", "evidence", "verifier", b)
    assert led.verify()["ok"]
    seq = data.draw(st.integers(min_value=1, max_value=len(evs)))
    led.tamper(seq, {"tampered": True, "seq": seq})
    v = led.verify()
    assert not v["ok"] and v["first_bad_seq"] == seq


def test_ledger_rejects_update_and_delete_and_anchors_catch_truncation(tmp_path):
    led = EvidenceLedger(tmp_path / "l.db")
    for i in range(5):
        led.append("tx", "evidence", "verifier", {"i": i})
    with pytest.raises(Exception):
        led.conn.execute("DELETE FROM events WHERE seq = 5")
    a = led.anchor(tmp_path / "anchors.jsonl")
    assert a["seq"] == 5 and led.verify()["ok"]
    led.conn.execute("DROP TRIGGER no_delete")
    led.conn.execute("DELETE FROM events WHERE seq = 5")
    v = led.verify()
    assert not v["ok"] and v.get("anchor_mismatch")


def _run(task, mode, policy):
    env = make_env_proc(task, "g")
    gw = ProcGateway(env, Contract.for_task(task), mode)
    policy(gw, task)
    return env, gw, episode_record_proc(task, env)


@pytest.mark.parametrize("policy", [naive, oracle])
def test_observe_never_changes_what_the_agent_sees_or_the_state(policy):
    for t in make_split("proc_dev", 60) + make_split("proc_stress", 20):
        env_p, _, _ = _run(t, "passthrough", policy)
        env_o, gw, _ = _run(t, "observe", policy)
        assert env_p.trace == env_o.trace and env_p.state == env_o.state
        assert gw.ledger.verify()["ok"]


@pytest.mark.parametrize("mode", ["observe", "enforce", "enforce_inform"])
@pytest.mark.parametrize("policy", [naive, oracle])
def test_gateway_never_finalizes_success_unless_the_goal_holds(mode, policy):
    for split in ("proc_dev", "proc_stress", "proc_heldout_service"):
        for t in make_split(split, 50):
            env, gw, rec = _run(t, mode, policy)
            if gw.disposition()["status"] == "completed":
                assert rec["score"]["goal_met"], t.task_id


def test_enforce_recovers_phantom_and_escalates_never_commit():
    base = make_split("proc_dev", 30)
    t = copy.deepcopy(next(x for x in base if "pay" in x.meta["requested"]))
    t.meta.update(faults=[{"tool": "pay_supplier", "k": 0, "kind": "phantom"}], never_commit=None)
    _, gw, rec = _run(t, "enforce", naive)
    assert rec["score"]["goal_met"] and gw.disposition()["status"] == "completed"
    assert any(d["action"] == "SAFE_RETRY" for d in gw.decisions)
    t2 = copy.deepcopy(t)
    t2.meta.update(faults=[], never_commit="pay_supplier")
    _, gw2, rec2 = _run(t2, "enforce", naive)
    assert gw2.disposition()["status"] == "escalated" and not rec2["score"]["goal_met"]
    assert any(d["action"] == "ESCALATE" for d in gw2.decisions)


def test_enforce_compensates_duplicate_payment_and_partial_payment():
    base = make_split("proc_dev", 40)
    t = copy.deepcopy(next(x for x in base if "pay" in x.meta["requested"]))
    for kind in ("dup_delivery", "partial"):
        t.meta.update(faults=[{"tool": "pay_supplier", "k": 0, "kind": kind}], never_commit=None)
        _, gw, rec = _run(t, "enforce", naive)
        assert rec["score"]["steps"]["pay"] == "VERIFIED_SUCCESS", kind
        assert any(d["action"] == "COMPENSATE" for d in gw.decisions)


def test_enforce_rejects_out_of_scope_and_over_limit_calls():
    t = make_split("proc_dev", 1)[0]
    env = make_env_proc(t, "g")
    gw = ProcGateway(env, Contract.for_task(t), "enforce")
    other = next(p["po_id"] for p in t.initial_state["pos"] if p["po_id"] != t.meta["po_id"])
    assert gw.call("get_purchase_order", {"po_id": other})["error"] == "operation_not_permitted"
    first = t.meta["requested"][0]
    args = {"po_id": t.meta["po_id"]}
    if first in ("pay", "post"):
        args["amount_cents"] = 10**12
    elif first == "receive":
        args["qty"] = 1
    r = gw.call(STEP_TOOL[first], args)
    if first in ("pay", "post"):
        assert r["error"] == "operation_not_permitted"
    assert all(e.kind != "dispatched" or e.body["step"] == first for e in gw.ledger.events())


def test_enforce_blocks_a_step_before_its_commit_point():
    t = next(x for x in make_split("proc_dev", 60) if x.workflow == "order_prepay")
    env = make_env_proc(t, "g")
    gw = ProcGateway(env, Contract.for_task(t), "enforce")
    po = next(p for p in t.initial_state["pos"] if p["po_id"] == t.meta["po_id"])
    r = gw.call("pay_supplier", {"po_id": po["po_id"], "amount_cents": po["amount_cents"]})
    assert r["error"] == "operation_not_permitted" and not env.state["payments"][len(t.initial_state["payments"]):]


def test_random_fault_plans_never_leave_two_live_payments_under_enforce():
    rng = random.Random(5)
    kinds = ("ok", "phantom", "partial", "lost_ack", "precommit_timeout", "dup_delivery")
    for t in make_split("proc_val", 60):
        t = copy.deepcopy(t)
        t.meta.update(faults=[{"tool": STEP_TOOL[s], "k": k, "kind": rng.choice(kinds)}
                              for s in t.meta["requested"] for k in range(2)], never_commit=None)
        for policy in (naive, oracle):
            _, gw, rec = _run(t, "enforce", policy)
            if "pay" in t.meta["requested"]:
                assert rec["score"]["steps"]["pay"] != "DUPLICATE", t.task_id
            assert gw.ledger.verify()["ok"]


def test_modes_are_the_registered_four():
    assert MODES == ("passthrough", "observe", "enforce", "enforce_inform")
