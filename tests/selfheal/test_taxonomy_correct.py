"""Taxonomy and B4-teacher corrections on scripted failure episodes (P1 dev and TXN dev tasks)."""
from __future__ import annotations

import copy

import pytest

from actionproof.integrity.sim.agents import _args, _po
from actionproof.integrity.sim.env import STEP_READ, STEP_TOOL
from actionproof.selfheal import taxonomy
from actionproof.selfheal.correct import SplitRefused, correct_episode
from actionproof.selfheal.envs import spec_for
from actionproof.selfheal.replay import ReplayMismatch, replay, turn_before
from actionproof.selfheal.teachers import B4Teacher
from tests.selfheal.helpers import find, simulate


def naive_calls(task, new_key_on_timeout: bool = True, report: str = "completed") -> list:
    """Write each step; on a non-ok ack write again with a new key; never read; report."""
    po = _po(task)
    out = []
    for step in task.meta["requested"]:
        out.append((STEP_TOOL[step], _args(step, po, f"{step}-{po['po_id']}")))
    return out + [("submit_report", {"status": report, "summary": "done"})]


def run_naive(task):
    """Like naive_calls, but reacting to acks: a timed-out write is retried once with a new key."""
    from actionproof.selfheal.replay import act_text
    po = _po(task)
    spec = spec_for(task)
    env = spec.make(task, "probe")
    calls = []
    for step in task.meta["requested"]:
        a = _args(step, po, f"{step}-{po['po_id']}")
        calls.append((STEP_TOOL[step], a))
        if env.call(STEP_TOOL[step], a).get("status") != "ok":
            b = _args(step, po, f"{step}-{po['po_id']}-new")
            calls.append((STEP_TOOL[step], b))
            env.call(STEP_TOOL[step], b)
    calls.append(("submit_report", {"status": "completed", "summary": "done"}))
    assert act_text  # the record is rebuilt by simulate from the same calls
    return simulate(task, calls)


def classify(rec):
    spec = spec_for(_task(rec))
    return taxonomy.classify(rec, spec.success(spec.annotate(rec)))


def _task(rec):
    from actionproof.selfheal.envs import task_from_record
    return task_from_record(rec)


def p1_task(cls, **kw):
    return find("p1_dev_full", lambda t: t.meta["fault_class"] == cls and all(t.meta.get(k) == v for k, v in kw.items()))


# ---------------------------------------------------------------------- taxonomy
def test_lost_ack_new_key_retry_is_unsafe_retry_and_correctable():
    t = find("p1_dev_full", lambda t: t.meta["fault_class"] == "lost_ack"
             and t.meta["faults"][0]["tool"] == "pay_supplier")
    rec = run_naive(t)
    c = classify(rec)
    assert c["primary"] == "unsafe_retry" and c["flags"]["lost_ack_seen"] and c["flags"]["false_completion"]
    second_pay = [s["idx"] for s in rec["trace"] if s["tool"] == "pay_supplier"][1]
    assert c["first_bad_idx"] == second_pay
    r = correct_episode(rec, B4Teacher())
    assert r["status"] == "accepted" and r["cut_kind"] == "primary", r
    assert r["verified"]["vsc"] == 1
    # the corrected trajectory never pays twice: the teacher read the payment instead
    teacher_tools = [m["content"] for m in r["conversation"][r["prefix_len"]::2]]
    assert not any("pay_supplier" in x for x in teacher_tools)


def test_phantom_trusted_is_phantom_write_and_retry_reuses_key():
    t = find("p1_dev_full", lambda t: t.meta["fault_class"] == "phantom"
             and t.meta["faults"][0]["tool"] == "pay_supplier")
    rec = simulate(t, naive_calls(t))
    c = classify(rec)
    assert c["primary"] == "phantom_write" and c["flags"]["phantom_ack_seen"]
    r = correct_episode(rec, B4Teacher())
    assert r["status"] == "accepted" and r["verified"]["vsc"] == 1
    pays = [m["content"] for m in r["conversation"][r["prefix_len"]::2] if "pay_supplier" in m["content"]]
    assert len(pays) == 1 and f"pay-{t.meta['po_id']}\"" in pays[0]  # same key as the agent's phantom write


def test_never_commit_correction_is_an_honest_failure():
    t = p1_task("never_commit")
    rec = simulate(t, naive_calls(t))
    c = classify(rec)
    assert c["primary"] == "phantom_write" and not c["success"]
    r = correct_episode(rec, B4Teacher())
    assert r["status"] == "accepted" and r["verified"]["report"] == "failed" and r["verified"]["vsc"] == 0


def test_txn_injected_duplicate_is_repaired_by_refund():
    t = find("txn_dev", lambda t: t.meta["fault_class"] == "dup_delivery")
    rec = simulate(t, naive_calls(t))
    c = classify(rec)
    assert c["primary"] == "duplicate_mutation" and c["flags"]["injected_duplicate"] and not c["flags"]["unsafe_retry"]
    r = correct_episode(rec, B4Teacher())
    assert r["status"] == "accepted" and r["verified"]["txn"]["vtc"]
    assert any("refund_payment" in m["content"] for m in r["conversation"][r["prefix_len"]::2])


def test_txn_concurrent_colleague_payment_is_refunded():
    t = find("txn_dev", lambda t: t.meta["fault_class"] == "concurrent")
    rec = simulate(t, naive_calls(t))
    assert classify(rec)["primary"] == "duplicate_mutation"
    r = correct_episode(rec, B4Teacher())
    assert r["status"] == "accepted" and r["verified"]["txn"]["vtc"] and not r["verified"]["txn"]["harm"]


def test_loop_without_report_backs_off_to_an_earlier_cut():
    t = p1_task("clean")
    po = _po(t)
    step = t.meta["requested"][0]
    calls = [(STEP_TOOL[step], _args(step, po, f"{step}-{po['po_id']}"))]
    calls += [(STEP_READ[step], {"po_id": po["po_id"]})] * 15
    rec = simulate(t, calls)
    assert rec["ended_by"] == "max_turns"
    c = classify(rec)
    assert c["primary"] == "loop_or_timeout" and c["first_bad_idx"] is None
    r = correct_episode(rec, B4Teacher())
    assert r["status"] == "accepted" and r["cut_kind"] == "backoff" and r["cut"] < 16
    assert r["verified"]["vsc"] == 1


def test_unparseable_final_turn_is_invalid_call():
    t = p1_task("clean")
    po = _po(t)
    step = t.meta["requested"][0]
    rec = simulate(t, [(STEP_TOOL[step], _args(step, po, f"{step}-{po['po_id']}")), "I think we are done here."])
    c = classify(rec)
    assert rec["ended_by"] == "no_call" and c["primary"] == "invalid_call"
    r = correct_episode(rec, B4Teacher())
    assert r["status"] == "accepted" and r["cut"] == 1 and r["rejected_turn"]["content"] == "I think we are done here."


def test_success_is_not_a_failure():
    from actionproof.selfheal.replay import act_text  # noqa: F401
    t = p1_task("clean")
    po = _po(t)
    calls = [(STEP_TOOL[s], _args(s, po, f"{s}-{po['po_id']}")) for s in t.meta["requested"]]
    rec = simulate(t, calls + [("submit_report", {"status": "completed", "summary": "ok"})])
    assert classify(rec)["primary"] == "success"
    assert correct_episode(rec, B4Teacher())["reason"] == "not_a_failure"


def test_wrong_amount_is_wrong_write_at_that_call():
    t = p1_task("clean")
    po = _po(t)
    calls = []
    for s in t.meta["requested"]:
        a = _args(s, po, f"{s}-{po['po_id']}")
        if s == "pay":
            a["amount_cents"] = po["amount_cents"] + 100
        calls.append((STEP_TOOL[s], a))
    rec = simulate(t, calls + [("submit_report", {"status": "completed", "summary": "ok"})])
    c = classify(rec)
    assert c["primary"] == "wrong_or_collateral_write"
    assert rec["trace"][c["first_bad_idx"]]["tool"] == "pay_supplier"


# ---------------------------------------------------------------------- replay and guards
def test_replay_reproduces_trace_and_detects_tampering():
    t = find("txn_dev", lambda t: t.meta["fault_class"] == "late_commit")
    rec = simulate(t, naive_calls(t))
    full = replay(rec, spec_for(t))
    assert full.turns == len(naive_calls(t)) and full.env.done
    bad = copy.deepcopy(rec)
    bad["trace"][0]["obs"] = {"status": "ok", "tampered": True}
    with pytest.raises(ReplayMismatch):
        replay(bad, spec_for(t))


def test_turn_before():
    ends = [0, 1, 3, 4]  # start, then turns ending at trace lengths 1, 3, 4
    assert turn_before(ends, 2, 3) == 1   # call 2 belongs to turn 2
    assert turn_before(ends, 3, 3) == 2
    assert turn_before(ends, None, 3) == 3
    assert turn_before(ends, 0, 3) == 0


def test_test_splits_are_refused():
    t = find("txn_test", lambda t: t.meta["txn"]["group"] == "core")
    rec = simulate(t, naive_calls(t))
    with pytest.raises(SplitRefused):
        correct_episode(rec, B4Teacher())


class _Visible:
    """An env handle that exposes only what an agent sees: calls, their observations, and whether it has ended."""

    def __init__(self, env):
        object.__setattr__(self, "_env", env)

    def __getattr__(self, name):
        if name not in ("call", "trace", "done"):
            raise AttributeError(f"teacher touched hidden env attribute {name!r}")
        return getattr(self._env, name)


def test_teacher_uses_only_agent_visible_env_surface():
    from actionproof.selfheal.replay import replay as rp
    from actionproof.selfheal.teachers import Context
    for cls in ("lost_ack", "phantom", "partial", "never_commit"):
        t = p1_task(cls)
        rec = simulate(t, naive_calls(t))
        rep = rp(rec, spec_for(t), upto_turn=1)
        rep.env = _Visible(rep.env)
        ctx = Context(spec=spec_for(t), rep=rep)
        B4Teacher().continue_from(ctx)
        assert ctx.actions and '"submit_report"' in ctx.actions[-2]["content"]
