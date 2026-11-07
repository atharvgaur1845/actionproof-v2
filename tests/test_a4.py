"""Amendment 4 splits, env, verifier and oracle (PREREGISTRATION_A4 §4, acceptance 2)."""
from __future__ import annotations

import statistics

import pytest

from actionproof.bench.generator import make_split
from actionproof.bench.generator_a4 import ALL_A4, EVAL_SPLITS, SEEN_WRITES, make_a4_split, split_bytes
from actionproof.env.db import snapshot
from actionproof.env.tools_a4 import HELD_OUT_TOOL, EcomEnvA4
from actionproof.env.tools_ecom import EcomEnv
from actionproof.env.verifier import verify, vsc
from actionproof.env.verifier_a4 import verify_a4
from actionproof.eval.a4_metrics import report_outcome
from actionproof.eval.a4_runtime import episode_record_a4, make_env_a4
from actionproof.rewards.sanity_agents import task_oracle
from actionproof.rewards.sanity_agents_a4 import a4_oracle

MAX_TURNS = 18


@pytest.fixture(scope="module")
def splits():
    return {s: make_a4_split(s) for s in ALL_A4}


def play(task):
    env = make_env_a4(task, f"t-{task.task_id}")
    a4_oracle(env, task)
    return env, episode_record_a4(task, env)


def test_deterministic():
    for s in ALL_A4:
        assert split_bytes(s) == split_bytes(s)


def test_sizes(splits):
    assert len(splits["a4_train_A"]) == len(splits["a4_train_B"]) == 400
    for s, c in EVAL_SPLITS.items():
        assert len(splits[s]) == c.get("n_ur", 0) + c.get("n_rp", 0) + c.get("n_ok", 0)


def test_pools_matched_except_fault_tool(splits):
    A, B = splits["a4_train_A"], splits["a4_train_B"]
    for a, b in zip(A, B):
        ma, mb = a.meta["a4"], b.meta["a4"]
        ga, gb = a.goal.to_json(), b.goal.to_json()
        ga.pop("task_id"); gb.pop("task_id")
        assert a.initial_state == b.initial_state and ga == gb and a.prompt == b.prompt
        for k in ("env_class", "fault_type", "k_max"):
            assert ma[k] == mb[k]
        assert (ma["fault_tool"] is None) == (mb["fault_tool"] is None)
        if ma["fault_tool"]:
            assert ma["fault_tool"] == "issue_refund" and mb["fault_tool"] in SEEN_WRITES
            assert len(a.schedule.by_call) == len(b.schedule.by_call)  # same number of faulted calls
            assert {k for _, k in a.schedule.by_call} == {k for _, k in b.schedule.by_call}
    tools_b = [b.meta["a4"]["fault_tool"] for b in B if b.meta["a4"]["fault_tool"]]
    for t in SEEN_WRITES:  # B spreads faults over every seen write tool
        assert 0.25 < tools_b.count(t) / len(tools_b) < 0.42


def test_held_out_tool_absent_from_training(splits):
    for s in ("a4_train_A", "a4_train_B"):
        for t in splits[s]:
            assert "domain" not in t.meta  # plain EcomEnv in training
            assert all(op != HELD_OUT_TOOL for op, _ in t.goal.required_writes)
            assert all(p["p"] != "credit" for p in t.goal.postconditions)
            assert all(tool != HELD_OUT_TOOL for tool, _ in t.schedule.by_call)
            assert "credit" not in t.prompt
    env = EcomEnv({}, splits["a4_train_B"][0].schedule, "x")
    assert HELD_OUT_TOOL not in [f.__name__ for f in env.tools()]
    assert env.call(HELD_OUT_TOOL, {"order_id": "O-1", "amount_cents": 100})["error"] == "unknown_tool"


def test_eval_env_lists_held_out_tool_everywhere(splits):
    for s in EVAL_SPLITS:
        t = splits[s][0]
        env = make_env_a4(t, "x")
        assert isinstance(env, EcomEnvA4) and HELD_OUT_TOOL in [f.__name__ for f in env.tools()]


@pytest.mark.parametrize("split", ALL_A4)
def test_oracle_solves_within_budget(splits, split):
    for t in splits[split]:
        env, rec = play(t)
        assert len(env.trace) <= MAX_TURNS, (t.task_id, len(env.trace))
        cls = t.meta["a4"]["env_class"]
        if cls == "UR":
            assert report_outcome(rec) == "CF", t.task_id
        else:
            assert rec["vsc"] == 1, (t.task_id, rec["verdict"])


def test_oracle_cost_matched_across_regimes(splits):
    cost = {s: statistics.mean(len(play(t)[0].trace) for t in splits[s]) for s in ("a4_train_A", "a4_train_B")}
    assert abs(cost["a4_train_A"] - cost["a4_train_B"]) <= 0.5, cost


def test_verify_a4_equals_verify_without_credit():
    for t in make_split("val")[:60]:
        env = EcomEnv(t.initial_state, t.schedule, "r", stale_status=t.stale_status, ambiguous_text=t.ambiguous_text)
        task_oracle(env, t.workflow, t.meta["ids"]["order"], t.meta["ids"]["ticket"])
        a = verify(t.goal, snapshot(env.conn), env.log, env.final_report)
        b = verify_a4(t.goal, snapshot(env.conn), env.log, env.final_report)
        assert a == b and vsc(a) == vsc(b)


def test_credit_visible_through_seen_reads(splits):
    t = next(x for x in splits["e_heldout"] if x.meta["a4"]["env_class"] == "OK")
    env = make_env_a4(t, "x")
    o = t.meta["ids"]["order"]
    assert env.call(HELD_OUT_TOOL, {"order_id": o, "amount_cents": 700})["status"] == "success"
    assert env.call("lookup_order", {"order_id": o})["store_credits"][0]["amount_cents"] == 700
    assert env.call("query_ledger", {"order_id": o})["credits"]


def test_held_out_phantom_is_a_lie(splits):
    t = next(x for x in splits["e_heldout"] if x.meta["a4"]["fault_type"] == "F1_phantom")
    env = make_env_a4(t, "x")
    o = t.meta["ids"]["order"]
    assert env.call(HELD_OUT_TOOL, {"order_id": o, "amount_cents": t.meta["ids"]["credit"]})["status"] == "success"
    assert env.call("query_ledger", {"order_id": o})["credits"] == []
