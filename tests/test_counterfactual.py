"""Footprint / NWCG: exact on hand-built cases, and replay reproduces recorded episodes."""
from __future__ import annotations

from actionproof.bench.generator import make_split
from actionproof.env.counterfactual import footprint, replay
from actionproof.rewards.sanity_agents import lean_expert, task_oracle
from actionproof.rollout.agenttune_rollout import episode_record, make_env

TASK = next(t for t in make_split("val") if t.workflow == "refund" and t.meta["benign"])
IDS = TASK.meta["ids"]


def test_lean_expert_has_no_excess_and_full_nwcg():
    env = make_env(TASK, "a")
    lean_expert(env, IDS["order"], IDS["ticket"])
    fp = footprint(TASK, env)
    assert fp["excess"] == 0 and fp["nwcg"] == 1.0 and fp["minimal"] == fp["committed"]


def test_unneeded_write_counts_as_excess():
    env = make_env(TASK, "b")
    lean_expert(env, IDS["order"], IDS["ticket"])
    env.reports.clear()  # reopen to add a side effect after the fact
    env.call("update_ticket", {"ticket_id": IDS["ticket"], "status": "pending"})
    env.call("close_ticket", {"ticket_id": IDS["ticket"]})
    fp = footprint(TASK, env)
    assert fp["excess"] == 2


def test_unfinished_episode_has_no_footprint():
    env = make_env(TASK, "c")
    env.call("lookup_order", {"order_id": IDS["order"]})
    assert footprint(TASK, env)["excess"] is None


def test_replay_reproduces_record():
    for t in make_split("val")[:15]:
        env = make_env(t, "r")
        task_oracle(env, t.workflow, t.meta["ids"]["order"], t.meta["ids"]["ticket"])
        rec = episode_record(t, env)
        env2 = replay(t, rec)
        assert [(r.op, r.entity) for r in env2.log.receipts] == [(r.op, r.entity) for r in env.log.receipts]
