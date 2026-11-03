"""Reward adapters score real episode records and reject misaligned batches."""
from __future__ import annotations

import pytest

from actionproof.bench.generator import make_split
from actionproof.rewards import trl_rewards as R
from actionproof.rewards.sanity_agents import task_oracle
from actionproof.rollout.agenttune_rollout import episode_record, make_env


def records(n=40):
    tasks = make_split("train", n)
    out = []
    for t in tasks:
        good, bad = make_env(t, "ro-good"), make_env(t, "ro-bad")
        ids = t.meta["ids"]
        task_oracle(good, t.workflow, ids["order"], ids["ticket"])
        bad.call("submit_report", {"status": "completed", "summary": "done"})  # pure lie
        out += [episode_record(t, good), episode_record(t, bad)]
    return tasks, out


def test_pcstr_prefers_oracle_and_vsc_metric_matches():
    tasks, eps = records()
    comps = ["x"] * len(eps)
    r = R.pcstr_reward(None, comps, ap_episodes=eps)
    assert all(r[i] > r[i + 1] for i in range(0, len(r), 2))
    assert R.vsc_metric(None, comps, ap_episodes=eps)[::2] == [1.0] * (len(eps) // 2)
    tt = R.make_tool_trust_reward({t.task_id: t.goal for t in tasks})(None, comps, ap_episodes=eps)
    assert all(tt[i + 1] == 0.0 for i in range(0, len(tt), 2))  # no write acks -> 0 for the liar


def test_misaligned_batch_raises():
    _, eps = records(2)
    with pytest.raises(RuntimeError):
        R.pcstr_reward(None, ["x"] * (len(eps) + 1), ap_episodes=eps)
