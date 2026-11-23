"""CPU tests for the costed-verification testbed and aliased-world Rao-Blackwellisation (AWRB)."""
from __future__ import annotations

import copy
import json
import math
from collections import defaultdict
from pathlib import Path

import pytest

from actionproof.ad import rewards as R
from actionproof.ad.scripted import POLICIES
from actionproof.ad.worlds import (P_CLEAN, make_env_cw, marginalise, replay, world_key, world_prior)
from actionproof.bench.generator_proc import DATA, load_tasks
from actionproof.integrity.sim.runtime import episode_record_proc

TASKS = load_tasks(DATA / "cw_dev.jsonl")[:40]


def with_world(task, faults):
    t = copy.deepcopy(task)
    t.meta["faults"] = copy.deepcopy(faults)
    return t


def run_policy(name, task, seed=7):
    env = make_env_cw(task, "t", seed=seed)
    POLICIES[name](env, task)
    return env


def test_prior_sums_to_one_and_spares_post():
    for req in (["order", "pay", "post"], ["receive", "pay", "post"], ["pay", "post"]):
        w = world_prior(req)
        assert math.isclose(sum(p for _, p in w), 1.0)
        assert math.isclose(dict((world_key(f), p) for f, p in w)[()], P_CLEAN)
        assert all(f["tool"] != "post_journal_entry" for fs, _ in w for f in fs)
        assert len({world_key(f) for f, _ in w}) == len(w)


def test_pool_worlds_are_in_prior():
    for name in ("cw_train", "cw_dev", "cw_conf"):
        for t in load_tasks(DATA / f"{name}.jsonl"):
            keys = {world_key(f) for f, _ in world_prior(t.meta["requested"])}
            assert world_key(t.meta["faults"]) in keys


def test_write_acks_identical_across_silent_worlds():
    t = next(t for t in TASKS if "pay" in t.meta["requested"])
    obs = []
    for fs, _ in world_prior(t.meta["requested"]):
        if any(f["kind"] in ("lost_ack", "precommit_timeout") for f in fs):
            continue
        env = run_policy("never", with_world(t, fs))
        obs.append(json.dumps([s["obs"] for s in env.trace]))
    assert len(set(obs)) == 1, "silent faults must leave a no-read trajectory byte-identical"


def test_replay_reproduces_actual_world():
    for t in TASKS:
        for pol in ("never", "always", "random_0.5"):
            env = run_policy(pol, t)
            rec = episode_record_proc(t, env)
            assert replay(t, env.trace, t.meta["faults"], 7) == rec["score"]["r_v"]


@pytest.mark.parametrize("pol", ["never", "always", "random_0.5", "pay_only", "timeout_only"])
def test_marginal_matches_independent_world_enumeration(pol):
    """Run the policy afresh in every world (no replay); group worlds by the observation sequence the policy saw;
    the prior-weighted mean R_V within a group must equal marginalise() for every member."""
    for t in TASKS[:20]:
        groups: dict[str, list[tuple[float, int]]] = defaultdict(list)
        members = []
        for fs, p in world_prior(t.meta["requested"]):
            tw = with_world(t, fs)
            env = run_policy(pol, tw)
            sig = json.dumps([(s["tool"], s["args"], s["obs"]) for s in env.trace], sort_keys=True)
            rv = episode_record_proc(tw, env)["score"]["r_v"]
            groups[sig].append((p, rv))
            members.append((sig, tw, env))
        for sig, tw, env in members:
            g = groups[sig]
            want = sum(p * rv for p, rv in g) / sum(p for p, _ in g)
            got = marginalise(tw, env.trace, 7)["r_bar"]
            assert math.isclose(got, want, abs_tol=1e-12), (pol, t.task_id, got, want)


@pytest.mark.parametrize("pol", ["never", "always", "random_0.5", "skip_post"])
def test_unbiased_and_rao_blackwell(pol):
    """E_w[r_bar] = E_w[R_V] (law of total expectation) and Var(r_bar) <= Var(R_V), exactly, over the prior."""
    for t in TASKS[:20]:
        rows = []
        for fs, p in world_prior(t.meta["requested"]):
            tw = with_world(t, fs)
            env = run_policy(pol, tw)
            rv = episode_record_proc(tw, env)["score"]["r_v"]
            rows.append((p, rv, marginalise(tw, env.trace, 7)["r_bar"]))
        m1 = sum(p * a for p, a, _ in rows)
        m2 = sum(p * b for p, _, b in rows)
        assert math.isclose(m1, m2, abs_tol=1e-12)
        v1 = sum(p * (a - m1) ** 2 for p, a, _ in rows)
        v2 = sum(p * (b - m2) ** 2 for p, _, b in rows)
        assert v2 <= v1 + 1e-12


def test_never_read_gets_prior_mean_and_reading_reveals():
    t = next(t for t in TASKS if t.meta["requested"] == ["pay", "post"] or "pay" in t.meta["requested"])
    env = run_policy("never", with_world(t, []))
    m = marginalise(with_world(t, []), env.trace, 7)
    assert m["r_bar"] < 1.0 and m["post_fault"] > 0
    env = run_policy("always", with_world(t, []))
    m2 = marginalise(with_world(t, []), env.trace, 7)
    assert m2["r_bar"] == 1.0 and m2["post_fault"] < m["post_fault"]


def test_cvar_below_mean():
    for t in TASKS[:10]:
        env = run_policy("never", t)
        m = marginalise(t, env.trace, 7)
        assert m["cvar"] <= m["r_bar"] + 1e-12


def test_reward_functions():
    t = TASKS[0]
    env = run_policy("always", t)
    rec = episode_record_proc(t, env)
    rec["ad"] = marginalise(t, env.trace, 7)
    kw = {"ap_episodes": [rec]}
    assert R.make_reward("r_v", 0.05)(None, [0], **kw)[0] == pytest.approx(rec["score"]["r_v"] - 0.05 * rec["n_verify"])
    assert R.make_reward("r_bar", 0.05)(None, [0], **kw)[0] == pytest.approx(rec["ad"]["r_bar"] - 0.05 * rec["n_verify"])
    assert set(R.ESTIMATORS) == {"grpo", "nostd", "batchstd", "awrb", "awrbn", "cvar"}


def test_exact_policy_gradient_identity():
    """Bandit with one aliased decision: E[(a - rho) * r_bar] = E[(a - rho) * R] = dJ/dtheta (exact enumeration)."""
    for p, c, s in ((0.3, 0.05, 1.0), (0.1, 0.02, 0.6)):
        for rho in (0.1, 0.5, 0.9):
            g_r = g_rb = 0.0
            for a, pa in ((1, rho), (0, 1 - rho)):
                for w, pw in ((1, p), (0, 1 - p)):
                    for rec, pr in ((1, s), (0, 1 - s)):
                        r = (rec if w else 1) - c if a else (0 if w else 1)
                        g_r += pa * pw * pr * (a - rho) * r
                rb = (1 - p) + p * s - c if a else 1 - p
                g_rb += pa * (a - rho) * rb
            exact = rho * (1 - rho) * (p * s - c)
            assert math.isclose(g_r, exact, abs_tol=1e-12) and math.isclose(g_rb, exact, abs_tol=1e-12)
