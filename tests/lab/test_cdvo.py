"""Tests of the CDVO testbed estimators (actionproof/lab/cdvo.py)."""
from __future__ import annotations

import math

import numpy as np
import pytest

import actionproof.ad2.tabular.policy as P
from actionproof.ad2.tabular import envs as E
from actionproof.ad2.tabular.env import (DTYPES, Config, OpType, init_state, exact_rows, sample_episode,
                                         sample_world, walk)
from actionproof.ad2.tabular.policy import Policy
from actionproof.ad2.tabular.train import fork_return, prior_of
from actionproof.ad2.tabular.train import train as v2_train
from actionproof.lab.cdvo import train

CORE = Config([OpType("pay", 0.3, 0.6, tid=0), OpType("post", 0.0, 1.0, tid=1)], c=0.05, Lf=1.0, mode="single",
              tag="core")


def _spec(est, **kw):
    tasks = E.rich("multi", 0.05, 1.0)
    return {"tasks": tasks, "eval_tasks": tasks[:1], "est": est, "seed": 3, "steps": 4, "n_mc": 50, "fork_k": 1, **kw}


@pytest.mark.parametrize("est", ["nostd", "grpo", "rloo", "igc", "critic_v", "critic_q", "fork_act", "cvc", "awrbn"])
def test_reproduces_v2_trainer_exactly(est):
    a = v2_train(_spec(est))
    b = train(_spec(est))
    assert a["episodes_used"] == b["episodes_used"]
    for k in a["theta"]:
        assert np.allclose(a["theta"][k], b["theta"][k], atol=0, rtol=0)
    ea, eb = next(iter(a["eval"].values())), next(iter(b["eval"].values()))
    assert ea["J"] == eb["J"]


def test_igc_v1_reproduces_tb7_gate():
    orig = P.informative

    def gate(dtype, j, st):
        if dtype in ("V2", "V2c"):
            return True
        return orig(dtype, j, st)
    P.informative = gate
    try:
        a = v2_train(_spec("igc"))
    finally:
        P.informative = orig
    b = train(_spec("igc_v1"))
    for k in a["theta"]:
        assert np.allclose(a["theta"][k], b["theta"][k], atol=0, rtol=0)
    assert P.informative is orig


def _exact_q(cfg, pol, a0):
    """Exact Q^pi(root, V1 at op 0 = a0) by expectimax-free policy expectation with the first action forced."""
    def ch(dtype, c, s):
        if c["j"] == 0 and dtype == "V1":
            return np.eye(2)[a0]
        return pol.probs(dtype, c)
    return walk(cfg, init_state(cfg, exact_rows(cfg)), ch, "mixed")


@pytest.mark.parametrize("crn", [True, False])
def test_paired_fork_contrast_is_unbiased_for_the_current_policy(crn):
    pol = Policy(2, "tab", rho0=0.4)
    want = _exact_q(CORE, pol, 1) - _exact_q(CORE, pol, 0)
    rng = np.random.default_rng(0)
    pr = prior_of(CORE)
    xs = []
    for _ in range(6000):
        ep = sample_episode(CORE, rng, sample_world(CORE, rng, pr), pr, pol.sample)
        rec = ep["log"][0]
        assert rec["d"] == "V1" and rec["j"] == 0
        s1 = int(rng.integers(2 ** 31))
        s0 = s1 if crn else int(rng.integers(2 ** 31))
        xs.append(fork_return(CORE, rec, pol, rng, 1, "act", s1, ep["z"]) -
                  fork_return(CORE, rec, pol, rng, 0, "act", s0, ep["z"]))
    xs = np.array(xs)
    assert abs(xs.mean() - want) < 4 * xs.std() / math.sqrt(len(xs))


def test_common_random_numbers_do_not_increase_variance():
    pol = Policy(2, "tab", rho0=0.4)
    rng = np.random.default_rng(1)
    pr = prior_of(CORE)
    out = {True: [], False: []}
    for _ in range(4000):
        ep = sample_episode(CORE, rng, sample_world(CORE, rng, pr), pr, pol.sample)
        rec = ep["log"][0]
        for crn in (True, False):
            s1 = int(rng.integers(2 ** 31))
            s0 = s1 if crn else int(rng.integers(2 ** 31))
            out[crn].append(fork_return(CORE, rec, pol, rng, 1, "act", s1, ep["z"]) -
                            fork_return(CORE, rec, pol, rng, 0, "act", s0, ep["z"]))
    assert np.var(out[True]) <= np.var(out[False]) * 1.05


def test_policy_inputs_never_contain_the_hidden_world():
    """Same observable history in two different hidden worlds -> identical action probabilities at every decision
    before the observations diverge (the actor sees only ctx: op descriptors, cost and 'known failed')."""
    pol = Policy(2, "tab", rho0=0.5)
    pr = prior_of(CORE)
    Z = pr[0]
    for z1 in Z:
        for z2 in Z:
            e1 = sample_episode(CORE, np.random.default_rng(5), z1, pr, pol.sample)
            e2 = sample_episode(CORE, np.random.default_rng(5), z2, pr, pol.sample)
            r1, r2 = e1["log"][0], e2["log"][0]
            assert np.array_equal(r1["phi"], r2["phi"]) and np.array_equal(r1["pr"], r2["pr"])
            for x in (r1, r2):
                assert len(x["phi"]) == pol.nf  # features: bias, op one-hot, kf; no z


def test_episode_budget_and_fork_accounting():
    r = train(_spec("cdvo_adapt", steps=None, episode_budget=200, branch_frac=0.5))
    assert 200 <= r["episodes_used"] < 200 + 32 * 3
    assert r["n_forks"] > 0
    n = train(_spec("nostd", steps=None, episode_budget=200))
    assert n["episodes_used"] == 224 and n["n_forks"] == 0  # 7 steps x 32 rollouts, first step reaching >= 200


def test_adaptive_allocation_uses_only_past_forks():
    """cdvo_adapt at b = 0 is exactly nostd (no forks, no peeking at the current batch's outcomes). cdvo_fixed at
    b = 0 makes no forks; its Bernoulli draws shift the random stream, so it is not bit-identical."""
    a = train(_spec("nostd"))
    b = train(_spec("cdvo_adapt", branch_frac=0.0))
    assert b["n_forks"] == 0
    for k in a["theta"]:
        assert np.allclose(a["theta"][k], b["theta"][k], atol=0, rtol=0)
    assert train(_spec("cdvo_fixed", branch_frac=0.0))["n_forks"] == 0


@pytest.mark.parametrize("est", ["cdvo_fixed", "cdvo_adapt", "cdvo_adapt_critic", "cdvo_ia", "fork_act_nocrn", "ppo",
                                 "ppo_norm", "indep_world_std", "indep_task_std", "igc_v1"])
def test_new_estimators_run(est):
    r = train(_spec(est))
    e = next(iter(r["eval"].values()))
    assert math.isfinite(e["J"]) and e["regret"] >= -1e-9


def test_unsupported_estimator_is_refused():
    with pytest.raises(ValueError):
        train(_spec("lagr"))
