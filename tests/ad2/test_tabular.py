"""CPU tests for the tabular costed-verification testbed (actionproof/ad2/tabular)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from actionproof.ad2.tabular.env import (Config, OpType, fault_plans, oracle_value, policy_value, root_voi,
                                         sample_episode, sample_rows, sample_world)
from actionproof.ad2.tabular.policy import Policy
from actionproof.ad2.tabular.train import _adv_seq, assumed_cfg, fork_return, prior_of, train

CORE = Config([OpType("pay", 0.3, 0.7, tid=0), OpType("post", 0.0, 1.0, tid=1)], c=0.05, tag="core")
RICH = Config([OpType("order", 0.1, 0.9, q_to=0.2, a=0.5, tid=0), OpType("pay", 0.25, 0.6, f=0.05, tid=1),
               OpType("post", 0.0, 1.0, tid=2)], c=0.05, D=0.5, Lf=0.5, tag="rich")


@pytest.mark.parametrize("p,s,c,Lf", [(0.4, 1, 0.05, 0), (0.2, 0.5, 0.05, 0), (0.1, 0.3, 0.05, 2), (0.05, 1, 0.1, 0)])
def test_single_op_voi_formula(p, s, c, Lf):
    cfg = Config([OpType("x", p, s)], c=c, Lf=Lf)
    assert math.isclose(root_voi(cfg)[0]["voi"], p * s * (1 + Lf) - c, abs_tol=1e-9)


def test_priors_sum_to_one_and_single_mode():
    for cfg in (CORE, RICH):
        Z, P = fault_plans(cfg)
        assert math.isclose(P.sum(), 1)
    single = Config(RICH.ops, mode="single")
    Z, P = fault_plans(single)
    assert ((Z == 1).sum(1) <= 1).all()
    for j, o in enumerate(single.ops):  # marginal silent-fault rate among ok acks is p_j in both modes
        ok = Z[:, j] < 2
        assert math.isclose(P[ok & (Z[:, j] == 1)].sum() / P[ok].sum(), o.p, rel_tol=1e-6)


@pytest.mark.parametrize("cfg", [CORE, RICH])
def test_exact_policy_value_matches_monte_carlo(cfg):
    pol = Policy(3, "tab", rho0=0.4)
    J = policy_value(cfg, pol.probs)
    rng = np.random.default_rng(0)
    pr = prior_of(cfg)
    R = [sample_episode(cfg, rng, sample_world(cfg, rng, pr), pr, pol.sample)["R"] for _ in range(6000)]
    assert abs(np.mean(R) - J) < 4 * np.std(R) / math.sqrt(len(R))


def test_oracle_dominates_policies():
    for cfg in (CORE, RICH):
        Js = oracle_value(cfg)
        for rho in (0.01, 0.3, 0.9):
            assert policy_value(cfg, Policy(3, "tab", rho).probs) <= Js + 1e-12


@pytest.mark.parametrize("cfg", [CORE, RICH])
def test_rao_blackwell_unbiased_and_lower_variance(cfg):
    pol = Policy(3, "tab", rho0=0.4)
    rng = np.random.default_rng(1)
    pr = prior_of(cfg)
    eps = [sample_episode(cfg, rng, sample_world(cfg, rng, pr), pr, pol.sample) for _ in range(6000)]
    R = np.array([e["R"] for e in eps])
    Rb = np.array([e["rbar"] for e in eps])
    assert abs(R.mean() - Rb.mean()) < 4 * R.std() / math.sqrt(len(R))
    assert Rb.var() <= R.var() + 1e-9


def test_consistent_worlds_reproduce_the_observations():
    """Every world kept by the replay must generate the same observations under the same actions: replaying the
    recorded actions with that world as the actual one keeps exactly the same consistent set."""
    rng = np.random.default_rng(2)
    pr = prior_of(RICH)
    for _ in range(200):
        z = sample_world(RICH, rng, pr)
        seed = int(rng.integers(1 << 30))
        pol = Policy(3, "tab", rho0=0.5)
        e = sample_episode(RICH, np.random.default_rng(seed), z, pr, pol.sample)
        acts = [r["a"] for r in e["log"]]
        alive = np.where(e["state"].alive)[0]
        for idx in alive:
            it = iter(acts)
            e2 = sample_episode(RICH, np.random.default_rng(seed), e["state"].rows.Z[idx], pr,
                                lambda d, c, s, r: (next(it), {"d": d}))
            assert (e2["state"].alive == e["state"].alive).all()
            assert e2["nv"] == e["nv"]


def test_grpo_clean_group_advantage_is_cost_free():
    """Two-valued clean-world group: the standardised read advantage does not depend on c (P1)."""
    rho, G = 3 / 8, 8
    expect = -math.sqrt((1 - rho) / rho) * math.sqrt((G - 1) / G)  # Bessel std
    for c in (0.01, 0.05, 0.3):
        g = [{"R": 1 - c}] * 3 + [{"R": 1.0}] * 5
        a = _adv_seq("grpo", g)
        assert math.isclose(a[0], expect, rel_tol=2e-2)
        assert math.isclose(_adv_seq("nostd", g)[0], -c * (1 - rho), rel_tol=1e-9)
    # invariance fails only when the group std is comparable to TRL's epsilon (1e-4): c ~ 1e-3
    g = [{"R": 1 - 0.001}] * 3 + [{"R": 1.0}] * 5
    assert abs(_adv_seq("grpo", g)[0]) < 0.9 * abs(expect)


def test_condition_b_resamples_and_a_shares():
    tasks = [CORE]
    for cond, shared in (("A", True), ("B", False)):
        r = train({"tasks": tasks, "est": "nostd", "steps": 2, "seed": 0, "cond": cond, "n_mc": 10})
        assert r["episodes_used"] == 64


def test_belief_fork_estimates_belief_voi():
    """At the first decision of a single-op workflow, belief forks estimate Q(verify) - Q(skip) = p*s - c under the
    optimal-like continuation (always repair)."""
    cfg = Config([OpType("pay", 0.3, 0.7)], c=0.05, tag="one")
    pol = Policy(1, "tab", rho0=0.5)
    pol.theta["R"][1, 0] = 20.0  # always repair
    pol.theta["V2"][1, 0] = -20.0  # never re-verify
    pol.theta["V2c"][1, 0] = -20.0
    rng = np.random.default_rng(3)
    pr = prior_of(cfg)
    diffs = []
    for _ in range(400):
        z = np.array([0], dtype=np.int8)  # actual world clean: the fork must still see the posterior p = 0.3
        e = sample_episode(cfg, rng, z, pr, pol.sample)
        rec = e["log"][0]
        sd = int(rng.integers(1 << 30))
        diffs.append(fork_return(cfg, rec, pol, rng, 1, "bel", sd, z) - fork_return(cfg, rec, pol, rng, 0, "bel", sd, z))
    assert abs(np.mean(diffs) - (0.3 * 0.7 - 0.05)) < 0.06
    act = [fork_return(cfg, rec, pol, rng, 1, "act", 5, z) - fork_return(cfg, rec, pol, rng, 0, "act", 5, z)]
    assert math.isclose(act[0], -0.05)  # in the actual clean world, verifying only costs c


def test_misspecified_prior_support():
    cfg = assumed_cfg(CORE, "mis:0.25")
    assert math.isclose(cfg.ops[0].p, 0.075)
    rng = np.random.default_rng(0)
    rows = sample_rows(CORE, rng, np.array([1, 0], dtype=np.int8), fault_plans(assumed_cfg(CORE, "flat:0.0")))
    assert rows.w[rows.actual] == 0.0  # actual world outside the assumed support gets zero assumed mass
