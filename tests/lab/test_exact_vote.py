"""Tests of the exact finite-G expected-update computation and of Theorem 1 / Proposition 2
(docs/research_lab/GRPO_MECHANISM_FORMAL.md)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from actionproof.lab.exact_vote import (_group_update, categories, expected_update, m_G, threshold, voi)


@pytest.mark.parametrize("G", [2, 4, 8])
@pytest.mark.parametrize("rho", [0.1, 0.3, 0.7])
def test_theorem1_grpo_is_a_stake_free_vote(G, rho):
    """Deterministic within-world returns (s = 1, q0 = 0), eps -> 0: E[update] = (G-1) m_G(rho) (P(beta>0) - P(beta<0))."""
    for p in (0.1, 0.4, 0.8):
        for c in (0.01, 0.2):
            for Lf in (0.0, 3.0):
                got = expected_update(p, c, 1.0, rho, G, "grpo", "A", Lf=Lf, eps=1e-12)
                want = (G - 1) * m_G(rho, G) * (p - (1 - p))
                assert math.isclose(got, want, rel_tol=1e-6, abs_tol=1e-9)


@pytest.mark.parametrize("G", [2, 5, 8])
@pytest.mark.parametrize("cond", ["A", "B"])
def test_theorem1_drgrpo_is_exactly_proportional_to_voi(G, cond):
    """For i.i.d. rollouts given the group's world(s): E[sum (R - Rbar)(r - rho)] = (G-1) rho (1-rho) VOI, any s, q0."""
    rho = 0.35
    for p, c, s, Lf, q0 in [(0.2, 0.05, 1.0, 0.0, 0.0), (0.6, 0.1, 0.5, 1.0, 0.0), (0.3, 0.02, 0.7, 0.5, 0.3)]:
        got = expected_update(p, c, s, rho, G, "nostd", cond, Lf=Lf, q0=q0)
        want = (G - 1) * rho * (1 - rho) * voi(p, c, s, Lf=Lf, q0=q0)
        assert math.isclose(got, want, rel_tol=1e-9, abs_tol=1e-12)


def test_rloo_is_proportional_to_voi():
    rho, G = 0.3, 6
    got = expected_update(0.3, 0.05, 0.6, rho, G, "rloo", "A", Lf=1.0)
    want = G * rho * (1 - rho) * voi(0.3, 0.05, 0.6, Lf=1.0)
    assert math.isclose(got, want, rel_tol=1e-9)


def test_vote_threshold_is_one_half_and_cost_free():
    for c in (0.001, 0.01, 0.05, 0.2):
        for Lf in (0.0, 1.0):
            t = threshold(c, 1.0, 0.3, 8, "grpo", "A", Lf=Lf, eps=1e-12)
            assert abs(t - 0.5) < 1e-6
    # Dr. GRPO threshold = VOI root c / (s (V + Lf))
    assert abs(threshold(0.05, 1.0, 0.3, 8, "nostd", "A") - 0.05) < 1e-6
    assert abs(threshold(0.05, 0.5, 0.3, 8, "nostd", "A", Lf=1.0) - 0.05) < 1e-6


def test_eps_restores_scale_dependence_only_for_tiny_costs():
    big = threshold(0.2, 1.0, 0.3, 8, "grpo", "A")
    tiny = threshold(1e-5, 1.0, 0.3, 8, "grpo", "A")
    assert abs(big - 0.5) < 1e-3 and tiny < 0.2


def test_exogenous_noise_counterexample_restores_cost_sensitivity():
    t1 = threshold(0.01, 1.0, 0.3, 8, "grpo", "A", q0=0.3)
    t2 = threshold(0.2, 1.0, 0.3, 8, "grpo", "A", q0=0.3)
    assert t1 < t2 - 0.05


def test_categories_probabilities_sum_to_one():
    for world in (None, 0, 1):
        cats = categories(0.3, 0.05, 0.6, 0.4, Lf=1.0, q0=0.2, world=world)
        assert math.isclose(sum(w for w, _, _ in cats), 1.0)


def test_group_update_matches_trl_formula():
    R = np.array([1.0, 0.95, 0.95, 1.0])
    S = np.array([-0.3, 0.7, 0.7, -0.3])
    A = (R - R.mean()) / (R.std(ddof=1) + 1e-4)
    assert math.isclose(_group_update(R, S, "grpo", 1e-4), float((A * S).sum()))
