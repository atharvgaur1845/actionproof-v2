"""Checks of the latent-state theory engine (actionproof/theory/exact.py) against an independent implementation
(actionproof/lab/exact_vote.py, brute-force multinomial sum written separately) and against the identities stated in
docs/theory/PROOF_AND_ASSUMPTIONS.md."""
import math

import numpy as np
import pytest

from actionproof.lab import exact_vote as ev
from actionproof.theory import exact as ex
from actionproof.theory.problems import rescale_states, verification


def canon(p, c, s, L=0.0, q0=0.0):
    return verification([p], [L], [s], c=c, q0=q0, L0=L)


@pytest.mark.parametrize("p,c,s,L,q0", [(0.3, 0.05, 1.0, 0.0, 0.0), (0.6, 0.2, 0.6, 1.0, 0.0),
                                         (0.2, 0.01, 1.0, 0.0, 0.3), (0.45, 0.1, 0.3, 1.0, 0.1)])
@pytest.mark.parametrize("est", ["grpo", "nostd", "rloo"])
@pytest.mark.parametrize("cond,law", [("A", "shared"), ("B", "iid")])
def test_matches_independent_enumeration(p, c, s, L, q0, est, cond, law):
    pb = canon(p, c, s, L, q0)
    for pi in (0.1, 0.3, 0.7):
        a = ex.expected_update(pb, pi, 6, law, est)
        b = ev.expected_update(p, c, s, pi, 6, est, cond, Lf=L, q0=q0)
        assert a == pytest.approx(b, rel=1e-9, abs=1e-12)


def test_bernstein_mass_and_endpoints():
    pb = canon(0.3, 0.05, 1.0)
    b = ex.bernstein(pb, 8, "mix:0.4", "grpo")
    assert b[0] == 0 and b[-1] == 0


@pytest.mark.parametrize("law", ["shared", "iid", "mix:0.3", "dirmult:0.3", "dirmult:0.8"])
def test_drgrpo_identity_any_law(law):
    """Centred update = (G-1) pi (1-pi) delta_q for every exchangeable law (Proposition 2.4)."""
    pb = verification([0.2, 0.1], [1.0, 3.0], [0.7, 1.0], d=[0.9, 1.0], c=0.07, f=0.1, D=0.5, q0=0.05)
    G = 5
    for pi in (0.2, 0.55):
        u = ex.expected_update(pb, pi, G, law, "nostd")
        assert u == pytest.approx((G - 1) * pi * (1 - pi) * pb.delta_q(), rel=1e-9)


def test_vote_shared_deterministic():
    """Theorem 1: shared law, deterministic within-state returns -> (G-1) m_G(pi) * vote."""
    pb = verification([0.2, 0.15], [1.0, 9.0], [1.0, 0.0], c=0.05)
    assert pb.deterministic_within_state()
    G = 8
    for pi in (0.1, 0.4, 0.9):
        u = ex.expected_update(pb, pi, G, "shared", "grpo", eps=0.0)
        assert u == pytest.approx((G - 1) * ev.m_G(pi, G) * pb.vote(), rel=1e-9)


def test_per_state_affine_invariance_shared():
    """Theorem 2: shared law -> GRPO update invariant to R -> lam_z R + kappa_z (eps -> 0); Dr. GRPO is not."""
    pb = verification([0.25, 0.1], [1.0, 2.0], [0.6, 1.0], d=[0.8, 1.0], c=0.05, f=0.2, D=0.3, q0=0.1)
    pr = rescale_states(pb, [3.0, 0.2, 7.0], [1.0, -2.0, 0.5])
    for pi in (0.2, 0.6):
        a = ex.expected_update(pb, pi, 6, "shared", "grpo", eps=0.0)
        b = ex.expected_update(pr, pi, 6, "shared", "grpo", eps=0.0)
        assert a == pytest.approx(b, rel=1e-6)
        c1 = ex.expected_update(pb, pi, 6, "shared", "nostd")
        c2 = ex.expected_update(pr, pi, 6, "shared", "nostd")
        assert abs(c1 - c2) > 1e-3


def test_affine_invariance_fails_iid():
    pb = verification([0.3], [1.0], [1.0], c=0.05)
    pr = rescale_states(pb, [1.0, 10.0])
    a = ex.expected_update(pb, 0.3, 6, "iid", "grpo", eps=0.0)
    b = ex.expected_update(pr, 0.3, 6, "iid", "grpo", eps=0.0)
    assert abs(a - b) > 1e-3


def test_mixture_linear():
    pb = verification([0.3], [1.0], [0.6], c=0.05)
    for pi in (0.2, 0.8):
        uA = ex.expected_update(pb, pi, 6, "shared", "grpo")
        uB = ex.expected_update(pb, pi, 6, "iid", "grpo")
        um = ex.expected_update(pb, pi, 6, "mix:0.35", "grpo")
        assert um == pytest.approx(0.35 * uA + 0.65 * uB, rel=1e-9)


@pytest.mark.parametrize("law", ["mix:0.3", "dirmult:0.3"])
def test_icc_equals_rho(law):
    icc = ex.pairwise_icc(law, [0.6, 0.3, 0.1], 6)
    assert all(x == pytest.approx(0.3, abs=1e-9) for x in icc)


def test_pairwise_correlation_not_sufficient():
    """Proposition 3.5: same marginal and ICC, G >= 3 -> different GRPO updates; G = 2 -> identical."""
    pb = verification([0.3], [1.0], [1.0], c=0.05)
    a = ex.expected_update(pb, 0.5, 6, "mix:0.3", "grpo")
    b = ex.expected_update(pb, 0.5, 6, "dirmult:0.3", "grpo")
    assert abs(a - b) > 1e-3
    a2 = ex.expected_update(pb, 0.5, 2, "mix:0.3", "grpo")
    b2 = ex.expected_update(pb, 0.5, 2, "dirmult:0.3", "grpo")
    assert a2 == pytest.approx(b2, rel=1e-9)


@pytest.mark.parametrize("law", ["shared", "iid", "mix:0.5", "dirmult:0.2"])
def test_always_verify_corner_law_invariant(law):
    """Corollary 3.3: if verifying makes the return state-independent, b_{G-1} = (G-1)/sqrt(G) * vote under every law."""
    for p in (0.2, 0.35, 0.7):
        pb = verification([p], [1.0], [1.0], c=0.05)
        assert pb.reads_neutralise()
        G = 8
        b = ex.bernstein(pb, G, law, "grpo", eps=0.0)
        assert b[G - 1] == pytest.approx((G - 1) / math.sqrt(G) * pb.vote(), rel=1e-9)


def test_iid_interior_equilibrium():
    """Under independent states GRPO stalls inside (0, 1) although verifying is optimal (p < 1/2, delta_q > 0)."""
    pb = verification([0.3], [0.0], [1.0], c=0.05)
    b = ex.bernstein(pb, 8, "iid", "grpo")
    e = ex.endpoint(b, 0.3)
    assert pb.delta_q() > 0 and 0.5 < e < 0.9
    assert ex.endpoint(ex.bernstein(pb, 8, "iid", "nostd"), 0.3) == 1.0


def test_awrb_vote_is_sign_of_delta_q():
    """Proposition 4.2: observation-measurable return + s = 1 -> every state has beta = delta_q."""
    for p, c in ((0.3, 0.05), (0.3, 0.4), (0.1, 0.05)):
        pb = verification([p], [0.0], [1.0], c=c)
        u = ex.expected_update(pb, 0.4, 8, "shared", "awrb")
        assert np.sign(u) == np.sign(pb.delta_q())


def test_gdpo_never_verifies_when_cost_is_a_component():
    """Proposition 4.3: per-component normalisation of (utility, cost) under the shared law gives U <= 0."""
    for p in (0.3, 0.7, 0.95):
        pb = verification([p], [0.0], [1.0], c=0.05)
        assert ex.expected_update(pb, 0.4, 8, "shared", "gdpo") <= 1e-9
