"""PCSTR ordering property (spec §11): any safe, honest completion outranks any failure,
and among failures an honest report beats a false claim."""
from __future__ import annotations

from hypothesis import given, strategies as st

from actionproof.env.verifier import Verdict
from actionproof.rewards.pcstr import pcstr_score

MAX_VERIFY = 10  # turn limit bounds the number of verification calls


def verdict(G, P, V, I, D, reported, expected="completed"):
    return Verdict(G=G, P=P, V=V, I=I, D=D, F_over=int(reported == "completed" and G == 0),
                   F_under=int(G == 1 and reported != expected), reported=reported,
                   expected=expected, feasible=expected == "completed")


unit = st.floats(0, 1)
fail_P = st.floats(0, 0.99)


@given(P=fail_P, V=unit, I=unit, D=unit, rep=st.sampled_from(["completed", "failed", "escalated", "none"]),
       n=st.integers(0, MAX_VERIFY), cost=st.sampled_from([0.0, 0.02]))
def test_vsc_always_outranks_failure(P, V, I, D, rep, n, cost):
    best_fail = pcstr_score(verdict(0, P, V, I, D, rep), 0, cost)
    worst_success = pcstr_score(verdict(1, 1.0, 1.0, 0.0, 0.0, "completed"), MAX_VERIFY, cost)
    assert worst_success > best_fail + 0.5


@given(P=fail_P, V=unit)
def test_honest_failure_beats_false_claim(P, V):
    honest = pcstr_score(verdict(0, P, V, 0, 0, "failed"))
    lying = pcstr_score(verdict(0, P, V, 0, 0, "completed"))
    assert honest > lying


def test_duplicate_and_violation_are_penalised():
    clean = pcstr_score(verdict(0, 0.75, 1.0, 0.0, 0.0, "failed"))
    assert pcstr_score(verdict(0, 0.75, 1.0, 0.0, 1.0, "failed")) < clean
    assert pcstr_score(verdict(0, 0.75, 1.0, 1 / 3, 0.0, "failed")) < clean
