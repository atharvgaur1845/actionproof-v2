"""A5 reward arms (draft Amendment 5): source-only difference, identical transcript-only K, predicted incentives."""
from __future__ import annotations

import copy

import pytest

from actionproof.bench.generator_a5 import make_a5_split
from actionproof.eval.a5_runtime import episode_record_a5, make_env_a5
from actionproof.rewards.a5_rewards import LAMBDA, a5_reward, consistency, t_ack, t_state
from actionproof.rewards.sanity_agents_a5 import a5_oracle, retry_blind


@pytest.fixture(scope="module")
def T():
    return make_a5_split("a5_eval")


def first(T, sc):
    return next(t for t in T if t.meta["a5"]["scenario"] == sc)


def rec(t, agent):
    env = make_env_a5(t, "r")
    agent(env, t)
    return episode_record_a5(t, env)


def test_oracle_consistent_and_state_rewarded_where_resolvable(T):
    for t in T:
        r = rec(t, a5_oracle)
        assert consistency(r["trace"], r["report"]) == 1.0, t.meta["a5"]["scenario"]
        if t.meta["a5"]["resolvable"]:
            assert t_state(r) == 1.0 and a5_reward(r, "state") == 1 + LAMBDA


def test_k_reads_no_hidden_state(T):
    """K is a function of the transcript alone: identical traces give identical K whatever the hidden world."""
    a, b = (first(T, s) for s in ("unbounded_late", "never_commit"))
    ra = rec(a, a5_oracle)
    rb = rec(next(t for t in T if t.meta["a5"]["pair_id"] == a.meta["a5"]["pair_id"] and t is not a), a5_oracle)
    assert ra["trace"] == rb["trace"] and consistency(ra["trace"], ra["report"]) == consistency(rb["trace"], rb["report"])


def test_ack_reward_pays_blind_retry_and_duplicates(T):
    for sc in ("lost_ack", "pre_satisfied", "bounded_late"):  # acks say success once; the world has two refunds
        r = rec(first(T, sc), retry_blind)
        assert r["a5v"]["duplicate"] == 1
        assert t_ack(r) == 1.0 and t_state(r) == -1.0  # the ack arm pays the duplicate; the state arm charges it


def test_ack_reward_cannot_credit_someone_elses_refund(T):
    r = rec(first(T, "pre_satisfied"), a5_oracle)  # correct behaviour: do not refund again
    assert t_state(r) == 1.0 and t_ack(r) == 0.0


def test_k_rejects_inconsistent_reports(T):
    r = rec(first(T, "lost_ack"), a5_oracle)
    for mutate in (lambda x: x.update(status="uncertain"),                                # abstain after proof
                   lambda x: x["certificate"]["effects"][0].update(actor="billing_bot"),  # misattribute
                   lambda x: x["certificate"].update(evidence=[0]),                       # wrong evidence
                   lambda x: x.update(status="failed")):                                 # deny a seen commit
        rep = copy.deepcopy(r["report"]); mutate(rep)
        assert consistency(r["trace"], rep) == 0.0
    r2 = rec(first(T, "unbounded_late"), a5_oracle)
    rep = copy.deepcopy(r2["report"]); rep["status"] = "completed"
    assert consistency(r2["trace"], rep) == 0.0


def test_expected_ordering_on_unresolvable_pair(T):
    """In expectation over the observation-equivalent pair, an honest 'uncertain' beats a blind retry."""
    late, never = first(T, "unbounded_late"), None
    never = next(t for t in T if t.meta["a5"]["pair_id"] == late.meta["a5"]["pair_id"] and t is not late)
    for src in ("state",):  # (the ack arm has no such ordering: that is the point of the comparison)
        honest = (a5_reward(rec(late, a5_oracle), src) + a5_reward(rec(never, a5_oracle), src)) / 2
        blind = (a5_reward(rec(late, retry_blind), src) + a5_reward(rec(never, retry_blind), src)) / 2
        assert honest > blind
