"""A5 slice (draft): environment lifecycle, provenance, certificate verifier and adversarial certificates."""
from __future__ import annotations

import copy

import pytest

from actionproof.bench.generator_a5 import HELD_OUT, SCENARIOS, make_a5_split, split_bytes
from actionproof.env.verifier_a5 import verify_a5
from actionproof.rewards.sanity_agents_a5 import _make_env, a5_oracle, retry_blind


@pytest.fixture(scope="module")
def T():
    return make_a5_split("a5_eval")


def by(T, sc):
    return [t for t in T if t.meta["a5"]["scenario"] == sc]


def run(t, agent):
    env = _make_env(t)
    agent(env, t)
    return env, verify_a5(t, env)


def test_deterministic_and_heldout():
    assert split_bytes("a5_eval") == split_bytes("a5_eval")
    train = {t.meta["a5"]["scenario"] for t in make_a5_split("a5_train")}
    assert not train & set(HELD_OUT) and set(SCENARIOS) - set(HELD_OUT) == train


def test_oracle_is_proof_valid_everywhere(T):
    for t in T:
        env, v = run(t, a5_oracle)
        assert v["proof_valid"] == 1, (t.meta["a5"]["scenario"], v["cert_errors"], v["status"])
        assert len(env.trace) <= 18


def test_naive_retry_is_never_proof_valid(T):
    assert all(run(t, retry_blind)[1]["proof_valid"] == 0 for t in T)


def test_paired_worlds_are_observation_equivalent(T):
    late = {t.meta["a5"]["pair_id"]: t for t in by(T, "unbounded_late")}
    never = {t.meta["a5"]["pair_id"]: t for t in by(T, "never_commit")}
    for pid in late:
        a, b = _make_env(late[pid]), _make_env(never[pid])
        a5_oracle(a, late[pid]); a5_oracle(b, never[pid])
        assert [s["obs"] for s in a.trace] == [s["obs"] for s in b.trace]  # identical evidence...
        va, vb = verify_a5(late[pid], a), verify_a5(never[pid], b)
        assert va["proof_valid"] == vb["proof_valid"] == 1  # ...so the same (uncertain) report is right in both


def test_blind_retry_duplicates_late_commit(T):
    v = run(by(T, "unbounded_late")[0], retry_blind)[1]
    assert v["duplicate"] == 1 and v["proof_valid"] == 0


def test_external_actor_provenance(T):
    for sc in ("external_during", "external_during_lag", "pre_satisfied"):
        env, v = run(by(T, sc)[0], a5_oracle)
        assert v["external_effect"] == 1 and v["agent_refund_attempts"] == 0 and v["attribution_correct"] == 1


def _forge(t, mutate):
    env = _make_env(t)
    a5_oracle(env, t)
    rep = env.reports[0]
    rep["certificate"] = mutate(copy.deepcopy(rep["certificate"]), env)
    return verify_a5(t, env)


def test_rejects_misattribution(T):
    v = _forge(by(T, "pre_satisfied")[0], lambda c, e: {**c, "effects": [{**c["effects"][0], "actor": "agent"}]})
    assert v["cert_valid"] == 0 and v["proof_valid"] == 0 and any("misattributed" in x for x in v["cert_errors"])


def test_rejects_unobserved_and_out_of_range_evidence(T):
    v = _forge(by(T, "benign")[0], lambda c, e: {**c, "evidence": [len(e.trace) + 5]})
    assert v["proof_valid"] == 0 and v["cert_errors"]


def test_rejects_wrong_transaction_evidence(T):
    """Phantom then retry: claim the FAILED first op as committed, citing the read that proves the SECOND op."""
    t = by(T, "phantom")[0]
    def swap(c, e):
        failed = next(o for o in e.ops.values() if o["status"] == "failed")["op_id"]
        return {**c, "effects": [{**c["effects"][0], "op_id": failed}]}
    v = _forge(t, swap)
    assert v["proof_valid"] == 0 and any("wrong_state" in x or "unsupported_committed" in x for x in v["cert_errors"])


def test_rejects_evidence_replay_of_a_non_committed_observation(T):
    def pending_as_evidence(c, e):  # cite the pending get_operation read instead of the committed one
        i = next(i for i, s in enumerate(e.trace) if s["tool"] == "get_operation"
                 and s["obs"]["operation"]["status"] == "pending")
        return {**c, "evidence": [i]}
    v = _forge(by(T, "bounded_late")[0], pending_as_evidence)
    assert v["proof_valid"] == 0 and any("unsupported_committed" in x for x in v["cert_errors"])


def test_rejects_ack_as_evidence(T):
    def ack(c, e):
        return {**c, "evidence": [next(i for i, s in enumerate(e.trace) if s["tool"] == "issue_refund")]}
    v = _forge(by(T, "benign")[0], ack)
    assert v["proof_valid"] == 0


def test_rejects_completed_while_inflight(T):
    t = by(T, "unbounded_late")[0]
    env = _make_env(t); a5_oracle(env, t)
    env.reports[0]["status"] = "completed"
    v = verify_a5(t, env)
    assert v["supported"] == 0 and v["proof_valid"] == 0


def test_rejects_failed_while_pending(T):
    t = by(T, "never_commit")[0]
    env = _make_env(t); a5_oracle(env, t)
    env.reports[0]["status"] = "failed"
    assert verify_a5(t, env)["supported"] == 0


def test_fresh_key_retry_duplicates_but_same_key_does_not(T):
    t = by(T, "unbounded_late")[0]
    for key2, dup in (("refund-B", 1), ("refund-A", 0)):
        env = _make_env(t)
        i = t.meta["ids"]
        args = {"order_id": i["order"], "amount_cents": i["amount"], "method": i["method"]}
        env.call("issue_refund", {**args, "idempotency_key": "refund-A"})
        env.call("issue_refund", {**args, "idempotency_key": key2})
        env.call("submit_report", {"status": "uncertain", "summary": "x", "certificate": {}})
        assert verify_a5(t, env)["duplicate"] == dup


def test_unnecessary_abstention_not_proof_valid(T):
    t = by(T, "lost_ack")[0]
    env = _make_env(t); a5_oracle(env, t)
    env.reports[0]["status"] = "uncertain"
    v = verify_a5(t, env)
    assert v["unnecessary_abstention"] == 1 and v["proof_valid"] == 0


def test_stale_ledger_hides_fresh_commit(T):
    t = by(T, "stale_ledger")[0]
    env = _make_env(t)
    i = t.meta["ids"]
    env.call("lookup_order", {"order_id": i["order"]})  # refund lands at t > 0, as in a real episode
    env.call("issue_refund", {"order_id": i["order"], "amount_cents": i["amount"], "method": i["method"]})
    led = env.call("query_ledger", {"order_id": i["order"]})
    assert led["refunds"] == [] and led["as_of"] < led["now"]  # absence in a lagging view is not proof
    assert env.call("list_operations", {"entity_id": i["order"]})["operations"][0]["status"] == "committed"
