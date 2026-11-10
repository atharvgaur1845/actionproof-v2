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


# ---------------------------------------------------------------- B' gates: parameterisation and observation equivalence
import random as _random

from actionproof.bench.generator_a5 import BOUNDED, EXT_AT, LAG, UNBOUNDED


def _random_policy(seed):
    def pol(env, task):
        rng = _random.Random(seed)
        i = task.meta["ids"]
        ops = []
        for _ in range(rng.randint(4, 12)):
            c = rng.random()
            if c < 0.25:
                obs = env.call("issue_refund", {"order_id": i["order"], "amount_cents": i["amount"], "method": i["method"],
                                                **({"idempotency_key": "k"} if rng.random() < 0.5 else {})})
                ops.append(obs.get("op_id") or obs.get("request_id"))
            elif c < 0.45 and ops:
                env.call("get_operation", {"op_id": rng.choice(ops)})
            elif c < 0.6:
                env.call("list_operations", {"entity_id": i["order"]})
            elif c < 0.75:
                env.call("query_ledger", {"order_id": i["order"]})
            elif c < 0.85:
                env.call("wait", {"seconds": rng.choice((10, 60, 300))})
            else:
                env.call("lookup_order", {"order_id": i["order"]})
        env.call("submit_report", {"status": "uncertain", "summary": "x", "certificate": {"unresolved": ops[:1]}})
    return pol


def test_parameters_are_randomised_within_registered_ranges():
    T = make_a5_split("a5_eval")
    seen = {"bounded": set(), "unbounded": set(), "lag": set(), "ext": set(), "seed": set(), "prompt": set()}
    for t in T:
        a5 = t.meta["a5"]
        f = a5["faults"].get("issue_refund:0", {})
        if a5["scenario"] == "bounded_late":
            assert BOUNDED[0] <= f["delay"] <= BOUNDED[1]; seen["bounded"].add(f["delay"])
        if a5["scenario"] == "unbounded_late":
            assert UNBOUNDED[0] <= f["delay"] <= UNBOUNDED[1]; seen["unbounded"].add(f["delay"])
        if a5["ledger_lag"]:
            assert LAG[0] <= a5["ledger_lag"] <= LAG[1]; seen["lag"].add(a5["ledger_lag"])
        for e in a5["external"]:
            if a5["scenario"] != "pre_satisfied":
                assert EXT_AT[0] <= e["at"] <= EXT_AT[1]; seen["ext"].add(e["at"])
        seen["seed"].add(a5["env_seed"]); seen["prompt"].add(t.prompt.split(":")[0][:8])
    assert all(len(v) > 3 for k, v in seen.items() if k != "prompt") and len(seen["seed"]) > 200


def test_heldout_absent_from_all_training_splits():
    for s in ("a5_train", "a5_val"):
        assert not {t.meta["a5"]["scenario"] for t in make_a5_split(s)} & set(HELD_OUT)


@pytest.mark.parametrize("policy", ["oracle", "blind", "rand0", "rand1", "rand2"])
def test_every_inflight_pair_is_observation_equivalent(policy):
    T = make_a5_split("a5_eval")
    pairs = {}
    for t in T:
        if t.meta["a5"]["pair_id"] and t.meta["a5"]["pair_id"].startswith("inflight"):
            pairs.setdefault(t.meta["a5"]["pair_id"], []).append(t)
    assert len(pairs) == 30
    pol = {"oracle": a5_oracle, "blind": retry_blind}.get(policy) or _random_policy(int(policy[-1]))
    for pid, (a, b) in pairs.items():
        assert a.initial_state == b.initial_state and a.prompt == b.prompt
        ea, eb = _make_env(a), _make_env(b)
        pol(ea, a); pol(eb, b)
        assert [(s["tool"], s["obs"]) for s in ea.trace] == [(s["tool"], s["obs"]) for s in eb.trace], pid


def test_malformed_certificate_fields_do_not_crash(T):
    """D-A5-3: a policy emitted `evidence` as a bare int during GRPO; malformed fields count as empty (invalid)."""
    from actionproof.rewards.a5_rewards import consistency
    t = by(T, "benign")[0]
    env = _make_env(t); a5_oracle(env, t)
    env.reports[0]["certificate"] = {"effects": 3, "evidence": 4, "unresolved": "OP-1"}
    assert verify_a5(t, env)["proof_valid"] == 0
    assert consistency(env.trace, env.reports[0]) == 0.0
