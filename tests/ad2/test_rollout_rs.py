"""CPU tests for group-generation condition B (actionproof/ad2/rollout_rs.py)."""
from __future__ import annotations

from collections import Counter

import actionproof.ad.rollout as R
from actionproof.ad.scripted import POLICIES
from actionproof.ad.worlds import P_CLEAN, world_key, world_prior
from actionproof.ad2 import rollout_rs as RS
from actionproof.bench.generator_proc import DATA, load_tasks

TASKS = load_tasks(DATA / "cw_train.jsonl")[:30]


def test_worlds_are_iid_from_the_prior():
    t = next(t for t in TASKS if len(t.meta["requested"]) == 3)
    prior = {world_key(f): p for f, p in world_prior(t.meta["requested"])}
    n = 6000
    c = Counter(world_key(RS.world_for(t, s)) for s in range(n))
    assert set(c) <= set(prior)
    assert abs(c[()] / n - P_CLEAN) < 0.025
    for k, p in prior.items():
        assert abs(c[k] / n - p) < 4 * (p * (1 - p) / n) ** 0.5 + 1e-3


def test_rollouts_of_one_task_get_different_worlds():
    t = TASKS[0]
    keys = {world_key(RS.world_for(t, s)) for s in range(200)}
    assert len(keys) > 3


def test_record_and_posterior_use_the_rollout_world():
    for t in TASKS:
        for seed in (11, 12, 13, 14, 15):
            env = RS.env_rs(t, f"ro-{seed}", seed=0)
            POLICIES["always"](env, t)
            rec = RS.record_rs(t, env)
            ad = RS.marginalise_rs(t, env.trace, 0)
            assert ad["consistent_actual"] and ad["in_prior"]
            assert rec["world"] == env._ad2_faults
            # a policy that reads after every write reveals the world: posterior mean = realised R_V
            assert abs(ad["r_bar"] - rec["score"]["r_v"]) < 1e-9 or ad["post_fault"] > 0


def test_never_read_posterior_is_world_independent():
    t = next(t for t in TASKS if "pay" in t.meta["requested"])
    vals = set()
    for seed in range(40):
        env = RS.env_rs(t, f"ro-{seed}", seed=0)
        POLICIES["never"](env, t)
        RS.record_rs(t, env)
        ad = RS.marginalise_rs(t, env.trace, 0)
        if not any(f["kind"] in ("lost_ack", "precommit_timeout") for f in env._ad2_faults):
            vals.add(round(ad["r_bar"], 9))
    assert len(vals) == 1


def test_install_rebinds_only_rollout_names():
    orig = (R.make_env_cw, R.marginalise, R.episode_record_proc)
    try:
        RS.install()
        assert R.make_env_cw is RS.env_rs and R.marginalise is RS.marginalise_rs
        assert R.episode_record_proc is RS.record_rs
    finally:
        R.make_env_cw, R.marginalise, R.episode_record_proc = orig
