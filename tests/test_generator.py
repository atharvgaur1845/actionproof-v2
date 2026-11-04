"""Generated tasks are deterministic, leak-free in their prompts, and solvable (spec §10, §24.3)."""
from __future__ import annotations

from collections import Counter

import pytest

from actionproof.bench.generator import SPLITS, Task, make_split, make_task
from actionproof.env.db import snapshot
from actionproof.env.tools_ecom import EcomEnv
from actionproof.env.verifier import verify, vsc
from actionproof.rewards.sanity_agents import lazy_liar, task_oracle, RefundTask


def run_oracle(task: Task) -> int:
    env = EcomEnv(task.initial_state, task.schedule, rollout_id=task.task_id, seed=0,
                  stale_status=task.stale_status, ambiguous_text=task.ambiguous_text)
    ids = task.meta["ids"]
    task_oracle(env, task.workflow, ids["order"], ids["ticket"])
    return vsc(verify(task.goal, snapshot(env.conn), env.log, env.final_report))


def test_deterministic_and_roundtrip():
    a, b = make_task("train", 17), make_task("train", 17)
    assert a.to_json() == b.to_json()
    assert Task.from_json(a.to_json()).to_json() == a.to_json()


@pytest.mark.parametrize("split", list(SPLITS))
def test_oracle_solves_every_task(split):
    tasks = make_split(split, 120)
    failed = [t.task_id for t in tasks if run_oracle(t) != 1]
    assert not failed, failed[:10]


def test_split_fault_menus_are_disjoint_where_promised():
    train = Counter(f for t in make_split("train") for f in t.meta["faults"])
    ood = Counter(f for t in make_split("test_ood_fault") for f in t.meta["faults"])
    assert {"F2_stale", "F8_ambiguous", "F6", "F7"}.isdisjoint(train)
    assert {"F1_phantom", "F3_partial", "F4_timeout", "F5"}.isdisjoint(ood)
    assert 0.3 < sum(t.meta["benign"] for t in make_split("train")) / 400 < 0.5


def test_prompts_do_not_mention_faults_or_amounts():
    for t in make_split("train", 200):
        assert str(t.meta["ids"]["amount"]) not in t.prompt
        for word in ("approval", "eligible", "timeout", "phantom", "stale", "fault"):
            assert word not in t.prompt.lower()


def test_entities_never_shared_between_train_and_test():
    ids = lambda s: {t.meta["ids"]["order"] + t.meta["ids"]["cust"] for t in make_split(s)}
    assert not ids("train") & ids("test_iid")


def test_probe_cues_split_disjoint_from_test_entities():
    ids = lambda s: {t.meta["ids"]["order"] + t.meta["ids"]["cust"] for t in make_split(s)}
    assert not ids("probe_cues") & ids("test_ood_fault") and not ids("probe_cues") & ids("test_iid")


def test_no_entity_shared_between_train_and_any_eval_split():
    ids = lambda s: {t.meta["ids"]["order"] + t.meta["ids"]["cust"] for t in make_split(s)}
    train = ids("train") | ids("val")
    for s in ("test_iid", "test_ood_fault", "test_iid_c", "test_ood_fault_c", "test_calib_swap"):
        assert not train & ids(s), s
    assert not ids("test_iid_c") & ids("test_iid") and not ids("test_ood_fault_c") & ids("test_ood_fault")
