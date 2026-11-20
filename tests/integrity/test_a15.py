"""A15 harness checks (PREREGISTRATION_A15_A17): the ratio environment equals ProcEnv at q = 1/2, branch states match
q, prefixes are identical across branches until the forced read's result, scripted references score as predicted,
splits are disjoint. Needs `transformers` for the system prompt; skipped without it."""
import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("transformers")

from actionproof.bench.generator_proc import load_tasks  # noqa: E402
from actionproof.integrity.sim.runtime import make_env_proc  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("a15", ROOT / "scripts/a15_eval.py")
a15 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a15)
cef = a15.cef
TASKS = load_tasks(ROOT / "data/bench/proc/proc_mech_dev.jsonl")[:12]


def test_half_ratio_equals_procenv():
    for t in TASKS:
        bt = a15.branch_task(t, "Q2")
        envs = [make_env_proc(bt, "x"), a15.make_env_ratio(bt, "x")]
        po = next(p for p in t.initial_state["pos"] if p["po_id"] == t.meta["po_id"])
        for e in envs:
            e.call("pay_supplier", {"po_id": po["po_id"], "amount_cents": po["amount_cents"], "idempotency_key": "k"})
        assert json.dumps(envs[0].state, sort_keys=True) == json.dumps(envs[1].state, sort_keys=True)
        assert envs[0].trace == envs[1].trace


def test_branch_state_matches_q():
    for t in TASKS:
        po = next(p for p in t.initial_state["pos"] if p["po_id"] == t.meta["po_id"])
        for b, q in a15.QUARTERS.items():
            env, _, _ = cef.build(t, b)
            paid = [p["amount_cents"] for p in env.state["payments"] if p["po_id"] == po["po_id"]]
            assert paid == ([] if q == 0 else [po["amount_cents"] * q // 4]), (t.task_id, b, paid)


def test_prefix_identical_until_evidence():
    for t in TASKS:
        convs = {b: cef.build(t, b)[1] for b in a15.BRANCHES}
        heads = [c[:-1] for c in convs.values()]
        assert all(h == heads[0] for h in heads), t.task_id
        assert len({c[-1]["content"] for c in convs.values()}) == len(convs), t.task_id


def test_amounts_aligned_and_splits_disjoint():
    mech = load_tasks(ROOT / "data/bench/proc/proc_mech.jsonl")
    dev = load_tasks(ROOT / "data/bench/proc/proc_mech_dev.jsonl")
    assert len(mech) == 200 and len(dev) == 60
    for t in mech + dev:
        po = next(p for p in t.initial_state["pos"] if p["po_id"] == t.meta["po_id"])
        assert a15.aligned(po["amount_cents"])
    other = {p["po_id"] for f in (ROOT / "data/bench/proc").glob("*.jsonl") if f.stem not in a15.SPLITS
             for t in load_tasks(f) for p in t.initial_state["pos"]}
    mine = [p["po_id"] for t in mech + dev for p in t.initial_state["pos"]]
    assert len(mine) == len(set(mine)) and not set(mine) & other


def test_scripted_references():
    recs = cef.run_scripted(TASKS[:6], "agent")
    rate = {(p, b): sum(r["correct"] for r in recs if r["policy"] == p and r["branch"] == b) / 6
            for p in ("oracle", "naive", "fixed_retry") for b in a15.BRANCHES}
    assert all(rate[("oracle", b)] == 1 for b in a15.BRANCHES)
    assert [rate[("naive", b)] for b in a15.BRANCHES] == [0, 0, 0, 0, 1]
    assert [rate[("fixed_retry", b)] for b in a15.BRANCHES] == [1, 0, 0, 0, 0]
