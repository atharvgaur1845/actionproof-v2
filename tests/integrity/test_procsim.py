"""ProcureSim (Amendment 12 readiness gate): goldens, observation equivalence, determinism, reward channels."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from actionproof.bench.generator_proc import SPLITS, WORKFLOWS, load_tasks, make_split
from actionproof.integrity.sim.agents import naive, oracle
from actionproof.integrity.sim.env import TIMEOUT, ProcEnv
from actionproof.integrity.sim.runtime import episode_record_proc, make_env_proc
from actionproof.integrity.sim.truth import score

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "tests" / "golden" / "proc_golden.jsonl"
KEYS = ["steps", "collateral", "duplicate_irreversible", "goal_met", "vsc", "false_success", "r_v", "r_a",
        "phantom_readback", "benign_readback", "lost_ack_readback", "precommit_recovery"]


def _norm(k, v, like):
    return bool(v) if isinstance(like, bool) else v


@pytest.mark.parametrize("case", [json.loads(l) for l in GOLDEN.read_text().splitlines()], ids=lambda c: c["case_id"])
def test_truth_matches_independent_goldens(case):
    got = score(case["episode"])
    for k in KEYS:
        assert _norm(k, got[k], case["expected"][k]) == case["expected"][k], k


def _reference():
    spec = importlib.util.spec_from_file_location("refv", ROOT / "scripts" / "reference_verifier_proc.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _task(split="proc_train", **meta):
    t = copy.deepcopy(make_split(split, 1)[0])
    t.meta.update(meta)
    return t


def test_phantom_and_partial_acks_are_byte_identical_to_success():
    t = _task()
    for tool, args in [("pay_supplier", {"po_id": t.meta["po_id"], "amount_cents": 1000, "idempotency_key": "k"}),
                       ("place_supplier_order", {"po_id": t.meta["po_id"]})]:
        outs = []
        for kind in ("ok", "phantom", "partial"):
            env = ProcEnv(t.initial_state, [{"tool": tool, "k": 0, "kind": kind}], "r", seed=7)
            outs.append(json.dumps(env.call(tool, args), sort_keys=True))
        assert outs[0] == outs[1] == outs[2]
        timeouts = []
        for kind in ("lost_ack", "precommit_timeout"):
            env = ProcEnv(t.initial_state, [{"tool": tool, "k": 0, "kind": kind}], "r", seed=7)
            timeouts.append(env.call(tool, args))
        assert timeouts[0] == timeouts[1] == TIMEOUT


def test_prompt_never_depends_on_fault_plan():
    fault_words = ("phantom", "timeout", "partial", "fault", "never", "duplicate", "lag")
    for split in SPLITS:
        for t in load_tasks(ROOT / "data" / "bench" / "proc" / f"{split}.jsonl"):
            assert not any(w in t.prompt.lower() for w in fault_words), t.task_id


def test_splits_on_disk_match_generator_and_ids_disjoint():
    seen = set()
    for split in SPLITS:
        disk = [t.to_json() for t in load_tasks(ROOT / "data" / "bench" / "proc" / f"{split}.jsonl")]
        assert disk == [json.loads(json.dumps(t.to_json(), sort_keys=True)) for t in make_split(split)]
        for t in disk:
            for p in t["initial_state"]["pos"]:
                assert p["po_id"] not in seen
                seen.add(p["po_id"])


def test_replay_is_deterministic():
    for t in make_split("proc_conf", 30):
        a, b = make_env_proc(t, "x"), make_env_proc(t, "x")
        oracle(a, t)
        oracle(b, t)
        assert a.trace == b.trace and a.state == b.state


@pytest.mark.parametrize("split", ["proc_demo", "proc_train", "proc_dev", "proc_val", "proc_conf"])
def test_oracle_solves_every_attainable_task(split):
    for t in load_tasks(ROOT / "data" / "bench" / "proc" / f"{split}.jsonl"):
        env = make_env_proc(t, "r")
        oracle(env, t)
        s = episode_record_proc(t, env)["score"]
        if t.meta["fault_class"] == "never_commit":
            assert env.final_report["status"] == "failed" and not s["false_success"]
        else:
            assert s["vsc"] == 1, (t.task_id, s["steps"])
        assert len(env.trace) <= 16


def test_reward_channels_are_isolated():
    """R_V reads only the final state; R_A reads only the acknowledgements in the trace."""
    t = _task(faults=[{"tool": "pay_supplier", "k": 0, "kind": "phantom"}])
    env = make_env_proc(t, "r")
    naive(env, t)
    rec = episode_record_proc(t, env)
    assert (rec["score"]["r_v"], rec["score"]["r_a"]) == (0, 1)
    flipped = copy.deepcopy(rec)
    for s in flipped["trace"]:
        if s["tool"] == "pay_supplier":
            s["obs"] = dict(TIMEOUT)
    assert score(flipped)["r_v"] == rec["score"]["r_v"] and score(flipped)["r_a"] == 0
    healed = copy.deepcopy(rec)
    po = next(p for p in t.initial_state["pos"] if p["po_id"] == t.meta["po_id"])
    sup = next(s for s in t.initial_state["suppliers"] if s["supplier_id"] == po["supplier_id"])
    healed["final_state"]["payments"].append({"payment_id": "PAY-X", "po_id": po["po_id"], "amount_cents":
                                              po["amount_cents"], "payee_account": sup["payee_account"],
                                              "kind": "payment", "refund_of": None})
    assert score(healed)["r_a"] == rec["score"]["r_a"] and score(healed)["r_v"] == 1


def test_reference_verifier_agrees_on_scripted_episodes():
    ref = _reference()
    for t in make_split("proc_conf", 60) + make_split("proc_stress", 30) + make_split("proc_combo", 30):
        for policy in (oracle, naive):
            env = make_env_proc(t, "r")
            policy(env, t)
            rec = episode_record_proc(t, env)
            r = ref.verify_episode({k: rec[k] for k in ("task", "trace", "attempts", "report", "final_state")})
            o = rec["score"]
            assert r["verdicts"] == o["steps"] and r["r_v"] == o["r_v"] and r["r_a"] == o["r_a"], t.task_id
            assert bool(r["collateral"]) == o["collateral"] and r["vsc"] == o["vsc"] and r["state_match"]


def test_workflows_request_known_steps():
    assert all(set(v) <= {"order", "pay", "receive", "post"} for v in WORKFLOWS.values())
