"""proc1/proc2 are unchanged by TXN: frozen sim files keep their registered hashes, the proc1 goldens still pass on the
frozen verifier and agree 100% with scripts/reference_verifier_proc.py, and TxnEnv reproduces proc2 exactly on proc2
tasks (same observations, state and attempts for every old tool)."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from actionproof.bench.generator_proc import DATA, load_tasks
from actionproof.integrity.sim import agents
from actionproof.integrity.sim.runtime import episode_record_proc
from actionproof.integrity.sim.truth import score
from actionproof.stage3 import scripted
from actionproof.stage3.env import make_env_s3
from actionproof.txn.env import TxnEnv, make_env_txn

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "tests/golden/proc_golden.jsonl"
KEYS = ["steps", "collateral", "duplicate_irreversible", "goal_met", "vsc", "false_success", "r_v", "r_a",
        "phantom_readback", "benign_readback", "lost_ack_readback", "precommit_recovery"]
CORE = ("idx", "tool", "po_id", "fault", "committed", "from_idempotency_store", "obs_status")


def _ref():
    spec = importlib.util.spec_from_file_location("refv_proc", ROOT / "scripts/reference_verifier_proc.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_frozen_sim_and_stage3_files_keep_registered_hashes():
    for hashes in ("results/a12_hashes.json", "results/s3_hashes.json"):
        files = json.loads((ROOT / hashes).read_text())["files"]
        for path, h in files.items():
            if path.startswith(("actionproof/integrity/sim/", "actionproof/stage3/env.py", "scripts/reference_verifier",
                                "tests/golden/proc_golden", "docs/stage2/VERIFIER_SPEC")):
                assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == h, path


def test_proc_goldens_frozen_verifier_and_reference_agree_100pct():
    ref = _ref()
    cases = [json.loads(line) for line in GOLDEN.read_text().splitlines()]
    assert len(cases) == 91
    for c in cases:
        got = score(c["episode"])
        for k in KEYS:
            assert (bool(got[k]) if isinstance(c["expected"][k], bool) else got[k]) == c["expected"][k], (c["case_id"], k)
        r = ref.verify_episode(c["episode"])
        assert r["verdicts"] == got["steps"], c["case_id"]
        assert r["vsc"] == got["vsc"] and bool(r["collateral"]) == got["collateral"], c["case_id"]


def _run(mk, t, pol):
    env = mk(t, "r")
    for tool, args in t.meta.get("start_prefix") or []:
        env.call(tool, dict(args))
    getattr(scripted if t.meta.get("start_prefix") else agents, pol)(env, t)
    return env


@pytest.mark.parametrize("split", ["proc_dev", "proc_stress", "proc_combo", "s3_pp_full", "s3_rec_dev", "p1_test_combo"])
def test_txn_env_reproduces_proc2(split):
    ref = _ref()
    for t in load_tasks(DATA / f"{split}.jsonl"):
        for pol in ("oracle", "naive"):
            a, b = _run(make_env_s3, t, pol), _run(make_env_txn, t, pol)
            assert [(s["tool"], s["args"], s["obs"]) for s in a.trace] == [(s["tool"], s["args"], s["obs"]) for s in b.trace]
            assert a.state == b.state
            assert [{k: x[k] for k in CORE} for x in a.attempts] == [{k: x[k] for k in CORE} for x in b.attempts]
            # the frozen reference verifier reads proc3 records exactly as it reads proc2 records (proc1's reference
            # replays a partial as one half, so on ratio partials its state_match is False for proc2 as well)
            ra, rb = (ref.verify_episode({k: rec[k] for k in ("task", "trace", "attempts", "report", "final_state")})
                      for rec in (episode_record_proc(t, a), episode_record_proc(t, b)))
            assert {k: ra[k] for k in ("verdicts", "vsc", "state_match")} == \
                {k: rb[k] for k in ("verdicts", "vsc", "state_match")}, t.task_id
            if not any(f.get("ratio") for f in t.meta.get("faults") or []):
                assert rb["state_match"] and rb["verdicts"] == episode_record_proc(t, b)["score"]["steps"], t.task_id


def test_old_tool_docstrings_and_ack_text_unchanged():
    from actionproof.integrity.sim.env import ProcEnv, READ_TOOLS, WRITE_TOOLS
    for name in WRITE_TOOLS + ("submit_report",):
        assert getattr(TxnEnv, name).__doc__ == getattr(ProcEnv, name).__doc__, name
    for name in READ_TOOLS:
        new, old = getattr(TxnEnv, name).__doc__, getattr(ProcEnv, name).__doc__
        assert new.startswith(old.rstrip()) and "consistent:" in new, name
    import inspect
    for name in READ_TOOLS:
        params = list(inspect.signature(getattr(TxnEnv, name)).parameters)
        assert params == list(inspect.signature(getattr(ProcEnv, name)).parameters) + ["consistent"], name
