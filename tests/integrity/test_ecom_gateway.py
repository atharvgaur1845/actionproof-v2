"""Amendment 11 gateway (actionproof/integrity/adapters/ecom.py), splits, taxonomy and offline observe. CPU only."""
from __future__ import annotations

import importlib.util
import inspect
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import pytest

from actionproof.bench.generator import Task
from actionproof.env.db import snapshot
from actionproof.env.verifier import verify
from actionproof.integrity.adapters.ecom import REFUSAL, EcomGateway, gateway_episode_record
from actionproof.rollout.agenttune_rollout import make_env

ROOT = Path(__file__).resolve().parents[2]
C2 = ROOT / "data/bench/c2"


def _module(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


EG = _module("eval_gateway")
CF = _module("classify_faults")
AGENTS = EG.SCRIPTED


def _tasks(split: str) -> list[Task]:
    return [Task.from_json(json.loads(line)) for line in (C2 / f"{split}.jsonl").open()]


IID, STRESS = _tasks("test_iid_c2"), _tasks("v2_stress_c2")
# 20 tasks covering every fault type: 12 IID (benign, F1, F3, F4, F5) + 8 timeout-before-commit
TWENTY = IID[:12] + STRESS[:8]
ALL = IID + STRESS


def _play(agent: str, t: Task, mode: str | None):
    env = make_env(t, f"ev-{t.task_id}-0", seed=0)
    target = env if mode is None else EcomGateway(env, t, mode)
    try:
        EG.scripted_agent(agent)(target, t)
    except (KeyError, TypeError, IndexError):
        pass
    return env, target


def _state(env) -> str:
    return json.dumps({"trace": env.trace, "snap": snapshot(env.conn), "receipts": [asdict(r) for r in env.log.receipts],
                       "attempts": [asdict(a) for a in env.log.attempts], "rng": repr(env.rng.getstate()),
                       "calls": dict(env.calls), "seq": env._seq, "reports": env.reports,
                       "idem": repr(sorted(env.idempotency.items(), key=repr))}, sort_keys=True, default=str)


def test_splits_shape_and_v2_format():
    assert len(IID) == 200 and len(STRESS) == 100
    assert all(t.meta["env_version"] == 2 for t in ALL)
    assert all(t.meta["precommit_timeouts"] for t in STRESS)
    assert all(type(make_env(t, "r")).__name__ == "EcomEnvV2" for t in TWENTY)
    assert {f for t in TWENTY for f in t.meta["faults"]} >= {"F1_phantom", "F3_partial", "F4_timeout", "F5"}


@pytest.mark.parametrize("agent", AGENTS)
def test_observe_is_byte_identical_to_passthrough(agent):
    for t in TWENTY:
        plain, _ = _play(agent, t, None)
        observed, gw = _play(agent, t, "observe")
        assert _state(plain) == _state(observed), (agent, t.task_id)
        assert gw.trace is observed.trace  # the agent-visible trace is the env trace itself
        passthrough, _ = _play(agent, t, "passthrough")
        assert _state(plain) == _state(passthrough), (agent, t.task_id)


def test_gateway_tools_have_identical_schemas():
    t = TWENTY[0]
    env = make_env(t, "r")
    gw = EcomGateway(make_env(t, "r"), t, "enforce_inform")
    a, b = env.tools(), gw.tools()
    assert [f.__name__ for f in a] == [f.__name__ for f in b]
    for f, g in zip(a, b):
        assert f.__doc__ == g.__doc__ and inspect.signature(f) == inspect.signature(g)
        assert f.__annotations__ == g.__annotations__
    assert not hasattr(gw, "adapt_prompt")


def _expected(at) -> str:
    if not at.committed:
        return "absent"
    return "partial" if at.fault == "F3_partial" else "committed"


@pytest.mark.parametrize("agent", AGENTS)
def test_observe_verdicts_agree_with_ground_truth_and_verify(agent):
    for t in ALL:
        env, gw = _play(agent, t, "observe")
        attempts = {a.action_idx: a for a in env.log.attempts}
        for w in gw.writes:
            if w["verdict"] == "rejected":
                assert w["env_idx"] not in attempts
                continue
            at = attempts.get(w["env_idx"])
            if at is None:  # exactly-once replay of a committed keyed write: no new attempt
                assert w["verdict"] == "committed" and env.log.committed(w["op"], w["entity"]), (t.task_id, w)
            else:
                assert w["verdict"] == _expected(at), (agent, t.task_id, w["op"], at)
            if w["verdict"] == "committed":
                assert w["effect_holds_a4"] is not False
        # per required write, against the frozen verifier: verified <=> receipt (V) and postconditions (P) hold
        v = verify(t.goal, snapshot(env.conn), env.log, env.final_report)
        req = gw.required_status()
        if t.goal.feasible:
            assert all(r["verified"] for r in req) == (v.V == 1.0 and v.P == 1.0), (agent, t.task_id, req, v)
        for r in req:
            if r["verified"]:
                assert env.log.committed(r["op"], r["entity"]), (agent, t.task_id, r)


@pytest.mark.parametrize("mode", ["enforce", "enforce_inform"])
@pytest.mark.parametrize("agent", AGENTS)
def test_enforce_never_duplicates_an_irreversible_effect(agent, mode):
    for t in ALL:
        env, gw = _play(agent, t, mode)
        dup = Counter((r.op, r.entity) for r in env.log.receipts if r.irreversible)
        assert max(dup.values(), default=0) <= 1, (agent, mode, t.task_id, dup)
        rec = gateway_episode_record(t, gw)
        assert rec["system_level"]["D"] == 0
        # a completed report reaches the environment only when every required write is verified
        if env.final_report and env.final_report["status"] == "completed":
            assert all(r["verified"] for r in gw.required_status())
        # gateway retries are tagged in the env trace and never counted as agent calls
        n_gw = sum(s.get("by") == "gateway" for s in env.trace)
        assert n_gw == rec["gateway"]["n_gateway_retries"]
        assert rec["agent_level"]["n_calls"] == len(gw.trace)


def _no_free_text(x) -> bool:
    if isinstance(x, str):
        return " " not in x
    if isinstance(x, dict):
        return all(_no_free_text(v) for v in x.values())
    if isinstance(x, list):
        return all(_no_free_text(v) for v in x)
    return True


@pytest.mark.parametrize("agent", AGENTS)
def test_enforce_inform_payload_schema(agent):
    seen = Counter()
    for t in ALL:
        env, gw = _play(agent, t, "enforce_inform")
        writes = {w["env_idx"]: w for w in gw.writes}
        for s in gw.trace:
            if env.META.get(s["tool"], (None, ""))[1] not in ("write", "terminal"):
                continue
            obs = s["obs"]
            if not s["forwarded"]:
                assert obs == REFUSAL and s["blocked"]
                seen["refusal"] += 1
                continue
            if s["tool"] == "submit_report":
                continue
            w = writes[s["env_idx"]]
            if w["final_verdict"] == "rejected":
                assert obs["status"] == "error" and obs["error"] not in ("not_committed", "timeout")
                continue
            if obs["status"] == "ok":
                assert obs["verdict"] == "VERIFIED_SUCCESS" and w["final_verdict"] == "committed"
                assert "detail" not in obs and "error" not in obs and obs["evidence_id"] == w["evidence_id"]
                seen["verified"] += 1
            else:
                assert set(obs) == {"status", "error", "verdict", "evidence_id", "observed"}, obs
                assert obs["error"] == "not_committed" and obs["verdict"] == w["final_verdict"].upper()
                assert obs["verdict"] in ("ABSENT", "PARTIAL")
                assert isinstance(obs["observed"], list) and all("table" in r for r in obs["observed"])
                seen["not_committed"] += 1
            assert _no_free_text(obs), obs
    if agent != "no_op":  # no_op never writes
        assert seen["verified"] > 0
    if agent in ("lazy_liar", "blind_retrier", "policy_violator", "lean_expert", "oracle"):
        assert seen["not_committed"] > 0  # these agents meet phantom or partial writes on these splits


def test_enforce_retries_timeout_before_commit_once_with_same_key():
    t = next(t for t in STRESS if t.schedule.by_call.get(("issue_refund", 0)))
    env, gw = _play("task_oracle", t, "enforce")
    retried = [w for w in gw.writes if "gateway_retry" in w]
    assert retried and retried[0]["op"] == "issue_refund" and retried[0]["final_verdict"] == "committed"
    g = env.trace[retried[0]["gateway_retry"]["env_idx"]]
    assert g["by"] == "gateway" and g["args"]["idempotency_key"] == retried[0]["args"]["idempotency_key"]
    assert gw.trace[retried[0]["agent_idx"]]["obs"]["error"] == "timeout"  # enforce: the agent sees the original ack
    assert gateway_episode_record(t, gw)["vsc"] == 1


def test_enforce_refuses_false_finalization():
    t = next(t for t in IID if t.meta["faults"] == ["F1_phantom"] and t.schedule.by_call.get(("issue_refund", 0)))
    env, gw = _play("lazy_liar", t, "enforce")
    rec = gateway_episode_record(t, gw)
    assert env.final_report is None and rec["gateway"]["n_refused"] == 1
    assert rec["gateway"]["finalization"] == "refused" and rec["system_level"]["F_over"] == 0
    assert rec["agent_level"]["report"] == "completed" and rec["agent_level"]["F_over"] == 1
    p_env, p_gw = _play("lazy_liar", t, "passthrough")
    assert gateway_episode_record(t, p_gw)["system_level"]["F_over"] == 1


def test_offline_observe_matches_live_observe(tmp_path):
    tasks = TWENTY
    recs = EG.run_scripted("lean_expert", tasks, "passthrough")
    p = tmp_path / "pt.jsonl"
    p.write_text("".join(json.dumps(r, sort_keys=True, default=str) + "\n" for r in recs))
    off = EG.offline_observe(str(p), tasks)
    live = EG.run_scripted("lean_expert", tasks, "observe")
    for a, b in zip(off, live):
        assert a["task_id"] == b["task_id"] and a["vsc"] == b["vsc"]
        assert json.dumps(a["gateway"]["writes"], sort_keys=True, default=str) == \
            json.dumps(b["gateway"]["writes"], sort_keys=True, default=str)


def test_classify_faults_deterministic_and_spec_only():
    rows = [json.loads(line) for p in ("test_iid_c2", "v2_stress_c2") for line in (C2 / f"{p}.jsonl").open()]
    a = [CF.label(d) for d in rows]
    b = [CF.label(json.loads(json.dumps(d))) for d in reversed(rows)][::-1]
    assert a == b
    on_disk = [json.loads(line) for line in (C2 / "fault_labels.jsonl").open()]
    assert on_disk == a
    for lab in a:
        for axis, values in CF.AXES.items():
            assert lab[axis] in values
    # the label depends only on the spec: adding outcome-like fields changes nothing
    d = dict(rows[0], vsc=1, verdict={"G": 1}, trace=[])
    assert CF.label(d) == a[0]
    # precommit timeouts are no-commit; IID timeouts commit
    st = next(l for l in a if l["split"] == "v2_stress_c2")
    assert all(s == "none" for _, s in st["faulted_ops"])


def test_freeze_a11_register_and_check(tmp_path, monkeypatch):
    import shutil
    import sys
    FZ = _module("freeze_a11")
    for f in FZ.FILES:
        (tmp_path / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / f, tmp_path / f)
    (tmp_path / "results").mkdir()
    pre = tmp_path / "PREREGISTRATION_A11.md"  # the real file is registered now; test the DRAFT guard on a copy
    pre.write_text("DRAFT (test copy)\n" + pre.read_text())
    monkeypatch.setattr(FZ, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["freeze_a11.py"])
    with pytest.raises(SystemExit, match="DRAFT"):
        FZ.main()  # a DRAFT pre-registration cannot be registered
    p = tmp_path / "PREREGISTRATION_A11.md"
    p.write_text("\n".join(p.read_text().splitlines()[1:]))
    FZ.main()
    monkeypatch.setattr(sys, "argv", ["freeze_a11.py", "--check"])
    with pytest.raises(SystemExit) as e:
        FZ.main()
    assert e.value.code == 0
    (tmp_path / "data/bench/c2/fault_labels.jsonl").write_text("{}\n")
    with pytest.raises(SystemExit) as e:
        FZ.main()
    assert e.value.code == 1
