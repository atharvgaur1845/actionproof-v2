"""No-model tests for the live Space logic. The repo's own modules (worktree root) are the reference."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SPACE = Path(__file__).resolve().parents[1]
REPO = SPACE.parent
for p in (str(SPACE), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

import core  # noqa: E402
from actionproof_min import lite_format as LITE  # noqa: E402
from actionproof_min.agents import naive as v_naive, oracle as v_oracle  # noqa: E402
from actionproof_min.env_s3 import make_env_s3 as v_make  # noqa: E402
from actionproof_min.runtime import episode_record_proc as v_rec  # noqa: E402
from actionproof_min.tasks import ProcTask as VTask  # noqa: E402
from actionproof_min.truth import score as v_score  # noqa: E402

REALITY_KEYS = list(core.REALITIES)
TASK_IDS = [t.task_id for t in core.load_curated()]


def _repo_task(task: VTask, reality: str):
    """The same reality built independently, through the repo's own task class and env factory."""
    from actionproof.bench.generator_proc import ProcTask
    spec = core.REALITIES[reality]
    d = json.loads(json.dumps(task.to_json()))
    d["meta"]["faults"] = [] if spec["kind"] is None else [
        {"tool": "pay_supplier", "k": 0, "kind": spec["kind"], **({"ratio": 0.5} if spec["kind"] == "partial" else {})}]
    d["meta"]["never_commit"] = None
    return ProcTask.from_json(d)


def test_six_tasks_all_with_pay():
    assert len(TASK_IDS) == 6
    for t in core.load_curated():
        assert "pay" in t.meta["requested"]


@pytest.mark.parametrize("policy", ["naive", "oracle"])
@pytest.mark.parametrize("reality", REALITY_KEYS)
@pytest.mark.parametrize("task_id", TASK_IDS)
def test_vendored_matches_repo_score(task_id, reality, policy):
    from actionproof.integrity.sim import agents as R_agents
    from actionproof.integrity.sim.runtime import episode_record_proc as r_rec
    from actionproof.stage3.env import make_env_s3 as r_make
    task = core.get_task(task_id)
    vt = core.apply_reality(task, reality)
    rt = _repo_task(task, reality)
    venv, renv = v_make(vt, "x"), r_make(rt, "x")
    getattr(R_agents, policy)(renv, rt)
    (v_naive if policy == "naive" else v_oracle)(venv, vt)
    assert venv.trace == renv.trace
    v, r = v_rec(vt, venv), r_rec(rt, renv)
    assert v["score"] == r["score"]
    assert v["final_state"] == r["final_state"]
    from actionproof.integrity.sim.truth import score as r_score
    assert v_score(r) == r_score(r)  # the vendored verifier on the repo's own record


def test_scripted_outcomes_make_the_point():
    """Ack-trusting policy reports success on phantom and partial; the oracle does not."""
    def vsc(policy, reality):
        ev = list(core.scripted_events(policy, core.get_task(TASK_IDS[0]), reality))[-1]["rec"]
        return ev["score"]
    assert vsc("scripted_naive", "Reality C")["false_success"] and vsc("scripted_naive", "Reality D")["false_success"]
    for r in REALITY_KEYS:
        assert vsc("scripted_oracle", r)["vsc"] == 1


SAMPLES = [
    '[{"tool_name": "pay_supplier", "parameters": {"po_id": "PO-1", "amount_cents": 5}}]',
    '```json\n[{"tool_name": "list_payments", "parameters": {"po_id": "PO-1"}}]\n```',
    'Sure, calling it now: [{"tool_name": "submit_report", "parameters": {"status": "completed", "summary": "ok"}}]',
    '[{"tool_name": "a", "parameters": {}}, {"tool_name": "b", "parameters": {"x": 1}}]',
    '{"name": "get_purchase_order", "arguments": {"po_id": "PO-2"}}',
    '[{"tool_name": "directly_answer", "parameters": {}}]',
    "I will now pay the supplier.", "", "[{broken json", '[{"tool_name": "", "parameters": {}}]',
    '[{"tool_name": "pay_supplier", "parameters": "{\\"po_id\\": \\"PO-3\\"}"}]',
]


def _norm(calls):
    return [(c["function"]["name"], json.loads(c["function"]["arguments"] or "{}")) for c in (calls or [])]


def test_parser_matches_repo_parser():
    from actionproof.rollout.episode import _agenttune
    _, _, repo_parse = _agenttune()  # AgentTune where installed, else the repo's lite copy
    for s in SAMPLES:
        assert _norm(LITE.extract_tool_calls_from_text(s)) == _norm(repo_parse(s)), s


def test_fallback_prompt_matches_repo():
    from actionproof.rollout.episode import _agenttune
    assert LITE.FALLBACK_PROMPT == _agenttune()[0]


def test_system_prompt_matches_repo():
    pytest.importorskip("transformers")
    from actionproof.integrity.sim.prompts import PROC_NEUTRAL
    from actionproof.rollout.episode import system_prompt as r_sys
    from actionproof.stage3.env import make_env_s3 as r_make
    from actionproof_min.episode import system_prompt as v_sys
    from actionproof_min.prompts import PROC_NEUTRAL as V_NEUTRAL
    t = core.apply_reality(core.get_task(TASK_IDS[0]), "Reality B")
    assert v_sys(v_make(t, "a"), V_NEUTRAL) == r_sys(r_make(_repo_task(t, "Reality B"), "a"), PROC_NEUTRAL)


def test_tool_result_format_matches_repo():
    from actionproof.bench.prompts import TOOL_RESULT_FORMAT as R
    from actionproof_min.prompts import TOOL_RESULT_FORMAT as V
    assert R == V


# ---------------------------------------------------------------- fake generator and the stream
class FakeGen:
    """Emits a fixed ack-trusting script: pay, post, report completed."""

    def __init__(self, po: dict):
        a = po["amount_cents"]
        self.script = [[{"tool_name": "pay_supplier", "parameters": {"po_id": po["po_id"], "amount_cents": a, "idempotency_key": "k"}}],
                       [{"tool_name": "post_journal_entry", "parameters": {"po_id": po["po_id"], "amount_cents": a, "idempotency_key": "j"}}],
                       [{"tool_name": "submit_report", "parameters": {"status": "completed", "summary": "paid and posted"}}]]
        self.i = 0

    def __call__(self, conv):
        out = json.dumps(self.script[min(self.i, 2)])
        self.i += 1
        return out


def _po(task):
    return next(p for p in task.initial_state["pos"] if p["po_id"] == task.meta["po_id"])


def _fake_stream_factory():
    def model_stream(model_key, task_id, reality):
        task = core.get_task(task_id)
        # tests have no transformers requirement: stub the schema-dependent system prompt
        yield from core.model_events(FakeGen(_po(task)), task, reality, system_fn=lambda env, base: base)
    return model_stream


HIDDEN_WORDS = ["phantom", "lost_ack", "partial", "precommit", "committed", "never_commit", "fault", "Reality was"]


@pytest.mark.parametrize("reality", REALITY_KEYS)
def test_hidden_state_not_streamed_before_end(reality):
    task_id = "p1_test_full-0003"  # pay_post
    state: dict = {}
    frames = list(core.run_session("trained", task_id, reality, state, _fake_stream_factory()))
    assert len(frames) > 3
    final = frames[-1]
    pre = frames[:-1]
    for f in pre:
        assert f["reveal"] == ""
        text = (f["banner_l"] + f["left"] + f["banner_r"] + f["right"]).lower()
        for w in HIDDEN_WORDS:
            assert w.lower() not in text, (w, reality)
        assert "payments (po)" not in text and "verifier verdict" not in text
    assert "Transaction decision record" in final["reveal"]
    assert core.REALITIES[reality]["name"] in final["reveal"]
    assert "LIVE MODEL RUN · " + core.TRAINED_REPO in final["banner_l"]
    assert "greedy, max 16 turns" in final["banner_l"]
    # the ack-trusting fake gets VSC only in the committed realities
    assert ("VSC <b>1</b>" in final["reveal"]) == (reality in ("Reality A", "Reality B"))


def test_scripted_label_and_no_banner_confusion():
    frames = list(core.run_session("scripted_oracle", "p1_test_full-0003", "Reality C", {}, None))
    assert "SCRIPTED" in frames[0]["banner_l"] and "LIVE MODEL RUN" not in frames[0]["banner_l"]


def test_compare_mode_two_columns_and_two_verdicts():
    frames = list(core.run_session("both", "p1_test_full-0003", "Reality C", {}, _fake_stream_factory()))
    last = frames[-1]
    assert core.BASE_REPO in last["banner_l"] and core.TRAINED_REPO in last["banner_r"]
    assert last["reveal"].count("Transaction decision record") == 2


def test_session_cap_and_free_scripted_runs():
    state: dict = {}
    ms = _fake_stream_factory()
    for _ in range(10):
        assert list(core.run_session("trained", "p1_test_full-0003", "Reality B", state, ms))[-1]["reveal"]
    over = list(core.run_session("trained", "p1_test_full-0003", "Reality B", state, ms))
    assert len(over) == 1 and "Session cap" in over[0]["status"]
    assert list(core.run_session("scripted_naive", "p1_test_full-0003", "Reality B", state, None))[-1]["reveal"]
    assert core.Budget().charge({"runs": 9}, 2) is False  # compare mode needs two runs


def test_ack_highlighted():
    assert '<mark class="ack">"status": "ok"</mark>' in core.render_reply({"status": "ok", "payment_id": "PAY-1"})
    assert "mark" not in core.render_reply({"status": "error", "error": "timeout"})


def test_curated_tasks_exist_in_main_p1_test_full():
    p = REPO / "data/bench/proc/p1_test_full.jsonl"
    if not p.exists():
        pytest.skip("p1_test_full.jsonl not present in this checkout")
    full = {json.loads(l)["task_id"]: json.loads(l) for l in p.read_text().splitlines() if l.strip()}
    for t in core.load_curated():
        assert full[t.task_id]["initial_state"] == t.initial_state and full[t.task_id]["prompt"] == t.prompt


# ------------------------------------------------------------------ judge mode, recorded fallback, export
@pytest.fixture()
def recorded(tmp_path, monkeypatch):
    """A recording made with the fake (ack-trusting) generator, in the format of space_live/record_runs.py."""
    from actionproof_min.lite_format import build_schema  # noqa: F401  (import check only)
    ms = _fake_stream_factory()
    runs = {}
    for key in ("base", "trained"):
        for tid in TASK_IDS:
            for r in REALITY_KEYS:
                runs[f"{key}|{tid}|{r}"] = list(ms(key, tid, r))
    f = tmp_path / "recorded.json"
    f.write_text(json.dumps({"meta": {"recorded_utc": "2026-11-27 06:00Z", "hardware": "test"}, "runs": runs}, default=str))
    monkeypatch.setattr(core, "RECORDED", f)
    return f


def test_judge_mode_script_left_trained_right_and_export():
    frames = list(core.run_session("judge", "p1_test_full-0003", "Reality C", {}, _fake_stream_factory()))
    last = frames[-1]
    assert "SCRIPTED: acknowledgement-trusting" in last["banner_l"] and core.TRAINED_REPO in last["banner_r"]
    assert last["reveal"].count("Transaction decision record") == 2
    ex = last["export"]
    assert [s["kind"] for s in ex["sides"]] == ["scripted", "live"]
    assert ex["sides"][1]["model"] == core.TRAINED_REPO and ex["sides"][1]["revision"] == core.TRAINED_REVISION
    assert ex["reality"] == "Reality C" and ex["run_id"] in last["status"]
    json.dumps(ex)  # serialisable
    assert ex["sides"][0]["score"]["false_success"]  # the ack-trusting script reports a phantom payment


def test_live_failure_falls_back_to_labelled_recording(recorded):
    def broken(model_key, task_id, reality):
        yield {"type": "start", "system": None, "prompt": "x"}
        raise RuntimeError("ZeroGPU quota exceeded")
    frames = list(core.run_session("trained", "p1_test_full-0003", "Reality C", {}, broken))
    last = frames[-1]
    assert "LIVE RUN FAILED (RuntimeError: ZeroGPU quota exceeded)" in last["banner_l"]
    assert "RECORDED MODEL RUN" in last["banner_l"] and "LIVE MODEL RUN" not in last["banner_l"]
    assert last["export"]["sides"][0]["kind"] == "recorded"
    assert "Transaction decision record" in last["reveal"]


def test_live_failure_without_recording_is_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "RECORDED", tmp_path / "none.json")

    def broken(model_key, task_id, reality):
        raise RuntimeError("no GPU")
        yield  # pragma: no cover
    with pytest.raises(RuntimeError):
        list(core.run_session("trained", "p1_test_full-0003", "Reality C", {}, broken))


@pytest.mark.parametrize("reality", REALITY_KEYS)
def test_recorded_option_is_labelled_and_hides_state_until_end(recorded, reality):
    frames = list(core.run_session("recorded_trained", "p1_test_full-0003", reality, {}, None))
    for f in frames[:-1]:
        text = (f["banner_l"] + f["left"]).lower()
        assert f["reveal"] == "" and "recorded model run" in text and "live model run" not in text
        for w in HIDDEN_WORDS:
            assert w.lower() not in f["left"].lower(), (w, reality)
    assert frames[-1]["export"]["sides"][0]["kind"] == "recorded"


def test_recorded_runs_need_no_budget(recorded):
    state = {"runs": core.SESSION_CAP}
    assert list(core.run_session("recorded_base", "p1_test_full-0003", "Reality A", state, None))[-1]["reveal"]
