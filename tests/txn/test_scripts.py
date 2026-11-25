"""scripts/txn_eval.py (patched p1_eval loop, API client budget) and scripts/txn_stats.py on synthetic inputs."""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

from actionproof.bench.generator_proc import DATA, load_tasks
from actionproof.txn.policies import run_policy

ROOT = Path(__file__).resolve().parents[2]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------- stats
def _cp_reference(k, n):
    from scipy.stats import beta
    return (0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)), (1.0 if k == n else beta.ppf(0.975, k + 1, n - k))


def test_clopper_pearson_and_holm():
    S = _load("txn_stats")
    assert S.cp(0, 10)[0] == 0.0 and S.cp(10, 10)[1] == 1.0
    lo, hi = S.cp(3, 20)
    assert (round(lo, 4), round(hi, 4)) == (0.0321, 0.3789)
    assert S.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    d = tmp_path_factory.mktemp("txnres")
    tasks = load_tasks(DATA / "txn_dev.jsonl")
    arms = {"D_s0": "B4", "D_s1": "A3", "D_s2": "B4", "B3": "B3", "B1": "B1", "B0": "B0", "B4": "B4"}
    for tag, pol in arms.items():
        rows = [run_policy(pol, t) for t in tasks]
        (d / f"{tag}__txn_dev.jsonl").write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))
    return d


def test_stats_end_to_end_and_h3_rule(synthetic):
    S = _load("txn_stats")
    res = S.main(["--split", "txn_dev", "--res-dir", str(synthetic), "--pairs", "D:B4"])
    h = res["H3"]
    assert len(h["tests"]) == 4 and {(t["comparator"], t["metric"]) for t in h["tests"]} == {
        (c, m) for c in ("B3", "B1") for m in ("dmr", "urr")}
    for t in h["tests"]:
        assert t["p_holm"] >= t["p_boot"]
        assert t["supported"] == (t["p_holm"] < 0.05 and t["ci95"][1] < 0 and t["vtc_noninferior"])
        assert t["n_clusters"] == len({json.loads(line)["task"]["meta"]["txn"]["template_id"]
                                       for line in (synthetic / "B4__txn_dev.jsonl").read_text().splitlines()})
    dmr_b3 = next(t for t in h["tests"] if t["comparator"] == "B3" and t["metric"] == "dmr")
    assert dmr_b3["diff"] < 0 and dmr_b3["supported"]  # D (B4/A3/B4) duplicates far less than B3 on dev
    assert isinstance(h["supported"], bool)
    assert res["H1"]["by_class"]["phantom"]["rate"] == 1.0
    assert "B4" in res["H5"]["per_model"] and res["H5"]["per_model"]["B4"]["CER"]["rate"] == 1.0
    assert (synthetic / "stats__txn_dev.md").exists()
    again = S.main(["--split", "txn_dev", "--res-dir", str(synthetic), "--out", str(synthetic / "again.json")])
    assert [t["ci95"] for t in again["H3"]["tests"]] == [t["ci95"] for t in h["tests"]]  # seeded: reproducible


def test_h3_pending_without_d_seeds(tmp_path, synthetic):
    S = _load("txn_stats")
    for tag in ("B3", "B1"):
        (tmp_path / f"{tag}__txn_dev.jsonl").write_text((synthetic / f"{tag}__txn_dev.jsonl").read_text())
    res = S.main(["--split", "txn_dev", "--res-dir", str(tmp_path)])
    assert res["H3"]["status"].startswith("PENDING")


# ---------------------------------------------------------------------- txn_eval plumbing
class FakeOut:
    def __init__(self, text):
        self.outputs = [types.SimpleNamespace(text=text)]


class ScriptedLLM:
    """Plays a fixed call sequence per conversation: pay, read the audit log, read payments, report."""

    llm_engine = types.SimpleNamespace(model_config=types.SimpleNamespace(max_model_len=10**6))

    def get_tokenizer(self):
        E = _load("txn_eval")
        return E._Tok()

    def chat(self, convs, params, use_tqdm=False):
        outs = []
        for c in convs:
            n = sum(m["role"] == "assistant" for m in c)
            po = c[1]["content"].split()[2]
            seq = [{"tool_name": "pay_supplier", "parameters": {"po_id": po, "amount_cents": 1, "idempotency_key": "k"}},
                   {"tool_name": "get_audit_log", "parameters": {"po_id": po}},
                   {"tool_name": "list_payments", "parameters": {"po_id": po, "consistent": True}},
                   {"tool_name": "submit_report", "parameters": {"status": "failed", "summary": "x"}}]
            outs.append(FakeOut(json.dumps([seq[min(n, 3)]])))
        return outs


@pytest.fixture()
def lite(monkeypatch):
    """p1_eval builds tool schemas with transformers (cluster only); a minimal schema builder stands in here."""
    from actionproof.rollout import episode, lite_format as L

    def schema(fn):
        import inspect
        return {"name": fn.__name__, "parameters": list(inspect.signature(fn).parameters), "doc": fn.__doc__}
    monkeypatch.setattr(episode, "_agenttune", lambda: (L.FALLBACK_PROMPT, schema, L.extract_tool_calls_from_text))
    E = _load("txn_eval")
    E._ensure_sampling_params()
    return E


def test_txn_eval_runs_p1_loop_on_txn_env_and_restores_factory(lite):
    import actionproof.stage3.env as S3
    orig = S3.make_env_s3
    tasks = load_tasks(DATA / "txn_dev.jsonl")[:3]
    rows = lite.run_txn(ScriptedLLM(), tasks, samples=1)
    assert S3.make_env_s3 is orig
    assert len(rows) == 6 and all(r["ended_by"] == "report" for r in rows)
    for r in rows:
        tools = [s["tool"] for s in r["trace"]]
        assert "get_audit_log" in tools and r["trace"][1]["obs"]["status"] == "ok"  # TxnEnv, not S3ProcEnv
        assert {"txn", "invariants", "decision_record"} <= set(r) and "str" in r["txn"]
        assert "get_audit_log" in r["conversation"][0]["content"] and "consistent" in r["conversation"][0]["content"]
    s = lite.txn_summary(rows)
    assert s["all"]["n"] == 6 and s["greedy"]["n"] == 3 and s["sampled"]["n"] == 3


def test_txn_eval_uses_surface_env_for_alias_tasks(lite):
    t = next(t for t in load_tasks(DATA / "txn_test.jsonl") if t.meta["txn"]["surface"] == "alias")
    rows = lite.run_txn(ScriptedLLM(), [t], samples=0)
    sys_prompt = rows[0]["conversation"][0]["content"]
    assert "execute_vendor_payment" in sys_prompt and "pay_supplier" not in sys_prompt
    assert rows[0]["trace"][0]["obs"]["error"] == "unknown_tool"  # canonical names are not part of that API


def test_api_client_hard_budget_stop(lite, monkeypatch):
    llm = lite.ApiLLM("http://x", "m", None, price_in=1.0, price_out=2.0, max_usd=0.002, ctx=10**6, concurrency=1)
    calls = []

    def post(body):
        calls.append(body)
        assert body["temperature"] == 0
        return {"choices": [{"message": {"content": "[]"}}], "usage": {"prompt_tokens": 400, "completion_tokens": 10}}
    monkeypatch.setattr(llm, "_post", post)
    P = sys.modules["vllm"].SamplingParams
    conv = [{"role": "user", "content": "x" * 300}]
    outs = [llm.chat([conv], [P(temperature=0, max_tokens=256)])[0].outputs[0].text for _ in range(6)]
    rep = llm.report()
    assert rep["usd_spent"] <= 0.002 and rep["budget_stopped"] and outs[-1] == "" and len(calls) < 6
    with pytest.raises(SystemExit):
        llm.chat([conv], [P(temperature=1.0, max_tokens=8)])
