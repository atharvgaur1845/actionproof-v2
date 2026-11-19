"""Gateway API (FastAPI TestClient): the endpoints the dashboard calls."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from actionproof.integrity import api as api_mod
from actionproof.integrity.api import PUBLIC_SPLITS, create_app
from actionproof.integrity.gateway import MODES

PHANTOM = [{"tool": "pay_supplier", "k": 0, "kind": "phantom"}]


@pytest.fixture()
def cl() -> TestClient:
    return TestClient(create_app(None, True))


def _task(cl: TestClient) -> dict:
    return next(s for s in cl.get("/v1/scenarios").json() if "pay" in s["requested"])


def test_scenarios_list(cl):
    r = cl.get("/v1/scenarios")
    assert r.status_code == 200
    sc = r.json()
    assert sc and {s["split"] for s in sc} <= set(PUBLIC_SPLITS)
    assert all({"task_id", "workflow", "fault_class", "prompt", "requested"} <= set(s) for s in sc)


@pytest.mark.parametrize("mode", MODES)
def test_run_each_mode(cl, mode):
    t = _task(cl)
    r = cl.post("/v1/test/runs", json={"task_id": t["task_id"], "split": t["split"], "agent": "naive", "mode": mode,
                                       "faults": PHANTOM})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["mode"] == mode and d["status"] == "finalized"
    assert d["agent_view"] and d["agent_view"][-1]["tool"] == "submit_report"
    assert "system_level" in d["truth"] and d["disposition"]["status"] in ("completed", "failed", "escalated")
    assert len(d["contract_hash"]) == 64
    ev = cl.get(f"/v1/transactions/{d['tx_id']}/events").json()
    assert ev[0]["kind"] == "contract_bound"
    if mode != "passthrough":
        assert d["disposition"]["claim"] in ("SUPPORTED", "CONTRADICTED_CLAIM", "UNDERCLAIM", "NO_CLAIM")
        assert any(e["kind"] == "finalized" for e in ev)


def test_reading_a_transaction_does_not_append_to_the_ledger(cl):
    t = _task(cl)
    tx = cl.post("/v1/test/runs", json={"task_id": t["task_id"], "split": t["split"], "mode": "enforce",
                                        "faults": PHANTOM}).json()
    n = cl.get("/v1/ledger/verify").json()["n"]
    again = cl.get(f"/v1/transactions/{tx['tx_id']}").json()
    cl.get("/v1/transactions")
    assert cl.get("/v1/ledger/verify").json()["n"] == n
    assert again["disposition"] == tx["disposition"]


def test_compare_phantom_enforce_not_worse_than_passthrough(cl):
    t = _task(cl)
    r = cl.post("/v1/test/compare", json={"task_id": t["task_id"], "split": t["split"], "agent": "naive",
                                          "faults": PHANTOM})
    assert r.status_code == 200, r.text
    rows = {x["mode"]: x for x in r.json()["rows"]}
    assert len(r.json()["rows"]) == 4 and set(rows) == set(MODES)
    assert rows["enforce"]["verified_completion"] >= rows["passthrough"]["verified_completion"]
    assert rows["passthrough"]["false_finalization"] == 1 and rows["enforce"]["false_finalization"] == 0
    for x in rows.values():
        assert cl.get(f"/v1/transactions/{x['tx_id']}").status_code == 200


def test_ledger_verify_then_tamper(cl):
    t = _task(cl)
    tx = cl.post("/v1/test/runs", json={"task_id": t["task_id"], "split": t["split"], "mode": "enforce",
                                        "faults": PHANTOM}).json()["tx_id"]
    v = cl.get("/v1/ledger/verify").json()
    assert v["ok"] and v["first_bad_seq"] is None
    seq = next(e["seq"] for e in cl.get(f"/v1/transactions/{tx}/events").json() if e["kind"] == "evidence")
    r = cl.post("/v1/test/tamper", json={"seq": seq}).json()
    assert r["ok"] is False and r["first_bad_seq"] == seq
    assert cl.get("/v1/ledger/verify").json()["first_bad_seq"] == seq


def test_evaluation_reports_pending_for_missing_files(cl, monkeypatch):
    monkeypatch.setitem(api_mod.EVAL_FILES, "missing_for_test", "results/does/not/exist.json")
    d = cl.get("/v1/evaluation").json()
    assert d["missing_for_test"] == {"path": "results/does/not/exist.json", "status": "pending"}
    for v in d.values():
        assert v["status"] in ("available", "pending") and ("data" in v) == (v["status"] == "available")


def test_static_files_served(cl):
    r = cl.get("/")
    assert r.status_code == 200 and "<title>ActionProof" in r.text and "/static/app.js" in r.text
    for name, kind in (("app.js", "javascript"), ("styles.css", "css"), ("index.html", "html")):
        r = cl.get(f"/static/{name}")
        assert r.status_code == 200 and kind in r.headers["content-type"]
    assert cl.get("/static/nope.js").status_code == 404
    assert cl.get("/static/..%2Fapi.py").status_code == 404


def test_confirmatory_splits_not_served(cl):
    assert not any(s["split"] == "proc_conf" for s in cl.get("/v1/scenarios").json())
    r = cl.post("/v1/test/runs", json={"task_id": "proc_conf-0000", "split": "proc_conf", "mode": "enforce"})
    assert r.status_code == 404
    r = cl.post("/v1/transactions", json={"task_id": "proc_conf-0000", "split": "proc_conf"})
    assert r.status_code == 404
