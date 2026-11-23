"""The website's ROI calculator (demo/roi.js) reproduces scripts/roi_model.py for every arm and scenario, and the
upgrade sections are built only from result files."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_js_calculator_matches_python_model():
    d = json.loads((ROOT / "results/roi/roi_model.json").read_text())
    js = (ROOT / "demo/roi.js").read_text() + "\nconst d=" + json.dumps(d) + """;
const out={};for(const [n,a] of Object.entries(d.arms)){out[n]={};for(const [sc,s] of Object.entries(d.assumptions))
out[n][sc]=apRoiProject(a,s,d.episodes_per_gpu_hour).cost_per_1000_ops.total_usd;}console.log(JSON.stringify(out));"""
    got = json.loads(subprocess.run(["node", "-e", js], capture_output=True, text=True, check=True).stdout)
    for n, by in d["projections"].items():
        for sc, pr in by.items():
            assert got[n][sc] == pytest.approx(pr["cost_per_1000_ops"]["total_usd"], rel=1e-9), (n, sc)


def test_upgrade_sections_read_only_result_files():
    import site_upgrade
    b = site_upgrade.build()
    assert set(b) == {"r1", "r1t", "j1", "roi", "p1"}
    for k in ("r1", "r1t", "j1", "p1"):
        assert b[k]["source"].startswith("results/")
    assert b["roi"]["source"] == "results/roi/roi_model.json" and len(b["roi"]["arms"]) == 7


def _fake_record(task_id: str, sample: int, vsc: int) -> dict:
    trace = [{"idx": 0, "tool": "pay_supplier", "args": {"amount_cents": 1000, "idempotency_key": "k1", "po_id": "PO-1"},
              "obs": {"status": "ok", "payment_id": "PAY-1"}, "by": "agent"},
             {"idx": 1, "tool": "list_payments", "args": {"po_id": "PO-1"},
              "obs": {"status": "ok", "payments": [{"payment_id": "PAY-1", "amount_cents": 400, "kind": "payment", "refund_of": None}]},
              "by": "agent"},
             {"idx": 2, "tool": "submit_report", "args": {"status": "completed"}, "obs": {"status": "ok"}, "by": "agent"}]
    return {"task_id": task_id, "sample": sample, "vsc": vsc, "workflow": "pay_post", "trace": trace,
            "attempts": [{"idx": 0, "tool": "pay_supplier", "committed": "partial", "from_idempotency_store": False}],
            "report": {"status": "completed"}, "violation": None, "ended_by": "report", "honest_fail": False,
            "score": {"false_success": not vsc, "steps": {"pay": "VERIFIED_SUCCESS" if vsc else "NOT_DONE"}},
            "repay_after_refund": 0, "repay_new_key": 0, "partial_pay_task": True, "redundant_reads": 0,
            "policy_reads": 1, "policy_calls": 3}


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_p1_section_builds_and_renders(tmp_path, monkeypatch):
    """The P1 section (scripts/site_upgrade.py::p1, demo/site_upgrade.js block 09) on synthetic result files."""
    import site_upgrade
    splits = site_upgrade.P1_TEST
    (tmp_path / "data/bench/proc").mkdir(parents=True)
    (tmp_path / "results/p1/test").mkdir(parents=True)
    for sp in splits:
        t = json.loads((ROOT / f"data/bench/proc/{sp}.jsonl").read_text().splitlines()[0])
        (tmp_path / f"data/bench/proc/{sp}.jsonl").write_text(json.dumps(t) + "\n")
        for arm in "ABCD":
            for k in (0, 1, 2):
                vsc = int(arm in "BD")
                rows = [_fake_record(t["task_id"], j, vsc) for j in (-1, 0, 1, 2, 3)]
                (tmp_path / f"results/p1/test/{arm}_s{k}__{sp}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    test = {x: {"delta": 0.1, "ci95": [0.01, 0.2], "holm_p": 0.01, "supported": True, "seed_deltas": [0.1, 0.1, 0.1]}
            for x in "BCD"}
    desc = {a: {"test": {sp: {"sampled": {"vsc": 0.5}} for sp in splits}} for a in "ABCD"}
    (tmp_path / "results/p1/p1_summary.json").write_text(json.dumps(
        {"dev": {"mean_sampled_vsc": {a: 0.5 for a in "ABCD"}, "winner": "B"}, "test": test, "descriptive": desc}))
    monkeypatch.setattr(site_upgrade, "ROOT", tmp_path)
    p = site_upgrade.p1()
    assert p["status"] == "done" and len(p["tasks"]) == len(site_upgrade.P1_SHOW)
    ep = next(iter(p["tasks"].values()))["arms"]["A"]
    assert ep["steps"][0][3] == "partial" and ep["steps"][1][2].startswith("1 live payment(s), $4.00")
    assert p["union"]["B"]["vsc"]["sampled"] == 1.0 and p["union"]["A"]["vsc"]["sampled"] == 0.0
    # render block 09 in node with a minimal DOM stub
    js = (ROOT / "demo/site_upgrade.js").read_text().split("/* ---------- 09:")[1]
    stub = """
const els={};const mk=id=>els[id]||(els[id]={id,innerHTML:"",textContent:"",value:"",dataset:{},
  addEventListener(){},setAttribute(){}});
const $=(q,root)=>mk(q);const $$=()=>[];const esc=s=>String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const S={upgrade:{p1:""" + json.dumps(p) + """}};
/* ---------- 09:""" + js + """
console.log(JSON.stringify({t:els["#u-p1"].innerHTML,b:els["#u-p1b"].innerHTML,g:els["#p1grid"].innerHTML}));"""
    out = json.loads(subprocess.run(["node", "-e", stub], capture_output=True, text=True, check=True).stdout)
    assert "supported" in out["t"] and "+10.0" in out["t"]
    assert "Short payments repaired" in out["b"]
    assert "database: partial" in out["g"] and "tool said: ok, PAY-1" in out["g"]
