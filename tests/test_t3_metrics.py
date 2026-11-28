"""scripts/t3_metrics.py on a synthetic row set: success, tool-call accuracy, calls, turns and wall time per episode."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("t3_metrics", ROOT / "scripts/t3_metrics.py")
T3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T3)


def call(tool: str, error: str | None = None) -> dict:
    return {"tool": tool, "args": {}, "obs": {"status": "error", "error": error} if error else {"status": "ok"},
            "by": "agent"}


def rows() -> list[dict]:
    # greedy: one scripted prefix call, then 3 policy calls (1 unknown tool) plus 1 invalid-JSON call -> 4 calls, 2 bad
    g = {"sample": -1, "vsc": 1, "turns": 4, "invalid": 2, "ended_by": "report", "policy_calls": 3,
         "trace": [call("list_payments"), call("pay_supplier"), call("pay_suplier", "unknown_tool"), call("report")]}
    # sampled: 2 valid calls, then a turn with no call
    s = {"sample": 0, "vsc": 0, "turns": 3, "invalid": 0, "ended_by": "no_call", "policy_calls": 2,
         "trace": [call("list_payments"), call("get_purchase_order"), call("pay_supplier")]}
    return [g, s]


def test_episode_counts_prefix_and_json_failures():
    e = T3.episode(rows()[0], "vsc")
    assert e == {"greedy": True, "success": True, "calls": 4, "invalid": 2, "turns": 4, "no_call": False}


def test_episode_without_policy_calls_uses_start_prefix_and_mode():
    r = {"mode": "t1_s0", "vsc": 1, "turns": 2, "invalid": 0, "ended_by": "report",
         "task": {"meta": {"start_prefix": [["list_payments", {}]]}},
         "trace": [call("list_payments"), call("pay_supplier"), call("report")]}
    e = T3.episode(r, "vsc")
    assert (e["greedy"], e["calls"]) == (False, 2)


def test_episode_txn_metric_and_scripted_policy():
    r = {"sample": -1, "vsc": 1, "txn": {"vtc": False}, "trace": [call("pay_supplier"), call("x", "unknown_tool")]}
    e = T3.episode(r, "vtc")
    assert (e["success"], e["calls"], e["invalid"], e["turns"]) == (False, 2, 1, None)


def test_invalid_below_trace_errors_is_rejected():
    r = rows()[0] | {"invalid": 0}
    with pytest.raises(ValueError):
        T3.episode(r, "vsc")


def test_aggregate():
    a = T3.aggregate([T3.episode(r, "vsc") for r in rows()])
    assert a["episodes"] == 2 and a["success"] == 0.5
    assert a["policy_calls"] == 6 and a["tool_call_accuracy"] == pytest.approx(1 - 2 / 6)
    assert a["calls_per_episode"] == 3 and a["turns_per_episode"] == 3.5 and a["no_call_end_rate"] == 0.5
    assert T3.aggregate([]) is None


def test_cell_wall_time_and_gpu(tmp_path):
    d = tmp_path / "results/p1/test"
    d.mkdir(parents=True)
    for tag, secs in (("X_s0", 4.0), ("X_s1", 8.0)):
        (d / f"{tag}__p1_test_full.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows()))
        (d / f"{tag}__p1_test_full.summary.json").write_text(json.dumps({"seconds": secs}))
    g = {"study": "P1 test", "subset": "p1_test_full", "arm": "X (test)", "show": True, "metric": "vsc",
         "runs": [(f"results/p1/test/X_s{k}__p1_test_full.jsonl", "t") for k in (0, 1)]}
    prov = {"results/p1/test/X_s0__p1_test_full.summary.json": {"node": "gpunode7", "gpu": "H200", "consistent": True},
            "results/p1/test/X_s1__p1_test_full.summary.json": {"node": "gpunode4", "gpu": "A100 80GB",
                                                                "consistent": False}}
    c = T3.cell(g, tmp_path, prov)
    assert c["n_models"] == 2 and c["greedy"]["episodes"] == 2 and c["sampled"]["success"] == 0
    w = c["wall_s_per_episode"]
    assert (w["min"], w["max"], w["pooled"]) == (2.0, 4.0, 3.0)
    assert w["gpus"] == ["H200", "unattributed"]  # a summary older than its task's claim gets no GPU
    assert T3.row_values(c)[5:] == ["2.00--4.00", "H200/?"]
    frag = T3.claims_fragment([c], {})
    assert '"P1|p1_test_full|X", "greedy", "success"' in frag and 'shown: ["2.00"]' in frag


def test_parse_owner_slurm_and_hf():
    node, when = T3.parse_owner("slurm:460633:gpunode7.sharanga.local:2026-11-24T13:30:01+05:30")
    assert node == "gpunode7" and when.utcoffset().total_seconds() == 19800
    node, when = T3.parse_owner("hf:  url: https://huggingface.co/jobs/u/abc:l40sx1:2026-11-20T12:56:25+05:30")
    assert node == "l40sx1" and T3.GPU[node] == "L40S" and when.hour == 12
    with pytest.raises(ValueError):
        T3.parse_owner("local:me")
