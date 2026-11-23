"""R1 scorer checks: known-answer cases through the official BFCL and IFEval checkers.

Run in the scoring venv: `.venv-r1/bin/python -m pytest tests/retention -q`. Skipped where bfcl-eval is absent.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("bfcl_eval")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import r1_score  # noqa: E402


@pytest.fixture(scope="module")
def tables():
    return r1_score._bfcl_tables()


def row(cat, id_, text):
    return {"bench": "bfcl", "category": cat, "id": id_, "text": text}


def test_gold_call_is_correct(tables):
    r = r1_score.score_bfcl(row("simple_python", "simple_python_0",
                                "[calculate_triangle_area(base=10, height=5)]"), *tables)
    assert r["correct"] and r["format_ok"]


def test_wrong_argument_is_incorrect_but_well_formed(tables):
    r = r1_score.score_bfcl(row("simple_python", "simple_python_0",
                                "[calculate_triangle_area(base=11, height=5)]"), *tables)
    assert not r["correct"] and r["format_ok"]


def test_prose_is_a_format_failure(tables):
    r = r1_score.score_bfcl(row("simple_python", "simple_python_0", "The area is 25 square units."), *tables)
    assert not r["correct"] and not r["format_ok"]


def test_irrelevance_abstention_is_correct(tables):
    irr_id = next(k for k in tables[0] if k.startswith("irrelevance_"))
    assert r1_score.score_bfcl(row("irrelevance", irr_id, "None of the functions can be used."), *tables)["correct"]
    fn = tables[0][irr_id]["function"][0]["name"]
    assert not r1_score.score_bfcl(row("irrelevance", irr_id, f"[{fn}(x=1)]"), *tables)["correct"]


def test_ifeval_no_comma():
    docs = {str(json.loads(line)["key"]): json.loads(line)
            for line in (ROOT / "data/retention/ifeval_input_data.jsonl").read_text().splitlines()}
    key = next(k for k, d in docs.items() if d["instruction_id_list"] == ["punctuation:no_comma"])
    ok = r1_score.score_ifeval({"id": key, "text": "A plain sentence without that mark."}, docs)
    bad = r1_score.score_ifeval({"id": key, "text": "Yes, it has one."}, docs)
    assert ok["prompt_strict"] and not bad["prompt_strict"]


def test_inputs_frozen_and_complete():
    items = [json.loads(line) for line in (ROOT / "data/retention/r1_inputs.jsonl").read_text().splitlines()]
    assert len(items) == 4032
    assert sum(i["bench"] == "ifeval" for i in items) == 541
    bfcl = [i for i in items if i["bench"] == "bfcl"]
    assert all(i["messages"][0]["role"] == "system" and "composing functions" in i["messages"][0]["content"]
               for i in bfcl)


def test_stats_known_delta(tmp_path, monkeypatch):
    import numpy as np
    import r1_stats
    monkeypatch.setattr(r1_stats, "SCORE", tmp_path)
    monkeypatch.setattr(r1_stats, "B", 500)

    def write(tag, rate):
        rows = [{"category": "ifeval", "id": str(i), "correct": i < rate * 100} for i in range(100)]
        (tmp_path / f"{tag}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    write("ref", 0.5)
    write("a", 0.6)
    write("b", 0.6)
    out = r1_stats.contrast(["a", "b"], "ref", ("ifeval",), np.random.default_rng(0))
    assert abs(out["delta"] - 0.1) < 1e-9 and out["per_seed_delta"] == pytest.approx([0.1, 0.1])
    assert out["p_noninferior"] < 0.01  # Δ = +0.1 is far above the −0.03 margin
    assert r1_stats.contrast(["a"], ["ref", "a"], ("ifeval",), np.random.default_rng(0))["delta"] == pytest.approx(0.05)
    assert r1_stats.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
