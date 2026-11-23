"""J1 analysis: known-answer checks for the agreement statistics, the committee rule and the reply parser."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

D = Path(__file__).resolve().parents[2] / "evaluation/llm_judge_committee"


def mod(name):
    spec = importlib.util.spec_from_file_location(name, D / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


an, cm = mod("analyze"), mod("committee")


def test_kappa_known():
    # 2x2 table: a=20 yes/yes, b=5 yes/no, c=10 no/yes, d=15 no/no -> po=0.7, pe=0.5 -> kappa=0.4
    a = ["y"] * 25 + ["n"] * 25
    b = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
    assert an.kappa(a, b) == pytest.approx(0.4)


def test_fleiss_perfect_and_wikipedia_example():
    assert an.fleiss([["a", "a", "a"], ["b", "b", "b"]]) == pytest.approx(1.0)
    # Fleiss (1971) style check: 3 raters, mixed
    rows = [["a", "a", "b"], ["a", "b", "b"], ["a", "a", "a"], ["b", "b", "b"]]
    # P_i = 1/3,1/3,1,1 -> Pbar=2/3; p_a=p_b=0.5 -> Pe=0.5 -> kappa=1/3
    assert an.fleiss(rows) == pytest.approx(1 / 3)


def test_binary_abstention_and_fpr():
    r = an.binary([True, None, False, True], [True, True, False, False])
    assert r["coverage"] == 0.75 and r["selective_accuracy"] == pytest.approx(2 / 3)
    assert r["fpr"] == 0.5 and r["recall"] == 0.5 and r["accuracy_abstain_wrong"] == 0.5


def test_committee_rule():
    assert an.committee([True, True, True, False]) is True
    assert an.committee([True, True, False, False]) is None
    assert an.committee([None, None, None, True]) is None


def test_parser_takes_last_valid_object_and_rejects_bad_values():
    good = {k: opts[0] for k, _, opts in cm.QUESTIONS} | {"confidence": 0.8, "rationale": "x"}
    text = "thinking {\"a\": 1} ... " + json.dumps(good)
    assert cm.parse(text)["operation_occurred"] == cm.QUESTIONS[0][2][0]
    bad = dict(good, operation_occurred="maybe")
    assert cm.parse(json.dumps(bad)) is None


def test_option_order_is_seeded_and_varies():
    p1 = cm.build_prompt("X", "c1", "gemma", "v1")
    assert p1 == cm.build_prompt("X", "c1", "gemma", "v1")
    assert len({cm.build_prompt("X", f"c{i}", "gemma", "v1") for i in range(20)}) > 1
