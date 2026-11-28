from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.p6_stats import bootstrap_contrast, read_model, test_ni as _test_ni


def test_bootstrap_contrast_detects_consistent_positive_effect():
    tasks = [f"t{i}" for i in range(12)]
    candidate = [{t: {"vsc": 0.6} for t in tasks} for _ in range(3)]
    baseline = [{t: {"vsc": 0.4} for t in tasks} for _ in range(3)]
    result = bootstrap_contrast(candidate, baseline, "vsc", np.random.default_rng(7))
    assert np.isclose(result["estimate"], 0.2)
    assert result["ci95"][0] > 0
    assert result["p_one_sided"] < 0.05
    assert all(x > 0 for x in result["seed_deltas"])


def test_noninferiority_requires_ci_and_each_seed_to_clear_margin():
    assert _test_ni({"ci95": [-0.02, 0.03], "seed_deltas": [-0.01, 0.00, 0.01]})
    assert not _test_ni({"ci95": [-0.04, 0.03], "seed_deltas": [-0.01, 0.00, 0.01]})
    assert not _test_ni({"ci95": [-0.02, 0.03], "seed_deltas": [-0.04, 0.00, 0.01]})


def test_read_model_requires_four_samples_and_keeps_greedy(tmp_path, monkeypatch):
    import json
    import scripts.p6_stats as stats

    monkeypatch.setattr(stats, "ROOT", tmp_path)
    out = tmp_path / "results/p6"
    out.mkdir(parents=True)
    rows = []
    for sample in (-1, 0, 1, 2, 3):
        rows.append({
            "task_id": "task-1", "sample": sample, "vsc": 1 if sample != 0 else 0,
            "score": {"false_success": False, "duplicate_irreversible": False},
            "violation": 0, "policy_calls": 3, "policy_reads": 1, "invalid": 0, "turns": 4,
        })
    (out / "test__split.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    result = read_model("test", "split", "p6")
    assert result["task-1"]["greedy_vsc"] == 1.0
    assert result["task-1"]["vsc"] == 0.75
    assert result["task-1"]["reads"] == 1.0
