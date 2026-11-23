"""Reporting pipeline: labels, freshness, canonical build (actionproof/reporting/)."""
from __future__ import annotations

import os
import time
from pathlib import Path


def test_labels():
    from actionproof.reporting.canonical import label
    assert label({"id": "X", "status": "supported", "prereg": "PREREGISTRATION_S3.md"}) == "confirmatory"
    assert label({"id": "X", "status": "not_supported", "prereg": "PREREGISTRATION_S3.md"}) == "unsupported"
    assert label({"id": "X", "status": "refuted", "prereg": "docs/research/algorithm/C5G_DEV.md"}) == "failed"
    assert label({"id": "X", "status": "supported", "prereg": "docs/research/x.md"}) == "development"
    assert label({"id": "J1-A1", "status": "supported", "prereg": "PREREGISTRATION_J1_ADDENDUM1.md"}) == "exploratory"
    assert label({"id": "X", "status": "supported", "prereg": None}) == "descriptive"
    assert label({"id": "X", "status": "pending", "prereg": "PREREGISTRATION_P1.md"}) == "incomplete"


def test_canonical_rereads_every_number():
    from actionproof.reporting.canonical import build
    b = build()
    assert b["metrics"] and not b["mismatches"], b["mismatches"][:5]
    assert all(m["error"] is None for m in b["metrics"])
    assert all(set(m) >= {"source", "claim", "label", "registration", "seeds", "tasks", "ci_type"} for m in b["metrics"])


def test_freshness_flags_a_newer_source(tmp_path, monkeypatch):
    import actionproof.reporting.freshness as F
    (tmp_path / "results").mkdir()
    (tmp_path / "b.py").write_text('X = "results/r.json"\n')
    surf = tmp_path / "s.json"
    surf.write_text("{}")
    src = tmp_path / "results/r.json"
    src.write_text("{}")
    os.utime(surf, (time.time() - 100, time.time() - 100))
    monkeypatch.setattr(F, "ROOT", tmp_path)
    monkeypatch.setattr(F, "SURFACES", {"s.json": ["b.py"]})
    assert F.check()["s.json"]["status"] == "stale"
    os.utime(surf, None)
    assert F.check()["s.json"]["status"] == "fresh"


def test_tables_from_synthetic_summaries():
    from actionproof.reporting import tables
    r = {"delta": 0.1, "ci95": [0.05, 0.15], "holm_p": 0.001, "seed_deltas": [0.1, 0.09, -0.01], "supported": False,
         "X_mean": 0.6, "A_mean": 0.5, "Y_mean": 0.5, "contrast": "E - D"}
    t = tables.p1({"test": {"B": r, "C": r, "D": r}, "dev": {"winner": "C"}})
    assert "$-$0.010" in t and "Development winner: C" in t and t.count("\\\\") >= 5
    t2 = tables.p2({"primary": {"H1": r}, "secondary": {"S1": r}, "references": {"O": {"mean_vsc": 0.9, "decoding": "greedy", "n_models": 1}}})
    assert "H1" in t2 and "S1" in t2 and "O: 0.900" in t2
