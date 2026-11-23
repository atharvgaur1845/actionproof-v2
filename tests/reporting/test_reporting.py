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
