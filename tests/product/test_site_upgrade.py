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
    assert set(b) == {"r1", "r1t", "j1", "roi"}
    for k in ("r1", "r1t", "j1"):
        assert b[k]["source"].startswith("results/")
    assert b["roi"]["source"] == "results/roi/roi_model.json" and len(b["roi"]["arms"]) == 7
