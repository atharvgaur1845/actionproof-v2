"""Website procurement chapter: every recorded trace equals its result file, every simulator scenario replays to the
same outcome, and every headline number is read from a registered result file (no hand-typed values)."""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import site_procure as P  # noqa: E402


def test_stories_match_result_files():
    for key, tid in {v: k for k, v in P.STORIES.items()}.items():
        st = P.story(tid)
        for m, rec in st["models"].items():
            r = next(json.loads(x) for x in (ROOT / f"results/s3/conf/{m}__s3_conf_full.jsonl").open()
                     if json.loads(x)["task_id"] == tid)
            assert rec["vsc"] == r["score"]["vsc"] and rec["report"] == r["report"]
            assert [c["tool"] for c in rec["calls"]] == [x["tool"] for x in r["trace"] if x.get("by") == "agent"]


def test_simulator_scenarios_are_deterministic_and_consistent():
    a, b = P.sim(), P.sim()
    assert a == b and len(a["runs"]) >= 30
    for key, run in a["runs"].items():
        wf, fault, step, ratio = key.split("|")
        v = run["verifying"]
        if fault not in ("never_commit",):
            assert v["vsc"] == 1, key  # the reference oracle completes every attainable scenario
        if fault in ("phantom", "partial", "never_commit", "dup_delivery"):
            assert run["trusting"]["vsc"] == 0 and any(r["div"] for r in run["trusting"]["rows"]), key


def test_headline_numbers_come_from_files():
    r = P.results()
    a12 = json.loads((ROOT / "results/s2/a12_summary.json").read_text())
    assert r["a12"]["vsc_diff"] == {k: v["diff"] for k, v in a12["vsc"]["per_seed"].items()}
    s3 = json.loads((ROOT / "results/s3/s3_summary.json").read_text())
    assert r["s3"]["recipe_full"]["est"] == s3["splits"]["s3_conf_full"]["across"]["vsc"]["seed_avg"]["est"]
    src = (ROOT / "scripts/site_procure.py").read_text()
    assert not re.search(r"\b0\.(21|22|09|27|31)\b", src), "a result value is typed into the site builder"
