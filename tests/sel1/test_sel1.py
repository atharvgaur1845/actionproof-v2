"""SEL1 checks (PREREGISTRATION_SEL1.md): the transient-outage env, the reference policies on every SEL1 split,
disjointness and hashes of the splits, and a dry run of the registered analysis on scripted stand-ins."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from actionproof.bench.generator_proc import DATA
from actionproof.sel1 import tasks as ST
from actionproof.sel1.env import SelEnv, make_env_sel
from actionproof.sel1.policy import run_policy
from actionproof.txn.env import UNAVAILABLE

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ("sel1_train", "sel1_ctl_pool", "sel1_test")


@pytest.fixture(scope="module")
def splits():
    return {s: ST.load(s) for s in SPLITS}


def test_manifest_hashes(splits):
    man = json.loads((DATA / "sel1_manifest.json").read_text())
    for s in SPLITS:
        f = DATA / f"{s}.jsonl"
        assert man["files"][f.name]["sha256"] == hashlib.sha256(f.read_bytes()).hexdigest()
        assert man["files"][f.name]["n"] == len(splits[s])


def test_regenerates_identically(splits):
    for s in SPLITS:
        assert "".join(ST.T.task_line(t) for t in ST.make_split(s)) == (DATA / f"{s}.jsonl").read_text()


def test_disjoint(splits):
    ST.check_disjoint(splits)


def test_train_never_faults_receive_or_has_heldout_regimes(splits):
    for t in splits["sel1_train"] + splits["sel1_ctl_pool"]:
        tx = t.meta["txn"]
        assert tx["regime"] in ("standard", "delayed", "unresolvable", "transient")
        if tx.get("quad_id"):
            assert tx["quad_step"] in ("order", "pay")
        if tx["regime"] == "transient":
            assert set(tx["transient_unavailable"].values()) == {1}
        assert "get_audit_log" not in tx["unavailable_reads"] or tx["regime"] == "unresolvable"
    assert all(t.meta["txn"]["regime"] in ("standard", "delayed") for t in splits["sel1_ctl_pool"])
    test = splits["sel1_test"]
    assert {t.meta["txn"]["group"] for t in test} == set(ST.TEST_GROUPS)
    assert all(t.meta["txn"]["quad_step"] == "receive" for t in test if t.meta["txn"]["group"] == "u_receive")
    assert all(set(t.meta["txn"]["transient_unavailable"].values()) == {2} for t in test if t.meta["txn"]["group"] == "t2")


def test_transient_outage_env(splits):
    t = next(t for t in splits["sel1_test"] if t.meta["txn"]["group"] == "t2")
    env = make_env_sel(t, "test-transient")
    assert isinstance(env, SelEnv)
    tool = next(iter(t.meta["txn"]["transient_unavailable"]))
    po = t.meta["po_id"]
    assert env.call(tool, {"po_id": po, "consistent": True}) == UNAVAILABLE
    assert env.call(tool, {"po_id": po, "consistent": True}) == UNAVAILABLE
    assert env.call(tool, {"po_id": po, "consistent": True})["status"] == "ok"
    assert env.call("get_audit_log", {"po_id": po})["status"] == "ok"


def test_policies_on_every_split(splits):
    """B5 resolves every task safely; B4 over-escalates exactly on transient tasks; ALWAYS escalates everywhere."""
    for s, ts in splits.items():
        for t in ts:
            tx = t.meta["txn"]
            b5, _ = run_policy("B5", t)
            assert b5["txn"]["str"] and b5["ended_by"] == "report", (t.task_id, tx["regime"])
            assert b5["txn"]["escalated"] == tx["escalation_correct"]
            b4, _ = run_policy("B4", t)
            assert b4["txn"]["escalated"] == (tx["regime"] in ("unresolvable", "transient"))
            al, _ = run_policy("ALWAYS", t)
            assert al["txn"]["escalated"] and al["txn"]["str"] == tx["escalation_correct"]


def test_b5_on_txn_test_matches_b4_outside_outages():
    from actionproof.bench.generator_proc import load_tasks
    for t in load_tasks(DATA / "txn_test.jsonl"):
        b5, _ = run_policy("B5", t, txn_env=True)
        b4, _ = run_policy("B4", t, txn_env=True)
        assert b5["txn"]["str"] == b4["txn"]["str"], t.task_id
        assert b5["txn"]["escalated"] == b4["txn"]["escalated"], t.task_id


def test_registered_analysis_dry_run(tmp_path, monkeypatch, splits):
    """Scripted stand-ins: SEL := B5, CTL := A5 (never escalates). H1-H3 must be supported, H4 must hold (both are
    B4-equivalent on resolvable TXN tasks), and the reverse assignment must fail H1-H3."""
    from actionproof.bench.generator_proc import load_tasks
    spec = importlib.util.spec_from_file_location("sel1_stats", ROOT / "scripts/sel1_stats.py")
    S = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(S)
    monkeypatch.setattr(S, "R", tmp_path)
    monkeypatch.setattr(S, "B", 400)
    eps = {}
    for split, ts in (("sel1_test", splits["sel1_test"]), ("txn_test", load_tasks(DATA / "txn_test.jsonl"))):
        for p in ("B5", "A5"):
            eps[(p, split)] = [run_policy(p, t, txn_env=(split == "txn_test"))[0] for t in ts]

    def write(sel, ctl):
        for f in tmp_path.glob("*"):
            f.unlink()
        for arm, p in (("SEL", sel), ("CTL", ctl)):
            for k in (0, 1, 2):
                for split in ("sel1_test", "txn_test"):
                    (tmp_path / f"{arm}_s{k}__{split}.jsonl").write_text(
                        "".join(json.dumps(r, default=str) + "\n" for r in eps[(p, split)]))

    write("B5", "A5")
    S.main()
    res = json.loads((tmp_path / "sel1_summary.json").read_text())
    assert res["H1"]["supported"] and res["H2"]["supported"] and res["H3"]["supported"] and res["H4"]["supported"]
    assert res["H3"]["J_SEL"] == pytest.approx(1.0) and res["H3"]["J_CTL"] == pytest.approx(0.0)
    assert res["SEL1_supported"]
    write("A5", "B5")
    S.main()
    res = json.loads((tmp_path / "sel1_summary.json").read_text())
    assert not (res["H1"]["supported"] or res["H2"]["supported"] or res["H3"]["supported"])
    assert not res["SEL1_supported"]
