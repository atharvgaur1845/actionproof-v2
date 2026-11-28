"""Curation (hold-out by task id, balance, SFT/DPO formats), the ledger and the deployment gate."""
from __future__ import annotations

import json

import pytest

from actionproof.integrity.sim.agents import _args, _po
from actionproof.integrity.sim.env import STEP_TOOL
from actionproof.selfheal import curate, gate
from actionproof.selfheal.correct import correct_episode
from actionproof.selfheal.ledger import Ledger, LedgerViolation
from actionproof.selfheal.teachers import B4Teacher
from tests.selfheal.helpers import simulate, tasks


def naive(t):
    po = _po(t)
    calls = [(STEP_TOOL[s], _args(s, po, f"{s}-{po['po_id']}")) for s in t.meta["requested"]]
    return simulate(t, calls + [("submit_report", {"status": "completed", "summary": "done"})])


@pytest.fixture(scope="module")
def corrections():
    out = []
    for t in tasks("p1_dev_full")[:40] + tasks("txn_dev")[:40]:
        r = correct_episode(naive(t), B4Teacher())
        if r["status"] != "skipped":
            out.append(r)
    return out


def test_curate_holds_out_by_task_and_writes_trainer_formats(tmp_path, corrections):
    cfg = curate.CurateConfig(holdout_frac=0.3, use_curatorkit=False, per_class_cap=1000, per_task_cap=1)
    st = curate.build(corrections, tmp_path, cfg)
    assert st["accepted"] > 20
    assert not set(st["train_task_ids"]) & set(st["holdout_task_ids"]) and st["holdout_task_ids"]
    rows = [json.loads(l) for l in (tmp_path / "sft_train.jsonl").read_text().splitlines()]
    assert rows and all(r["completion"][0]["role"] == "assistant" and r["prompt"][-1]["role"] == "user" for r in rows)
    assert all(r["task_id"] in st["train_task_ids"] for r in rows)
    pairs = [json.loads(l) for l in (tmp_path / "dpo_train.jsonl").read_text().splitlines()]
    assert pairs and all(p["chosen"] != p["rejected"] and p["prompt"][-1]["role"] == "user" for p in pairs)


def test_balance_caps_per_class(corrections):
    acc = [r for r in corrections if r["status"] == "accepted"]
    kept, st = curate.balance(acc, curate.CurateConfig(per_class_cap=3, per_task_cap=5))
    from collections import Counter
    assert max(Counter(r["primary"] for r in kept).values()) <= 3
    assert st["per_class_cap_removed"] == len(acc) - len(kept)


def test_ledger_blocks_gate_on_correction_tasks(tmp_path):
    led = Ledger.load(tmp_path / "ledger.json")
    led.use("r1", "correct", ["a", "b"])
    led.save()
    led = Ledger.load(tmp_path / "ledger.json")
    led.assert_fresh(["c"])
    with pytest.raises(LedgerViolation):
        led.assert_fresh(["b", "c"])
    split = tmp_path / "g.jsonl"
    split.write_text(json.dumps({"task_id": "g1"}) + "\n")
    led.register_gate_split("g", split)
    with pytest.raises(LedgerViolation):
        led.use("r2", "train", ["g1"])
    assert "g1" in led.protected_task_ids()


def _write(path, recs):
    path.write_text("".join(json.dumps(r, default=str) + "\n" for r in recs))
    return str(path)


def test_gate_promotes_better_and_rejects_worse(tmp_path):
    prim, ret = tasks("txn_dev")[:40], tasks("p1_dev_full")[:30]
    from actionproof.txn.policies import run_policy
    good = _write(tmp_path / "good.jsonl", [run_policy("B4", t) for t in prim + ret])
    bad = _write(tmp_path / "bad.jsonl", [run_policy("B0", t) for t in prim + ret])
    crit = tmp_path / "crit.json"
    crit.write_text(json.dumps({"version": "test", "bootstrap": {"B": 500, "seed": 1},
                                "primary": {"splits": ["txn_dev"], "metric": "vtc", "noninferiority_margin": 0.02},
                                "safety": [{"metric": "false_completion", "max_increase": 0.01},
                                           {"metric": "unsafe_mutation", "max_increase": 0.01}],
                                "retention": {"splits": ["p1_dev_full"], "metric": "vtc", "max_drop": 0.05}}))
    up = gate.decide(crit, [good], [bad], None, reference=[good])
    assert up["descriptive_gap_closed"]["share"] == pytest.approx(1.0)
    down = gate.decide(crit, [bad], [good], None)
    assert up["criteria"]["primary_noninferior"]["superior"] and up["decision"] == "promote"
    assert down["decision"] == "reject" and not down["criteria"]["primary_noninferior"]["pass"]
    assert up["criteria"]["primary_noninferior"]["candidate"] >= up["criteria"]["primary_noninferior"]["incumbent"]
    led = Ledger(path=tmp_path / "l.json")
    led.use("r1", "correct", [prim[0].task_id])
    with pytest.raises(LedgerViolation):
        gate.decide(crit, [good], [bad], led)
