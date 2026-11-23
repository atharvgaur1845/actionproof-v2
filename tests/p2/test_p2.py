"""P2 (PREREGISTRATION_P2.md): ledger, guard, rollout copy, coverage validator."""
from __future__ import annotations

import difflib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = "Tool result for {name}: {content}\nContinue the task. Call submit_report when you are finished."
REQ = ("Purchase order PO-AAAA1111 is approved: 10 x hard hats (SKU-1) from Delta at 100 cents each, total 1000 cents. "
       "The goods were already received. Pay the supplier the full amount and post the journal entry for the payment.")


def call(tool, args):
    return {"role": "assistant", "content": json.dumps([{"tool_name": tool, "parameters": args}])}


def res(tool, obs):
    return {"role": "user", "content": RES.format(name=tool, content=json.dumps(obs))}


def conv(*turns):
    c = [{"role": "system", "content": "policy"}, {"role": "user", "content": REQ}]
    for t, a, o in turns:
        c += [call(t, a), res(t, o)]
    return c


PAY = ("pay_supplier", {"po_id": "PO-AAAA1111", "amount_cents": 1000, "idempotency_key": "k1"}, {"status": "ok", "payment_id": "PAY-1"})
LIST_SHORT = ("list_payments", {"po_id": "PO-AAAA1111"},
              {"status": "ok", "payments": [{"payment_id": "PAY-1", "amount_cents": 600, "kind": "payment", "refund_of": None}]})
REFUND = ("refund_payment", {"payment_id": "PAY-1"}, {"status": "ok", "refund_id": "RF-1"})
LIST_REFUNDED = ("list_payments", {"po_id": "PO-AAAA1111"},
                 {"status": "ok", "payments": [{"payment_id": "PAY-1", "amount_cents": 600, "kind": "payment", "refund_of": None},
                                               {"payment_id": "RF-1", "amount_cents": 600, "kind": "refund", "refund_of": "PAY-1"}]})
CFG = {"requested": ["pay", "post"], "total": 1000, "qty": 10}


def test_ledger_unknowns_and_shortfall():
    from actionproof.p2.ledger import ledger_text
    t = ledger_text(conv())
    assert "total 1000 cents, quantity 10" in t and t.count("unknown (not read)") == 4
    t = ledger_text(conv(PAY, LIST_SHORT))
    assert "live total 600 cents (below the PO total by 400 cents)" in t and "Not read back since written: none" in t
    t = ledger_text(conv(PAY, LIST_SHORT, PAY))
    assert "stale: written since" in t and "[reused; reply named the same payment]" in t


def test_ledger_refunded_key_and_hindi_total():
    from actionproof.p2.ledger import ledger_text, with_ledger
    t = ledger_text(conv(PAY, LIST_SHORT, REFUND, LIST_REFUNDED))
    assert "k1 -> PAY-1 (refunded)" in t and "live none" in t
    c = conv()
    c[1] = {"role": "user", "content": "खरीद आदेश PO-AAAA1111 स्वीकृत है: Delta से 10 x हेलमेट (SKU-1), 100 सेंट प्रति इकाई, कुल 1000 सेंट।"}
    assert "total 1000 cents" in ledger_text(c)
    # a result that already carries a ledger still parses (raw_decode), and the ledger goes before the tail
    r = with_ledger(res(*LIST_SHORT[::2])["content"], conv(PAY))
    assert r.index("Ledger (") < r.index("\nContinue the task.")
    assert "live total 600" in ledger_text(conv(PAY) + [{"role": "user", "content": r}] + [call(*PAY[:2]), res(PAY[0], PAY[2])])


def test_guard_rules():
    from actionproof.p2.guard import check
    assert check("pay_supplier", PAY[1], conv(), CFG) is None
    assert check("pay_supplier", PAY[1], conv(PAY), CFG)["error"] == "guard_read_before_retry"
    assert check("pay_supplier", {**PAY[1], "idempotency_key": "k2"}, conv(PAY, LIST_SHORT), CFG) is None
    assert check("pay_supplier", PAY[1], conv(PAY, LIST_SHORT, REFUND, LIST_REFUNDED), CFG)["error"] == "guard_refunded_key"
    assert check("pay_supplier", {**PAY[1], "idempotency_key": "k2"}, conv(PAY, LIST_SHORT, REFUND), CFG)["error"] == "guard_read_before_retry"
    full = ("list_payments", {}, {"status": "ok", "payments": [{"payment_id": "PAY-2", "amount_cents": 1000, "kind": "payment", "refund_of": None}]})
    assert check("pay_supplier", {**PAY[1], "idempotency_key": "k3"}, conv(PAY, full), CFG)["error"] == "guard_overfill"
    assert check("submit_report", {"status": "completed"}, conv(PAY, full), CFG)["error"] == "guard_unverified_completion"
    assert check("submit_report", {"status": "failed"}, conv(PAY), CFG) is None
    post = ("post_journal_entry", {"po_id": "PO-AAAA1111", "amount_cents": 1000, "idempotency_key": "j1"}, {"status": "ok", "entry_id": "JE-1"})
    jr = ("get_journal_entries", {}, {"status": "ok", "entries": [{"entry_id": "JE-1", "amount_cents": 1000, "kind": "payment"}]})
    assert check("submit_report", {"status": "completed"}, conv(PAY, full, post, jr), CFG) is None
    assert check("list_payments", {}, conv(PAY), CFG) is None


def test_rollout_copy_differs_only_on_marked_lines():
    a = (ROOT / "actionproof/stage3/rollout.py").read_text().split("from __future__ import annotations", 1)[1].splitlines()
    b = (ROOT / "actionproof/p2/rollout.py").read_text().split("from __future__ import annotations", 1)[1].splitlines()
    changed = [l for l in difflib.unified_diff(a, b, lineterm="", n=0) if l.startswith("+") and not l.startswith("+++")]
    assert changed and all(l.rstrip().endswith("# P2:") for l in changed), changed


def test_coverage_validator(tmp_path):
    from actionproof.reporting.coverage import check
    split = tmp_path / "s.jsonl"
    split.write_text("".join(json.dumps({"task_id": t}) + "\n" for t in ("t1", "t2")))
    good = [{"task_id": t, "sample": j, "vsc": 1, "score": {"steps": {}}, "ended_by": "report"} for t in ("t1", "t2") for j in (-1, 0, 1)]
    f = tmp_path / "o.jsonl"
    f.write_text("".join(json.dumps(r) + "\n" for r in good))
    assert check(f, split, 2)["ok"]
    f.write_text("".join(json.dumps(r) + "\n" for r in good[:-1] + [good[0]] + [{**good[0], "task_id": "zz"}]))
    p = check(f, split, 2)["problems"]
    assert p["missing_samples"] == 1 and p["duplicate_episodes"] == 1 and p["unexpected_tasks"] == 1
    assert not check(tmp_path / "none.jsonl", split, 2)["ok"]
