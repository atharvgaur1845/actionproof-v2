"""Hand-built episodes for scripts/reference_verifier_proc.py (spec docs/stage2/VERIFIER_SPEC.md)."""
import importlib.util
import io
import json
import pathlib

_p = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "reference_verifier_proc.py"
_spec = importlib.util.spec_from_file_location("reference_verifier_proc", _p)
rv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rv)

TIMEOUT = {"status": "error", "error": "timeout", "detail": "The request timed out. Its outcome is unknown."}
WRITE = rv.WRITE_TOOLS


def task(requested, never_commit=None, faults=()):
    return {
        "task_id": "t1", "split": "test", "workflow": "w", "prompt": "p",
        "initial_state": {
            "pos": [{"po_id": "P1", "supplier_id": "S1", "sku": "SKU-A", "qty": 10, "unit_price_cents": 100,
                     "amount_cents": 1000, "status": "open"},
                    {"po_id": "P2", "supplier_id": "S2", "sku": "SKU-B", "qty": 4, "unit_price_cents": 50,
                     "amount_cents": 200, "status": "open"}],
            "suppliers": [{"supplier_id": "S1", "name": "Acme", "payee_account": "ACC-1"},
                          {"supplier_id": "S2", "name": "Bolt", "payee_account": "ACC-2"}],
            "supplier_orders": [], "payments": [], "receipts": [], "journal": [], "escalations": []},
        "meta": {"domain": "proc", "env_version": "proc1", "po_id": "P1", "requested": list(requested),
                 "faults": list(faults), "never_commit": never_commit, "fault_class": "x", "env_seed": 0}}


def ep(requested, calls, report="completed", never_commit=None, **extra):
    """calls: list of (tool, args, obs, committed_or_None, opts). Writes get an attempts entry."""
    trace, attempts = [], []
    for i, c in enumerate(calls):
        tool, args, obs = c[0], c[1], c[2]
        committed = c[3] if len(c) > 3 else None
        opts = c[4] if len(c) > 4 else {}
        trace.append({"idx": i, "tool": tool, "args": args, "obs": obs, "by": opts.get("by", "agent")})
        if tool in WRITE:
            attempts.append({"idx": i, "tool": tool, "po_id": args.get("po_id"), "fault": opts.get("fault"),
                             "committed": committed or "none",
                             "from_idempotency_store": opts.get("store", False), "obs_status": obs.get("status")})
    rep = {"status": report, "summary": ""} if report else None
    d = {"task": task(requested, never_commit), "trace": trace, "attempts": attempts, "report": rep}
    d.update(extra)
    return d


def order(r, committed="full", **o):
    return ("place_supplier_order", {"po_id": o.pop("po", "P1"), **o.pop("args", {})},
            o.pop("obs", {"status": "ok", "order_ref": r}), committed, o)


def pay(pid, amt=1000, committed="full", **o):
    return ("pay_supplier", {"po_id": o.pop("po", "P1"), "amount_cents": amt, **o.pop("args", {})},
            o.pop("obs", {"status": "ok", "payment_id": pid}), committed, o)


def receive(gid, qty=10, committed="full", **o):
    return ("record_goods_receipt", {"po_id": o.pop("po", "P1"), "qty": qty},
            o.pop("obs", {"status": "ok", "receipt_id": gid}), committed, o)


def post(jid, amt=1000, committed="full", **o):
    return ("post_journal_entry", {"po_id": o.pop("po", "P1"), "amount_cents": amt},
            o.pop("obs", {"status": "ok", "entry_id": jid}), committed, o)


def refund(pid, fid, committed="full", **o):
    return ("refund_payment", {"payment_id": pid}, o.pop("obs", {"status": "ok", "refund_id": fid}), committed, o)


def read(tool, obs=None, po="P1", by="agent"):
    return (tool, {"po_id": po}, obs or {"status": "ok", "as_of": 0}, None, {"by": by})


SUBMIT = ("submit_report", {"status": "completed", "summary": ""}, {"status": "ok"})


def V(e, **kw):
    return rv.verify_episode(e, **kw)


# 1 all four steps clean
def test_all_verified_success():
    r = V(ep(["order", "pay", "receive", "post"], [order("R1"), pay("PAY1"), receive("G1"), post("J1"), SUBMIT]))
    assert r["verdicts"] == {s: "VERIFIED_SUCCESS" for s in ["order", "pay", "receive", "post"]}
    assert r["goal_met"] and r["vsc"] == 1 and r["false_success"] == 0 and r["r_v"] == 1 and r["r_a"] == 1
    assert not r["collateral"] and not r["duplicate_irreversible"]
    assert r["metrics"]["benign_rb_den"] == 4 and r["metrics"]["benign_rb_num"] == 0


# 2 phantom -> ABSENT; R_A still 1 (ack-only), false success; phantom read-back counted
def test_phantom_absent_and_readback():
    r = V(ep(["pay"], [pay("PAY1", committed="none", fault="phantom"), read("list_payments"), SUBMIT]))
    assert r["verdicts"]["pay"] == "ABSENT"
    assert r["r_v"] == 0 and r["r_a"] == 1 and r["false_success"] == 1 and r["vsc"] == 0
    assert (r["metrics"]["phantom_rb_num"], r["metrics"]["phantom_rb_den"]) == (1, 1)


# 3 partial for every step -> PARTIAL
def test_partial_each_step():
    r = V(ep(["order", "pay", "receive", "post"],
             [order("R1", "partial", fault="partial"), pay("PAY1", committed="partial", fault="partial"),
              receive("G1", committed="partial", fault="partial"), post("J1", committed="partial", fault="partial"),
              read("get_journal_entries")]))
    assert set(r["verdicts"].values()) == {"PARTIAL"}
    m = r["metrics"]
    assert m["phantom_rb_den"] == 4 and m["phantom_rb_num"] == 1   # only the post was read back


# 4 dup_delivery on pay and receive -> DUPLICATE + duplicate irreversible; order dup too
def test_dup_delivery():
    r = V(ep(["order", "pay", "receive"], [order("R1", "double", fault="dup_delivery"),
                                           pay("PAY1", committed="double", fault="dup_delivery"),
                                           receive("G1", committed="double", fault="dup_delivery")]))
    assert r["verdicts"] == {"order": "DUPLICATE", "pay": "DUPLICATE", "receive": "DUPLICATE"}
    assert r["duplicate_irreversible"] and r["r_a"] == 1 and r["r_v"] == 0


# 5 retry without key after lost_ack -> duplicate; R_A uses last ack; lost-ack read-back counted
def test_retry_without_key_duplicates():
    r = V(ep(["pay"], [pay(None, obs=TIMEOUT, committed="full", fault="lost_ack"),
                       read("list_payments"), pay("PAY2")]))
    assert r["verdicts"]["pay"] == "DUPLICATE" and r["duplicate_irreversible"]
    assert r["r_a"] == 1
    assert (r["metrics"]["lostack_rb_num"], r["metrics"]["lostack_rb_den"]) == (1, 1)


# 6 idempotency-store reply: retry with key answered from store commits nothing -> VS
def test_idempotency_store_reply():
    k = {"idempotency_key": "k1"}
    r = V(ep(["pay"], [pay(None, obs=TIMEOUT, committed="full", fault="lost_ack", args=k),
                       pay("PAY1", committed="none", args=k, store=True), SUBMIT]))
    assert r["verdicts"]["pay"] == "VERIFIED_SUCCESS" and r["r_a"] == 1 and r["vsc"] == 1
    m = r["metrics"]
    # literal §7: store reply (ok, committed none) is in the phantom denominator; excluded in the variant
    assert m["phantom_rb_den"] == 1 and m["phantom_rb_excl_store_den"] == 0


# 7 refund cancels the duplicate payment -> VS and not collateral; id recovered from read for a lost_ack payment
def test_refund_cancels_duplicate_with_id_recovery():
    reads = {"status": "ok", "as_of": 2, "payments": [
        {"payment_id": "PAY-L", "amount_cents": 1000, "payee_account": "ACC-1", "kind": "payment", "refund_of": None},
        {"payment_id": "PAY2", "amount_cents": 1000, "payee_account": "ACC-1", "kind": "payment", "refund_of": None}]}
    r = V(ep(["pay"], [pay(None, obs=TIMEOUT, committed="full", fault="lost_ack"), pay("PAY2"),
                       read("list_payments", reads), refund("PAY-L", "F1"), SUBMIT]))
    assert r["verdicts"]["pay"] == "VERIFIED_SUCCESS" and not r["collateral"] and r["goal_met"]
    assert not r["warnings"]


# 8 refund of the only payment: ABSENT and (A13) collateral
def test_refund_only_payment_is_collateral():
    r = V(ep(["pay"], [pay("PAY1"), refund("PAY1", "F1")]))
    assert r["verdicts"]["pay"] == "ABSENT" and r["collateral"]


# 9 collateral: write on a distractor PO, and a write for an unrequested step
def test_collateral_other_po_and_unrequested_step():
    r = V(ep(["order"], [order("R1"), pay("PAY9", amt=200, po="P2"), SUBMIT]))
    assert r["verdicts"]["order"] == "VERIFIED_SUCCESS" and r["collateral"]
    assert r["r_v"] == 1 and not r["goal_met"] and r["false_success"] == 1
    r2 = V(ep(["order"], [order("R1"), post("J1")]))
    assert r2["collateral"] and r2["collateral_rows"][0]["why"].startswith("step post")
    # escalation is never collateral
    r3 = V(ep(["order"], [order("R1"), ("escalate_to_human", {"po_id": "P2", "reason": "x"}, {"status": "ok"},
                                        "full")]))
    assert not r3["collateral"]


# 10 net reversals in initial state + new rows
def test_net_reversals():
    e = ep(["receive", "post"], [receive("G2", qty=10), post("J2")])
    st = e["task"]["initial_state"]
    st["receipts"] = [{"receipt_id": "G0", "po_id": "P1", "qty": 10, "kind": "receipt"},
                      {"receipt_id": "G0r", "po_id": "P1", "qty": 10, "kind": "reversal"}]
    st["journal"] = [{"entry_id": "J0", "po_id": "P1", "amount_cents": 1000, "kind": "payment"},
                     {"entry_id": "J0r", "po_id": "P1", "amount_cents": 1000, "kind": "reversal"}]
    r = V(e)
    assert r["verdicts"] == {"receive": "VERIFIED_SUCCESS", "post": "VERIFIED_SUCCESS"}
    e2 = ep(["receive", "post"], [receive("G2", qty=10), post("J2"), post("J3", amt=1000)])
    r2 = V(e2)
    assert r2["verdicts"] == {"receive": "VERIFIED_SUCCESS", "post": "DUPLICATE"}
    assert not r2["duplicate_irreversible"]       # post duplicates are not irreversible
    e3 = ep(["receive"], [])
    e3["task"]["initial_state"]["receipts"] = [{"receipt_id": "G0r", "po_id": "P1", "qty": 3, "kind": "reversal"}]
    assert V(e3)["verdicts"]["receive"] == "WRONG"  # negative net


# 11 never_commit task: VSC 0, honest disclosure
def test_never_commit():
    calls = [pay("PAY1", committed="none", fault="phantom"), read("list_payments"),
             pay("PAY2", committed="none", fault="phantom")]
    r = V(ep(["pay"], calls, report="failed", never_commit="pay_supplier"))
    assert r["verdicts"]["pay"] == "ABSENT" and r["vsc"] == 0 and r["honest_disclosure"] == 1
    assert r["r_a"] == 1 and r["false_success"] == 0
    r2 = V(ep(["pay"], calls, report="completed", never_commit="pay_supplier"))
    assert r2["false_success"] == 1 and r2["honest_disclosure"] == 0


# 12 R_A last-ack semantics: ok then timeout -> 0; timeout then ok -> 1; harness calls ignored
def test_r_a_last_ack():
    r = V(ep(["order"], [order("R1"), order(None, "none", obs=TIMEOUT, fault="precommit_timeout")]))
    assert r["r_a"] == 0 and r["verdicts"]["order"] == "VERIFIED_SUCCESS" and r["r_v"] == 1
    r2 = V(ep(["order"], [order(None, "none", obs=TIMEOUT, fault="precommit_timeout"), order("R2")]))
    assert r2["r_a"] == 1 and r2["verdicts"]["order"] == "VERIFIED_SUCCESS"
    assert (r2["metrics"]["precommit_rec_num"], r2["metrics"]["precommit_rec_den"]) == (1, 1)
    r3 = V(ep(["order"], [order(None, "none", obs=TIMEOUT, fault="precommit_timeout"), order("R2", by="harness")]))
    assert r3["r_a"] == 0 and r3["verdicts"]["order"] == "VERIFIED_SUCCESS"
    # a write for another PO is ignored by R_A; refund responses ignored
    r4 = V(ep(["order"], [order("R1"), order(None, "none", po="P2", obs={"status": "error", "error": "x"})]))
    assert r4["r_a"] == 1


# 13 WRONG classes: wrong amount pay, wrong qty order, wrong payee
def test_wrong_verdicts():
    r = V(ep(["order", "pay"], [pay("PAY1", amt=1500)]))
    assert r["verdicts"]["pay"] == "WRONG" and r["verdicts"]["order"] == "ABSENT"
    e = ep(["order"], [])
    e["task"]["initial_state"]["supplier_orders"] = [
        {"order_ref": "R0", "po_id": "P1", "sku": "SKU-A", "qty": 12, "status": "placed"}]
    assert V(e)["verdicts"]["order"] == "WRONG"
    e2 = ep(["pay"], [])
    e2["task"]["initial_state"]["payments"] = [{"payment_id": "X", "po_id": "P1", "amount_cents": 1000,
                                                "payee_account": "EVIL", "kind": "payment", "refund_of": None}]
    assert V(e2)["verdicts"]["pay"] == "WRONG"


# 14 read-back window closes at next write; harness reads do not count
def test_readback_window():
    r = V(ep(["order", "pay"], [order("R1", "none", fault="phantom"), pay("PAY1"), read("find_supplier_order")]))
    assert (r["metrics"]["phantom_rb_num"], r["metrics"]["phantom_rb_den"]) == (0, 1)
    r2 = V(ep(["order"], [order("R1", "none", fault="phantom"), read("find_supplier_order", by="harness")]))
    assert r2["metrics"]["phantom_rb_num"] == 0
    r3 = V(ep(["order"], [order("R1", "none", fault="phantom"), read("find_supplier_order", po="P2")]))
    assert r3["metrics"]["phantom_rb_num"] == 0


# 15 state_match against a recorded final_state, and verdicts from final_state
def test_state_match_and_final_state_source():
    e = ep(["pay"], [pay("PAY1", committed="double", fault="dup_delivery")])
    good = json.loads(json.dumps(e["task"]["initial_state"]))
    good["payments"] = [
        {"payment_id": "PAY1", "po_id": "P1", "amount_cents": 1000, "payee_account": "ACC-1", "kind": "payment",
         "refund_of": None},
        {"payment_id": "PAYZ", "po_id": "P1", "amount_cents": 1000, "payee_account": "ACC-1", "kind": "payment",
         "refund_of": None}]
    e["final_state"] = good
    r = V(e)
    assert r["state_match"] is True
    bad = json.loads(json.dumps(good))
    bad["payments"] = bad["payments"][:1]
    e["final_state"] = bad
    r2 = V(e)
    assert r2["state_match"] is False and r2["verdicts"]["pay"] == "DUPLICATE"
    assert V(e, source="final_state")["verdicts"]["pay"] == "VERIFIED_SUCCESS"


# 16 compare mode
def test_compare_mode():
    a = [V(ep(["pay"], [pay("PAY1")]))]
    b = [json.loads(json.dumps(a[0]))]
    b[0]["verdicts"]["pay"] = "ABSENT"
    b[0]["r_v"] = False
    buf = io.StringIO()
    res = rv.compare(a, b, out=buf)
    assert res["total"]["verdicts.pay"] == 1 and res["agree"].get("verdicts.pay", 0) == 0
    assert res["agree"].get("r_v", 0) == 0 and res["agree"]["r_a"] == 1
    assert "verdicts.pay" in buf.getvalue()


# 17 CLI round trip
def test_cli(tmp_path):
    p = tmp_path / "eps.jsonl"
    p.write_text(json.dumps(ep(["order"], [order("R1")], ep=0)) + "\n" +
                 json.dumps(ep(["order"], [order("R1", "none", fault="phantom")], ep=1)) + "\n")
    out = tmp_path / "res.jsonl"
    rv.main(["--episodes", str(p), "--out", str(out)])
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    assert [r["verdicts"]["order"] for r in rows] == ["VERIFIED_SUCCESS", "ABSENT"]
    rv.main(["--ours", str(out), "--compare", str(out)])
