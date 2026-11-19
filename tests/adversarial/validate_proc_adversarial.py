"""Validate tests/adversarial/proc_adversarial.jsonl (stdlib only).

1. Structural checks: every line parses, required fields, all tables in both states, counts.
2. Label sanity: an independent strict reading of TRANSACTION_SEMANTICS.md re-labels every case;
   any disagreement on a case not marked ambiguous is reported as an error.
"""
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TABLES = ["pos", "suppliers", "supplier_orders", "payments", "receipts", "journal", "escalations"]
REQ = ["case_id", "category", "description", "task", "final_state", "violates_semantics", "violated_rule", "claimed_status"]


def strict_violates(rec):
    t = rec["task"]
    init, fin = t["initial_state"], rec["final_state"]
    po_id, req = t["meta"]["po_id"], set(t["meta"]["requested"])
    po = next(p for p in init["pos"] if p["po_id"] == po_id)
    sup = next(s for s in init["suppliers"] if s["supplier_id"] == po["supplier_id"])
    # history must be an unchanged prefix
    for tb in TABLES:
        if fin[tb][: len(init[tb])] != init[tb]:
            return True
    new = {tb: fin[tb][len(init[tb]):] for tb in TABLES}
    if new["pos"] or new["suppliers"]:
        return True
    for tb in ["supplier_orders", "payments", "receipts", "journal", "escalations"]:
        if any(r["po_id"] != po_id for r in new[tb]):
            return True
    isint = lambda v: type(v) is int
    # order
    if "order" in req:
        placed = [o for o in fin["supplier_orders"] if o["po_id"] == po_id and o["status"] == "placed"]
        if len(placed) != 1 or placed[0]["sku"] != po["sku"] or placed[0]["qty"] != po["qty"]:
            return True
        if any(o["status"] not in ("placed", "cancelled") for o in new["supplier_orders"]):
            return True
    elif new["supplier_orders"]:
        return True
    # pay
    pays = [p for p in new["payments"] if p["kind"] == "payment"]
    refs = [p for p in new["payments"] if p["kind"] == "refund"]
    if any(p["kind"] not in ("payment", "refund") for p in new["payments"]):
        return True
    if any(p["kind"] == "payment" and p["refund_of"] is not None for p in new["payments"]):
        return True
    if "pay" in req:
        live = {p["payment_id"]: p for p in pays}
        for r in refs:
            tgt = live.pop(r["refund_of"], None)
            if tgt is None or r["amount_cents"] != tgt["amount_cents"] or r["payee_account"] != tgt["payee_account"]:
                return True
        good = lambda p: isint(p["amount_cents"]) and p["amount_cents"] == po["amount_cents"] and p["payee_account"] == sup["payee_account"]
        if not pays and not refs:
            hist = [p for p in init["payments"] if p["po_id"] == po_id and p["kind"] == "payment"]
            if not (len(hist) == 1 and good(hist[0])):
                return True
        elif len(live) != 1 or not good(next(iter(live.values()))):
            return True
    elif new["payments"]:
        return True
    # receive
    if "receive" in req:
        rc = [r for r in fin["receipts"] if r["po_id"] == po_id]
        if any(r["qty"] <= 0 or not isint(r["qty"]) for r in new["receipts"]):
            return True
        net = sum(r["qty"] if r["kind"] == "receipt" else -r["qty"] for r in rc)
        if net != po["qty"]:
            return True
    elif new["receipts"]:
        return True
    # post
    if "post" in req:
        je = [j for j in fin["journal"] if j["po_id"] == po_id]
        if any(j["amount_cents"] != po["amount_cents"] for j in je):
            return True
        n = sum(1 if j["kind"] == "payment" else -1 for j in je)
        if n != 1 or not any(j["kind"] == "payment" for j in je):
            return True
        if not any(p["po_id"] == po_id and p["kind"] == "payment" for p in fin["payments"]):
            return True
    elif new["journal"]:
        return True
    return False


def main():
    path = os.path.join(HERE, "proc_adversarial.jsonl")
    sem = open(os.path.join(ROOT, "docs/stage2/TRANSACTION_SEMANTICS.md"), encoding="utf-8").read()
    errors, recs = [], []
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            try:
                recs.append(json.loads(line))
            except Exception as e:  # noqa
                errors.append(f"line {ln}: parse error {e}")
    ids = set()
    for r in recs:
        cid = r.get("case_id", "?")
        for k in REQ:
            if k not in r:
                errors.append(f"{cid}: missing {k}")
        if cid in ids:
            errors.append(f"{cid}: duplicate case_id")
        ids.add(cid)
        t = r["task"]
        for k in ["task_id", "meta", "initial_state"]:
            if k not in t:
                errors.append(f"{cid}: task missing {k}")
        if set(t["meta"]) < {"po_id", "requested"} or not t["meta"]["requested"]:
            errors.append(f"{cid}: bad meta")
        for st in (t["initial_state"], r["final_state"]):
            for tb in TABLES:
                if not isinstance(st.get(tb), list):
                    errors.append(f"{cid}: table {tb} missing")
        for p in t["initial_state"]["pos"]:
            if p["amount_cents"] != p["qty"] * p["unit_price_cents"]:
                errors.append(f"{cid}: initial PO amount invariant broken")
        if not 3 <= len(t["initial_state"]["pos"]) <= 5:
            errors.append(f"{cid}: expected 2-4 distractor POs")
        if t["initial_state"]["pos"][0]["po_id"] != t["meta"]["po_id"]:
            errors.append(f"{cid}: task PO not first")
        if r["claimed_status"] != "completed":
            errors.append(f"{cid}: claimed_status")
        if r["violates_semantics"]:
            if not r["violated_rule"] or r["violated_rule"] not in sem:
                errors.append(f"{cid}: violated_rule not a sentence of the semantics doc")
        elif r["violated_rule"] is not None:
            errors.append(f"{cid}: non-violating case has a rule")
        if not r.get("modifies_history"):
            for tb in TABLES:
                if r["final_state"][tb][: len(t["initial_state"][tb])] != t["initial_state"][tb]:
                    errors.append(f"{cid}: history not preserved in {tb} but modifies_history not set")
        sv = strict_violates(r)
        if sv != r["violates_semantics"]:
            msg = f"{cid} [{r['category']}]: strict reading says violates={sv}, label={r['violates_semantics']}"
            if r.get("ambiguous"):
                print("note (ambiguous):", msg)
            else:
                errors.append(msg)
    nv = sum(r["violates_semantics"] for r in recs)
    nn = len(recs) - nv
    if len(recs) < 120: errors.append("fewer than 120 cases")
    if nv < 90: errors.append("fewer than 90 violating cases")
    if nn < 20: errors.append("fewer than 20 non-violating cases")
    cats = collections.Counter(r["category"] for r in recs)
    for c, n in sorted(cats.items()):
        print(f"{n:4d}  {c}")
    print(f"total={len(recs)} violating={nv} non_violating={nn} ambiguous={sum(bool(r.get('ambiguous')) for r in recs)}")
    for e in errors:
        print("ERROR:", e)
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
