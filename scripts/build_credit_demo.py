"""Demo tab "Same task, three templates" (Phase 3): recorded greedy `e_heldout` episodes, rendered side by side.

  A6 warm start : calls apply_store_credit with the refund's key -> the call replays the refund's response (D-ENV-1)
  A6 state arm  : drops the credit step, re-issues the refund, claims the credit in its summary
  A7 state arm  : same reward as the A6 state arm, warm start with 20 credit demos -> credit with its own key, committed

Only fault-free (class OK) tasks are used, so every difference is the template, not a fault.

    python scripts/build_credit_demo.py   # -> demo/replays/credit_templates.json
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

SOURCES = {
    "A6 warm start": "results/raw/a6/init_a6__neutral__e_heldout.jsonl",
    "A6 state arm (pcstr_d_noreport)": "results/raw/a6/a6_pcstr_d_noreport_s0__neutral__e_heldout.jsonl",
    "A7 state arm (same reward + 20 credit demos in SFT)": "results/eval/a6/a7_pcstr_d_noreport_s0__neutral__e_heldout.jsonl",
}
OUT = Path("demo/replays/credit_templates.json")
N = 6


def _d(x):
    if isinstance(x, dict):
        return x
    try:
        return ast.literal_eval(x) if isinstance(x, str) else {}
    except (ValueError, SyntaxError):
        return {}


def classify(rec: dict) -> str:
    t = [s for s in rec["trace"] if not s.get("by")]
    rk = next((_d(s["args"]).get("idempotency_key") for s in t if s["tool"] == "issue_refund"), None)
    credits = [s for s in t if s["tool"] == "apply_store_credit"]
    committed = any(r["op"] == "apply_store_credit" for r in rec["receipts"])
    if committed:
        return "committed"
    if credits and all(_d(s["args"]).get("idempotency_key") == rk for s in credits):
        return "void"
    if not credits and sum(s["tool"] == "issue_refund" for s in t) >= 2:
        return "substituted"
    return "other"


def panel(rec: dict) -> str:
    t = [s for s in rec["trace"] if not s.get("by")]
    receipts = {(r["op"], r["entity"]) for r in rec["receipts"]}
    rk = next((_d(s["args"]).get("idempotency_key") for s in t if s["tool"] == "issue_refund"), None)
    lines = []
    for s in t:
        a, o = _d(s["args"]), _d(s.get("obs"))
        if s["tool"] == "submit_report":
            continue
        note = ""
        if s["tool"] in ("issue_refund", "apply_store_credit"):
            key = a.get("idempotency_key")
            note = f" · key `{key}`"
            if s["tool"] == "apply_store_credit":
                same = key == rk
                note += " **(the refund's key: the tool replays the refund's response)**" if same else " (its own key)"
                note += f" · tool says `{o.get('status')}`" + (f", returns `{o.get('refund_id')}`" if o.get("refund_id") else "")
            elif sum(x["tool"] == "issue_refund" for x in t[: t.index(s)]):
                note += " **(re-issued: the ledger already shows this refund)**"
        elif s["tool"] == "query_ledger":
            note = f" → refunds {len(o.get('refunds') or [])}, credits {len(o.get('credits') or [])}"
        lines.append(f"{len(lines) + 1}. `{s['tool']}`{note}")
    rep = rec.get("report") or {}
    credit_db = any(op == "apply_store_credit" for op, _ in receipts)
    v = rec["verdict"]
    lines += ["", f"**Agent reports:** `{rep.get('status')}`: _{str(rep.get('summary', ''))[:220]}_",
              f"**Database:** store credit applied: **{'yes' if credit_db else 'no'}** · duplicate refund: "
              f"**{'yes' if v['D'] > 0 else 'no'}** · goal met (G): **{v['G']}**"]
    return "\n".join(lines)


def main() -> None:
    recs = {m: {json.loads(l)["task_id"]: json.loads(l) for l in open(p)} for m, p in SOURCES.items()}
    want = ("void", "substituted", "committed")
    common = sorted(set.intersection(*(set(r) for r in recs.values())))
    items = []
    for tid in common:
        rs = [recs[m][tid] for m in SOURCES]
        if (rs[0].get("a4") or {}).get("env_class") != "OK":
            continue
        if tuple(classify(r) for r in rs) != want:
            continue
        items.append({"task_id": tid, "prompt": rs[0]["task_prompt"],
                      "panels": {m: panel(r) for m, r in zip(SOURCES, rs)}})
        if len(items) == N:
            break
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"models": list(SOURCES), "items": items}, indent=1))
    print(f"{len(items)} tasks with the void / substituted / committed pattern -> {OUT}")


if __name__ == "__main__":
    main()
