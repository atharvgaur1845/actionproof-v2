"""Shared prefix sets for the reward-source mechanistic analysis (docs/INTERP_PLAN.md). CPU only.

A1 post-ack : the conversation up to the tool result of a SUCCESS-acked issue_refund (no read since). Label: did it
              land? Unobservable on F1/F3, so a probe must be at chance (observation-equivalence control).
A2 post-read: up to the result of the first model read of the order after a success-acked refund. Label:
              observable discrepancy (the read contradicts the ack).
M2 (A5)     : every turn prefix of observation-equivalent pairs (unbounded_late vs never_commit).
CD (A5)     : prefixes ending with a get_operation / list_operations result; labels C (observed lifecycle of the
              agent's own op) and D (the oracle's appropriate next action).

    python -m actionproof.interp.rs_prefixes --v2 results/eval/a4/h11/*__val_T1x8.jsonl \
        --a5 results/eval/a5/init__neutral__none__a5_eval.jsonl --out results/interp/rs/prefixes.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

READS = ("query_ledger", "lookup_order", "get_refund_status")


def blocks(rec):
    """Yield (j, steps): conversation index j of each tool-result user message and the trace steps it reports."""
    conv, trace, t = rec["conversation"], rec["trace"], 0
    for j, m in enumerate(conv):
        if m["role"] == "user" and j >= 2 and "Tool result for " in m["content"]:
            k = m["content"].count("Tool result for ")
            yield j, trace[t:t + k]
            t += k


def discrepancy(step) -> int | None:
    o = step.get("obs") if isinstance(step.get("obs"), dict) else {}
    if step["tool"] == "query_ledger":
        return int(not o.get("refunds"))
    if step["tool"] == "lookup_order" and isinstance(o.get("order"), dict):
        return int(o["order"].get("status") != "refunded")
    if step["tool"] == "get_refund_status":
        return int(o.get("refund_status") != "completed")
    return None


def v2_prefixes(rec, source: str) -> list[dict]:
    att = {a["action_idx"]: a for a in rec.get("attempts", [])}
    out, acked_order = [], None
    for j, steps in blocks(rec):
        if any(s["tool"] == "submit_report" for s in steps):
            break
        for s in steps:
            args = s.get("args") if isinstance(s.get("args"), dict) else {}
            if s["tool"] == "issue_refund" and isinstance(s.get("obs"), dict) and s["obs"].get("status") == "success":
                a = att.get(s["idx"])
                if a is not None and acked_order is None:
                    landed = int(a["committed"] and a["fault"] != "F3_partial")
                    out.append({"kind": "A1", "prefix": j + 1, "label": 1 - landed, "fault": a["fault"] or "none",
                                "order": args.get("order_id")})
                    acked_order = args.get("order_id")
            elif acked_order and s["tool"] in READS and args.get("order_id") == acked_order:
                d = discrepancy(s)
                if d is not None:
                    out.append({"kind": "A2", "prefix": j + 1, "label": d, "read": s["tool"],
                                "fault": "+".join(rec.get("faults") or ["benign"]), "order": acked_order})
                acked_order = "__done__"  # first read only
    for e in out:
        e.update(task_id=rec["task_id"], source=source, conv=rec["conversation"][:e["prefix"]])
    return out


def a5_prefixes(rec) -> list[dict]:
    a5 = rec["a5"]
    out = []
    own = set()
    for j, steps in blocks(rec):
        for s in steps:
            o = s.get("obs") if isinstance(s.get("obs"), dict) else {}
            if s["tool"] == "issue_refund":
                x = o.get("op_id") or o.get("request_id")
                if x:
                    own.add(x)
            views = [o.get("operation")] if s["tool"] == "get_operation" else \
                o.get("operations", []) if s["tool"] == "list_operations" else []
            mine = [v for v in views if isinstance(v, dict) and v.get("op_id") in own]
            if mine:
                st, bound = mine[-1]["status"], mine[-1].get("settle_bound_s")
                d = {"committed": "proceed", "failed": "retry"}.get(st, "wait" if bound else "report_uncertain")
                out.append({"kind": "CD", "prefix": j + 1, "C": st, "D": d})
        if a5.get("pair_id", "") and str(a5.get("pair_id")).startswith("inflight"):
            out.append({"kind": "M2", "prefix": j + 1, "label": int(a5["scenario"] == "unbounded_late"),
                        "pair_id": a5["pair_id"], "turn": j})
    for e in out:
        e.update(task_id=rec["task_id"], family=a5["scenario"], source="a5_init", conv=rec["conversation"][:e["prefix"]])
    return out


def key(conv) -> str:
    return hashlib.sha1(json.dumps(conv[1:], sort_keys=True).encode()).hexdigest()


def balance(rows, rng):
    pos = [r for r in rows if r["label"] == 1]
    neg = [r for r in rows if r["label"] == 0]
    n = min(len(pos), len(neg))
    return rng.sample(pos, n) + rng.sample(neg, n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", nargs="+", required=True)
    ap.add_argument("--a5", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap", type=int, default=600, help="max balanced prefixes per kind (A1, A2)")
    a = ap.parse_args()
    rng = random.Random(0)
    seen, rows = set(), {"A1": [], "A2": []}
    for f in a.v2:
        src = Path(f).name.split("__")[0]
        for l in open(f):
            for e in v2_prefixes(json.loads(l), src):
                k = key(e["conv"])
                if k not in seen:
                    seen.add(k)
                    rows[e["kind"]].append(e)
    out = []
    for kind in ("A1", "A2"):
        b = balance(rows[kind], rng)
        rng.shuffle(b)
        out += b[: a.cap]
    a5 = [e for l in open(a.a5) for e in a5_prefixes(json.loads(l))]
    out += a5
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for i, e in enumerate(out):
            f.write(json.dumps({"id": i, **e}) + "\n")
    c = Counter((e["kind"], e.get("label", e.get("C"))) for e in out)
    print(json.dumps({"rows": len(out), "counts": {f"{k}:{v}": n for (k, v), n in sorted(c.items(), key=str)},
                      "A2_by_fault": dict(Counter(f"{e['fault']}|{e['label']}" for e in out if e["kind"] == "A2").most_common(8)),
                      "A1_by_fault": dict(Counter(f"{e['fault']}|{e['label']}" for e in out if e["kind"] == "A1"))}, default=str))


if __name__ == "__main__":
    main()
