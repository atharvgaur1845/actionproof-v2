"""Per-fault breakdown of episode files: VSC, false completions, post-ack check rate.

post_ack_check = share of episodes in which a refund acknowledged as SUCCESS is followed by a
read of the same order (query_ledger / get_refund_status / lookup_order). It isolates the behaviour the 32B
teacher never shows (it verifies before writing and after errors, never after a success ack).

    python -m actionproof.eval.by_fault results/eval/r1/a.jsonl [b.jsonl ...]
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

from actionproof.env.tools_ecom import TOOL_META, VERIFICATION_TOOLS


ID_ARGS = ("order_id", "ticket_id", "sku")


def post_ack_check(trace: list[dict], op: str | None = "issue_refund") -> bool:
    """A success-acked write of `op` (any write if None) is followed by a read of the SAME entity."""
    acked: set = set()
    for s in trace:
        ids = {s["args"].get(k) for k in ID_ARGS} - {None} if isinstance(s.get("args"), dict) else set()
        if s["tool"] in VERIFICATION_TOOLS and ids & acked:
            return True
        if TOOL_META.get(s["tool"], (None, ""))[1] == "write" and (op is None or s["tool"] == op) \
                and isinstance(s["obs"], dict) and s["obs"].get("status") == "success":
            acked |= ids
    return False


def breakdown(path: str) -> dict[str, dict[str, float]]:
    g = defaultdict(lambda: defaultdict(float))
    for line in open(path):
        e = json.loads(line)
        v = e["verdict"]
        for k in ("+".join(e.get("faults") or []) or "benign", "ALL"):
            g[k]["n"] += 1
            g[k]["vsc"] += e["vsc"]
            g[k]["false_completion"] += v["G"] == 0 and v["reported"] == "completed"
            g[k]["no_report"] += v["reported"] == "none"
            g[k]["post_ack_check"] += post_ack_check(e["trace"])
    return {k: {"n": int(d["n"]), **{m: round(d[m] / d["n"], 3) for m in d if m != "n"}} for k, d in sorted(g.items())}


def main() -> None:
    for p in sys.argv[1:]:
        print(p.rsplit("/", 1)[-1])
        for k, d in breakdown(p).items():
            print(f"  {k:<14} " + "  ".join(f"{m}={v}" for m, v in d.items()))


if __name__ == "__main__":
    main()
