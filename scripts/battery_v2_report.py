"""Report for task 416a: registered battery (v1, lone '"' close) vs corrected battery (v2, '",' close, vLLM) vs the
greedy next action on the same prefix (the behaviour the readout is meant to predict).

Per model x prefix type x schema: mean target probability under v1 and v2, the greedy rate of that target, and the
share of prefixes where each readout's argmax equals the greedy action. A cell is citable only if v2 agrees with
greedy on >= 90% of its prefixes.

    python scripts/battery_v2_report.py results/a7/battery_v2_scores.jsonl results/a7/battery_v2_greedy.jsonl \
        results/a7/forced_prefix_all.jsonl
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

TARGETS = {"a": ("status", "failed"), "b": ("status", "failed"), "c": ("tool", "query_ledger"),
           "d": ("tool", "query_ledger"), "e": ("tool", "apply_store_credit")}


def load(p):
    return {(r["model"], r["id"]): r for r in map(json.loads, open(p))}


def main() -> None:
    v2, gr, v1 = load(sys.argv[1]), load(sys.argv[2]), load(sys.argv[3]) if len(sys.argv) > 3 else {}
    v2 = {k: {**r, **gr[k]} for k, r in v2.items() if k in gr}
    cells = defaultdict(list)
    for (m, i), r in v2.items():
        cells[m, r["type"], r["schema"]].append((r, v1.get((m, i))))
    models = list(dict.fromkeys(m for m, _ in v2))
    print("| model | type | schema | n | target | P v1 | P v2 | greedy rate | argmax=greedy v1 | argmax=greedy v2 |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for m in models:
        for t in "abcde":
            for s in ("a4", "plain"):
                rows = cells.get((m, t, s))
                if not rows:
                    continue
                key, tgt = TARGETS[t]
                g = [(r["greedy_status"] if key == "status" else r["greedy_tool"]) for r, _ in rows]
                # status readouts compare against the greedy report status only where greedy reported
                n = len(rows)
                p2 = sum(r[key][tgt] for r, _ in rows) / n
                have1 = [o for _, o in rows if o and key in o]
                p1 = sum(o[key][tgt] for o in have1) / len(have1) if have1 else float("nan")
                gr = sum(x == tgt for x in g) / n
                ag2 = sum(max(r[key], key=r[key].get) == x for (r, _), x in zip(rows, g)) / n
                ag1 = (sum(max(o[key], key=o[key].get) == x for (_, o), x in zip(rows, g) if o and key in o)
                       / len(have1)) if have1 else float("nan")
                flag = "" if ag2 >= 0.9 else " ⚠"
                print(f"| {m} | {t} | {s} | {n} | {tgt} | {p1:.3f} | {p2:.3f} | {gr:.2f} | {ag1:.2f} | {ag2:.2f}{flag} |")
    print("\n⚠ = v2 argmax agrees with the greedy action on < 90% of prefixes: not citable as a behavioural readout.")
    print("Status readouts: greedy 'status' is None when the greedy action is not submit_report (counts as disagreement).")


if __name__ == "__main__":
    main()
