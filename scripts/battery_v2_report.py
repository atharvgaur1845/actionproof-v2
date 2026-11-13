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
    print("| model | type | schema | n | target | P v1 | P v2 | greedy rate | next-tool argmax = greedy (v1 / v2) |"
          " report-status argmax = greedy status (v1 / v2; where greedy reports) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    flags = []
    for m in models:
        for t in "abcde":
            for s in ("a4", "plain"):
                rows = cells.get((m, t, s))
                if not rows:
                    continue
                key, tgt = TARGETS[t]
                n = len(rows)
                g = [(r["greedy_status"] if key == "status" else r["greedy_tool"]) for r, _ in rows]
                p2 = sum(r[key][tgt] for r, _ in rows) / n
                have1 = [o for _, o in rows if o and key in o]
                p1 = sum(o[key][tgt] for o in have1) / len(have1) if have1 else float("nan")
                gr = sum(x == tgt for x in g) / n

                def agree(readout, rs):
                    ok1 = [max(o[readout], key=o[readout].get) == want for (r, o), want in rs if o and readout in o]
                    ok2 = [max(r[readout], key=r[readout].get) == want for (r, _), want in rs]
                    return ok1, ok2

                tool_rows = [((r, o), r["greedy_tool"]) for r, o in rows if "tool" in r]
                st_rows = [((r, o), r["greedy_status"]) for r, o in rows if "status" in r and r["greedy_status"]]
                t1, t2 = agree("tool", tool_rows) if tool_rows else ([], [])
                s1, s2 = agree("status", st_rows) if st_rows else ([], [])
                fmt = lambda a, b: (f"{sum(a)}/{len(a)} / {sum(b)}/{len(b)}" if b else "– (greedy never reports)")
                bad = (t2 and sum(t2) < 0.9 * len(t2)) or (s2 and sum(s2) < 0.9 * len(s2))
                if bad:
                    flags.append(f"{m} {t}/{s}")
                print(f"| {m} | {t} | {s} | {n} | {tgt} | {p1:.3f} | {p2:.3f} | {gr:.2f} | "
                      f"{fmt(t1, t2) if tool_rows else '–'} | {fmt(s1, s2) if 'status' in rows[0][0] else '–'}"
                      f"{' ⚠' if bad else ''} |")
    print("\n⚠ = the corrected (v2) readout agrees with the greedy action on < 90% of the prefixes where they can be "
          "compared: not citable as a behavioural readout. A status readout is compared only where the greedy action "
          "is `submit_report`; elsewhere it is a counterfactual (\"if it reported now\") with no behavioural check.")
    print("\nFlagged cells: " + (", ".join(flags) if flags else "none"))


if __name__ == "__main__":
    main()
