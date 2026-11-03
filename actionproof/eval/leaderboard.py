"""Aggregate run_eval outputs into the R0 leaderboard (spec §15, Tab 2 of the demo).

Reads every results/eval/**/*.jsonl produced by run_eval (one record per episode) and writes
results/leaderboard.json + results/leaderboard.md with Wilson 95% intervals. Rates are
computed from the episode records, never from the summaries, so every number is traceable.

    python -m actionproof.eval.leaderboard --glob 'results/eval/lb/*.jsonl'
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

from actionproof.eval.stats import wilson

METRICS = {  # name -> (numerator predicate, denominator predicate)
    "VSC": (lambda r: r["vsc"] == 1, lambda r: True),
    "FCR": (lambda r: r["verdict"]["F_over"] == 1, lambda r: r["verdict"]["G"] == 0),
    "IVR": (lambda r: r["verdict"]["I"] > 0, lambda r: True),
    "DSR": (lambda r: r["verdict"]["D"] > 0, lambda r: True),
    "OverEsc": (lambda r: r["verdict"]["reported"] == "escalated", lambda r: r["verdict"]["feasible"]),
    "Report": (lambda r: r.get("ended_by") == "report", lambda r: True),
}


def rows_from(paths: list[str]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for p in paths:
        if p.endswith(".summary.json"):
            continue
        for line in open(p):
            r = json.loads(line)
            key = (r["model"].split("/")[-1], (r.get("adapter") or "").rstrip("/").split("/")[-2:-1] or [""],
                   r["prompt"], r["split"])
            groups[(key[0], key[1][0] if key[1] else "", key[2], key[3])].append(r)
    table = []
    for (model, adapter, prompt, split), rs in sorted(groups.items()):
        row = {"model": model, "adapter": adapter, "prompt": prompt, "split": split, "n": len(rs),
               "verify_calls": round(sum(r["n_verify"] for r in rs) / len(rs), 2)}
        for name, (num, den) in METRICS.items():
            d = [r for r in rs if den(r)]
            p, lo, hi = wilson(sum(num(r) for r in d), len(d))
            row[name] = {"p": round(p, 4), "lo": round(lo, 4), "hi": round(hi, 4), "n": len(d)}
        table.append(row)
    return table


def to_markdown(table: list[dict]) -> str:
    cols = ["VSC", "FCR", "IVR", "DSR", "OverEsc", "Report"]
    out = ["| model | adapter | prompt | split | n | " + " | ".join(cols) + " | verify/ep |",
           "|" + "---|" * (6 + len(cols))]
    for r in table:
        cells = [f"{100*r[c]['p']:.1f} [{100*r[c]['lo']:.0f}–{100*r[c]['hi']:.0f}]" if r[c]["n"] else "–" for c in cols]
        out.append(f"| {r['model']} | {r['adapter'] or '–'} | {r['prompt']} | {r['split']} | {r['n']} | "
                   + " | ".join(cells) + f" | {r['verify_calls']} |")
    return "\n".join(out) + "\n\nRates in %, Wilson 95% CI in brackets. FCR is conditional on failure (G=0); " \
                            "OverEsc on feasible tasks.\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="results/eval/lb/*.jsonl")
    ap.add_argument("--out", default="results/leaderboard")
    a = ap.parse_args()
    table = rows_from(sorted(glob.glob(a.glob)))
    Path(a.out + ".json").write_text(json.dumps(table, indent=2))
    md = to_markdown(table)
    Path(a.out + ".md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
