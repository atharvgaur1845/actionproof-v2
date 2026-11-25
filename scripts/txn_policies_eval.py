"""Scripted TXN baselines on a split, on CPU (docs/txn/TXN_SPEC.md §9): B0-B4 and the B4 ablations A1-A6.

Each policy runs on the canonical TxnEnv (the alias/schema surfaces only rename what a model sees; the scripted
policies call canonical names, so their OOD alias/schema rows measure the semantics, not the renaming). Output uses the
model layout, so scripts/txn_stats.py reads policies and models alike: results/txn/<policy>__<split>.jsonl and
.summary.json. Existing files are never overwritten.

    python scripts/txn_policies_eval.py --split txn_dev [--policies B0,B1,B2,B3,B4] [--table]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from actionproof.bench.generator_proc import DATA, load_tasks  # noqa: E402
from actionproof.txn.policies import POLICIES, run_policy  # noqa: E402
from actionproof.txn.truth import breakdown, summarise  # noqa: E402

COLS = ("VTC", "STR", "FCR", "DMR_all", "DMR_dup_possible", "URR_all", "FCompR_all", "PRS", "CER", "unverified_claim")


def table(summaries: dict[str, dict]) -> str:
    head = "| policy | " + " | ".join(COLS) + " | calls | inv. viol. |"
    lines = [head, "|" + "---|" * (len(COLS) + 3)]
    for name, s in summaries.items():
        a = s["txn"]["all"]
        cells = []
        for c in COLS:
            m = a[c]
            cells.append("n/a" if m["rate"] is None else f"{m['rate']:.3f} ({m['num']}/{m['den']})")
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {a['cost']['calls']:.1f} | {s['episodes_with_violation']} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--policies", default=",".join(POLICIES))
    ap.add_argument("--n-tasks", type=int, default=None)
    ap.add_argument("--out-dir", default=str(ROOT / "results/txn"))
    ap.add_argument("--table", action="store_true", help="print a markdown table of the main metrics")
    a = ap.parse_args()
    tasks = load_tasks(DATA / f"{a.split}.jsonl")[: a.n_tasks]
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for name in a.policies.split(","):
        out = out_dir / f"{name}__{a.split}.jsonl"
        if out.exists():
            raise SystemExit(f"{out} exists")
        rows = [run_policy(name, t) for t in tasks]
        for r in rows:
            r["tag"] = name
        out.write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))
        sc = [r["txn"] for r in rows]
        inv: dict[str, int] = {}
        for r in rows:
            for v in r["invariants"]:
                inv[v["code"]] = inv.get(v["code"], 0) + 1
        s = {"tag": name, "split": a.split, "n_tasks": len(tasks), "policy": True,
             "txn": {"all": summarise(sc), "by_group": breakdown(sc, "group"), "by_class": breakdown(sc, "fault_class"),
                     "by_regime": breakdown(sc, "regime")},
             "invariant_violations": inv, "episodes_with_violation": sum(bool(r["invariants"]) for r in rows)}
        out.with_suffix(".summary.json").write_text(json.dumps(s, indent=2))
        summaries[name] = s
        print(json.dumps({"policy": name, "STR": s["txn"]["all"]["STR"]["rate"], "VTC": s["txn"]["all"]["VTC"]["rate"]}))
    if a.table:
        print(table(summaries))


if __name__ == "__main__":
    main()
