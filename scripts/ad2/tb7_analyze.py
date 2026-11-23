"""V2-TB7 analysis as registered (V2TB7_IGC_ABLATION.md §3).

    python scripts/ad2/tb7_analyze.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "results/ad2/tabular"


def per_seed(rows, prefix):
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        p = r["id"].split("|")
        name, mult, seed = p[4], p[5], p[6]
        evs = list(r["eval"].values())
        acc[f"{name}/{mult}"][seed].append((mean(e["regret"] for e in evs), mean(e["safe_verify_rate"] or 0 for e in evs)))
    return {k: {s: (mean(x[0] for x in v), mean(x[1] for x in v)) for s, v in d.items()} for k, d in acc.items()}


def main():
    tb7 = [json.loads(l) for l in (D / "TB7.jsonl").read_text().splitlines() if l.strip()]
    tb2 = [json.loads(l) for l in (D / "TB2.jsonl").read_text().splitlines() if l.strip()]
    tb2 = [r for r in tb2 if r["id"].split("|")[4] == "nostd" and r["id"].split("|")[5] in ("x1", "x4")]
    a = per_seed(tb7, "TB7")
    b = per_seed(tb2, "TB2")
    out = {"table": {}, "predictions": {}}
    for k, d in {**a, **b}.items():
        out["table"][k] = {"regret": mean(x[0] for x in d.values()), "safe_verify": mean(x[1] for x in d.values()),
                           "regret_by_seed": {s: x[0] for s, x in sorted(d.items())}}
    for m in ("x1", "x4"):
        n = out["table"][f"nostd/{m}"]["regret"]
        full = out["table"][f"igc_full/{m}"]["regret"]
        v1 = out["table"][f"igc_v1/{m}"]["regret"]
        gain_full, gain_v1 = n - full, n - v1
        out["table"][f"share_of_gain_from_v1_{m}"] = gain_v1 / gain_full if gain_full > 0 else None
        out["predictions"][f"P7a_v1_retains_80pct_{m}"] = gain_full > 0 and gain_v1 >= 0.8 * gain_full
    (D / "TB7_summary.json").write_text(json.dumps(out, indent=1))
    for k, v in sorted(out["table"].items()):
        print(k, v if not isinstance(v, dict) else {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                                                     for kk, vv in v.items() if kk != "regret_by_seed"})
    print(json.dumps(out["predictions"], indent=1))


if __name__ == "__main__":
    main()
