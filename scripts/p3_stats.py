"""P3 registered analysis (PREREGISTRATION_P3.md §4). Run once, after every Dn test file exists.

    python scripts/p3_stats.py      # writes results/p3/p3_summary.json
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("p1_test_full", "p1_test_pp", "p1_test_rec", "p1_test_wf4", "p1_test_combo")
SEEDS = (0, 1, 2)


def task_scores(tag: str) -> dict | None:
    out = {}
    for sp in SPLITS:
        f = ROOT / f"results/p1/test/{tag}__{sp}.jsonl"
        if not f.exists():
            return None
        by = defaultdict(list)
        for r in map(json.loads, f.read_text().splitlines()):
            if r["sample"] >= 0:
                by[r["task_id"]].append(r["vsc"])
        out.update({(sp, k): float(np.mean(v)) for k, v in by.items()})
    return out


def main() -> None:
    D = [task_scores(f"D_s{s}") for s in SEEDS]
    N = [task_scores(f"Dn_s{s}") for s in SEEDS]
    missing = [f"{a}_s{s}" for a, L in (("D", D), ("Dn", N)) for s, x in zip(SEEDS, L) if x is None]
    if missing:
        raise SystemExit(f"incomplete: {missing}")
    keys = sorted(D[0])
    assert all(sorted(x) == keys for x in D + N), "task sets differ"
    d_avg = np.array([np.mean([x[k] for x in D]) for k in keys])
    n_avg = np.array([np.mean([x[k] for x in N]) for k in keys])
    diff = n_avg - d_avg
    rng = np.random.default_rng(20261125)
    boots = diff[rng.integers(0, len(diff), size=(10_000, len(diff)))].mean(axis=1)
    p = float((np.sum(boots <= 0) + 1) / (len(boots) + 1))
    seed_d = [float(np.mean([N[i][k] - D[i][k] for k in keys])) for i in range(len(SEEDS))]
    res = {"H1": {"contrast": "Dn - D", "D_mean": float(d_avg.mean()), "Dn_mean": float(n_avg.mean()), "delta": float(diff.mean()),
                  "ci95": [float(np.quantile(boots, .025)), float(np.quantile(boots, .975))], "p_one_sided": p,
                  "seed_deltas": seed_d, "n_tasks": len(keys), "supported": bool(p < 0.05 and all(x > 0 for x in seed_d))}}
    per = {}
    for sp in SPLITS:
        ks = [k for k in keys if k[0] == sp]
        per[sp] = {"D": float(np.mean([np.mean([x[k] for x in D]) for k in ks])), "Dn": float(np.mean([np.mean([x[k] for x in N]) for k in ks]))}
    res["descriptive_per_split_sampled_vsc"] = per
    out = ROOT / "results/p3/p3_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res["H1"], indent=1))


if __name__ == "__main__":
    main()
