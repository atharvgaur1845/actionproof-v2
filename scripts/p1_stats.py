"""P1 registered analysis (PREREGISTRATION_P1.md §7). Frozen before any P1 output. Writes results/p1/p1_summary.json.

**Units.**
- A model's score on a task is its sampled VSC: the mean over its 4 samples at T = 1.
- An arm's score on a task is the mean of that over its three seed models.
- The test union is p1_test_full, p1_test_pp, p1_test_rec, p1_test_wf4 and p1_test_combo (450 tasks).

**Primary (H1–H3).** For each X in {B, C, D}: Δ_X = mean over test tasks of (arm X − arm A).
- Paired task bootstrap, B = 10,000, seed 20261124; one-sided p = P(Δ* ≤ 0); Holm over the three contrasts.
- X is supported iff its Holm p < 0.05 and all three seed-paired differences (X_sk − A_sk on the union) are > 0.

**Development winner.** On the dev union (p1_dev_full + p1_dev_rec), the arm among B, C, D with the highest mean sampled
VSC, provided it beats A by at least 0.03; otherwise none (A stays the reference agent). This choice only labels the
demonstration agent. Every arm runs the frozen test once regardless.

**Descriptive.** Per split, greedy VSC, and the §6 behaviour metrics per arm.

    python scripts/p1_stats.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("A", "B", "C", "D")
SEEDS = (0, 1, 2)
TEST = ("p1_test_full", "p1_test_pp", "p1_test_rec", "p1_test_wf4", "p1_test_combo")
DEV = ("p1_dev_full", "p1_dev_rec")
METRICS = ("vsc", "false_success", "duplicate_unrefunded", "partial_pay_repaired", "new_key_share_on_repay",
           "redundant_reads_per_episode", "reads_per_episode", "calls_per_success")


def task_scores(phase: str, tag: str, splits: tuple[str, ...]) -> dict | None:
    out = {}
    for sp in splits:
        f = ROOT / f"results/p1/{phase}/{tag}__{sp}.jsonl"
        if not f.exists():
            return None
        by = defaultdict(list)
        for r in map(json.loads, f.read_text().splitlines()):
            if r["sample"] >= 0:
                by[r["task_id"]].append(r["vsc"])
        out.update({(sp, k): float(np.mean(v)) for k, v in by.items()})
    return out


def arm_scores(phase: str, arm: str, splits: tuple[str, ...]) -> tuple[dict, list[dict]] | None:
    per_seed = [task_scores(phase, f"{arm}_s{s}", splits) for s in SEEDS]
    if any(p is None for p in per_seed):
        return None
    keys = sorted(set.intersection(*(set(p) for p in per_seed)))
    return {k: float(np.mean([p[k] for p in per_seed])) for k in keys}, per_seed


def holm(ps: list[float]) -> list[float]:
    order = np.argsort(ps)
    adj, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def summaries(phase: str, arm: str, splits: tuple[str, ...]) -> dict:
    out = {}
    for sp in splits:
        rows = []
        for s in SEEDS:
            f = ROOT / f"results/p1/{phase}/{arm}_s{s}__{sp}.summary.json"
            if f.exists():
                rows.append(json.loads(f.read_text()))
        if rows:
            out[sp] = {mode: {m: (float(np.mean([r[mode][m] for r in rows if r[mode][m] is not None]))
                                  if any(r[mode][m] is not None for r in rows) else None) for m in METRICS}
                       for mode in ("greedy", "sampled")}
    return out


def main() -> None:
    rng = np.random.default_rng(20261124)
    res: dict = {"dev": {}, "test": {}, "descriptive": {}}
    dev = {a: arm_scores("dev", a, DEV) for a in ARMS}
    if all(dev.values()):
        means = {a: float(np.mean(list(dev[a][0].values()))) for a in ARMS}
        cand = max(("B", "C", "D"), key=lambda a: means[a])
        res["dev"] = {"mean_sampled_vsc": means, "winner": cand if means[cand] - means["A"] >= 0.03 else None}
    test = {a: arm_scores("test", a, TEST) for a in ARMS}
    if all(test.values()):
        A, A_seeds = test["A"]
        ps, rows = [], {}
        for x in ("B", "C", "D"):
            X, X_seeds = test[x]
            keys = sorted(set(A) & set(X))
            d = np.array([X[k] - A[k] for k in keys])
            boots = d[rng.integers(0, len(d), size=(10_000, len(d)))].mean(axis=1)
            seed_d = [float(np.mean([X_seeds[i][k] - A_seeds[i][k] for k in keys])) for i in range(len(SEEDS))]
            p = float((1 + np.sum(boots <= 0)) / (len(boots) + 1))
            ps.append(p)
            rows[x] = {"delta": float(d.mean()), "ci95": [float(np.quantile(boots, .025)), float(np.quantile(boots, .975))],
                       "p_one_sided": p, "seed_deltas": seed_d, "n_tasks": len(keys), "A_mean": float(np.mean([A[k] for k in keys])),
                       "X_mean": float(np.mean([X[k] for k in keys]))}
        for (x, r), p in zip(rows.items(), holm(ps)):
            r["holm_p"] = p
            r["supported"] = bool(p < 0.05 and all(v > 0 for v in r["seed_deltas"]))
        res["test"] = rows
    for a in ARMS:
        res["descriptive"][a] = {"test": summaries("test", a, TEST), "dev": summaries("dev", a, DEV)}
    (ROOT / "results/p1").mkdir(parents=True, exist_ok=True)
    (ROOT / "results/p1/p1_summary.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({"dev": res["dev"], "test": {k: {kk: v[kk] for kk in ("delta", "ci95", "holm_p", "supported")}
                                                  for k, v in res["test"].items()}}, indent=1))


if __name__ == "__main__":
    main()
