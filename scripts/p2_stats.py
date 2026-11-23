"""P2 registered analysis (PREREGISTRATION_P2.md §7). Frozen before any P2 output. Writes results/p2/p2_summary.json.

**Coverage gate.** Every required output is checked by `actionproof/reporting/coverage.py` against its split, with the
greedy run plus 4 samples per task. If any file is missing, incomplete or duplicated, the script writes only the
coverage report and exits with code 2; no statistic is computed.

**Units** (as in P1).
- A model's task score is its mean sampled VSC (4 samples, T = 1).
- An arm's task score is the mean over its three seed models.
- The test union is the five p1_test_* splits (450 tasks).

**Primary family (Holm over four).** For each contrast X − Y: a paired task bootstrap (B = 10,000, seed 20261125)
with a one-sided p = P(Δ* ≤ 0). X is supported iff Holm p < 0.05 and all three seed-indexed differences
X_sk − Y_sk are > 0.

| | Contrast |
|---|---|
| H1 | E − D: the ledger, given the same SFT targets and RL |
| H2 | E − A: the full candidate against the existing agent |
| H3 | B2 − A: the ledger at inference only |
| H4 | Ag − A: the deterministic guard on the existing agent |

**Secondary family (Holm over three, same rule).** Cg − C, Dg − D, Eg − E.

**Descriptive.**
- The shift splits (s4_lang_hi, s4_lang_hinglish, s4_api): each model's sampled VSC and its paired difference from
  A, with a 95% task-bootstrap interval. No test.
- References on the test union: warm start W, class-matched control M (3 seeds), oracle O (greedy only).
- Greedy VSC for every arm, reported separately from sampled.

    python scripts/p2_stats.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from actionproof.reporting.coverage import matrix  # noqa: E402

SEEDS = (0, 1, 2)
TEST = ("p1_test_full", "p1_test_pp", "p1_test_rec", "p1_test_wf4", "p1_test_combo")
SHIFT = ("s4_lang_hi", "s4_lang_hinglish", "s4_api")
DATA = ROOT / "data/bench/proc"
# arm -> directory holding its frozen test outputs (P1 arms are reused exactly as P1 ran them)
WHERE = {"A": "results/p1/test", "C": "results/p1/test", "D": "results/p1/test", "B2": "results/p2/test",
         "E": "results/p2/test", "Ag": "results/p2/test", "Cg": "results/p2/test", "Dg": "results/p2/test",
         "Eg": "results/p2/test", "M": "results/p2/ref"}
PRIMARY = (("H1", "E", "D"), ("H2", "E", "A"), ("H3", "B2", "A"), ("H4", "Ag", "A"))
SECONDARY = (("S1", "Cg", "C"), ("S2", "Dg", "D"), ("S3", "Eg", "E"))
SHIFT_MODELS = ("A", "C", "D", "E")


def path(arm: str, seed: int | None, split: str, phase_dir: str | None = None) -> Path:
    d = phase_dir or WHERE[arm]
    tag = arm if seed is None else f"{arm}_s{seed}"
    return ROOT / d / f"{tag}__{split}.jsonl"


def scores(p: Path, sampled: bool = True) -> dict:
    by: dict = {}
    for r in map(json.loads, p.read_text().splitlines()):
        if (r["sample"] >= 0) == sampled:
            by.setdefault(r["task_id"], []).append(r["vsc"])
    return {k: float(np.mean(v)) for k, v in by.items()}


def arm(a: str, splits, phase_dir=None, sampled=True):
    per = []
    for s in SEEDS:
        d = {}
        for sp in splits:
            d.update({(sp, k): v for k, v in scores(path(a, s, sp, phase_dir), sampled).items()})
        per.append(d)
    keys = sorted(set.intersection(*(set(p) for p in per)))
    return {k: float(np.mean([p[k] for p in per])) for k in keys}, per


def holm(ps):
    order = np.argsort(ps)
    adj, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def contrast(rng, X, Xs, Y, Ys):
    keys = sorted(set(X) & set(Y))
    d = np.array([X[k] - Y[k] for k in keys])
    boots = d[rng.integers(0, len(d), size=(10_000, len(d)))].mean(axis=1)
    seed_d = [float(np.mean([Xs[i][k] - Ys[i][k] for k in keys])) for i in range(len(SEEDS))]
    return {"delta": float(d.mean()), "ci95": [float(np.quantile(boots, .025)), float(np.quantile(boots, .975))],
            "p_one_sided": float((1 + np.sum(boots <= 0)) / (len(boots) + 1)), "seed_deltas": seed_d,
            "n_tasks": len(keys), "X_mean": float(np.mean([X[k] for k in keys])), "Y_mean": float(np.mean([Y[k] for k in keys]))}


def family(rng, rows, A):
    out = {}
    for h, x, y in rows:
        out[h] = {"contrast": f"{x} - {y}", **contrast(rng, *A[x], *A[y])}
    for (h, r), p in zip(out.items(), holm([r["p_one_sided"] for r in out.values()])):
        r["holm_p"] = p
        r["supported"] = bool(p < 0.05 and all(v > 0 for v in r["seed_deltas"]))
    return out


def main() -> None:
    arms = sorted({a for _, x, y in PRIMARY + SECONDARY for a in (x, y)})
    spec = [(path(a, s, sp), DATA / f"{sp}.jsonl", 4) for a in arms for s in SEEDS for sp in TEST]
    spec += [(path(a, s, sp, "results/p2/shift"), DATA / f"{sp}.jsonl", 4) for a in SHIFT_MODELS for s in SEEDS for sp in SHIFT]
    ok, reps = matrix(spec)
    out_dir = ROOT / "results/p2"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "p2_coverage.json").write_text(json.dumps(reps, indent=1, default=str))
    if not ok:
        bad = [r for r in reps if not r["ok"]]
        print(f"coverage: {len(bad)} of {len(reps)} required outputs incomplete; no statistics computed")
        for r in bad[:10]:
            print(" ", r["path"], r["problems"])
        sys.exit(2)
    rng = np.random.default_rng(20261125)
    A = {a: arm(a, TEST) for a in arms}
    res = {"primary": family(rng, PRIMARY, A), "secondary": family(rng, SECONDARY, A),
           "greedy_test_union": {a: float(np.mean(list(arm(a, TEST, sampled=False)[0].values()))) for a in arms},
           "sampled_test_union": {a: float(np.mean(list(A[a][0].values()))) for a in arms}, "shift": {}, "references": {}}
    S = {a: arm(a, SHIFT, "results/p2/shift") for a in SHIFT_MODELS}
    for sp in SHIFT:
        for a in SHIFT_MODELS:
            X = {k: v for k, v in S[a][0].items() if k[0] == sp}
            Y = {k: v for k, v in S["A"][0].items() if k[0] == sp}
            r = {"sampled_vsc": float(np.mean(list(X.values())))}
            if a != "A":
                d = np.array([X[k] - Y[k] for k in sorted(set(X) & set(Y))])
                b = d[rng.integers(0, len(d), size=(10_000, len(d)))].mean(axis=1)
                r.update(delta_vs_A=float(d.mean()), ci95=[float(np.quantile(b, .025)), float(np.quantile(b, .975))])
            res["shift"].setdefault(sp, {})[a] = r
    for name, seeds in (("W", [None]), ("M", list(SEEDS)), ("O", [None])):
        vals = []
        for s in seeds:
            fs = [path(name, s, sp, "results/p2/ref") for sp in TEST]
            if all(f.exists() for f in fs):
                d = {}
                for f in fs:
                    d.update(scores(f, sampled=name != "O"))
                vals.append(float(np.mean(list(d.values()))))
        res["references"][name] = {"mean_vsc": float(np.mean(vals)) if vals else None, "n_models": len(vals),
                                   "decoding": "greedy" if name == "O" else "sampled"}
    (out_dir / "p2_summary.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: {h: {kk: v[kk] for kk in ("contrast", "delta", "ci95", "holm_p", "supported")}
                          for h, v in res[k].items()} for k in ("primary", "secondary")}, indent=1))


if __name__ == "__main__":
    main()
