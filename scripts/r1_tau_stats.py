"""R1-T analysis (PREREGISTRATION_R1_TAU.md §5). Frozen before any τ-bench output. Writes results/r1_tau/r1_tau_summary.json.

Per model: pass^1 (mean reward), τ-bench's pass^k for k = 1..trials (per task c successes of n trials: C(c,k)/C(n,k)),
overflow and error rates, mean messages. Contrasts (descriptive, no test): warm − base; {a12_rv, rsc} (seed-averaged)
− warm, on per-task pass^1, with a paired bootstrap over the 115 tasks (B = 10,000, seed 20261123).
"""
from __future__ import annotations

import json
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "results/r1_tau"
ARMS = {"base": ["base"], "warm": ["warm_start"], "a12_rv": [f"a12_rv_s{i}" for i in range(3)],
        "rsc": [f"rsc_s{i}" for i in range(3)]}


def per_task(tag: str) -> dict | None:
    f = D / f"{tag}.jsonl"
    if not f.exists():
        return None
    by: dict[int, list] = {}
    for r in map(json.loads, f.read_text().splitlines()):
        by.setdefault(r["task_index"], []).append(r)
    return by


def model_summary(by: dict) -> dict:
    rows = [r for v in by.values() for r in v]
    n = min(len(v) for v in by.values())
    pk = {k: float(np.mean([comb(sum(r["reward"] == 1.0 for r in v[:n]), k) / comb(n, k) for v in by.values()]))
          for k in range(1, n + 1)}
    return {"episodes": len(rows), "tasks": len(by), "trials": n, "pass1": float(np.mean([r["reward"] for r in rows])),
            "pass_hat_k": pk, "overflow_rate": float(np.mean([r["overflow"] for r in rows])),
            "error_rate": float(np.mean([bool(r["error"]) and not r["overflow"] for r in rows])),
            "mean_messages": float(np.mean([r["n_messages"] for r in rows]))}


def task_vec(tags: list[str]) -> np.ndarray | None:
    bys = [per_task(t) for t in tags]
    if any(b is None for b in bys):
        return None
    idx = sorted(set.intersection(*(set(b) for b in bys)))
    return np.array([np.mean([np.mean([r["reward"] for r in b[i]]) for b in bys]) for i in idx])


def main() -> None:
    rng = np.random.default_rng(20261123)
    out: dict = {"models": {}, "contrasts": {}}
    for f in sorted(D.glob("*.jsonl")):
        out["models"][f.stem] = model_summary(per_task(f.stem))
    for name, (a, b) in {"warm_vs_base": ("warm", "base"), "a12_rv_vs_warm": ("a12_rv", "warm"),
                         "rsc_vs_warm": ("rsc", "warm")}.items():
        x, y = task_vec(ARMS[a]), task_vec(ARMS[b])
        if x is None or y is None or len(x) != len(y):
            continue
        d = x - y
        boots = d[rng.integers(0, len(d), size=(10_000, len(d)))].mean(axis=1)
        out["contrasts"][name] = {"delta": float(d.mean()), "ci95": [float(np.quantile(boots, .025)),
                                                                    float(np.quantile(boots, .975))], "n_tasks": len(d)}
    (D / "r1_tau_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["contrasts"], indent=1))


if __name__ == "__main__":
    main()
