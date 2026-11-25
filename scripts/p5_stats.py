"""P5 registered analysis (PREREGISTRATION_P5.md), written before any P5 output. Run once.

For each base model m: Dw_m - base_m on the frozen P1 test (450 tasks), sampled VSC, one model per arm (one SFT seed).
Paired task bootstrap, B = 10,000, seed 20261126, one-sided p = P(delta* <= 0); Holm over the four models.
Supported for model m if its Holm-adjusted p < 0.05. Descriptive: Dw_m vs Dw (Tiny Aya Global) and per split.

    python scripts/p5_stats.py      # -> results/p5/p5_summary.json
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TEST = ("p1_test_full", "p1_test_pp", "p1_test_rec", "p1_test_wf4", "p1_test_combo")
MODELS = {"fire": ("results/p5/base_fire", "results/p5/dw_fire"), "water": ("results/p5/base_water", "results/p5/dw_water"),
          "earth": ("results/p5/base_earth", "results/p5/dw_earth"), "expanse8b": ("results/b1/aya8b", "results/p5/dw_expanse8b")}
REF = {"global_base": "results/b1/tag_base", "global_dw": "results/p1/test/Dw"}


def scores(prefix: str) -> dict:
    out = {}
    for sp in TEST:
        by = defaultdict(list)
        for r in map(json.loads, (ROOT / f"{prefix}__{sp}.jsonl").read_text().splitlines()):
            if r["sample"] >= 0:
                by[r["task_id"]].append(r["vsc"])
        out.update({(sp, t): float(np.mean(v)) for t, v in by.items()})
    return out


def main() -> None:
    rng = np.random.default_rng(20261126)
    res, ps = {}, {}
    for m, (b, d) in MODELS.items():
        B, D = scores(b), scores(d)
        keys = sorted(B)
        assert sorted(D) == keys, m
        diff = np.array([D[k] - B[k] for k in keys])
        bs = diff[rng.integers(0, len(diff), size=(10_000, len(diff)))].mean(1)
        ps[m] = float((np.sum(bs <= 0) + 1) / (len(bs) + 1))
        res[m] = {"base": float(np.mean(list(B.values()))), "dw": float(np.mean(list(D.values()))), "delta": float(diff.mean()),
                  "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))], "p_one_sided": ps[m], "n_tasks": len(keys),
                  "per_split": {sp: {"base": float(np.mean([B[k] for k in keys if k[0] == sp])),
                                     "dw": float(np.mean([D[k] for k in keys if k[0] == sp]))} for sp in TEST}}
    order, run, m_ = sorted(ps, key=ps.get), 0.0, len(ps)
    for i, m in enumerate(order):
        run = max(run, min(1.0, (m_ - i) * ps[m]))
        res[m]["p_holm"], res[m]["supported"] = run, bool(run < 0.05)
    res["reference_tiny_aya_global"] = {k: float(np.mean(list(scores(v).values()))) for k, v in REF.items()}
    (ROOT / "results/p5/p5_summary.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({m: {k: res[m][k] for k in ("base", "dw", "delta", "supported")} for m in MODELS}, indent=1))


if __name__ == "__main__":
    main()
