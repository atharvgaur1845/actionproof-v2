"""P4 registered analysis (PREREGISTRATION_P4.md §4), written before any P4 output. Run once.

Unit: a model's task score = mean sampled VSC over its 4 sampled episodes (sample >= 0). Arm score = mean over its
three seed models. Paired task bootstrap, B = 10,000, seed 20261126.
  H1 renamed tools   s4_api                         Dw_aug - Dw > 0: one-sided p < 0.05 after Holm (H1, H2) AND all
  H2 new workflow    s4_wf4 + p1_test_wf4 (160)     three same-index seed differences > 0
  H3 non-inferiority P1 test, 5 splits (450)       lower 95% bound of Dw_aug - Dw > -0.03
Dw seed 0 on the P1 test reuses results/p1/test/Dw__*.jsonl (evaluated once, NOTE_DWARM_EVAL.md).

    python scripts/p4_stats.py      # -> results/p4/p4_summary.json
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
P1T = ("p1_test_full", "p1_test_pp", "p1_test_rec", "p1_test_wf4", "p1_test_combo")
FAMILIES = {"H1": ("s4_api",), "H2": ("s4_wf4", "p1_test_wf4"), "H3": P1T}


def path(tag: str, split: str) -> Path:
    if tag == "Dw_s0" and split.startswith("p1_test_"):
        return ROOT / f"results/p1/test/Dw__{split}.jsonl"
    return ROOT / f"results/p4/{tag}__{split}.jsonl"


def scores(tag: str, splits) -> dict:
    out = {}
    for sp in splits:
        by = defaultdict(list)
        for r in map(json.loads, path(tag, sp).read_text().splitlines()):
            if r["sample"] >= 0:
                by[r["task_id"]].append(r["vsc"])
        out.update({(sp, t): float(np.mean(v)) for t, v in by.items()})
    return out


def holm(ps: dict) -> dict:
    order, m, adj, run = sorted(ps, key=ps.get), len(ps), {}, 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (m - i) * ps[k]))
        adj[k] = run
    return adj


def main() -> None:
    rng = np.random.default_rng(20261126)
    res, raw_p = {}, {}
    for h, splits in FAMILIES.items():
        D = [scores(f"Dw_s{s}", splits) for s in range(3)]
        A = [scores(f"Dwaug_s{s}", splits) for s in range(3)]
        keys = sorted(D[0])
        assert all(sorted(x) == keys for x in D + A), f"{h}: task sets differ"
        d = np.array([np.mean([x[k] for x in A]) - np.mean([x[k] for x in D]) for k in keys])
        bs = d[rng.integers(0, len(d), size=(10_000, len(d)))].mean(1)
        seed_d = [float(np.mean([A[i][k] - D[i][k] for k in keys])) for i in range(3)]
        res[h] = {"splits": list(splits), "n_tasks": len(keys), "Dw": float(np.mean([np.mean(list(x.values())) for x in D])),
                  "Dwaug": float(np.mean([np.mean(list(x.values())) for x in A])), "delta": float(d.mean()),
                  "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
                  "p_one_sided": float((np.sum(bs <= 0) + 1) / (len(bs) + 1)), "seed_deltas": seed_d}
        if h != "H3":
            raw_p[h] = res[h]["p_one_sided"]
    adj = holm(raw_p)
    for h in ("H1", "H2"):
        res[h]["p_holm"] = adj[h]
        res[h]["supported"] = bool(adj[h] < 0.05 and all(x > 0 for x in res[h]["seed_deltas"]))
    res["H3"]["supported"] = bool(res["H3"]["ci95"][0] > -0.03)
    out = ROOT / "results/p4/p4_summary.json"
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps({h: {k: res[h][k] for k in ("delta", "ci95", "supported")} for h in res}, indent=1))


if __name__ == "__main__":
    main()
