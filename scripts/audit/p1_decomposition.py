"""P1/P2 decomposition audit (research_audit/P1_DECOMPOSITION.md). Descriptive re-analysis, written 2026-11-25
after every P1/P2 output was known, so every contrast not registered in PREREGISTRATION_P1/P2 is EXPLORATORY.

Reads only the frozen-test episode files (results/p1/test, results/p2/test, results/p2/ref); writes
results/audit/p1_decomposition.json. Never touches a registered result file.

Units follow scripts/p1_stats.py exactly:
  model score on a task = mean VSC over its sampled episodes (sample >= 0; T = 1; 4 samples), greedy = sample -1;
  arm score = mean over its seed models; test union = 450 tasks over five splits.
Paired task bootstrap, B = 10,000, seed 20261125, two-sided 95% percentile interval and one-sided p = P(delta* <= 0).

    python scripts/audit/p1_decomposition.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TEST = ("p1_test_full", "p1_test_pp", "p1_test_rec", "p1_test_wf4", "p1_test_combo")
SEEDS = (0, 1, 2)
# arm -> (directory, tags)
ARMS = {
    "W": ("results/p2/ref", ["W"]),                      # warm start rft_proc (no recovery data, no RL)
    "M": ("results/p2/ref", [f"M_s{s}" for s in SEEDS]),  # class-matched full-episode R_V GRPO (s3_matched)
    "A": ("results/p1/test", [f"A_s{s}" for s in SEEDS]),  # RSC: warm start + recovery-start GRPO
    "B": ("results/p1/test", [f"B_s{s}" for s in SEEDS]),  # A + reconciliation line at inference
    "C": ("results/p1/test", [f"C_s{s}" for s in SEEDS]),  # A + recovery SFT (after RL)
    "D": ("results/p1/test", [f"D_s{s}" for s in SEEDS]),  # (orig + recovery demos) warm start + RSC recipe
    "E": ("results/p2/test", [f"E_s{s}" for s in SEEDS]),  # D with ledger-formatted demos and ledger rollouts
    "B2": ("results/p2/test", [f"B2_s{s}" for s in SEEDS]),
    "Ag": ("results/p2/test", [f"Ag_s{s}" for s in SEEDS]),
    "Cg": ("results/p2/test", [f"Cg_s{s}" for s in SEEDS]),
    "Dg": ("results/p2/test", [f"Dg_s{s}" for s in SEEDS]),
    "Eg": ("results/p2/test", [f"Eg_s{s}" for s in SEEDS]),
    "O": ("results/p2/ref", ["O"]),                      # scripted oracle, greedy only
}
CONTRASTS = [  # (X, Y, registered?)  delta = X - Y
    ("A", "W", "no"), ("M", "W", "no"), ("A", "M", "no"),
    ("B", "A", "P1-H1"), ("C", "A", "P1-H2"), ("D", "A", "P1-H3"), ("E", "A", "P2-H2"), ("E", "D", "P2-H1"),
    ("D", "C", "no"), ("C", "M", "no"), ("D", "M", "no"), ("O", "D", "no"),
]


def load(dirname: str, tag: str):
    per = {"sampled": {}, "greedy": {}}
    meta = {}
    for sp in TEST:
        f = ROOT / dirname / f"{tag}__{sp}.jsonl"
        if not f.exists():
            return None
        by = defaultdict(lambda: {"sampled": [], "greedy": []})
        for r in map(json.loads, f.read_text().splitlines()):
            by[r["task_id"]]["greedy" if r["sample"] < 0 else "sampled"].append(r["vsc"])
            meta[(sp, r["task_id"])] = (r.get("fault_class"), r.get("workflow"), r.get("start_type"))
        for t, d in by.items():
            for mode in ("sampled", "greedy"):
                if d[mode]:
                    per[mode][(sp, t)] = float(np.mean(d[mode]))
    return per, meta


def boot(x: np.ndarray, y: np.ndarray, rng, B=10_000):
    d = x - y
    idx = rng.integers(0, len(d), size=(B, len(d)))
    bs = d[idx].mean(1)
    return {"delta": float(d.mean()), "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
            "p_one_sided": float((bs <= 0).mean()), "n_tasks": int(len(d))}


def main():
    rng = np.random.default_rng(20261125)
    models, meta = {}, {}
    for arm, (dn, tags) in ARMS.items():
        ms = []
        for tag in tags:
            got = load(dn, tag)
            if got is None:
                print("missing", arm, tag)
                continue
            ms.append(got[0]); meta.update(got[1])
        if ms:
            models[arm] = ms
    keys = sorted(set.intersection(*(set(m[0]["greedy"]) | set(m[0]["sampled"]) for m in models.values())))
    out = {"n_tasks_union": len(keys), "arms": {}, "contrasts": {}, "per_split_contrasts": {}, "seed_pairs": {},
           "by_fault_class": {}}

    def arm_vec(arm, mode, ks):
        ms = models[arm]
        if mode == "sampled" and not ms[0]["sampled"]:
            return None
        return np.array([np.mean([m[mode][k] for m in ms]) for k in ks])

    for arm, ms in models.items():
        a = {"n_models": len(ms)}
        for mode in ("sampled", "greedy"):
            if not ms[0][mode]:
                continue
            a[mode] = {"union": float(arm_vec(arm, mode, keys).mean()),
                       "per_split": {sp: float(arm_vec(arm, mode, [k for k in keys if k[0] == sp]).mean()) for sp in TEST},
                       "per_seed_union": [float(np.mean([m[mode][k] for k in keys])) for m in ms]}
        out["arms"][arm] = a

    for x, y, reg in CONTRASTS:
        mode = "greedy" if "O" in (x, y) else "sampled"
        if x not in models or y not in models:
            continue
        name = f"{x}-{y}"
        r = boot(arm_vec(x, mode, keys), arm_vec(y, mode, keys), rng)
        r.update({"registered": reg, "decoding": mode})
        out["contrasts"][name] = r
        out["per_split_contrasts"][name] = {
            sp: boot(arm_vec(x, mode, [k for k in keys if k[0] == sp]), arm_vec(y, mode, [k for k in keys if k[0] == sp]), rng)
            for sp in TEST}
        mx, my = models[x], models[y]
        if len(mx) == 3 and len(my) == 3:
            same = [float(np.mean([mx[i][mode][k] - my[i][mode][k] for k in keys])) for i in range(3)]
            cross = [float(np.mean([mx[i][mode][k] - my[j][mode][k] for k in keys])) for i, j in product(range(3), range(3))]
            out["seed_pairs"][name] = {"same_index": same, "all_9_pairs_min": min(cross), "all_9_pairs_max": max(cross),
                                       "all_9_positive": int(sum(c > 0 for c in cross))}

    fcs = sorted({meta[k][0] or "none" for k in keys})
    for fc in fcs:
        ks = [k for k in keys if (meta[k][0] or "none") == fc]
        out["by_fault_class"][fc] = {"n": len(ks), **{arm: float(arm_vec(arm, "sampled", ks).mean())
                                                       for arm in ("W", "M", "A", "C", "D", "E") if arm in models}}
    starts = sorted({meta[k][2] or "full" for k in keys})
    out["by_start_type"] = {st: {"n": len([k for k in keys if (meta[k][2] or "full") == st]),
                                 **{arm: float(arm_vec(arm, "sampled", [k for k in keys if (meta[k][2] or "full") == st]).mean())
                                    for arm in ("W", "M", "A", "C", "D", "E") if arm in models}} for st in starts}
    dst = ROOT / "results/audit/p1_decomposition.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: {kk: round(vv["delta"], 3) for kk, vv in out["contrasts"].items()} if k == "contrasts" else None
                      for k in ("contrasts",)}, indent=1))
    print({a: round(v["sampled"]["union"], 3) if "sampled" in v else round(v["greedy"]["union"], 3) for a, v in out["arms"].items()})


if __name__ == "__main__":
    main()
