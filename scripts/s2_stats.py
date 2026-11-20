"""A12 statistics (PREREGISTRATION_A12 §8): per-seed paired task bootstrap and exact McNemar, between-seed spread,
exact paired sign-flip over seeds, and a two-level hierarchical bootstrap reported as descriptive only.

Inputs: confirmatory files `results/eval/a12/conf/a12_<arm>_s<seed>__neutral__<split>.jsonl` (one per frozen model).
Primary endpoints (Holm over 2): phantom read-back R_V - R_A, verified completion R_V - R_A.
  phantom read-back: ratio of sums over the split (attempts read back / phantom attempts), paired by task;
  its episode-level McNemar outcome is "every phantom attempt in the episode was read back", over episodes that had
  at least one phantom attempt in both arms.
    python scripts/s2_stats.py --split proc_conf --out results/s2/a12_summary.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
B = 10_000


def load(arm: str, seed: int, split: str) -> dict[str, dict] | None:
    p = ROOT / f"results/eval/a12/conf/a12_{arm}_s{seed}__neutral__{split}.jsonl"
    if not p.exists():
        return None
    return {r["task_id"]: r["score"] for r in map(json.loads, p.open())}


def ratio(scores: list[dict], key: str) -> float | None:
    n = sum(s[key][0] for s in scores)
    d = sum(s[key][1] for s in scores)
    return n / d if d else None


def metric(scores: list[dict], m: str) -> float | None:
    return ratio(scores, "phantom_readback") if m == "phantom_readback" else mean(s["vsc"] for s in scores)


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def per_seed(rv: dict, ra: dict, m: str, rng: random.Random) -> dict:
    tasks = sorted(set(rv) & set(ra))
    a, b = [rv[t] for t in tasks], [ra[t] for t in tasks]
    ma, mb = metric(a, m), metric(b, m)
    diff = None if ma is None or mb is None else ma - mb  # a metric with no denominator is missing, never 0
    boots = []
    for _ in range(B):
        idx = [rng.randrange(len(tasks)) for _ in tasks]
        x, y = metric([a[i] for i in idx], m), metric([b[i] for i in idx], m)
        if x is not None and y is not None:
            boots.append(x - y)
    boots.sort()
    if m == "phantom_readback":
        pairs = [(s1, s2) for s1, s2 in zip(a, b) if s1[m][1] and s2[m][1]]
        o1 = [int(s[m][0] == s[m][1]) for s, _ in pairs]
        o2 = [int(s[m][0] == s[m][1]) for _, s in pairs]
    else:
        o1, o2 = [s["vsc"] for s in a], [s["vsc"] for s in b]
    bb = sum(1 for x, y in zip(o1, o2) if x and not y)
    cc = sum(1 for x, y in zip(o1, o2) if y and not x)
    return {"n_tasks": len(tasks), "rv": metric(a, m), "ra": metric(b, m), "diff": diff,
            "ci95": [boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1]] if boots else None,
            "mcnemar": {"rv_only": bb, "ra_only": cc, "p": mcnemar_exact(bb, cc)}}


def sign_flip(ds: list[float], two_sided: bool = False) -> float:
    """Exact sign-flip test over seeds. Smallest attainable p: 1/2^k one-sided, 2/2^k two-sided (k seeds)."""
    obs = mean(ds)
    flips = [mean(s * d for s, d in zip(signs, ds)) for signs in itertools.product((1, -1), repeat=len(ds))]
    if two_sided:
        return sum(abs(f) >= abs(obs) - 1e-12 for f in flips) / len(flips)
    return sum(f >= obs - 1e-12 for f in flips) / len(flips)


def hierarchical(pairs: list[tuple[dict, dict]], m: str, rng: random.Random) -> list[float]:
    out = []
    for _ in range(B):
        seeds = [rng.randrange(len(pairs)) for _ in pairs]
        # tasks are crossed with seeds (every seed is evaluated on the same tasks), so one task draw per replicate is
        # shared by all resampled seeds; independent draws per seed would shrink the task variance by 1/S
        common = sorted(set.intersection(*(set(rv) & set(ra) for rv, ra in pairs)))
        idx = [common[rng.randrange(len(common))] for _ in common]
        ds = []
        for s in seeds:
            rv, ra = pairs[s]
            x, y = metric([rv[t] for t in idx], m), metric([ra[t] for t in idx], m)
            if x is not None and y is not None:
                ds.append(x - y)
        if ds:
            out.append(mean(ds))
    out.sort()
    return [out[int(0.025 * len(out))], out[int(0.975 * len(out)) - 1]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="proc_conf")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--out", default="results/s2/a12_summary.json")
    a = ap.parse_args()
    rng = random.Random(12)
    pairs, seeds = [], []
    for s in map(int, a.seeds.split(",")):
        rv, ra = load("proc_rv", s, a.split), load("proc_ra", s, a.split)
        if rv and ra:
            pairs.append((rv, ra))
            seeds.append(s)
    res: dict = {"split": a.split, "seeds": seeds, "status": "complete" if len(seeds) >= 3 else "pending (< 3 seeds)"}
    raw_p = {}
    for m in ("phantom_readback", "vsc"):
        ps = [per_seed(rv, ra, m, rng) for rv, ra in pairs]
        ds = [p["diff"] for p in ps if p["diff"] is not None]
        res[m] = {"per_seed": dict(zip(map(str, seeds), ps)),
                  "between_seed_sd": stdev(ds) if len(ds) > 1 else None,
                  "same_sign_all_seeds": bool(ds) and (all(d > 0 for d in ds) or all(d < 0 for d in ds)),
                  "sign_flip_p_one_sided": sign_flip(ds) if ds else None,
                  "sign_flip_p_two_sided": sign_flip(ds, two_sided=True) if ds else None,
                  "hierarchical_ci95_descriptive": hierarchical(pairs, m, rng) if pairs else None}
        raw_p[m] = res[m]["sign_flip_p_one_sided"]
    order = sorted((p, m) for m, p in raw_p.items() if p is not None)
    holm = {}
    for i, (p, m) in enumerate(order):
        holm[m] = min(1.0, max(p2 * (len(order) - j) for j, (p2, _) in enumerate(order[: i + 1])))
    res["holm_sign_flip"] = holm
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2)[:3000])


if __name__ == "__main__":
    main()
