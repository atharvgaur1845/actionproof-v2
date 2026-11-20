"""Amendment 14 statistics (PREREGISTRATION_A14 §3–5), on results/eval/a14/<policy>__proc_cf__<rep>.jsonl.

Per policy: BCA (success family, primary for R_V seeds), signed S, timeout-family BCA, P correct, per-branch end-to-end
rates and first-action distributions; task bootstrap (10,000, seed 0) 95% CIs; interpretation category with delta 0.10.
Contrasts per seed: R_V − warm start and R_V − R_A on BCA, same task indices for both policies, 95% and 90% CIs,
exact McNemar on tasks whose both-branch correctness differs, category (different / equivalent / inconclusive).
Across seeds: direction consistency and exact sign-flip p (one- and two-sided). Thesis check (§4).
    python scripts/a14_stats.py --rep agent --out results/a14/a14_summary.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
B, DELTA = 10_000, 0.10
CORRECT = {"C": {"next_step"}, "L": {"next_step"}, "A": {"repay_same_key", "repay_new_key"},
           "T": {"repay_same_key", "repay_new_key"}, "P": {"refund", "escalate"}}
REPAY = {"repay_same_key", "repay_new_key"}


def load(policy: str, rep: str, split: str = "proc_cf") -> dict[str, dict[str, dict]] | None:
    p = ROOT / f"results/eval/a14/{policy}__{split}__{rep}.jsonl"
    if not p.exists():
        return None
    out: dict[str, dict[str, dict]] = {}
    for r in map(json.loads, p.open()):
        out.setdefault(r["task_id"], {})[r["branch"]] = r
    return out


def per_task(d: dict, fam: tuple[str, str]) -> dict[str, float]:
    return {t: sum(b[x]["first"]["action"] in CORRECT[x] for x in fam) / 2 for t, b in d.items() if all(x in b for x in fam)}


def s_task(d: dict) -> dict[str, float]:
    return {t: float(b["A"]["first"]["action"] in REPAY) - float(b["C"]["first"]["action"] in REPAY)
            for t, b in d.items() if "A" in b and "C" in b}


def boot(vals: dict[str, float], tasks: list[str], rng: random.Random, levels=(0.95,)) -> dict:
    est = sum(vals[t] for t in tasks) / len(tasks)
    reps = sorted(sum(vals[tasks[rng.randrange(len(tasks))]] for _ in tasks) / len(tasks) for _ in range(B))
    out = {"est": est}
    for lv in levels:
        a = (1 - lv) / 2
        out[f"ci{int(lv * 100)}"] = [reps[int(a * B)], reps[int((1 - a) * B) - 1]]
    return out


def policy_category(ci: list[float]) -> str:
    lo, hi = ci
    if lo > 0.5 + DELTA:
        return "evidence-conditioned"
    if hi < 0.5 - DELTA:
        return "inverted"
    if lo >= 0.5 - DELTA and hi <= 0.5 + DELTA:
        return "evidence-insensitive"
    return "inconclusive"


def contrast_category(ci95: list[float], ci90: list[float]) -> str:
    diff = not (ci95[0] <= 0 <= ci95[1])
    equiv = -DELTA < ci90[0] and ci90[1] < DELTA
    if diff and equiv:
        return "different but smaller than delta"
    return "different" if diff else "equivalent" if equiv else "inconclusive"


def mcnemar(x: dict[str, float], y: dict[str, float], tasks: list[str]) -> dict:
    b = sum(1 for t in tasks if x[t] == 1 and y[t] < 1)
    c = sum(1 for t in tasks if y[t] == 1 and x[t] < 1)
    n, k = b + c, min(b, c)
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"first_only": b, "second_only": c, "p": p}


def sign_flip(ds: list[float], two_sided: bool) -> float | None:
    if not ds:
        return None
    obs = sum(ds) / len(ds)
    flips = [sum(s * d for s, d in zip(sg, ds)) / len(ds) for sg in itertools.product((1, -1), repeat=len(ds))]
    hit = (lambda f: abs(f) >= abs(obs) - 1e-12) if two_sided else (lambda f: f >= obs - 1e-12)
    return sum(map(hit, flips)) / len(flips)


def describe(d: dict, rng: random.Random) -> dict:
    tasks = sorted(t for t, b in d.items() if "A" in b and "C" in b)
    bca = boot(per_task(d, ("A", "C")), tasks, rng)
    s = boot(s_task(d), tasks, rng)
    tf = per_task(d, ("T", "L"))
    res = {"n_tasks": len(tasks), "bca": {**bca, "category": policy_category(bca["ci95"])}, "signed_S": s,
           "bca_timeout": boot(tf, sorted(tf), rng) if tf else None, "by_branch": {}}
    for br in CORRECT:
        rs = [b[br] for b in d.values() if br in b]
        if rs:
            res["by_branch"][br] = {"n": len(rs), "correct": sum(r["first"]["action"] in CORRECT[br] for r in rs) / len(rs),
                                    "goal_met": sum(bool(r["goal_met"]) for r in rs) / len(rs),
                                    "duplicate": sum(bool(r["duplicate"]) for r in rs) / len(rs),
                                    "report_accurate": sum(bool(r["report_accurate"]) for r in rs) / len(rs),
                                    "first_actions": dict(Counter(r["first"]["action"] for r in rs))}
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", default="agent")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--out", default="results/a14/a14_summary.json")
    a = ap.parse_args()
    rng = random.Random(0)
    names = ["warm_start", "oracle", "naive", "fixed_retry"] + \
        [f"a12_proc_{arm}_s{s}" for s in a.seeds.split(",") for arm in ("rv", "ra")]
    data = {n: load(n, a.rep) for n in names}
    res: dict = {"rep": a.rep, "delta": DELTA, "missing": [n for n, d in data.items() if d is None], "policies": {},
                 "contrasts": {}, "across_seeds": {}}
    for n, d in data.items():
        if d is not None:
            res["policies"][n] = describe(d, rng)
    w = data["warm_start"]
    diffs: dict[str, list[float]] = {"rv_minus_warm": [], "rv_minus_ra": []}
    for s in a.seeds.split(","):
        rv, ra = data.get(f"a12_proc_rv_s{s}"), data.get(f"a12_proc_ra_s{s}")
        for label, other in (("rv_minus_warm", w), ("rv_minus_ra", ra)):
            if rv is None or other is None:
                continue
            x, y = per_task(rv, ("A", "C")), per_task(other, ("A", "C"))
            tasks = sorted(set(x) & set(y))
            dv = {t: x[t] - y[t] for t in tasks}
            ci = boot(dv, tasks, rng, levels=(0.95, 0.90))
            res["contrasts"][f"{label}[s{s}]"] = {**ci, "category": contrast_category(ci["ci95"], ci["ci90"]),
                                                  "mcnemar": mcnemar(x, y, tasks)}
            diffs[label].append(ci["est"])
    for label, ds in diffs.items():
        res["across_seeds"][label] = {"per_seed": ds, "same_sign": bool(ds) and (all(d > 0 for d in ds) or all(d < 0 for d in ds)),
                                      "sign_flip_p_one_sided": sign_flip(ds, False),
                                      "sign_flip_p_two_sided": sign_flip(ds, True)}
    trained = [v for k, v in res["policies"].items() if k.startswith("a12_")]
    res["thesis_check"] = {"abandon_seeing_is_not_knowing": bool(trained) and all(
        v["bca"]["ci95"][0] >= 0.85 and v["signed_S"]["ci95"][0] >= 0.70 for v in trained),
        "rule": "abandon if every trained policy has BCA lower bound >= 0.85 and signed S lower bound >= 0.70"}
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    for k, v in res["policies"].items():
        print(f"{k:22s} BCA {v['bca']['est']:.3f} {[round(x, 3) for x in v['bca']['ci95']]} {v['bca']['category']:22s} "
              f"S {v['signed_S']['est']:+.3f}")
    for k, v in res["contrasts"].items():
        print(f"{k:22s} dBCA {v['est']:+.3f} 95% {[round(x, 3) for x in v['ci95']]} 90% {[round(x, 3) for x in v['ci90']]} "
              f"{v['category']}")
    print("missing:", res["missing"])


if __name__ == "__main__":
    main()
