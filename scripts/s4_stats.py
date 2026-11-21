"""S4-EVAL statistics (PREREGISTRATION_S4_EVAL.md as amended by PREREGISTRATION_S4_EVAL_ADDENDUM1.md).

Per split and per seed s: RSC s - A12 R_V s on the same tasks (paired task bootstrap, 10,000, seed 0). Across seeds:
number positive, seed-averaged task bootstrap (one task draw shared by all seeds), the one-sided unpaired seed
permutation p over all C(10, 5) = 252 assignments (decision statistic; Holm across G1-G4), the exact sign-flip p
(descriptive), a two-level bootstrap (seeds, then tasks; descriptive) and the fraction of tasks on which two seeds
make identical tool calls. Secondary: language cost per model, RSC - class-matched control, per-class contributions.
    python scripts/s4_stats.py                       # results/s4/eval -> results/s4/s4_summary.json
    python scripts/s4_stats.py --dir results/s3/conf --splits s3_conf_rec --out <tmp>   # self-test on S3 data
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("a14_stats", ROOT / "scripts/a14_stats.py")
a14 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(a14)
B = 10_000
FAMILY = {"G1": ["s4_api"], "G2": ["s4_lang_hi", "s4_lang_hinglish"], "G3": ["s4_wf4"], "G4": ["s4_combo"]}
ALL_SPLITS = ["s4_lang_en", "s4_lang_hi", "s4_lang_hinglish", "s4_lang_para", "s4_api", "s4_wf4", "s4_combo"]
METRICS = ("vsc", "false_success", "violation", "duplicate", "calls")
CAND, BASE, MATCHED = "s3_rv_mix", "a12_rv", "s3_matched"


def load(tag: str, split: str, d: str) -> dict[str, dict] | None:
    p = ROOT / d / f"{tag}__{split}.jsonl"
    if not p.exists():
        return None
    out = {}
    for r in map(json.loads, p.open()):
        calls = tuple((e["tool"], json.dumps({k: v for k, v in (e.get("args") or {}).items() if k != "summary"},
                                             sort_keys=True)) for e in r["trace"] if e.get("by") == "agent")
        out[r["task_id"]] = {"vsc": float(r["score"]["vsc"]), "false_success": float(r["score"]["false_success"]),
                             "violation": float(r["violation"]), "duplicate": float(r["score"]["duplicate_irreversible"]),
                             "calls": float(r["policy_calls"]), "cls": r.get("fault_class") or "none", "seq": calls}
    return out


def mean(d: dict, k: str) -> float:
    return sum(v[k] for v in d.values()) / len(d)


def unpaired_perm(a: list[float], b: list[float]) -> float:
    """One-sided p for mean(a) > mean(b) over every relabelling of the pooled per-seed values."""
    pool, n = a + b, len(a)
    obs = sum(a) / n - sum(b) / len(b)
    hits = tot = 0
    for idx in itertools.combinations(range(len(pool)), n):
        s = set(idx)
        x = [pool[i] for i in idx]
        y = [pool[i] for i in range(len(pool)) if i not in s]
        tot += 1
        hits += sum(x) / n - sum(y) / len(y) >= obs - 1e-12
    return hits / tot


def holm(ps: dict[str, float]) -> dict[str, float]:
    order = sorted(ps, key=ps.get)
    out, run = {}, 0.0
    for i, g in enumerate(order):
        run = max(run, min(1.0, (len(order) - i) * ps[g]))
        out[g] = run
    return out


def contrast(cand: list[dict], base: list[dict], k: str, rng: random.Random, paired: bool = True) -> dict:
    """cand[i], base[i]: per-seed task dicts. paired=True pairs by seed index (same count)."""
    tasks = sorted(set.intersection(*[set(x) for x in cand + base]))
    per_seed = []
    if paired:
        for x, y in zip(cand, base):
            per_seed.append(a14.boot({t: x[t][k] - y[t][k] for t in tasks}, tasks, rng))
    cm = [sum(x[t][k] for t in tasks) / len(tasks) for x in cand]
    bm = [sum(y[t][k] for t in tasks) / len(tasks) for y in base]

    def avg(ts):
        return (sum(sum(x[t][k] for t in ts) / len(ts) for x in cand) / len(cand)
                - sum(sum(y[t][k] for t in ts) / len(ts) for y in base) / len(base))

    reps = sorted(avg([tasks[rng.randrange(len(tasks))] for _ in tasks]) for _ in range(B))
    two = []
    for _ in range(2000):
        xs = [cand[rng.randrange(len(cand))] for _ in cand]
        ys = [base[rng.randrange(len(base))] for _ in base]
        ts = [tasks[rng.randrange(len(tasks))] for _ in tasks]
        two.append(sum(sum(x[t][k] for t in ts) for x in xs) / len(ts) / len(xs)
                   - sum(sum(y[t][k] for t in ts) for y in ys) / len(ts) / len(ys))
    two.sort()
    out = {"n_tasks": len(tasks), "cand_per_seed": cm, "base_per_seed": bm,
           "seed_avg": {"est": avg(tasks), "ci95": [reps[int(0.025 * B)], reps[int(0.975 * B) - 1]]},
           "unpaired_perm_p_one_sided": unpaired_perm(cm, bm),
           "two_level_ci95": [two[int(0.025 * len(two))], two[int(0.975 * len(two)) - 1]]}
    if paired:
        ds = [e["est"] for e in per_seed]
        out.update({"per_seed": ds, "per_seed_ci95": [e["ci95"] for e in per_seed],
                    "n_positive": sum(d > 0 for d in ds), "sign_flip_p_one_sided": a14.sign_flip(ds, False)})
    return out


def by_class(cand: list[dict], base: list[dict]) -> dict:
    """Each class's contribution to the seed-averaged VSC difference, in units of the whole split."""
    tasks = sorted(set.intersection(*[set(x) for x in cand + base]))
    cls = sorted({cand[0][t]["cls"] for t in tasks})
    out = {}
    for c in cls:
        ts = [t for t in tasks if cand[0][t]["cls"] == c]
        out[c] = {"n": len(ts), "contribution": (sum(sum(x[t]["vsc"] for t in ts) for x in cand) / len(cand)
                                                 - sum(sum(y[t]["vsc"] for t in ts) for y in base) / len(base))
                  / len(tasks)}
    return out


def identical(models: list[dict]) -> list[float]:
    out = []
    for x, y in itertools.combinations(models, 2):
        ts = sorted(set(x) & set(y))
        out.append(sum(x[t]["seq"] == y[t]["seq"] for t in ts) / len(ts))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/s4/eval")
    ap.add_argument("--out", default="results/s4/s4_summary.json")
    ap.add_argument("--splits", default=None, help="comma list; default: the S4-EVAL splits and family")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--matched-seeds", default="0,1,2")
    a = ap.parse_args()
    rng = random.Random(0)
    seeds, mseeds = a.seeds.split(","), a.matched_seeds.split(",")
    splits = a.splits.split(",") if a.splits else ALL_SPLITS
    family = FAMILY if not a.splits else {f"T{i}": [s] for i, s in enumerate(splits)}
    res: dict = {"dir": a.dir, "splits": {}, "family": {}, "missing": []}
    data: dict = {}
    for sp in splits:
        tags = ["warm_start"] + [f"{p}_s{s}" for s in seeds for p in (CAND, BASE)] + [f"{MATCHED}_s{s}" for s in mseeds]
        data[sp] = {t: load(t, sp, a.dir) for t in tags}
        res["missing"] += [f"{t}__{sp}" for t, v in data[sp].items() if v is None]
        D = data[sp]
        S: dict = {"models": {t: {k: mean(d, k) for k in METRICS} for t, d in D.items() if d}}
        cand = [D.get(f"{CAND}_s{s}") for s in seeds]
        base = [D.get(f"{BASE}_s{s}") for s in seeds]
        if all(cand) and all(base):
            S["rsc_minus_rv"] = {k: contrast(cand, base, k, rng) for k in METRICS}
            S["rsc_minus_rv_by_class"] = by_class(cand, base)
            S["identical_calls"] = {"rsc_pairs": identical(cand), "rv_pairs": identical(base)}
        mt = [D.get(f"{MATCHED}_s{s}") for s in mseeds]
        cand_m = [D.get(f"{CAND}_s{s}") for s in mseeds]
        if all(mt) and all(cand_m):
            S["rsc_minus_matched"] = {k: contrast(cand_m, mt, k, rng) for k in ("vsc", "violation")}
            S["rsc_minus_matched"]["vsc_rsc5_vs_matched"] = contrast(cand, mt, "vsc", rng, paired=False) \
                if all(cand) else None
        res["splits"][sp] = S
    if not a.splits and all(data.get("s4_lang_en", {}).values()):
        cost = {}
        for sp in ("s4_lang_hi", "s4_lang_hinglish", "s4_lang_para"):
            for t, d in data[sp].items():
                en = data["s4_lang_en"][t]
                ts = sorted(set(d) & set(en))
                cost[f"{t}:{sp}"] = a14.boot({x: en[x]["vsc"] - d[x]["vsc"] for x in ts}, ts, rng)
        res["language_cost_vsc"] = cost
    ps, ok = {}, {}
    for g, sps in family.items():
        rs = [res["splits"][sp].get("rsc_minus_rv", {}).get("vsc") for sp in sps]
        if not all(rs):
            continue
        ps[g] = max(r["unpaired_perm_p_one_sided"] for r in rs)
        ok[g] = all(r["n_positive"] >= 4 and r["seed_avg"]["ci95"][0] > 0 for r in rs)
    adj = holm(ps) if ps else {}
    for g in ps:
        res["family"][g] = {"splits": family[g], "p_unpaired": ps[g], "p_holm": adj[g],
                            "seeds_and_ci_rule": ok[g], "supported": ok[g] and adj[g] < 0.05}
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    for sp, S in res["splits"].items():
        v = S.get("rsc_minus_rv", {}).get("vsc")
        if v:
            print(f"{sp:18s} RSC-R_V {v['seed_avg']['est']:+.3f} {[round(x, 3) for x in v['seed_avg']['ci95']]} "
                  f"pos {v['n_positive']}/5 p_unpaired {v['unpaired_perm_p_one_sided']:.4f} "
                  f"sign-flip {v['sign_flip_p_one_sided']:.4f} two-level {[round(x, 3) for x in v['two_level_ci95']]}")
    for g, r in res["family"].items():
        print(g, r)
    print("missing:", len(res["missing"]))


if __name__ == "__main__":
    main()
