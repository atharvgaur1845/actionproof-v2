"""A15 statistics (PREREGISTRATION_A15 §3), on results/eval/a15/<tag>__proc_mech.jsonl (strict `correct` field).

Per policy: unseen-ratio accuracy (mean of Q1 and Q3 per task), trained-pattern accuracy (Q2), binary accuracy (mean of
Q0 and Q4), per-branch accuracy, the A14-rule unseen accuracy, with task bootstrap (10,000, seed 0) 95% CIs.
Per R_V seed: R_V - warm start and R_V - R_A (same seed) on unseen accuracy (95% and 90% CIs, exact McNemar),
pattern gap Q2 - unseen (paired), and the registered category. Across seeds: sign consistency and exact sign-flip p.
    python scripts/a15_stats.py --out results/a15/a15_summary.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("a14_stats", ROOT / "scripts/a14_stats.py")
a14 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(a14)
boot, mcnemar, sign_flip, DELTA = a14.boot, a14.mcnemar, a14.sign_flip, a14.DELTA

BRANCHES = ("Q0", "Q1", "Q2", "Q3", "Q4")
UNSEEN, TRAINED, BINARY = ("Q1", "Q3"), ("Q2",), ("Q0", "Q4")


DIR, SPLIT = "results/eval/a15", "proc_mech"


def load(tag: str) -> dict[str, dict[str, dict]] | None:
    p = ROOT / f"{DIR}/{tag}__{SPLIT}.jsonl"
    if not p.exists():
        return None
    out: dict[str, dict[str, dict]] = {}
    for r in map(json.loads, p.open()):
        out.setdefault(r["task_id"], {})[r["branch"]] = r
    return out


def per_task(d: dict, fam: tuple[str, ...], key: str = "correct") -> dict[str, float]:
    return {t: sum(bool(b[x][key]) for x in fam) / len(fam) for t, b in d.items() if all(x in b for x in fam)}


def describe(d: dict, rng: random.Random) -> dict:
    res = {}
    for name, fam, key in (("unseen", UNSEEN, "correct"), ("trained", TRAINED, "correct"), ("binary", BINARY, "correct"),
                           ("unseen_a14_rule", UNSEEN, "correct_a14_rule")):
        v = per_task(d, fam, key)
        res[name] = boot(v, sorted(v), rng)
    res["by_branch"] = {}
    for br in BRANCHES:
        rs = [b[br] for b in d.values() if br in b]
        res["by_branch"][br] = {"n": len(rs), "correct": sum(r["correct"] for r in rs) / len(rs),
                                "goal_met": sum(bool(r["goal_met"]) for r in rs) / len(rs),
                                "duplicate": sum(bool(r["duplicate"]) for r in rs) / len(rs),
                                "report_accurate": sum(bool(r["report_accurate"]) for r in rs) / len(rs),
                                "first_actions": dict(Counter(r["first"]["action"] for r in rs))}
    return res


def main() -> None:
    global DIR, SPLIT
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--out", default="results/a15/a15_summary.json")
    ap.add_argument("--dir", default="results/eval/a15")
    ap.add_argument("--split", default="proc_mech")
    a = ap.parse_args()
    DIR, SPLIT = a.dir, a.split
    rng = random.Random(0)
    seeds = a.seeds.split(",")
    names = ["warm_start", "oracle", "naive", "fixed_retry"] + [f"{arm}_s{s}" for s in seeds for arm in ("rv", "ra")]
    data = {n: load(n) for n in names}
    res: dict = {"delta": DELTA, "missing": [n for n, d in data.items() if d is None], "policies": {}, "contrasts": {},
                 "across_seeds": {}}
    for n, d in data.items():
        if d is not None:
            res["policies"][n] = describe(d, rng)
    w = data["warm_start"] or {}
    diffs: dict[str, list[float]] = {"rv_minus_warm": [], "rv_minus_ra": [], "pattern_gap_rv": []}
    for s in seeds:
        rv, ra = data.get(f"rv_s{s}"), data.get(f"ra_s{s}")
        if rv is None:
            continue
        x = per_task(rv, UNSEEN)
        for label, other in (("rv_minus_warm", w), ("rv_minus_ra", ra)):
            if other is None:
                continue
            y = per_task(other, UNSEEN)
            tasks = sorted(set(x) & set(y))
            ci = boot({t: x[t] - y[t] for t in tasks}, tasks, rng, levels=(0.95, 0.90))
            entry = {**ci, "category": a14.contrast_category(ci["ci95"], ci["ci90"]), "mcnemar": mcnemar(x, y, tasks)}
            if label == "rv_minus_warm":
                wb = boot(per_task(w, UNSEEN), sorted(per_task(w, UNSEEN)), random.Random(0))["est"]
                lo = res["policies"][f"rv_s{s}"]["unseen"]["ci95"][0]
                entry["generalises"] = lo > wb + DELTA
            res["contrasts"][f"{label}[s{s}]"] = entry
            diffs[label].append(ci["est"])
        q2 = per_task(rv, TRAINED)
        tasks = sorted(set(q2) & set(x))
        gap = boot({t: q2[t] - x[t] for t in tasks}, tasks, rng, levels=(0.95, 0.90))
        res["contrasts"][f"pattern_gap_rv[s{s}]"] = {**gap, "pattern_specific": gap["ci95"][0] > DELTA}
        diffs["pattern_gap_rv"].append(gap["est"])
    for label, ds in diffs.items():
        res["across_seeds"][label] = {"per_seed": ds, "same_sign": bool(ds) and (all(d > 0 for d in ds) or all(d < 0 for d in ds)),
                                      "sign_flip_p_one_sided": sign_flip(ds, False),
                                      "sign_flip_p_two_sided": sign_flip(ds, True)}
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    for k, v in res["policies"].items():
        print(f"{k:12s} unseen {v['unseen']['est']:.3f} {[round(z, 3) for z in v['unseen']['ci95']]} "
              f"Q2 {v['trained']['est']:.3f} binary {v['binary']['est']:.3f} "
              + " ".join(f"{b}:{v['by_branch'][b]['correct']:.2f}" for b in BRANCHES))
    for k, v in res["contrasts"].items():
        print(f"{k:22s} {v['est']:+.3f} 95% {[round(z, 3) for z in v['ci95']]}", v.get("category", ""),
              v.get("generalises", v.get("pattern_specific", "")))
    print("missing:", res["missing"])


if __name__ == "__main__":
    main()
