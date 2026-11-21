"""Stage 3 confirmatory statistics (PREREGISTRATION_S3), on results/s3/conf/<tag>__<split>.jsonl.

Per model and split: VSC, false completion, violation (duplicate, collateral, wrong amount or payee), honest failure,
escalation, policy calls. Per seed s: candidate s - A12 R_V s on the same tasks (paired task bootstrap, 10,000, seed 0;
exact McNemar on VSC). Across seeds: per-seed differences, number positive, exact sign-flip p (1/32 one-sided at five
seeds), and a seed-averaged task bootstrap (one task draw shared by all seeds).
    python scripts/s3_stats.py --cand s3_final --base a12_rv --out results/s3/s3_summary.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("a14_stats", ROOT / "scripts/a14_stats.py")
a14 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(a14)
B = 10_000
SPLITS = ("s3_conf_rec", "s3_conf_full", "s3_conf_stress")
METRICS = ("vsc", "false_success", "violation", "honest_fail", "duplicate")


def load(tag: str, split: str, d: str) -> dict[str, dict] | None:
    p = ROOT / d / f"{tag}__{split}.jsonl"
    if not p.exists():
        return None
    out = {}
    for r in map(json.loads, p.open()):
        out[r["task_id"]] = {"vsc": float(r["score"]["vsc"]), "false_success": float(r["score"]["false_success"]),
                             "violation": float(r["violation"]), "honest_fail": float(r["honest_fail"]),
                             "duplicate": float(r["score"]["duplicate_irreversible"]), "calls": r["policy_calls"]}
    return out


def mean(d: dict, k: str) -> float:
    return sum(v[k] for v in d.values()) / len(d)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True, help="tag prefix of the candidate, e.g. s3_final (tags <prefix>_s<seed>)")
    ap.add_argument("--base", default="a12_rv")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--dir", default="results/s3/conf")
    ap.add_argument("--out", default="results/s3/s3_summary.json")
    a = ap.parse_args()
    rng = random.Random(0)
    seeds = a.seeds.split(",")
    res: dict = {"cand": a.cand, "base": a.base, "splits": {}, "missing": []}
    for sp in SPLITS:
        S: dict = {"models": {}, "paired": {}, "across": {}}
        tags = ["warm_start"] + [f"{p}_s{s}" for s in seeds for p in (a.cand, a.base, "a12_ra")]
        data = {t: load(t, sp, a.dir) for t in tags}
        res["missing"] += [f"{t}__{sp}" for t, v in data.items() if v is None]
        for t, d in data.items():
            if d:
                S["models"][t] = {k: mean(d, k) for k in METRICS + ("calls",)}
        for k in METRICS:
            diffs, pairs = [], []
            for s in seeds:
                x, y = data.get(f"{a.cand}_s{s}"), data.get(f"{a.base}_s{s}")
                if not (x and y):
                    continue
                tasks = sorted(set(x) & set(y))
                dv = {t: x[t][k] - y[t][k] for t in tasks}
                ci = a14.boot(dv, tasks, rng)
                entry = {**ci}
                if k == "vsc":
                    entry["mcnemar"] = a14.mcnemar({t: x[t][k] for t in tasks}, {t: y[t][k] for t in tasks}, tasks)
                S["paired"][f"{k}[s{s}]"] = entry
                diffs.append(ci["est"])
                pairs.append((x, y, tasks))
            if pairs:
                tasks = sorted(set.intersection(*[set(p[2]) for p in pairs]))
                def avg(ts):
                    return sum(sum(x[t][k] - y[t][k] for t in ts) / len(ts) for x, y, _ in pairs) / len(pairs)
                reps = sorted(avg([tasks[rng.randrange(len(tasks))] for _ in tasks]) for _ in range(B))
                S["across"][k] = {"per_seed": diffs, "n_positive": sum(d > 0 for d in diffs),
                                  "n_negative": sum(d < 0 for d in diffs),
                                  "sign_flip_p_one_sided": a14.sign_flip(diffs, False),
                                  "sign_flip_p_two_sided": a14.sign_flip(diffs, True),
                                  "seed_avg": {"est": avg(tasks), "ci95": [reps[int(0.025 * B)], reps[int(0.975 * B) - 1]]}}
        res["splits"][sp] = S
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    for sp, S in res["splits"].items():
        print("==", sp)
        for t, m in S["models"].items():
            print(f"  {t:16s} " + " ".join(f"{k} {v:.3f}" for k, v in m.items()))
        for k, v in S["across"].items():
            print(f"  {a.cand}-{a.base} {k}: per seed {[round(x, 3) for x in v['per_seed']]} avg {v['seed_avg']['est']:+.3f} "
                  f"{[round(x, 3) for x in v['seed_avg']['ci95']]} p1 {v['sign_flip_p_one_sided']}")
    print("missing:", res["missing"])


if __name__ == "__main__":
    main()
