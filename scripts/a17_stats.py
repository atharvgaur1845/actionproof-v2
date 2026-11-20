"""A17 statistics (PREREGISTRATION_A17 §4-5), on results/eval/a17/<tag>__<split>.jsonl.

Main splits: per model and domain, phantom read-back (pooled num/den), VSC and false success on transfer classes,
collateral, correct escalation; R_V - R_A and R_V - warm start per seed with a template-cluster bootstrap (templates
resampled, then instances within each template; 10,000 resamples, seed 0); across seeds: sign consistency and exact
sign-flip p. Forks: accuracy per q with a task bootstrap.
    python scripts/a17_stats.py --out results/a17/a17_summary.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("a14_stats", ROOT / "scripts/a14_stats.py")
a14 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(a14)
B = 10_000
DOMAINS = ("it_ops", "care_admin")
OOT = ("wrong_target", "stale_read", "conflicting", "unauthorised")


def load(tag: str, split: str) -> list[dict] | None:
    p = ROOT / f"results/eval/a17/{tag}__{split}.jsonl"
    return [json.loads(l) for l in p.open()] if p.exists() else None


def per_task(recs: list[dict]) -> dict[str, dict]:
    out = {}
    for r in recs:
        s = r["score"]
        out[r["task_id"]] = {"template": r["task"]["template_id"], "cls": r["fault_class"],
                             "pr": s["phantom_readback"], "vsc": s["vsc"], "fs": float(s["false_success"])}
    return out


def pooled(rows: list[dict]) -> float | None:
    d = sum(r["pr"][1] for r in rows)
    return sum(r["pr"][0] for r in rows) / d if d else None


def cluster_boot(x: dict, y: dict | None, rng: random.Random) -> dict:
    """Template-cluster bootstrap of pooled phantom read-back (x) or its difference x - y on the same tasks."""
    tasks = sorted(set(x) & set(y)) if y else sorted(x)
    by = defaultdict(list)
    for t in tasks:
        by[x[t]["template"]].append(t)
    temps = sorted(by)

    def stat(ts):
        a = pooled([x[t] for t in ts])
        if y is None:
            return a
        b = pooled([y[t] for t in ts])
        return None if a is None or b is None else a - b
    est = stat(tasks)
    reps = []
    for _ in range(B):
        ts = []
        for tp in (temps[rng.randrange(len(temps))] for _ in temps):
            inst = by[tp]
            ts += [inst[rng.randrange(len(inst))] for _ in inst]
        v = stat(ts)
        if v is not None:
            reps.append(v)
    reps.sort()
    ci = [reps[int(0.025 * len(reps))], reps[int(0.975 * len(reps)) - 1]] if reps else None
    return {"est": est, "ci95": ci, "n_tasks": len(tasks), "n_templates": len(temps)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--out", default="results/a17/a17_summary.json")
    a = ap.parse_args()
    rng = random.Random(0)
    seeds = a.seeds.split(",")
    tags = ["warm_start"] + [f"{arm}_s{s}" for s in seeds for arm in ("rv", "ra")]
    res: dict = {"domains": {}, "forks": {}, "missing": []}
    for d in DOMAINS:
        data = {t: load(t, d) for t in tags}
        res["missing"] += [f"{t}__{d}" for t, v in data.items() if v is None]
        pt = {t: per_task(v) for t, v in data.items() if v}
        dres: dict = {"policies": {}, "contrasts": {}, "across_seeds": {}}
        for t, x in pt.items():
            tr = [r for r in x.values() if r["cls"] not in OOT]
            dres["policies"][t] = {"phantom_readback": cluster_boot(x, None, rng),
                                   "vsc_transfer": sum(r["vsc"] for r in tr) / len(tr),
                                   "false_success_transfer": sum(r["fs"] for r in tr) / len(tr),
                                   "summary": json.loads((ROOT / f"results/eval/a17/{t}__{d}.summary.json").read_text())}
        diffs = defaultdict(list)
        for s in seeds:
            rv = pt.get(f"rv_s{s}")
            for label, other in (("rv_minus_ra", pt.get(f"ra_s{s}")), ("rv_minus_warm", pt.get("warm_start"))):
                if rv and other:
                    c = cluster_boot(rv, other, rng)
                    dres["contrasts"][f"{label}[s{s}]"] = c
                    if c["est"] is not None:
                        diffs[label].append(c["est"])
        for label, ds in diffs.items():
            dres["across_seeds"][label] = {"per_seed": ds, "n_positive": sum(x > 0 for x in ds),
                                           "sign_flip_p_one_sided": a14.sign_flip(ds, False),
                                           "sign_flip_p_two_sided": a14.sign_flip(ds, True)}
        res["domains"][d] = dres
        fres = {}
        for t in tags + ["naive", "fixed_retry"]:
            v = load(t, f"{d}_cf")
            if not v:
                continue
            byq = defaultdict(dict)
            for r in v:
                byq[str(r["fork_q"])][r["base_task"]] = float(r["correct"])
            fres[t] = {q: a14.boot(vals, sorted(vals), rng) for q, vals in sorted(byq.items())}
        res["forks"][d] = fres
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    for d, dres in res["domains"].items():
        for t, v in dres["policies"].items():
            pr = v["phantom_readback"]
            print(f"{d:10s} {t:10s} phantom read-back {pr['est']} {pr['ci95']} VSC(tr) {v['vsc_transfer']:.3f}")
        for k, v in dres["across_seeds"].items():
            print(d, k, [round(x, 3) for x in v["per_seed"]], "p1", v["sign_flip_p_one_sided"])
    print("missing:", res["missing"])


if __name__ == "__main__":
    main()
