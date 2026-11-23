"""Registered analysis of LAB1 (PREREGISTRATION_LAB1.md §6). Run once after the families finish.

    python scripts/lab/lab1_analyze.py --out results/lab/lab1_summary.json
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "results/lab/tabular"
FIELDS = ("regret", "J", "safe_verify_rate", "risky_verify_rate", "mc_dupl", "recovery_given_silent",
          "uninformative_read_frac", "mc_cost", "mc_fail", "mc_nv", "precision", "recall")
SFB_NP = ("nostd", "rloo", "critic_v", "critic_q", "ppo")
SFB_ALL = SFB_NP + ("igc_v1",)
PRIMARY = {"H1": ("cdvo_adapt@0.25", SFB_NP), "H2": ("cdvo_ia@0.25", SFB_ALL)}
VOI_MARGIN = 0.02


def load(fam: str) -> list[dict]:
    p = D / f"{fam}.jsonl"
    if not p.exists():
        p = D / f"{fam}.jsonl.gz"
        return [json.loads(l) for l in gzip.open(p, "rt") if l.strip()]
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def _m(evs, f):
    xs = [e[f] for e in evs if e.get(f) is not None]
    return mean(xs) if xs else None


def l1_table(rows):
    """(arm, budget) -> {'unit': {(setting, seed): metrics}, 'seed': {seed: metrics}, means, resources}."""
    acc = defaultdict(lambda: defaultdict(list))
    res = defaultdict(lambda: defaultdict(list))
    for r in rows:
        _, mode, c, L, arm, bud, seed = r["id"].split("|")
        evs = list(r["eval"].values())
        acc[(arm, bud)][(f"{mode}|{c}|{L}", int(seed))] = {f: _m(evs, f) for f in FIELDS}
        for k in ("episodes_used", "n_updates", "n_forks"):
            res[(arm, bud)][k].append(r[k])
    tab = {}
    for k, units in acc.items():
        seeds = defaultdict(list)
        for (st, sd), m in units.items():
            seeds[sd].append(m)
        seed_m = {sd: {f: _m(ms, f) for f in FIELDS} for sd, ms in seeds.items()}
        tab[k] = {"units": units, "seed": seed_m, "n_units": len(units),
                  **{f: _m(list(seed_m.values()), f) for f in FIELDS},
                  **{kk: mean(v) for kk, v in res[k].items()}}
    return tab


def signflip(diffs, n=20000, seed=0):
    """One-sided paired sign-flip permutation p for mean(diff) > 0."""
    rnd = random.Random(seed)
    obs = mean(diffs)
    ge = 0
    for _ in range(n):
        s = mean(d if rnd.random() < 0.5 else -d for d in diffs)
        ge += s >= obs - 1e-15
    return (ge + 1) / (n + 1)


def holm(ps: dict) -> dict:
    items = sorted(ps.items(), key=lambda kv: kv[1])
    m, out, run = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


def gates(tab, cand, pool, bud):
    if (cand, bud) not in tab:
        return {"status": "not_run"}
    avail = [a for a in pool if (a, bud) in tab]
    sfb = min(avail, key=lambda a: tab[(a, bud)]["regret"])
    C, S = tab[(cand, bud)], tab[(sfb, bud)]
    seeds = sorted(set(C["seed"]) & set(S["seed"]))
    better = sum(C["seed"][s]["regret"] < S["seed"][s]["regret"] for s in seeds)
    rel = (S["regret"] - C["regret"]) / S["regret"]
    units = sorted(set(C["units"]) & set(S["units"]))
    diffs = [S["units"][u]["regret"] - C["units"][u]["regret"] for u in units]
    return {"sfb": sfb, "cand_regret": C["regret"], "sfb_regret": S["regret"], "rel_improvement": rel,
            "seeds_better": better, "n_seeds": len(seeds),
            "T1": rel >= 0.20 and better >= 4,
            "T2": C["safe_verify_rate"] <= S["safe_verify_rate"] + 0.02,
            "T3": C["mc_dupl"] <= S["mc_dupl"] + 0.01,
            "cand_episodes": C["episodes_used"], "sfb_episodes": S["episodes_used"],
            "mean_unit_diff": mean(diffs), "p_signflip": signflip(diffs), "n_units": len(units)}


def l2n(rows):
    by = defaultdict(list)
    for r in rows:
        _, p, c, s, est, seed = r["id"].split("|")
        e = next(iter(r["eval"].values()))
        by[est].append((float(c), e["voi"][0], e["verify_prob"][0], e["regret"]))
    out = {}
    for est, xs in by.items():
        pos = [x for x in xs if x[0] >= 0.01 and abs(x[1]) > VOI_MARGIN]
        wrong = [(x[2] > 0.5) != (x[1] > 0) for x in pos]
        out[est] = {"n": len(pos), "wrong_rate": sum(wrong) / len(pos) if pos else None,
                    "mean_regret": mean(x[3] for x in xs)}
    out["H4_supported"] = bool((out.get("indep_world_std", {}).get("wrong_rate") or 0) >= 0.30 and
                               (out.get("indep_task_std", {}).get("wrong_rate") or 1) <= 0.15)
    return out


def l3g(rows):
    acc = defaultdict(lambda: defaultdict(list))
    eps = defaultdict(list)
    for r in rows:
        _, arm, seed = r["id"].split("|")
        for tag, e in r["eval"].items():
            sp = "_".join(tag.split("_")[1:-1])
            acc[arm][sp].append(e["regret"])
        eps[arm].append(r["episodes_used"])
    out = {a: {sp: mean(v) for sp, v in d.items()} | {"episodes": mean(eps[a])} for a, d in acc.items()}
    for cand in ("cdvo_adapt@0.25", "cdvo_ia@0.25"):
        pool = SFB_NP if cand == "cdvo_adapt@0.25" else SFB_ALL
        pool = [a for a in pool if a in out]
        if cand in out and pool:
            for sp in ("test_shift", "test_long"):
                best = min(pool, key=lambda a: out[a].get(sp, 9))
                out.setdefault("T4", {})[f"{cand}|{sp}"] = {"best_baseline": best, "baseline": out[best][sp],
                                                           "cand": out[cand][sp],
                                                           "pass": out[cand][sp] <= out[best][sp]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/lab/lab1_summary.json")
    ap.add_argument("--dir", default=None, help="smoke tests only: read the families from another directory")
    a = ap.parse_args()
    global D
    if a.dir:
        D = Path(a.dir)
    out = {}
    rows = load("L1")
    tab = l1_table(rows)
    out["L1_table"] = {f"{k[0]}/{k[1]}": {kk: v for kk, v in t.items() if kk not in ("units",)}
                       for k, t in sorted(tab.items())}
    g, ps = {}, {}
    for h, (cand, pool) in PRIMARY.items():
        for bud in ("E1", "E4"):
            g[f"{h}|{bud}"] = gates(tab, cand, pool, bud)
            if "p_signflip" in g[f"{h}|{bud}"]:
                ps[f"{h}|{bud}"] = g[f"{h}|{bud}"]["p_signflip"]
    hp = holm(ps)
    for k in g:
        g[k]["p_holm"] = hp.get(k)
    for h in PRIMARY:
        e4 = g[f"{h}|E4"]
        g[h + "_supported"] = bool(e4.get("T1") and e4.get("T2") and e4.get("T3") and (e4.get("p_holm") or 1) < 0.05)
    out["gates"] = g
    # descriptive contrasts (no tests)
    desc = {}
    for bud in ("E1", "E4"):
        for x, y in (("cdvo_adapt@0.25", "cdvo_fixed@0.25"), ("fork_act", "fork_act_nocrn"),
                     ("cdvo_adapt_critic@0.25", "cdvo_adapt@0.25"), ("cdvo_adapt@0.1", "cdvo_adapt@0.25"),
                     ("ppo", "ppo_norm"), ("cdvo_ia@0.25", "igc_v1")):
            if (x, bud) in tab and (y, bud) in tab:
                desc[f"{x} vs {y} | {bud}"] = {"x": tab[(x, bud)]["regret"], "y": tab[(y, bud)]["regret"],
                                               "x_better_seeds": sum(tab[(x, bud)]["seed"][s]["regret"] <
                                                                     tab[(y, bud)]["seed"][s]["regret"]
                                                                     for s in tab[(x, bud)]["seed"])}
    out["descriptive"] = desc
    for fam, fn in (("L2N", l2n), ("L3G", l3g)):
        try:
            out[fam] = fn(load(fam))
        except FileNotFoundError:
            out[fam] = {"status": "not_run"}
    (ROOT / a.out).write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({"gates": g, "L2N": out["L2N"], "L3G_T4": out["L3G"].get("T4")}, indent=1, default=str))


if __name__ == "__main__":
    main()
