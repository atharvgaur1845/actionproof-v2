"""AD-C1 statistics, exactly as registered (PREREGISTRATION_AD_C1.md §3 and addendum 1).

Arms: grpo plus every candidate whose development summary says it passed G1-G4 (results/ad/d1_summary.json,
d2_summary.json, d3_summary.json). For each hypothesis (cand vs base, metric):
  statistic   seed-averaged difference (5 vs 5 runs) on cw_conf greedy
  test        one-sided unpaired seed-level permutation, exact over the C(10,5) = 252 assignments
  Holm        over every hypothesis tested
  supported   iff Holm p < 0.05 and the seed-averaged paired task bootstrap 95% interval (runs fixed) excludes 0
Also reported: two-level bootstrap interval (seeds, then tasks), per-seed differences, positive seeds.

    python scripts/ad/c1_stats.py --out results/ad/c1_summary.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONF = ROOT / "results/ad/conf"


def per_task(tag: str, metric: str, cost: float = 0.05) -> dict[str, float] | None:
    p = CONF / f"{tag}__cw_conf.jsonl"
    if not p.exists():
        return None
    out = {}
    for line in p.read_text().splitlines():
        r = json.loads(line)
        out[r["task_id"]] = float(r["score"]["vsc"]) if metric == "vsc" else float(r["score"]["r_v"]) - cost * r["n_verify"]
    return out


def perm_p(c: list[float], b: list[float]) -> float:
    allv = c + b
    obs = sum(c) / len(c) - sum(b) / len(b)
    n, k = len(allv), len(c)
    cnt = tot = 0
    for idx in itertools.combinations(range(n), k):
        s = set(idx)
        x = [allv[i] for i in idx]
        y = [allv[i] for i in range(n) if i not in s]
        tot += 1
        cnt += (sum(x) / k - sum(y) / (n - k)) >= obs - 1e-12
    return cnt / tot


def test(cand: str, base: str, metric: str, B: int = 4000) -> dict | None:
    C = [per_task(f"ad_{cand}_s{s}", metric) for s in range(5)]
    Bs = [per_task(f"ad_{base}_s{s}", metric) for s in range(5)]
    if None in C or None in Bs:
        return {"missing": [f"ad_{a}_s{s}" for a, L in ((cand, C), (base, Bs)) for s, x in enumerate(L) if x is None]}
    tasks = sorted(set.intersection(*[set(x) for x in C + Bs]))
    cm = [sum(x[t] for t in tasks) / len(tasks) for x in C]
    bm = [sum(x[t] for t in tasks) / len(tasks) for x in Bs]
    est = sum(cm) / 5 - sum(bm) / 5
    rng = random.Random(0)
    fixed, two = [], []
    for _ in range(B):
        ts = [rng.choice(tasks) for _ in tasks]
        fixed.append(sum(sum(x[t] for t in ts) for x in C) / (5 * len(ts)) - sum(sum(x[t] for t in ts) for x in Bs) / (5 * len(ts)))
        sc, sb = [rng.choice(C) for _ in C], [rng.choice(Bs) for _ in Bs]
        two.append(sum(sum(x[t] for t in ts) for x in sc) / (5 * len(ts)) - sum(sum(x[t] for t in ts) for x in sb) / (5 * len(ts)))
    fixed.sort(); two.sort()
    q = lambda v: [v[int(0.025 * B)], v[int(0.975 * B)]]
    d = [x - y for x, y in zip(cm, bm)]
    return {"metric": metric, "n_tasks": len(tasks), "cand_per_seed": cm, "base_per_seed": bm, "est": est,
            "p_perm_one_sided": perm_p(cm, bm), "ci95_runs_fixed": q(fixed), "ci95_two_level": q(two),
            "per_seed_diff_same_index": d, "n_positive_same_index": sum(x > 0 for x in d)}


def passed() -> list[str]:
    out = []
    for f in ("d1_summary.json", "d2_summary.json"):
        p = ROOT / "results/ad" / f
        if p.exists():
            out += [c for c, g in json.loads(p.read_text())["gates"].items() if g.get("pass")]
    p = ROOT / "results/ad/d3_summary.json"
    if p.exists() and json.loads(p.read_text()).get("gates", {}).get("pass"):
        out.append("nostd_ema_irc")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    arms = passed()
    hyps = {f"H_{c}": (c, "grpo", "vsc") for c in arms if c != "nostd_ema_irc"}
    if "awrbn" in arms and "nostd" in arms:
        hyps["H_AWRB_none"] = ("awrbn", "nostd", "vsc")
    if "awrbn_ema" in arms and "nostd_ema" in arms:
        hyps["H_AWRB_ema"] = ("awrbn_ema", "nostd_ema", "vsc")
    if "nostd_ema_irc" in arms:
        hyps["H_D3"] = ("nostd_ema_irc", "grpo", "vsc")
        if "nostd_ema" in arms:
            hyps["H_D3b"] = ("nostd_ema_irc", "nostd_ema", "j_c")
    res = {h: test(*v) for h, v in hyps.items()}
    ok = {h: r for h, r in res.items() if r and "p_perm_one_sided" in r}
    order = sorted(ok, key=lambda h: ok[h]["p_perm_one_sided"])
    m, prev = len(order), 0.0
    for i, h in enumerate(order):
        adj = min(1.0, max(prev, (m - i) * ok[h]["p_perm_one_sided"]))
        prev = adj
        ok[h]["p_holm"] = adj
        ok[h]["supported"] = adj < 0.05 and ok[h]["ci95_runs_fixed"][0] > 0
    out = {"arms": ["grpo"] + arms, "hypotheses": {h: list(v) for h, v in hyps.items()}, "results": res}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({h: {k: (round(x, 4) if isinstance(x, float) else x) for k, x in (r or {}).items()
                          if k in ("est", "p_perm_one_sided", "p_holm", "ci95_runs_fixed", "ci95_two_level",
                                   "supported", "missing", "n_positive_same_index")} for h, r in res.items()}, indent=1))


if __name__ == "__main__":
    main()
