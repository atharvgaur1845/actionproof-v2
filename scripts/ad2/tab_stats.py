"""Secondary (not registered as a decision rule) statistics for TB2: paired comparisons of each candidate variant
with each baseline at its matched budget, over units (setting, seed).

For each pair: mean paired regret difference (baseline - candidate; > 0 favours the candidate), 95% bootstrap CI
(10,000 resamples of units), Cohen's d_z, an exact-by-Monte-Carlo sign-flip permutation p-value (one-sided), and
Holm adjustment over all pairs reported. The registered decision is still V2TB_PROTOCOL §5 (tab_analyze.py TB2).

    python scripts/ad2/tab_stats.py --out results/ad2/tabular/TB2_stats.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def units(rows):
    acc = defaultdict(dict)  # variant -> (setting, seed) -> regret
    eps = defaultdict(list)
    for r in rows:
        _, mode, c, L, name, var, seed = r["id"].split("|")
        acc[f"{name}/{var}"][(mode, c, L, seed)] = mean(e["regret"] for e in r["eval"].values())
        eps[f"{name}/{var}"].append(r["episodes_used"])
    return acc, {k: mean(v) for k, v in eps.items()}


def compare(a: dict, b: dict, rng) -> dict:
    keys = sorted(set(a) & set(b))
    d = np.array([b[k] - a[k] for k in keys])
    boot = [rng.choice(d, len(d)).mean() for _ in range(10000)]
    flips = rng.choice([-1, 1], size=(20000, len(d)))
    perm = (flips * d).mean(1)
    sd = d.std(ddof=1) if len(d) > 1 else float("nan")
    return {"n_units": len(d), "mean_diff": float(d.mean()), "ci95": [float(np.percentile(boot, 2.5)),
            float(np.percentile(boot, 97.5))], "d_z": float(d.mean() / sd) if sd > 0 else None,
            "p_one_sided": float((perm >= d.mean() - 1e-15).mean()), "frac_units_favouring": float((d > 0).mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = [json.loads(l) for l in (ROOT / "results/ad2/tabular/TB2.jsonl").read_text().splitlines() if l.strip()]
    acc, eps = units(rows)
    rng = np.random.default_rng(0)
    cands = [k for k in acc if k.split("/")[0] in ("cvc", "fork_bel", "fork_act", "igc", "awrbn", "ctxema")]
    bases = ("nostd", "rloo", "critic_v", "critic_q", "ema", "ctxema", "lagr", "igc")
    out = []
    for cv in cands:
        for b in bases:
            opts = sorted([k for k in acc if k.split("/")[0] == b and k.split("/")[1].startswith("x")], key=lambda k: eps[k])
            if not opts or cv.split("/")[0] == b:
                continue
            m = next((k for k in opts if eps[k] >= eps[cv]), opts[-1])
            r = compare(acc[cv], acc[m], rng)
            out.append({"candidate": cv, "baseline": m, "cand_episodes": eps[cv], "base_episodes": eps[m],
                        "budget_matched": eps[m] >= eps[cv], **r})
    ps = sorted(range(len(out)), key=lambda i: out[i]["p_one_sided"])
    m = len(out)
    running = 0.0
    for rank, i in enumerate(ps):
        running = max(running, min(1.0, (m - rank) * out[i]["p_one_sided"]))
        out[i]["p_holm"] = running
    Path(a.out).write_text(json.dumps(out, indent=1))
    for r in sorted(out, key=lambda r: (r["candidate"], r["baseline"])):
        print(f"{r['candidate']:16s} vs {r['baseline']:14s} matched={r['budget_matched']!s:5s} diff={r['mean_diff']:+.4f} "
              f"CI[{r['ci95'][0]:+.4f},{r['ci95'][1]:+.4f}] dz={r['d_z'] if r['d_z'] is None else round(r['d_z'], 2)} "
              f"p={r['p_one_sided']:.4f} holm={r['p_holm']:.4f} fav={r['frac_units_favouring']:.2f}")


if __name__ == "__main__":
    main()
