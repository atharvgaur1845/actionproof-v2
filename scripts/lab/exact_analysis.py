"""Exact (enumeration) analysis of the group-relative verification update, and independent re-derivation of the
v2 headline numbers from raw files.

    python scripts/lab/exact_analysis.py --out results/lab/exact_analysis.json

Parts:
  thresholds    exact decision thresholds p* for grpo / nostd / rloo, conditions A and B, over c, s, Lf, q0, rho, G
  mc_check      the v2 Monte Carlo thresholds (results/ad2/derivation_numeric.json) against the exact ones
  tb1_recount   TB1 wrong-decision rates recomputed from results/ad2/tabular/TB1.jsonl.gz with independent code
  tb1_predict   does the exact expected-update sign at the warm start predict the decision GRPO actually learned?
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from actionproof.lab.exact_vote import expected_update, threshold, voi  # noqa: E402

C = (0.001, 0.01, 0.025, 0.05, 0.1, 0.2)
S = (0.3, 0.6, 1.0)


def thresholds() -> list[dict]:
    rows = []
    for est in ("grpo", "nostd", "rloo"):
        for cond in ("A", "B"):
            for s in S:
                for Lf in (0.0, 1.0):
                    for q0 in (0.0, 0.3):
                        for c in C:
                            rows.append({"est": est, "cond": cond, "s": s, "Lf": Lf, "q0": q0, "c": c, "rho": 0.3, "G": 8,
                                         "p_star": threshold(c, s, 0.3, 8, est, cond, Lf=Lf, q0=q0),
                                         "voi_p_star": min(1.0, c / (s * (1 - q0) * (1 + Lf)))})
    # rho and G dependence (grpo, s = 1 and s = 0.6, c = 0.05)
    for G in (2, 4, 8, 16):
        for rho in (0.05, 0.3, 0.6, 0.9):
            for s in (1.0, 0.6):
                for cond in (("A", "B") if G <= 8 else ("A",)):
                    rows.append({"est": "grpo", "cond": cond, "s": s, "Lf": 0.0, "q0": 0.0, "c": 0.05, "rho": rho,
                                 "G": G, "p_star": threshold(0.05, s, rho, G, "grpo", cond),
                                 "voi_p_star": 0.05 / s, "family": "rho_G"})
    return rows


def mc_check(rows: list[dict]) -> list[dict]:
    mc = json.loads((ROOT / "results/ad2/derivation_numeric.json").read_text())["rows"]
    out = []
    idx = {(r["est"], r["cond"], r["s"], r["Lf"], r["q0"], r["c"]): r for r in rows if r.get("family") is None}
    for m in mc:
        if m["est"] not in ("grpo", "nostd"):
            continue
        k = (m["est"], m["cond"], m["s"], m["Lf"], m["q0"], m["c"])
        if k in idx:
            out.append({**{kk: m[kk] for kk in ("est", "cond", "s", "Lf", "q0", "c")},
                        "mc_grid_threshold": m["learned_threshold_p"], "exact_threshold": idx[k]["p_star"]})
    return out


def tb1_load() -> list[dict]:
    out = []
    with gzip.open(ROOT / "results/ad2/tabular/TB1.jsonl.gz", "rt") as f:
        for line in f:
            r = json.loads(line)
            _, p, c, s, L, name, seed = r["id"].split("|")
            e = next(iter(r["eval"].values()))
            out.append({"p": float(p), "c": float(c), "s": float(s), "L": float(L), "arm": name, "seed": int(seed),
                        "vp": e["verify_prob"][0], "vp_safe": e["verify_prob"][1], "voi": e["voi"][0],
                        "regret": e["regret"], "cost": e["mc_cost"], "fail": e["mc_fail"]})
    return out


def tb1_recount(tb: list[dict]) -> dict:
    """Independent re-implementation of the registered definition: cells with c >= 0.01 and |VOI| > 0.02, decision =
    final verify prob > 0.5, wrong iff it disagrees with sign(VOI); pooled over cells and seeds."""
    res = {}
    by = defaultdict(list)
    for r in tb:
        by[r["arm"]].append(r)
    for arm, xs in by.items():
        pos = [x for x in xs if x["c"] >= 0.01 and abs(x["voi"]) > 0.02]
        wrong = [(x["vp"] > 0.5) != (x["voi"] > 0) for x in pos]
        res[arm] = {"n": len(pos), "wrong": sum(wrong), "wrong_rate": sum(wrong) / len(pos)}
    # closed-form VOI check on the core workflow: VOI(op 0) = p s (1 + Lf) - c (repair attempted; no 2nd repair)
    dev = max(abs(x["voi"] - voi(x["p"], x["c"], x["s"], Lf=x["L"])) for x in tb)
    res["_voi_closed_form_max_abs_dev"] = dev
    # where does nostd go wrong?
    nos = [x for x in by["nostd"] if x["c"] >= 0.01 and abs(x["voi"]) > 0.02 and (x["vp"] > 0.5) != (x["voi"] > 0)]
    res["_nostd_errors"] = {
        "n": len(nos),
        "voi_positive": sum(x["voi"] > 0 for x in nos),
        "abs_voi_le_0.05": sum(abs(x["voi"]) <= 0.05 for x in nos),
        "mean_abs_voi": sum(abs(x["voi"]) for x in nos) / max(1, len(nos)),
        "mean_regret": sum(x["regret"] for x in nos) / max(1, len(nos)),
        "by_s": {s: sum(x["s"] == s for x in nos) for s in S},
    }
    grp = [x for x in by["grpo"] if x["c"] >= 0.01 and abs(x["voi"]) > 0.02 and (x["vp"] > 0.5) != (x["voi"] > 0)]
    res["_grpo_errors"] = {"n": len(grp), "voi_positive_never_verified": sum(x["voi"] > 0 for x in grp),
                           "voi_negative_verified": sum(x["voi"] < 0 for x in grp),
                           "mean_regret": sum(x["regret"] for x in grp) / max(1, len(grp))}
    return res


def tb1_predict(tb: list[dict]) -> dict:
    """Prediction: GRPO ends verifying iff the exact expected group update on the risky read logit is positive at the
    warm start (rho0 = 0.3). Ignores the never-faulted op's reads (a second cost-only read), so it is approximate."""
    cache = {}
    out = {}
    for arm, est, cond in (("grpo", "grpo", "A"), ("grpo_B", "grpo", "B"), ("nostd", "nostd", "A")):
        hit = n = 0
        for x in tb:
            if x["arm"] != arm or x["c"] < 0.01 or abs(x["voi"]) <= 0.02:
                continue
            k = (est, cond, x["p"], x["c"], x["s"], x["L"])
            if k not in cache:
                cache[k] = expected_update(x["p"], x["c"], x["s"], 0.3, 8, est, cond, Lf=x["L"])
            n += 1
            hit += (cache[k] > 0) == (x["vp"] > 0.5)
        out[arm] = {"n": n, "agree": hit / n}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/lab/exact_analysis.json")
    a = ap.parse_args()
    t0 = time.time()
    th = thresholds()
    tb = tb1_load()
    res = {"thresholds": th, "mc_check": mc_check(th), "tb1_recount": tb1_recount(tb), "tb1_predict": tb1_predict(tb),
           "seconds": round(time.time() - t0, 1)}
    Path(ROOT / a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("tb1_recount", "tb1_predict", "seconds")}, indent=1)[:4000])


if __name__ == "__main__":
    main()
