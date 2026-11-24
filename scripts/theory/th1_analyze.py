"""TH1 analysis (PREREGISTRATION_TH1.md §4): registered criteria H1-H7 plus descriptive tables.

    python scripts/theory/th1_analyze.py --split heldout     # -> results/theory/th1_heldout_summary.json

Reads results/theory/th1_<split>/shard_*.jsonl. A cell's simulated value is the mean over its 10 seeds; seed-level
values and a 95% percentile bootstrap over cells are reported for every rate.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/theory"))
from th1_grid import grid  # noqa: E402

ESTS = ("grpo", "nostd", "rloo", "batchstd", "reinforce", "ppo", "dapo", "grpo_kl", "awrb", "gdpo")


def boot_rate(x: list[bool], rng, n: int = 2000) -> dict:
    a = np.array(x, dtype=float)
    if len(a) == 0:
        return {"n": 0, "rate": None, "ci95": None}
    bs = rng.choice(a, size=(n, len(a))).mean(axis=1)
    return {"n": int(len(a)), "k": int(a.sum()), "rate": float(a.mean()),
            "ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]}


def crit(name: str, value, threshold, op: str) -> dict:
    ok = None if value is None else (value >= threshold if op == ">=" else value <= threshold)
    return {"criterion": name, "value": value, "threshold": threshold, "op": op, "met": ok}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(7)
    cells = {c["id"]: c for c in grid(a.split)}
    rows = defaultdict(dict)
    for f in sorted((ROOT / f"results/theory/th1_{a.split}").glob("shard_*.jsonl")):
        for line in f.open():
            r = json.loads(line)
            rows[r["cell"]][r["est"]] = r
    missing = [cid for cid in cells if cid not in rows or len(rows[cid]) < len(ESTS)]
    out: dict = {"split": a.split, "cells_expected": len(cells), "cells_complete": len(cells) - len(missing),
                 "missing": missing[:50]}

    def sim_mean(r):
        return float(np.mean([s["pi_T"] for s in r["sim"]]))

    def sgn(x, tol=1e-9):
        return 0 if abs(x) < tol else (1 if x > 0 else -1)

    done = [cid for cid in cells if cid not in missing]
    # ---------------- H1 quantitative accuracy of the mean-field map (grpo, nostd, rloo)
    h1 = {}
    for est in ("grpo", "nostd", "rloo", "awrb", "gdpo"):
        err = [abs(sim_mean(rows[c][est]) - rows[c][est]["pred"]["pi_T"]) for c in done]
        h1[est] = {"median_abs_err": float(np.median(err)), "p90_abs_err": float(np.quantile(err, 0.9)),
                   "mean_abs_err": float(np.mean(err)), "max_abs_err": float(np.max(err)), "n": len(err)}
    out["H1"] = {"by_est": h1, "criteria": [crit(f"{e} median |sim - pred|", h1[e]["median_abs_err"], 0.03, "<=")
                                           for e in ("grpo", "nostd", "rloo")] +
                 [crit(f"{e} p90 |sim - pred|", h1[e]["p90_abs_err"], 0.10, "<=") for e in ("grpo", "nostd", "rloo")]}
    # ---------------- H2 decision prediction for grpo (cells with |pred - 0.5| > 0.1)
    agree = [(sim_mean(rows[c]["grpo"]) > 0.5) == (rows[c]["grpo"]["pred"]["pi_T"] > 0.5) for c in done
             if abs(rows[c]["grpo"]["pred"]["pi_T"] - 0.5) > 0.1]
    h2 = boot_rate(agree, rng)
    out["H2"] = {**h2, "criteria": [crit("grpo decision agreement", h2["rate"], 0.95, ">=")]}
    # ---------------- H3 vote rule: shared law, deterministic within-state returns, eps-safe, vote != 0
    sel = [c for c in done if cells[c]["law"] == "shared" and rows[c]["grpo"]["deterministic"]
           and rows[c]["grpo"]["min_abs_beta"] >= 0.005 and abs(rows[c]["grpo"]["vote"]) > 1e-9]
    v_agree = [sgn(sim_mean(rows[c]["grpo"]) - cells[c]["pi0"]) == sgn(rows[c]["grpo"]["vote"]) for c in sel]
    dq_cells = [c for c in sel if sgn(rows[c]["grpo"]["vote"]) != sgn(rows[c]["grpo"]["delta_q"])]
    v_disagree_follow_vote = [sgn(sim_mean(rows[c]["grpo"]) - cells[c]["pi0"]) == sgn(rows[c]["grpo"]["vote"])
                              for c in dq_cells]
    h3 = boot_rate(v_agree, rng)
    out["H3"] = {**h3, "cells_where_vote_and_delta_q_disagree": boot_rate(v_disagree_follow_vote, rng),
                 "criteria": [crit("grpo moves with sign(vote)", h3["rate"], 0.95, ">=")]}
    # ---------------- H4 endogenous homogeneity: non-shared law, reads neutralise, vote < 0 < delta_q, pi0 >= 0.3
    sel4 = [c for c in done if cells[c]["law"] != "shared" and rows[c]["grpo"]["reads_neutralise"]
            and rows[c]["grpo"]["vote"] < 0 < rows[c]["grpo"]["delta_q"] and cells[c]["pi0"] >= 0.3]
    a4 = boot_rate([sim_mean(rows[c]["grpo"]) < 0.95 for c in sel4], rng)
    b4 = boot_rate([sim_mean(rows[c]["nostd"]) > sim_mean(rows[c]["grpo"]) for c in sel4], rng)
    out["H4"] = {"grpo_below_0.95": a4, "nostd_above_grpo": b4,
                 "criteria": [crit("grpo pi_T < 0.95", a4["rate"], 0.9, ">="),
                              crit("nostd pi_T > grpo pi_T", b4["rate"], 0.9, ">=")]}
    # ---------------- H5 Dr. GRPO follows sign(delta_q), |delta_q| >= 0.05, any law
    sel5 = [c for c in done if abs(rows[c]["nostd"]["delta_q"]) >= 0.05]
    h5 = boot_rate([sgn(sim_mean(rows[c]["nostd"]) - cells[c]["pi0"]) == sgn(rows[c]["nostd"]["delta_q"])
                    for c in sel5], rng)
    out["H5"] = {**h5, "criteria": [crit("nostd moves with sign(delta_q)", h5["rate"], 0.95, ">=")]}
    # ---------------- H6 GRPO fine when vote and delta_q agree (shared, deterministic)
    sel6 = [c for c in sel if sgn(rows[c]["grpo"]["vote"]) == sgn(rows[c]["grpo"]["delta_q"])]
    reg = lambda c, e: float(np.mean([s["regret"] for s in rows[c][e]["sim"]]))
    h6 = boot_rate([reg(c, "grpo") <= reg(c, "nostd") + 0.01 for c in sel6], rng)
    out["H6"] = {**h6, "criteria": [crit("grpo regret <= nostd regret + 0.01", h6["rate"], 0.9, ">=")]}
    # ---------------- H7 pairwise insufficiency (mix vs dirmult twins)
    pairs = [(c, cells[c]["twin_of"]) for c in done if cells[c].get("twin_of") and cells[c]["twin_of"] in done]
    big = [(d, m) for d, m in pairs if abs(rows[d]["grpo"]["pred"]["pi_T"] - rows[m]["grpo"]["pred"]["pi_T"]) >= 0.05]
    h7 = boot_rate([sgn(sim_mean(rows[d]["grpo"]) - sim_mean(rows[m]["grpo"])) ==
                    sgn(rows[d]["grpo"]["pred"]["pi_T"] - rows[m]["grpo"]["pred"]["pi_T"]) for d, m in big], rng)
    out["H7"] = {"pairs": len(pairs), "pairs_pred_diff_ge_0.05": len(big), **h7,
                 "criteria": [crit("twin difference direction", h7["rate"], 0.9, ">="),
                              crit("pairs with predicted difference >= 0.05", len(big), 10, ">=")]}
    # ---------------- descriptive: every estimator
    desc = {}
    for est in ESTS:
        rs = [rows[c][est] for c in done]
        seeds = [s for r in rs for s in r["sim"]]
        desc[est] = {"agrees_bayes_cell": boot_rate([np.mean([s["agrees_bayes"] for s in r["sim"]]) > 0.5 for r in rs], rng),
                     "mean_regret": float(np.mean([s["regret"] for s in seeds])),
                     "median_regret": float(np.median([np.mean([s["regret"] for s in r["sim"]]) for r in rs])),
                     "mean_false_completion": float(np.mean([s["false_completion"] for s in seeds])),
                     "mean_grad_sd": float(np.mean([s["grad_sd"] for s in seeds])),
                     "mean_abs_grad": float(np.mean([s["grad_abs_mean"] for s in seeds])),
                     "rollouts_per_run": float(np.mean([s["rollouts"] for s in seeds]))}
        for law in ("shared", "iid", "mix", "dirmult"):
            rl = [rows[c][est] for c in done if cells[c]["law"].split(":")[0] == law]
            if rl:
                desc[est][f"regret|{law}"] = float(np.mean([s["regret"] for r in rl for s in r["sim"]]))
                desc[est][f"agrees_bayes|{law}"] = float(np.mean([np.mean([s["agrees_bayes"] for s in r["sim"]]) > 0.5
                                                                  for r in rl]))
    out["descriptive"] = desc
    # DAPO and PPO direction predictions (descriptive predictions stated in the registration)
    out["P_dapo_follows_grpo_pred"] = boot_rate([(sim_mean(rows[c]["dapo"]) > 0.5) ==
                                                 (rows[c]["grpo"]["pred"]["pi_T"] > 0.5) for c in done
                                                 if abs(rows[c]["grpo"]["pred"]["pi_T"] - 0.5) > 0.1], rng)
    out["P_ppo_follows_delta_q"] = boot_rate([sgn(sim_mean(rows[c]["ppo"]) - cells[c]["pi0"]) == sgn(rows[c]["ppo"]["delta_q"])
                                              for c in sel5], rng)
    out["P_batchstd_follows_delta_q"] = boot_rate([sgn(sim_mean(rows[c]["batchstd"]) - cells[c]["pi0"]) ==
                                                   sgn(rows[c]["batchstd"]["delta_q"]) for c in sel5], rng)
    out["by_stratum_grpo_decision"] = {s: boot_rate([(sim_mean(rows[c]["grpo"]) > 0.5) ==
                                                     (rows[c]["grpo"]["pred"]["pi_T"] > 0.5) for c in done
                                                     if cells[c]["stratum"] == s], rng) for s in ("broad", "adversarial")}
    out["all_criteria"] = [x for k in ("H1", "H2", "H3", "H4", "H5", "H6", "H7") for x in out[k]["criteria"]]
    p = ROOT / f"results/theory/th1_{a.split}_summary.json"
    p.write_text(json.dumps(out, indent=1))
    for x in out["all_criteria"]:
        print(x)
    print("wrote", p)


if __name__ == "__main__":
    main()
