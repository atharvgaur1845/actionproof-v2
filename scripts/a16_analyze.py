"""A16 analysis (PREREGISTRATION_A15 §4-5) of results/a16/{trajectory,causal}__*.json.

Trajectory: mean margin M by branch per checkpoint; probe AUROC (refit and frozen warm-start) at the chosen layer and
its range over layers. Causal, per model: pairs eligible (|M_complete - M_partial| >= 1 nat), median normalised
final-token and span effects in both directions, self-patch effect, the 95th percentile of random same-norm effects,
wrong-task effect; registered rule: median final complete->partial effect > random 95th percentile and |self| < 0.05.
Ablation: mean change in M per branch for the difference-of-means direction vs the largest random-direction change;
registered rule: partial-branch mean dM < 0 and |dM| > max random |dM|, with |dM(Q4)| < half the partial effect.
    python scripts/a16_analyze.py --out results/a16/a16_summary.json
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "results/a16"
PARTIAL = ("Q1", "Q2", "Q3")


def med(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else None


def pct(xs, q):
    xs = sorted(x for x in xs if x is not None)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


def trajectory(layer: int) -> dict:
    out = {}
    for f in sorted(D.glob("trajectory__*.json")):
        t = json.loads(f.read_text())
        rows = t["examples"]["test"]
        m = {b: st.mean(r["M"] for r in rows if r["branch"] == b) for b in ("Q0", "Q1", "Q2", "Q3", "Q4")}
        refit, frozen = t["probe"]["refit"], t["probe"]["frozen_warm"]
        out[t["tag"]] = {"M_by_branch": m, "M_partial_minus_complete": st.mean(m[b] for b in PARTIAL) - m["Q4"],
                         "probe_refit_at_layer": refit[layer - 1], "probe_frozen_at_layer": frozen[layer - 1],
                         "probe_refit_min_max": [min(refit), max(refit)],
                         "probe_frozen_min_max": [min(x for x in frozen if x is not None), max(x for x in frozen if x is not None)]}
    return out


def causal() -> dict:
    out = {}
    for f in sorted(D.glob("causal__*.json")):
        c = json.loads(f.read_text())
        p = c["patch"]
        elig = [r for r in p if abs(r["M_complete"] - r["M_partial"]) >= 1.0]
        rand = [x for r in c["controls"] for x in r["random"]]
        res = {"layer": c["layer"], "n_pairs": len(p), "n_eligible": len(elig),
               "n_aligned": sum(r["aligned"] for r in p),
               "median_final_c2p": med(r["final_c2p"] for r in elig), "median_final_p2c": med(r["final_p2c"] for r in elig),
               "median_span_c2p": med(r.get("span_c2p") for r in elig), "median_span_p2c": med(r.get("span_p2c") for r in elig),
               "self_median": med(r["self"] for r in c["controls"]), "random_p95": pct(rand, 0.95),
               "random_median": med(rand), "wrong_task_median": med(r["wrong_task"] for r in c["controls"]),
               "n_control_pairs": len(c["controls"])}
        res["patch_rule_met"] = (res["median_final_c2p"] is not None and res["random_p95"] is not None and
                                 res["median_final_c2p"] > res["random_p95"] and abs(res["self_median"] or 0) < 0.05)
        ab = c["ablation"]
        dm = ab["diff_means"]
        part = st.mean(dm[b] for b in PARTIAL)
        rmax = max(abs(st.mean(v[b] for b in PARTIAL)) for k, v in ab.items() if k.startswith("random"))
        res["ablation"] = {"dM_partial": part, "dM_Q4": dm["Q4"], "dM_Q0": dm["Q0"], "max_random_abs_dM_partial": rmax}
        res["ablation_rule_met"] = part < 0 and abs(part) > rmax and abs(dm["Q4"]) < abs(part) / 2
        out[c["tag"]] = res
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/a16/a16_summary.json")
    a = ap.parse_args()
    sweep = json.loads((D / "sweep_dev.json").read_text())
    L = sweep["chosen_layer"]
    res = {"chosen_layer": L, "trajectory": trajectory(L), "causal": causal()}
    rv = {k: v for k, v in res["causal"].items() if k.startswith("rv_")}
    res["registered"] = {"patch_rule_seeds": sum(v["patch_rule_met"] for v in rv.values()),
                         "ablation_rule_seeds": sum(v["ablation_rule_met"] for v in rv.values()), "rv_models": len(rv),
                         "rule": "causal role claimed only if met in at least 4 of 5 R_V seeds"}
    (ROOT / a.out).write_text(json.dumps(res, indent=2))
    print("layer", L)
    for k, v in res["trajectory"].items():
        print(f"{k:12s} M(partial)-M(complete) {v['M_partial_minus_complete']:+.2f}  probe refit {v['probe_refit_at_layer']:.3f} "
              f"frozen {v['probe_frozen_at_layer']}")
    for k, v in res["causal"].items():
        print(k, {x: (round(y, 3) if isinstance(y, float) else y) for x, y in v.items() if x != "ablation"},
              {x: round(y, 3) for x, y in v["ablation"].items()}, v["ablation_rule_met"])
    print(res["registered"])


if __name__ == "__main__":
    main()
