"""Generate docs/research/TESTBED_RESULTS.md tables from results/theory/th1_<split>_summary.json.

    python scripts/theory/th1_report.py --split heldout > docs/research/_th1_tables.md
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("--split", default="heldout")
a = ap.parse_args()
d = json.loads((ROOT / f"results/theory/th1_{a.split}_summary.json").read_text())


def f(x, n=3):
    if x is None:
        return "—"
    return f"{x:.{n}f}" if isinstance(x, float) else str(x)


def rate(r):
    if not r or r.get("rate") is None:
        return "— (n = 0)"
    return f"{r['k']}/{r['n']} = {r['rate']:.3f} [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]"


print(f"Split: {d['split']}; cells complete {d['cells_complete']}/{d['cells_expected']}.\n")
print("### Registered criteria\n")
print("| Criterion | Value | Threshold | Met |")
print("|---|---|---|---|")
for c in d["all_criteria"]:
    print(f"| {c['criterion']} | {f(c['value'], 4)} | {c['op']} {c['threshold']} | {c['met']} |")
print("\n### Hypothesis detail\n")
print("| H | Cells / rate [95% CI] |")
print("|---|---|")
print(f"| H2 GRPO decision | {rate(d['H2'])} |")
print(f"| H3 vote direction | {rate(d['H3'])} |")
print(f"| H3 subset where vote and ΔQ disagree | {rate(d['H3']['cells_where_vote_and_delta_q_disagree'])} |")
print(f"| H4a GRPO π_T < 0.95 | {rate(d['H4']['grpo_below_0.95'])} |")
print(f"| H4b nostd π_T > GRPO π_T | {rate(d['H4']['nostd_above_grpo'])} |")
print(f"| H5 nostd follows sign ΔQ | {rate(d['H5'])} |")
print(f"| H6 GRPO regret ≤ nostd + 0.01 | {rate(d['H6'])} |")
print(f"| H7 twins (pairs {d['H7']['pairs']}, with predicted diff ≥ 0.05: {d['H7']['pairs_pred_diff_ge_0.05']}) | {rate(d['H7'])} |")
print("\n### H1: mean-field accuracy |sim − pred| of π_T\n")
print("| Estimator | median | p90 | mean | max | n |")
print("|---|---|---|---|---|---|")
for e, v in d["H1"]["by_est"].items():
    print(f"| {e} | {f(v['median_abs_err'], 4)} | {f(v['p90_abs_err'], 4)} | {f(v['mean_abs_err'], 4)} | {f(v['max_abs_err'], 4)} | {v['n']} |")
print("\n### Descriptive: all estimators (seed-level means over cells)\n")
print("| Estimator | Bayes-action agreement (cells) | mean regret | median cell regret | regret shared / iid / mix / dirmult | false completion | mean |grad| | grad sd | rollouts/run |")
print("|---|---|---|---|---|---|---|---|---|")
for e, v in d["descriptive"].items():
    laws = " / ".join(f(v.get(f"regret|{l}"), 4) for l in ("shared", "iid", "mix", "dirmult"))
    print(f"| {e} | {rate(v['agrees_bayes_cell'])} | {f(v['mean_regret'], 4)} | {f(v['median_regret'], 4)} | {laws} | "
          f"{f(v['mean_false_completion'])} | {f(v['mean_abs_grad'], 4)} | {f(v['mean_grad_sd'], 4)} | {f(v['rollouts_per_run'], 0)} |")
print("\n### Descriptive predictions\n")
print(f"- DAPO follows GRPO's predicted decision: {rate(d['P_dapo_follows_grpo_pred'])}")
print(f"- PPO follows sign ΔQ: {rate(d['P_ppo_follows_delta_q'])}")
print(f"- batch-std follows sign ΔQ: {rate(d['P_batchstd_follows_delta_q'])}")
for s, r in d["by_stratum_grpo_decision"].items():
    print(f"- GRPO decision prediction, stratum {s}: {rate(r)}")
