"""AD-C2 registered analysis (PREREGISTRATION_AD_C2.md §4). Run once, after the five cw_c2 evaluations exist.

Unit: a task. A model's task score is the mean over its 5 episodes on that task (greedy + 4 sampled at T = 1).
Differences are paired on the 300 tasks; 95% percentile interval from B = 10,000 task resamples (seed 20261128).
Every new arm is one training seed, so intervals describe evaluation uncertainty with each model fixed; they are not
seed-level replications.

    python scripts/ad/c2_stats.py   ->  results/ad/c2/c2_summary.{json,md}
"""
from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "results/ad/c2"
TAGS = {"grpo_c005": "ad_grpo_s0", "nostd_c005": "ad_nostd_s0", "grpo_c001": "ad_c2_grpo_c001_s0",
        "nostd_c001": "ad_c2_nostd_c001_s0", "grpo_c000": "ad_c2_grpo_c000_s0"}
COST = {"grpo_c005": 0.05, "nostd_c005": 0.05, "grpo_c001": 0.01, "nostd_c001": 0.01, "grpo_c000": 0.0}
MODES = ["", ".t1_s0", ".t1_s1", ".t1_s2", ".t1_s3"]
B, SEED = 10_000, 20261128


def episodes(tag: str, suffix: str) -> list[dict]:
    return [json.loads(l) for l in (D / f"{tag}__cw_c2{suffix}.jsonl").open()]


def per_task(tag: str) -> dict[str, dict[str, float]]:
    acc: dict[str, dict[str, list[float]]] = {}
    for suf in MODES:
        for r in episodes(tag, suf):
            a = acc.setdefault(r["task_id"], {"vsc": [], "reads": [], "fs": [], "faulted": []})
            a["vsc"].append(float(r["score"]["vsc"])); a["reads"].append(float(r["n_verify"]))
            a["fs"].append(float(r["score"]["false_success"])); a["faulted"].append(float(r["fault_class"] != "cw:clean"))
    return {t: {k: sum(v) / len(v) for k, v in a.items()} for t, a in acc.items()}


def boot(x: dict, y: dict, key: str) -> dict:
    tasks = sorted(x)
    assert tasks == sorted(y) and len(tasks) == 300
    d = [x[t][key] - y[t][key] for t in tasks]
    est = sum(d) / len(d)
    rng = random.Random(SEED)
    bs = sorted(sum(d[rng.randrange(len(d))] for _ in d) / len(d) for _ in range(B))
    return {"diff": est, "lo": bs[int(0.025 * B)], "hi": bs[int(0.975 * B) - 1]}


def main() -> None:
    greedy = {k: json.loads((D / f"{t}__cw_c2.summary.json").read_text()) for k, t in TAGS.items()}
    pt = {k: per_task(t) for k, t in TAGS.items()}
    arms = {}
    for k in TAGS:
        g = greedy[k]
        m = lambda key: sum(v[key] for v in pt[k].values()) / 300
        arms[k] = {"train_cost": COST[k], "greedy_reads": g["reads"], "greedy_vsc": g["vsc"],
                   "greedy_false_success": g["false_success"], "greedy_faulted_vsc": (g.get("faulted") or {}).get("vsc"),
                   "all5_reads": m("reads"), "all5_vsc": m("vsc"), "all5_false_success": m("fs"),
                   "net_utility_at_train_cost": m("vsc") - COST[k] * m("reads"),
                   "verify_after": g.get("verify_after")}
    diffs = {"D1_vsc_nostd_minus_grpo_c001": boot(pt["nostd_c001"], pt["grpo_c001"], "vsc"),
             "D2_reads_grpo_c001_minus_c005": boot(pt["grpo_c001"], pt["grpo_c005"], "reads"),
             "D3_reads_nostd_c001_minus_c005": boot(pt["nostd_c001"], pt["nostd_c005"], "reads"),
             "D4_reads_grpo_c000_minus_c001": boot(pt["grpo_c000"], pt["grpo_c001"], "reads"),
             "D5_vsc_grpo_c000_minus_c001": boot(pt["grpo_c000"], pt["grpo_c001"], "vsc")}
    preds = {
        "P1_grpo_c001_cost_blind (greedy reads <= 0.2)": arms["grpo_c001"]["greedy_reads"] <= 0.2,
        "P2_nostd_c001_reads (greedy reads >= 0.8)": arms["nostd_c001"]["greedy_reads"] >= 0.8,
        "P3_vsc_nostd_gt_grpo_at_c001 (D1 lower bound > 0)": diffs["D1_vsc_nostd_minus_grpo_c001"]["lo"] > 0,
        "P4_boundary_grpo_c000_reads (greedy reads > 0.2) [secondary]": arms["grpo_c000"]["greedy_reads"] > 0.2,
        "P5_nostd_reads_not_lower_at_c001 (D3 estimate >= 0) [secondary]":
            diffs["D3_reads_nostd_c001_minus_c005"]["diff"] >= 0,
    }
    out = {"split": "cw_c2", "n_tasks": 300, "episodes_per_task": 5, "bootstrap": {"B": B, "seed": SEED},
           "arms": arms, "differences": diffs, "predictions": preds}
    (D / "c2_summary.json").write_text(json.dumps(out, indent=2))
    md = ["# AD-C2 registered analysis (scripts/ad/c2_stats.py)", "",
          "cw_c2: 300 sealed tasks (120 faulted). Seed 0 per arm; 5 episodes per task (greedy + 4 sampled).", "",
          "| Arm | train cost | greedy reads/ep | greedy VSC | greedy false success | all-5 reads | all-5 VSC | net utility |",
          "|---|---|---|---|---|---|---|---|"]
    for k, a in arms.items():
        md.append(f"| {k} | {a['train_cost']} | {a['greedy_reads']:.3f} | {a['greedy_vsc']:.3f} | "
                  f"{a['greedy_false_success']:.3f} | {a['all5_reads']:.3f} | {a['all5_vsc']:.3f} | "
                  f"{a['net_utility_at_train_cost']:.3f} |")
    md += ["", "| Difference | estimate [95% CI] |", "|---|---|"]
    md += [f"| {k} | {v['diff']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}] |" for k, v in diffs.items()]
    md += ["", "| Prediction | met |", "|---|---|"] + [f"| {k} | {'yes' if v else 'no'} |" for k, v in preds.items()]
    (D / "c2_summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
