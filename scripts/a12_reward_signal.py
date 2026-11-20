"""A12 addendum 2 §3, reward-signal analysis (exploratory). From the training rollout logs
(results/a12/rollouts/a12_<arm>_s<seed>.jsonl[.gz]; one record per rollout, 8 rollouts per task per step), per arm,
fault class and block of 50 steps:
  mean_reward          mean training reward of the arm (r_v or r_a)
  zero_var_share       share of (step, task) groups whose 8 rewards are all equal (no GRPO gradient)
  informative_share    this class's share of all informative groups in the block
  read_rate            share of rollouts with at least one authoritative read (n_verify > 0)
  read_gap             within informative groups that contain both readers and non-readers: mean reward of readers
                       minus non-readers (the sign tells which behaviour the group's advantage favours)
    python scripts/a12_reward_signal.py --seed 0 --out results/a12/reward_signal_s0.json
"""
from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARMS = {"proc_rv": "r_v", "proc_ra": "r_a"}
BLOCK = 50
PER_STEP = 32  # 4 tasks x 8 rollouts per GRPO step


def load(arm: str, seed: int) -> list[dict]:
    # the .gz is the complete log written at the end of training; a plain .jsonl may be a partial periodic upload
    gz = ROOT / f"results/a12/rollouts/a12_{arm}_s{seed}.jsonl.gz"
    f = gzip.open(gz, "rt") if gz.exists() else (ROOT / f"results/a12/rollouts/a12_{arm}_s{seed}.jsonl").open()
    keep = ("step", "task_id", "fault_class", "n_verify", "score")
    recs = [{k: r[k] for k in keep} for r in map(json.loads, f)]
    # a resumed run (D-A12-2) logs some steps twice: keep the last PER_STEP records of each step (the resumed ones)
    by_step: dict[int, list] = defaultdict(list)
    for r in recs:
        by_step[r["step"]].append(r)
    return [r for st in sorted(by_step) for r in by_step[st][-PER_STEP:]]


def analyse(recs: list[dict], rkey: str) -> dict:
    groups: dict[tuple, list] = defaultdict(list)
    for r in recs:
        groups[(r["step"], r["task_id"])].append(r)
    out: dict = defaultdict(lambda: defaultdict(lambda: {"groups": 0, "zero_var": 0, "rollouts": 0, "reward": 0.0,
                                                         "reads": 0, "gap_sum": 0.0, "gap_n": 0}))
    for (step, _), g in groups.items():
        b = f"{(step // BLOCK) * BLOCK}-{(step // BLOCK) * BLOCK + BLOCK - 1}"
        c = out[b][g[0]["fault_class"]]
        rewards = [float(r["score"][rkey]) for r in g]
        c["groups"] += 1
        c["zero_var"] += len(set(rewards)) == 1
        c["rollouts"] += len(g)
        c["reward"] += sum(rewards)
        c["reads"] += sum(r["n_verify"] > 0 for r in g)
        rd = [x for x, r in zip(rewards, g) if r["n_verify"] > 0]
        nr = [x for x, r in zip(rewards, g) if r["n_verify"] == 0]
        if rd and nr and len(set(rewards)) > 1:
            c["gap_sum"] += sum(rd) / len(rd) - sum(nr) / len(nr)
            c["gap_n"] += 1
    res = {}
    for b, classes in sorted(out.items(), key=lambda kv: int(kv[0].split("-")[0])):
        informative = sum(c["groups"] - c["zero_var"] for c in classes.values())
        res[b] = {cls: {"groups": c["groups"], "mean_reward": round(c["reward"] / c["rollouts"], 4),
                        "zero_var_share": round(c["zero_var"] / c["groups"], 4),
                        "informative_share": round((c["groups"] - c["zero_var"]) / informative, 4) if informative else None,
                        "read_rate": round(c["reads"] / c["rollouts"], 4),
                        "read_gap": round(c["gap_sum"] / c["gap_n"], 4) if c["gap_n"] else None,
                        "read_gap_groups": c["gap_n"]}
                  for cls, c in sorted(classes.items())}
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {"note": "exploratory (PREREGISTRATION_A12_ADDENDUM2 §3); read = any authoritative read in the rollout",
           "block_steps": BLOCK}
    for arm, rkey in ARMS.items():
        res[arm] = analyse(load(arm, a.seed), rkey)
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    for arm in ARMS:
        for b, classes in res[arm].items():
            print(arm, b, {k: (v["mean_reward"], v["zero_var_share"], v["informative_share"], v["read_rate"], v["read_gap"])
                           for k, v in classes.items()})


if __name__ == "__main__":
    main()
