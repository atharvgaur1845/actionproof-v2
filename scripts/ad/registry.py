"""Regenerate docs/research/algorithm_discovery/EXPERIMENT_REGISTRY.csv from the static run list below plus the
result files (status, results and wall-clock are read, never typed). CPU-only rows are static.

    python scripts/ad/registry.py [--worker-logs logs/ad_worker_times.txt]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/research/algorithm_discovery/EXPERIMENT_REGISTRY.csv"
COLS = ["exp_id", "candidate", "version", "hypothesis", "commit", "config_hash_file", "model", "tokenizer", "seed",
        "task_split", "train_steps", "train_episodes", "train_tokens", "wall_clock", "gpu", "memory", "checkpoint",
        "eval_results", "status", "failure_diagnosis", "next_decision"]
STATIC = [
    ["AD-T0", "theory", "toy v1", "group-std GRPO makes verification cost-blind", "6fa46d8", "results/ad_hashes.json",
     "tabular", "-", "0-3", "toy", "3000", "-", "-", "~3 min CPU", "none", "-", "-", "results/ad/t0/toy_bandit.json",
     "done", "modelled AWRB read rollouts as full-world Q (F1)", "superseded by AD-T0b"],
    ["AD-T0b", "theory", "toy v2", "corrected AWRB model; mirror bias; no-std correct", "ae50f4b",
     "results/ad_hashes_a1.json", "tabular", "-", "0-2", "toy", "3000", "-", "-", "~6 min CPU", "none", "-", "-",
     "results/ad/t0/toy_bandit_v2.json", "done", "-", "Amendment 1"],
    ["AD-S0", "baselines", "scripted", "verification-rule frontier", "6fa46d8", "results/ad_hashes.json", "scripted",
     "-", "-", "cw_dev", "-", "-", "-", "<1 min CPU", "none", "-", "-", "results/ad/scripted/cw_dev_frontier.json",
     "done", "-", "skip_post is the VoI optimum among rules (J_c 0.898)"],
    ["AD-AUD", "audit", "recompute", "independent recomputation of headline results", "bc24b5e", "-", "-", "-", "-",
     "s3_conf_*/s4_*/s3_rec_dev", "-", "-", "-", "~2 min CPU", "none", "-", "-", "results/ad/audit/recompute.json",
     "done", "point estimates reproduce; reported CIs hold runs fixed", "-"],
    ["AD-RW", "diagnostic", "readinfo", "share of provably uninformative reads", "7c91f5a", "-", "-", "-", "-", "cw_dev",
     "-", "-", "-", "~1 min CPU", "none", "-", "-", "results/ad/read_waste_cw_dev.json", "done",
     "42-49% of reads uninformative in every reading model", "D3"],
]
# (task id, tag, candidate, round, hash file, hypothesis)
RUNS = [
    (1200, "ad_smoke", "all", "D1a1", "results/ad_hashes_a1.json", "smoke"),
    (1210, "ad_base_*", "baselines", "D1a1", "results/ad_hashes_a1.json", "free-read models on cw_dev"),
    (1201, "ad_grpo_s0", "grpo", "D1a1", "results/ad_hashes_a1.json", "baseline (group std)"),
    (1202, "ad_nostd_s0", "B nostd", "D1a1", "results/ad_hashes_a1.json", "no-std restores VoI sign"),
    (1203, "ad_awrbn_s0", "A awrbn", "D1a1", "results/ad_hashes_a1.json", "RB returns + no std"),
    (1204, "ad_awrb_s0", "A awrb", "D1a1", "results/ad_hashes_a1.json", "RB returns + group std (mirror bias)"),
    (1205, "ad_awrb_s1", "A awrb", "D1a1", "results/ad_hashes_a1.json", "seed 1"),
    (1206, "ad_grpo_s1", "grpo", "D1a1", "results/ad_hashes_a1.json", "seed 1"),
    (1207, "ad_nostd_s1", "B nostd", "D1a1", "results/ad_hashes_a1.json", "seed 1"),
    (1208, "ad_awrbn_s1", "A awrbn", "D1a1", "results/ad_hashes_a1.json", "seed 1"),
    (1209, "ad_awrbc30_s0", "A awrb", "D1a1", "results/ad_hashes_a1.json", "c=0.30 intervention"),
    (1211, "ad_awrbnc30_s0", "A awrbn", "D1a1", "results/ad_hashes_a1.json", "c=0.30 intervention"),
    (1212, "ad_nostd_ema_s0", "B nostd_ema", "D2", "results/ad_hashes_d2.json", "EMA scale, raw returns"),
    (1213, "ad_awrbn_ema_s0", "A awrbn_ema", "D2", "results/ad_hashes_d2.json", "EMA scale, RB returns"),
    (1214, "ad_awrbn_ema_s1", "A awrbn_ema", "D2", "results/ad_hashes_d2.json", "seed 1"),
    (1215, "ad_nostd_ema_s1", "B nostd_ema", "D2", "results/ad_hashes_d2.json", "seed 1"),
    (1216, "ad_nostd_ema_irc_s0", "D3 irc", "D3", "results/ad_hashes_d3.json", "uninformative-read local credit"),
    (1217, "ad_nostd_ema_irc_s1", "D3 irc", "D3", "results/ad_hashes_d3.json", "seed 1"),
]


def result_of(tag: str) -> tuple[str, str]:
    for d, split in (("results/ad/dev", "cw_dev"), ("results/ad/conf", "cw_conf")):
        p = ROOT / d / f"{tag}__{split}.summary.json"
        if p.exists():
            s = json.loads(p.read_text())
            return "done", f"{split} VSC {s['vsc']:.3f} J_c {s['j_c']:.3f} reads {s['reads']:.2f}"
    return "queued/running", ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", nargs="*", default=[], help="extra C1 tags (training-only runs)")
    a = ap.parse_args()
    rows = [dict(zip(COLS, r)) for r in STATIC]
    for tid, tag, cand, rnd, h, hyp in RUNS + [(0, t, "C1", "C1", "results/ad_c1_hashes.json", "C1 seed") for t in a.extra]:
        st, res = result_of(tag) if "*" not in tag else ("done" if (ROOT / "results/ad/dev/ad_base_warm__cw_dev.summary.json").exists() else "queued", "see results/ad/dev/ad_base_*")
        seed = re.search(r"_s(\d)$", tag)
        rows.append(dict(zip(COLS, [str(tid or tag), cand, rnd, hyp, "see git log", h, "runs/rft_proc (CohereLabs/tiny-aya-global)",
                                    "CohereLabs/tiny-aya-global", seed.group(1) if seed else "-", "cw_train->cw_dev",
                                    "200" if "smoke" not in tag and "base" not in tag else "-", "6400" if "base" not in tag else "-",
                                    "see train_state num_tokens", "see SMOKE_TEST_LOG", "H200 lane (1 GPU)",
                                    "vLLM colocate 0.30", f"runs/{tag}" if "*" not in tag else "-", res, st, "", ""])))
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    print(OUT, len(rows), "rows")


if __name__ == "__main__":
    main()
