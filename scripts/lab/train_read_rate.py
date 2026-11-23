"""Descriptive: sampled (training-time) read behaviour of the first-stage AD arms, from their logged rollouts.

For each tag and step window: share of rollouts with >= 1 read, mean reads per rollout, and the share of groups with
zero reward variance (R = R_V - 0.05 n_verify). Input results/ad/rollouts/<tag>.jsonl.gz (one record per rollout).

    python scripts/lab/train_read_rate.py --out results/lab/train_read_rate.json
"""
from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TAGS = ["ad_grpo_s0", "ad_grpo_s1", "ad_nostd_s0", "ad_nostd_s1", "ad_awrb_s0", "ad_awrb_s1", "ad_awrbn_s0",
        "ad_awrbn_s1"]
WINDOWS = [(0, 20), (20, 50), (50, 100), (100, 150), (150, 200)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/lab/train_read_rate.json")
    a = ap.parse_args()
    out = {}
    for tag in TAGS:
        p = ROOT / "results/ad/rollouts" / f"{tag}.jsonl.gz"
        if not p.exists():
            out[tag] = {"status": "missing"}
            continue
        groups = defaultdict(list)
        with gzip.open(p, "rt") as f:
            for line in f:
                r = json.loads(line)
                groups[(int(r["step"] or 0), r["task_id"])].append(r)
        res = {}
        for lo, hi in WINDOWS:
            gs = [g for (s, _), g in groups.items() if lo <= s < hi]
            rs = [x for g in gs for x in g]
            if not rs:
                continue
            R = [[x["score"]["r_v"] - 0.05 * x["n_verify"] for x in g] for g in gs]
            res[f"{lo}-{hi}"] = {"rollouts": len(rs), "share_reading": sum(x["n_verify"] > 0 for x in rs) / len(rs),
                                 "reads_per_rollout": sum(x["n_verify"] for x in rs) / len(rs),
                                 "zero_var_groups": sum(max(r) - min(r) < 1e-12 for r in R) / len(R)}
        out[tag] = res
    (ROOT / a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
