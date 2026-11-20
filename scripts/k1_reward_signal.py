"""K1 key secondary (PREREGISTRATION_K1 §3): sampled read rate over training and per-class informative-group shares,
from the K1 rollout logs (results/k1/rollouts/k1_<pool>_ra_s<seed>.jsonl.gz), with the A12 R_A runs (full pool) as the
reference. Reuses `analyse` from scripts/a12_reward_signal.py unchanged.
    python scripts/k1_reward_signal.py --out results/k1/k1_reward_signal.json
"""
from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("rs", ROOT / "scripts/a12_reward_signal.py")
rs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rs)


def load(path: Path) -> list[dict]:
    keep = ("step", "task_id", "fault_class", "n_verify", "score")
    with gzip.open(path, "rt") as f:
        recs = [{k: r[k] for k in keep} for r in map(json.loads, f)]
    by: dict[int, list] = defaultdict(list)
    for r in recs:
        by[r["step"]].append(r)
    return [r for st in sorted(by) for r in by[st][-rs.PER_STEP:]]


def pooled_read_rate(recs: list[dict]) -> dict:
    by: dict[str, list] = defaultdict(list)
    for r in recs:
        b = (r["step"] // rs.BLOCK) * rs.BLOCK
        by[f"{b}-{b + rs.BLOCK - 1}"].append(r["n_verify"] > 0)
    return {b: round(sum(v) / len(v), 4) for b, v in sorted(by.items(), key=lambda kv: int(kv[0].split("-")[0]))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/k1/k1_reward_signal.json")
    a = ap.parse_args()
    runs = {f"k1_{p}_ra_s{s}": ROOT / f"results/k1/rollouts/k1_{p}_ra_s{s}.jsonl.gz" for p in ("k1t", "k1p") for s in (0, 1, 2)}
    runs |= {f"a12_proc_ra_s{s}": ROOT / f"results/a12/rollouts/a12_proc_ra_s{s}.jsonl.gz" for s in (0, 1, 2)}
    res = {"note": "K1 key secondary; read = any authoritative read in the rollout", "runs": {}, "missing": []}
    for tag, p in runs.items():
        if not p.exists():
            res["missing"].append(tag)
            continue
        recs = load(p)
        res["runs"][tag] = {"read_rate_by_block": pooled_read_rate(recs), "by_class": rs.analyse(recs, "r_a")}
        print(tag, res["runs"][tag]["read_rate_by_block"])
    (ROOT / a.out).write_text(json.dumps(res, indent=2))
    print("missing:", res["missing"])


if __name__ == "__main__":
    main()
