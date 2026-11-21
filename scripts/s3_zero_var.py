"""S3 addendum 2, Part C: how often GRPO groups on partial faults carry no learning signal.

Input: the training rollout logs `results/s3/rollouts/<tag>.jsonl.gz` (one line per rollout; a group is one task at
one step). For every group whose fault is partial, by workflow, faulted tool, start (recovery start or full episode)
and training half: the number of groups, the share with zero reward variance (all 0 or all 1), and the share all 0.
R_V is the episode's VSC, so the variance of `vsc` is the reward variance.
    python scripts/s3_zero_var.py --tags s3_rv_mix_s0,...,s3_matched_s0,... --out results/s3/s3_zero_var.json
"""
from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def groups(path: Path):
    g: dict = defaultdict(list)
    meta: dict = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            k = (int(r["step"]), r["task_id"])
            g[k].append(float(r["vsc"]))
            t = r["task"]
            fl = (t["meta"].get("faults") or [{}])[0]
            meta[k] = {"kind": fl.get("kind"), "tool": fl.get("tool"), "workflow": t["workflow"],
                       "start": "recovery" if (r.get("start_type") or t["meta"].get("start_prefix")) else "full"}
    return g, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True)
    ap.add_argument("--dir", default="results/s3/rollouts")
    ap.add_argument("--out", default="results/s3/s3_zero_var.json")
    a = ap.parse_args()
    res = {}
    for tag in a.tags.split(","):
        p = ROOT / a.dir / f"{tag}.jsonl.gz"
        if not p.exists():
            res[tag] = None
            continue
        g, meta = groups(p)
        cells: dict = defaultdict(lambda: {"groups": 0, "zero_var": 0, "all_zero": 0, "rollouts": 0, "vsc_sum": 0.0})
        for k, v in g.items():
            m = meta[k]
            if m["kind"] != "partial":
                continue
            half = "steps_0_99" if k[0] < 100 else "steps_100_199"
            for key in (f"{m['start']}|{m['workflow']}|{m['tool']}|all", f"{m['start']}|{m['workflow']}|{m['tool']}|{half}",
                        f"{m['start']}|all|all|all"):
                c = cells[key]
                c["groups"] += 1
                c["zero_var"] += len(set(v)) == 1
                c["all_zero"] += max(v) == 0
                c["rollouts"] += len(v)
                c["vsc_sum"] += sum(v)
        res[tag] = {k: {"groups": c["groups"], "zero_var_share": c["zero_var"] / c["groups"],
                        "all_zero_share": c["all_zero"] / c["groups"], "mean_vsc": c["vsc_sum"] / c["rollouts"]}
                    for k, c in sorted(cells.items())}
    out = ROOT / a.out
    out.write_text(json.dumps(res, indent=2))
    for tag, cells in res.items():
        if cells is None:
            print(tag, "missing")
            continue
        for k, c in cells.items():
            if k.endswith("|all") and "|pay_supplier|" in k or k.endswith("all|all|all"):
                print(f"{tag:16s} {k:44s} n {c['groups']:3d} zero-var {c['zero_var_share']:.2f} "
                      f"all-0 {c['all_zero_share']:.2f} vsc {c['mean_vsc']:.2f}")


if __name__ == "__main__":
    main()
