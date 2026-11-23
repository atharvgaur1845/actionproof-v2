"""Mechanism check of P1 on LLM training data (descriptive). For each training group in a step range, recompute every
rollout's advantage under two normalisations, applied to the same logged returns R_c = R_V - c * reads:
  group-std  (R - mean_g) / (std_g + 1e-4)   (TRL default; the population std is used here)
  none       R - mean_g
Average the advantage of rollouts that read vs did not, split by the group's hidden world (clean vs faulted). Report
also each world class's share of the summed read-advantage: under P1, group std makes clean-world groups pull against
reading about as hard as faulted-world groups pull for it.

    python scripts/ad/advantage_decomposition.py --tag ad_grpo_s0 --steps 0 40
"""
from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--steps", nargs=2, type=int, default=[0, 40])
    ap.add_argument("--cost", type=float, default=0.05)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = {}
    for tag in a.tags:
        groups = defaultdict(list)
        with gzip.open(ROOT / "results/ad/rollouts" / f"{tag}.jsonl.gz", "rt") as f:
            for line in f:
                r = json.loads(line)
                s = int(r["step"] or 0)
                if a.steps[0] <= s < a.steps[1]:
                    groups[(s, r["task_id"])].append(r)
        acc = {k: defaultdict(list) for k in ("std", "none")}
        for g in groups.values():
            if len(g) < 2:
                continue
            R = [x["score"]["r_v"] - a.cost * x["n_verify"] for x in g]
            mu, sd = mean(R), pstdev(R)
            world = "clean" if g[0]["fault_class"] == "cw:clean" else "faulted"
            for x, rr in zip(g, R):
                key = (world, "read" if x["n_verify"] > 0 else "no_read")
                acc["std"][key].append((rr - mu) / (sd + 1e-4))
                acc["none"][key].append(rr - mu)
        out = {}
        for norm, d in acc.items():
            out[norm] = {f"{w}/{r}": {"mean_adv": mean(v), "n": len(v), "sum": sum(v)} for (w, r), v in sorted(d.items())}
            pull_clean = sum(d[("clean", "read")]) if d[("clean", "read")] else 0.0
            pull_fault = sum(d[("faulted", "read")]) if d[("faulted", "read")] else 0.0
            out[norm]["summed_read_advantage"] = {"clean": pull_clean, "faulted": pull_fault, "net": pull_clean + pull_fault}
        res[tag] = out
        print(tag, json.dumps({n: {k: (round(v["mean_adv"], 3), v["n"]) if "mean_adv" in v else
                                   {kk: round(vv, 2) for kk, vv in v.items()} for k, v in o.items()}
                               for n, o in out.items()}))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
