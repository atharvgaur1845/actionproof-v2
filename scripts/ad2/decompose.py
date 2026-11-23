"""Independent reproduction and refinement of the first-stage advantage decomposition (P1 in the LLM).

Input: the logged training rollouts results/ad/rollouts/<tag>.jsonl.gz (one record per rollout; a group is the 8
rollouts with the same (step, task_id)). Returns R = R_V - c * n_verify, exactly the training reward of the grpo /
nostd arms (actionproof/ad/rewards.py, c = 0.05).

Denominators:
  pop    (R - mean_g) / (pstdev_g + 1e-4)          first-stage script (scripts/ad/advantage_decomposition.py)
  trl    (R - mean_g) / (std_g(Bessel) + 1e-4)     TRL 1.7.1 scale_rewards="group" (trainer/utils.py:nanstd)
  none   R - mean_g                                TRL scale_rewards="none"

Group types (by what varies inside the group):
  clean          hidden world clean
  fault_outcome  faulted world and R_V differs between rollouts (some rollout's behaviour changed the outcome)
  fault_cost     faulted world and R_V is constant (every rollout succeeded or every rollout failed; only reads vary)
  zero           R constant in the group (no gradient under any normalisation)

Aggregation: the summed advantage of rollouts that made >= 1 read, per group type. Also token-weighted with the
number of agent calls as the length proxy (TRL's 'dapo' loss weights every completion token equally; logs carry no
token counts, so this is a proxy, flagged as such).

    python scripts/ad2/decompose.py --tags ad_grpo_s0 ad_nostd_s0 --steps 0 40 --out results/ad2/decomposition.json
"""
from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev, stdev

ROOT = Path(__file__).resolve().parents[2]
EPS = 1e-4


def groups_of(tag: str, lo: int, hi: int, rdir: str = "results/ad/rollouts") -> list[list[dict]]:
    g = defaultdict(list)
    with gzip.open(ROOT / rdir / f"{tag}.jsonl.gz", "rt") as f:
        for line in f:
            r = json.loads(line)
            s = int(r["step"] or 0)
            if lo <= s < hi:
                g[(s, r["task_id"])].append(r)
    return [v for v in g.values() if len(v) > 1]


def gtype(g: list[dict], R: list[float]) -> str:
    if max(R) - min(R) < 1e-12:
        return "zero"
    if len({x["fault_class"] for x in g}) > 1:
        return "mixed"  # condition B: worlds differ within the group
    if g[0]["fault_class"] == "cw:clean":
        return "clean"
    rv = {x["score"]["r_v"] for x in g}
    return "fault_outcome" if len(rv) > 1 else "fault_cost"


def decompose(tag: str, lo: int, hi: int, cost: float, rdir: str = "results/ad/rollouts") -> dict:
    acc = {n: defaultdict(float) for n in ("pop", "trl", "none")}
    accw = {n: defaultdict(float) for n in ("pop", "trl", "none")}
    cnt = defaultdict(int)
    nread = defaultdict(int)
    for g in groups_of(tag, lo, hi, rdir):
        R = [x["score"]["r_v"] - cost * x["n_verify"] for x in g]
        t = gtype(g, R)
        cnt[t] += 1
        mu = mean(R)
        den = {"pop": pstdev(R) + EPS, "trl": stdev(R) + EPS, "none": 1.0}
        for x, r in zip(g, R):
            if x["n_verify"] == 0:
                continue
            nread[t] += 1
            L = len(x["trace"])
            for n, d in den.items():
                acc[n][t] += (r - mu) / d
                accw[n][t] += L * (r - mu) / d
    out = {"groups": dict(cnt), "read_rollouts": dict(nread)}
    for n in acc:
        a, w = acc[n], accw[n]
        out[n] = {"by_type": {k: round(v, 3) for k, v in a.items()},
                  "clean": round(a["clean"], 3), "faulted": round(a["fault_outcome"] + a["fault_cost"], 3),
                  "net": round(sum(a.values()), 3),
                  "token_weighted_net": round(sum(w.values()), 3)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--steps", nargs=2, type=int, default=[0, 40])
    ap.add_argument("--windows", type=int, default=40)
    ap.add_argument("--cost", type=float, default=0.05)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dir", default="results/ad/rollouts")
    a = ap.parse_args()
    res = {}
    for tag in a.tags:
        res[tag] = {"main": decompose(tag, a.steps[0], a.steps[1], a.cost, a.dir),
                    "windows": {f"{s}-{s + a.windows}": decompose(tag, s, s + a.windows, a.cost, a.dir)
                                for s in range(0, 200, a.windows)}}
        m = res[tag]["main"]
        print(tag, "groups", m["groups"], "read rollouts", m["read_rollouts"])
        for n in ("pop", "trl", "none"):
            print(f"  {n:5s}", m[n])
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
