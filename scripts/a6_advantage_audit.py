"""Step-0 GRPO advantage audit for the A6 report-term arms (CPU; exploratory). The gate file is the warm start sampled
T=1 x 4 on 100 pool-B tasks, i.e. 100 GRPO groups as the first training step would see them. Each episode is
rescored under every arm's reward, advantages are group-normalised as in GRPO ((r - mean) / std), and the advantage
mass on each report behaviour is summed. Positive mass on "failed" means the arm pushes the shared "failed" report
up; negative means it pushes it down.

    python scripts/a6_advantage_audit.py results/raw/a6/gate_init__a4_train_B_T1x4.jsonl > results/a6/advantage_audit.md
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from types import SimpleNamespace

from actionproof.rewards import a6_rewards as A
from actionproof.rewards.pcstr import pcstr_score

ARMS = {"M1 pcstr_d": lambda v: pcstr_score(v, disclosure=True), "M2b noND (F_under kept)": A.pcstr_d_noND_score,
        "M2 noreport": A.pcstr_d_noreport_score, "M3' outcome_G": A.outcome_G_score,
        "M3 outcome": lambda v: float(v.P == 1 and v.I == 0)}


def label(rec: dict) -> str:
    v, cls = rec["verdict"], (rec.get("a4") or {}).get("env_class", "?")
    rep = v["reported"] or "none"
    if rep in ("failed", "escalated"):
        return f"{rep} on {cls}" + (" (underclaim)" if v["F_under"] else "")
    return f"{rep} on {cls}" + (" (G=1)" if v["G"] == 1 else " (G=0)")


def main() -> None:
    recs = [json.loads(l) for l in open(sys.argv[1])]
    groups = defaultdict(list)
    for r in recs:
        groups[r["task_id"]].append(r)
    print(f"Groups: {len(groups)} tasks x {len(recs) // len(groups)} samples (warm start, T=1).\n")
    mass = {a: defaultdict(float) for a in ARMS}
    count = defaultdict(int)
    informative = {a: 0 for a in ARMS}
    for g in groups.values():
        for r in g:
            count[label(r)] += 1
        for a, fn in ARMS.items():
            rs = [fn(SimpleNamespace(**r["verdict"])) for r in g]
            sd = statistics.pstdev(rs)
            if sd < 1e-9:
                continue
            informative[a] += 1
            mu = statistics.mean(rs)
            for r, x in zip(g, rs):
                mass[a][label(r)] += (x - mu) / sd
    print("Informative groups (non-zero reward variance): " + ", ".join(f"{a} {n}" for a, n in informative.items()))
    print("\n| behaviour | n | " + " | ".join(ARMS) + " |")
    print("|---|---|" + "---|" * len(ARMS))
    for lab in sorted(count, key=lambda k: (-count[k], k)):
        print(f"| {lab} | {count[lab]} | " + " | ".join(f"{mass[a][lab]:+.2f}" for a in ARMS) + " |")
    print("\n**Net advantage mass on the report token 'failed' (all classes):**")
    for a in ARMS:
        print(f"- {a}: {sum(v for k, v in mass[a].items() if k.startswith('failed')):+.2f}")


if __name__ == "__main__":
    main()
