"""A11 H4: draw the 60-episode human-audit sample from the judge pool and allocate conditions (between subjects).

Strata (from the existing 815-episode pool, results/judge/pool.jsonl; every pooled episode reported "completed"):
  invisible failure: G = 0 and no verification read after the faulted write (failure not visible in the transcript)
  visible failure:   G = 0 and a verification read followed the faulted write
  true completion:   G = 1
20 per stratum, drawn with seed 1111; within each stratum 10 go to "transcript" and 10 to "ledger" (the ledger
condition also shows the receipt ledger the grounded judge saw). Each item is labelled once, in random order.
Writes data/h4/items.jsonl (what the annotator sees, no truth) and data/h4/key.jsonl (truth and judge scores).
Run on the cluster (needs the pool's source files):  python scripts/h4_sample.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from actionproof.eval.judge_audit import attach_attempts, with_policy
from actionproof.rewards.llm_judge import render_ledger, render_report, render_trajectory

ROOT = Path(__file__).resolve().parents[1]
SEED, PER = 1111, 20


def main() -> None:
    pool = [json.loads(l) for l in open(ROOT / "results/judge/pool.jsonl")]
    attach_attempts(pool)
    strata = {"invisible_failure": [p for p in pool if p["G"] == 0 and not p["visible"]],
              "visible_failure": [p for p in pool if p["G"] == 0 and p["visible"]],
              "true_completion": [p for p in pool if p["G"] == 1]}
    judge = {}
    for tag in ("aya32b_t", "aya32b_grounded"):
        for name in ("auditkit_tc", "llm_judge"):
            for line in open(ROOT / f"results/judge/{tag}__{name}.jsonl"):
                r = json.loads(line)
                judge.setdefault((r["source"], r["task_id"], r["fault"]), {})[f"{tag}__{name}"] = r["score"]
    rng = random.Random(SEED)
    items, key = [], []
    for s, rows in strata.items():
        pick = rng.sample(rows, PER)
        conds = ["transcript"] * (PER // 2) + ["ledger"] * (PER // 2)
        rng.shuffle(conds)
        for p, c in zip(pick, conds):
            text = (f"TASK AND POLICY\n{with_policy(p['task'])}\n\nAGENT TOOL CALLS\n{render_trajectory(p['trace'])}\n\n"
                    f"AGENT REPORT\n{render_report(p['report'])}")
            if c == "ledger":
                text += f"\n\nRECEIPT LEDGER (from the database: which writes actually committed)\n{render_ledger(p['attempts'])}"
            items.append({"condition": c, "text": text})
            key.append({"stratum": s, "condition": c, "source": p["source"], "task_id": p["task_id"], "fault": p["fault"],
                        "G": p["G"], "visible": p["visible"],
                        "judges": judge.get((p["source"], p["task_id"], p["fault"]), {})})
    order = list(range(len(items)))
    rng.shuffle(order)
    out = ROOT / "data/h4"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "items.jsonl", "w") as fi, open(out / "key.jsonl", "w") as fk:
        for n, i in enumerate(order):
            fi.write(json.dumps({"item": n, **items[i]}) + "\n")
            fk.write(json.dumps({"item": n, **key[i]}) + "\n")
    print({s: len(v) for s, v in strata.items()}, "->", len(items), "items")


if __name__ == "__main__":
    main()
