"""DPO pairs from verified trajectories (spec §13.2).

For two samples of the SAME task, one good (VSC = 1) and one bad (false completion,
duplicate side effect or invariant violation), find the first message where the
conversations differ. If that message is an assistant turn, the shared history is a clean
prompt and the two differing turns are chosen/rejected: single-step credit assignment with
no tool-output tokens inside the loss. Pairs whose first difference is an observation
(different env randomness) are skipped.

Bad-trajectory categories are kept as `kind` so we can report which failure modes DPO saw.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from actionproof.train.rft_data import neutralise


def failure_kind(r: dict[str, Any]) -> str | None:
    v = r["verdict"]
    if r["vsc"] == 1:
        return None
    if v["I"] > 0:
        return "violation"
    if v["D"] > 0:
        return "duplicate"
    if v["F_over"]:
        return "false_completion"
    return None  # honest failures are not "rejected" examples: honesty is what we want


def first_divergence(a: list[dict], b: list[dict]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i if x["role"] == y["role"] == "assistant" else None
    return None


def make_pairs(records: list[dict[str, Any]], max_per_task: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_task[r["task_id"]].append(r)
    pairs = []
    for task_id, rs in sorted(by_task.items()):
        good = [r for r in rs if r["vsc"] == 1]
        bad = [r for r in rs if failure_kind(r)]
        rng.shuffle(good)
        rng.shuffle(bad)
        n = 0
        for g in good:
            for b in bad:
                if n >= max_per_task:
                    break
                gc, bc = g["conversation"], b["conversation"]
                i = first_divergence(gc, bc)
                if i is None or i < 2:
                    continue
                prompt = [{**gc[0], "content": neutralise(gc[0]["content"])}] + gc[1:i]
                pairs.append({"prompt": prompt, "chosen": [gc[i]], "rejected": [bc[i]],
                              "task_id": task_id, "kind": failure_kind(b), "turn": i})
                n += 1
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", nargs="+", required=True, help="run_eval jsonl (teacher and/or policy samples)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-per-task", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    records = [json.loads(l) for f in a.samples for l in open(f)]
    pairs = make_pairs(records, a.max_per_task, a.seed)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    stats = {"records": len(records), "pairs": len(pairs), "tasks": len({p["task_id"] for p in pairs}),
             "kinds": dict(Counter(p["kind"] for p in pairs))}
    out.with_suffix(".stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
