"""RFT data: verifier-filtered teacher trajectories -> per-turn SFT rows (spec §13.1).

Pipeline
  1. keep teacher episodes with VSC == 1          (the verifier as a curation signal)
  2. swap the teacher's verify-prompt for NEUTRAL  (context distillation: the student must
     internalise verification, because training and evaluation use the neutral prompt)
  3. dedup identical action sequences, cap per task (no task dominates)
  4. one conversational prompt/completion row per assistant turn (loss on completions only)

`--raw` builds the curation ablation: the same number of episodes sampled WITHOUT the VSC
filter, so "curated vs raw" differs only in selection.

Rows keep the exact Cohere fallback format the teacher produced (AgentTune's
`[{"tool_name": ..., "parameters": ...}]` + tool results folded into user turns), i.e. the
format GRPO and evaluation use. AgentTune's create_distill_trainer is NOT used: it renders
`<tool_call>` tags and `role: tool` turns, which Aya templates reject (logged deviation).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from actionproof.bench.prompts import NEUTRAL, VERIFY_PROMPTS


def neutralise(system: str) -> str:
    """Replace a verify-prompt prefix with NEUTRAL, keeping the tool-schema block intact."""
    for vp in sorted(VERIFY_PROMPTS.values(), key=len, reverse=True):  # longest match first
        if system.startswith(vp):
            return NEUTRAL + system[len(vp):]
    if system.startswith(NEUTRAL):
        return system
    raise ValueError("system prompt does not start with a known ActionProof prompt")


def action_signature(conv: list[dict[str, str]]) -> str:
    acts = [m["content"].strip() for m in conv if m["role"] == "assistant"]
    return hashlib.sha1(json.dumps(acts).encode()).hexdigest()


def turn_rows(conv: list[dict[str, str]], meta: dict[str, Any]) -> list[dict[str, Any]]:
    conv = [{**conv[0], "content": neutralise(conv[0]["content"])}] + conv[1:]
    rows = []
    for i, m in enumerate(conv):
        if m["role"] == "assistant":
            rows.append({"prompt": conv[:i], "completion": [m], **meta, "turn": sum(
                x["role"] == "assistant" for x in conv[:i])})
    return rows


def select(records: list[dict[str, Any]], raw: bool, per_task_cap: int, n_target: int | None,
           seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    pool = list(records) if raw else [r for r in records if r["vsc"] == 1]
    rng.shuffle(pool)
    seen, per_task, out = set(), defaultdict(int), []
    for r in pool:
        sig = action_signature(r["conversation"])
        if sig in seen or per_task[r["task_id"]] >= per_task_cap:
            continue
        seen.add(sig)
        per_task[r["task_id"]] += 1
        out.append(r)
    return out[:n_target] if n_target else out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", nargs="+", required=True, help="run_eval jsonl files from the teacher")
    ap.add_argument("--out", required=True)
    ap.add_argument("--raw", action="store_true", help="curation ablation: no VSC filter")
    ap.add_argument("--match-curated", default=None, help="curated rows file whose episode count raw must match")
    ap.add_argument("--per-task-cap", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    records = [json.loads(l) for f in a.teacher for l in open(f)]
    n_target = None
    if a.match_curated:
        n_target = len({json.loads(l)["episode"] for l in open(a.match_curated)})
    eps = select(records, a.raw, a.per_task_cap, n_target, a.seed)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    with out.open("w") as f:
        for r in eps:
            meta = {"episode": f"{r['task_id']}#{r.get('sample', 0)}", "task_id": r["task_id"],
                    "vsc": r["vsc"], "faults": r.get("faults", [])}
            for row in turn_rows(r["conversation"], meta):
                f.write(json.dumps(row) + "\n")
                n_rows += 1
    keep_rate = sum(r["vsc"] for r in records) / max(len(records), 1)
    stats = {"teacher_episodes": len(records), "teacher_vsc": keep_rate, "selected_episodes": len(eps),
             "rows": n_rows, "raw": a.raw, "tasks_covered": len({r["task_id"] for r in eps}),
             "selected_vsc": sum(r["vsc"] for r in eps) / max(len(eps), 1)}
    out.with_suffix(".stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
