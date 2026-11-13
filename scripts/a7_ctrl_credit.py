"""Credit calls on the control split `e_heldout_ctrl` (CPU; exploratory; Fable 5.2 next-steps review, analysis 3).

On `e_heldout` the A6 state arms never call `apply_store_credit` (0/200). On `e_heldout_ctrl` (same credit workflow,
faults moved to familiar tools) they call it in 25-27/200 episodes, all committed. This table asks which histories
those calls follow: by the task's fault (tool x type), and by what the model saw in its last tool result before the
credit call. If the calls concentrate on one fault history, the credit step is a history-conditional branch of the
inherited script, not a step RL removed everywhere.

    python scripts/a7_ctrl_credit.py results/raw/a6 > results/a7/ctrl_credit.md
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

MODELS = ("init_a6", "a6_pcstr_d_s0", "a6_pcstr_d_noreport_s0", "a6_pcstr_d_noND_s0", "a6_outcome_s0",
          "a6_ack_outcome_s0")


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "results/raw/a6")
    by_fault = defaultdict(Counter)
    tasks_with_credit = {}
    prev = defaultdict(Counter)
    totals = Counter()
    for m in MODELS:
        p = root / f"{m}__neutral__e_heldout_ctrl.jsonl"
        if not p.exists():
            continue
        tasks_with_credit[m] = set()
        for line in open(p):
            r = json.loads(line)
            a4 = r.get("a4") or {}
            fault = f"{a4.get('fault_tool') or 'none'} / {a4.get('fault_type') or '-'}"
            t = [s for s in r["trace"] if not s.get("by")]
            idx = next((i for i, s in enumerate(t) if s["tool"] == "apply_store_credit"), None)
            by_fault[m, fault]["n"] += 1
            totals[fault] += m == MODELS[0]
            if idx is None:
                continue
            by_fault[m, fault]["credit"] += 1
            tasks_with_credit[m].add(r["task_id"])
            prev_obs = t[idx - 1] if idx else None
            prev[m][f"{prev_obs['tool']} → {(prev_obs.get('obs') or {}).get('status')}" if prev_obs else "start"] += 1
    faults = sorted({f for _, f in by_fault})
    models = [m for m in MODELS if m in tasks_with_credit]
    print("## Credit calls on `e_heldout_ctrl` by task fault (episodes calling `apply_store_credit` / n)\n")
    print("| fault (tool / type) | " + " | ".join(models) + " |")
    print("|---|" + "---|" * len(models))
    for f in faults:
        print(f"| {f} | " + " | ".join(f"{by_fault[m, f]['credit']}/{by_fault[m, f]['n']}" for m in models) + " |")
    print("\n## The call immediately before the first credit call\n")
    for m in models:
        print(f"- **{m}**: " + ", ".join(f"{k} ×{v}" for k, v in prev[m].most_common(4)))
    state = [m for m in models if m not in ("init_a6", "a6_ack_outcome_s0")]
    if state:
        common = set.intersection(*(tasks_with_credit[m] for m in state))
        union = set.union(*(tasks_with_credit[m] for m in state))
        print(f"\nState arms ({len(state)}): {len(common)} tasks in which every state arm calls the credit tool; "
              f"{len(union)} in which any does.")


if __name__ == "__main__":
    main()
