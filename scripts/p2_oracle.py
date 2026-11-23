"""P2 reference policy (PREREGISTRATION_P2.md §4): the scripted oracle on the test and shift splits, one
deterministic episode per task, scored by the same verifier and metrics as the models.

The oracle (`actionproof/integrity/sim/agents.py::oracle`, `actionproof/stage3/scripted.py::oracle` for recovery
starts) decides from read results and the request; it never reads the fault log. It encodes the fault taxonomy by
hand, so it is a reference for what written checks achieve, not a learned policy. It has no sampled decoding: its
single episode is recorded as sample −1 only.

    python scripts/p2_oracle.py
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("p1_eval", ROOT / "scripts/p1_eval.py")
P1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P1)
SPLITS = ("p1_test_full", "p1_test_pp", "p1_test_rec", "p1_test_wf4", "p1_test_combo", "s4_lang_hi", "s4_lang_hinglish")


def main() -> None:
    from actionproof.bench.generator_proc import DATA, load_tasks
    from actionproof.integrity.sim import agents
    from actionproof.stage3 import scripted
    from actionproof.stage3.env import make_env_s3
    for sp in SPLITS:
        out = ROOT / f"results/p2/ref/O__{sp}.jsonl"
        if out.exists():
            print("exists", out)
            continue
        rows = []
        for t in load_tasks(DATA / f"{sp}.jsonl"):
            env = make_env_s3(t, f"p2-oracle-{t.task_id}")
            for tool, args in t.meta.get("start_prefix") or []:
                env.call(tool, dict(args))
            n0 = len(env.trace)
            (scripted.oracle if t.meta.get("start_prefix") else agents.oracle)(env, t)
            r = P1.E1.S3E.episode_record_proc(t, env)
            status = (r["report"] or {}).get("status")
            r.update(sample=-1, turns=len(env.trace) - n0, invalid=0, ended_by="report" if env.done else "max_turns",
                     conversation=None, workflow=t.workflow, start_type=t.meta.get("start_type"),
                     violation=P1.E1.S3E.R.violation(r),
                     honest_fail=status in ("failed", "escalated") and not r["score"]["goal_met"],
                     policy_calls=len(env.trace) - n0, guard_blocks=[], ledger=False, guard=False,
                     **P1.episode_metrics({**r, "task": {"meta": t.meta}}, n0))
            rows.append(r)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))
        s = P1.summarise(rows)
        out.with_suffix(".summary.json").write_text(json.dumps({"tag": "O", "split": sp, **s}, indent=2))
        print(sp, "greedy vsc", round(s["greedy"]["vsc"], 3))


if __name__ == "__main__":
    main()
