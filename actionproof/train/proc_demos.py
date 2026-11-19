"""Amendment 12 warm-start demonstrations on ProcureSim -> SFT rows in the `train/rft.py` row format.

Each `proc_demo` task (a split disjoint from every training and evaluation split) gets one episode from one
scripted demonstrator, drawn with a fixed seed: the verifying `oracle` with probability VERIFY_SHARE, else the
acknowledgement-trusting `naive` policy. The plan is replayed through the real `run_episode` loop with the
neutral procurement prompt, so the conversations are byte-identical to what training and evaluation show.
All episodes are kept (the warm start must carry both behaviours for GRPO to select between them).
Report summaries vary over three templates so the report text is not one fixed string.

    python -m actionproof.train.proc_demos --out data/rft/proc_demo_rows.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from actionproof.bench.generator_proc import make_split
from actionproof.integrity.sim.agents import POLICIES
from actionproof.integrity.sim.prompts import PROC_NEUTRAL
from actionproof.integrity.sim.runtime import episode_record_proc, make_env_proc
from actionproof.rollout.episode import run_episode

VERIFY_SHARE, SEED, MAX_TURNS = 0.30, 1201, 16
SUMMARIES = {
    "completed": ("{steps} done for {po}.", "Finished {steps} for purchase order {po}.", "{po}: {steps} completed."),
    "failed": ("Could not complete {steps} for {po}.", "{po}: the requested changes did not all take effect.",
               "Stopped on {po}: a requested step does not show in the records."),
}


def plan(task, policy: str, rng: random.Random) -> list[str]:
    probe = make_env_proc(task, f"plan-{task.task_id}")
    POLICIES[policy](probe, task)
    acts = []
    for s in probe.trace:
        args = dict(s["args"])
        if s["tool"] == "submit_report":
            tpl = rng.choice(SUMMARIES.get(args["status"], SUMMARIES["completed"]))
            args["summary"] = tpl.format(steps=", ".join(task.meta["requested"]), po=task.meta["po_id"])
        acts.append(json.dumps([{"tool_name": s["tool"], "parameters": args}]))
    return acts


def turn_rows(conv: list[dict[str, str]], meta: dict) -> list[dict]:
    return [{"prompt": conv[:i], "completion": [m], **meta,
             "turn": sum(x["role"] == "assistant" for x in conv[:i])}
            for i, m in enumerate(conv) if m["role"] == "assistant"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--verify-share", type=float, default=VERIFY_SHARE)
    a = ap.parse_args()
    rng = random.Random(SEED)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    stats = {"episodes": 0, "rows": 0, "by_policy": {}, "vsc_by_policy": {}}
    with out.open("w") as f:
        for t in make_split("proc_demo"):
            policy = "oracle" if rng.random() < a.verify_share else "naive"
            acts = plan(t, policy, rng)
            assert len(acts) <= MAX_TURNS, (t.task_id, len(acts))
            it = iter(acts)
            env = make_env_proc(t, f"demo-{t.task_id}")
            conv, st = run_episode(env, lambda c: next(it), PROC_NEUTRAL, t.prompt, max_turns=MAX_TURNS)
            assert st.ended_by == "report" and len(env.trace) == len(acts), t.task_id
            rec = episode_record_proc(t, env)
            meta = {"episode": f"{t.task_id}#{policy}", "task_id": t.task_id, "policy": policy,
                    "fault_class": t.meta["fault_class"], "vsc": rec["vsc"]}
            for row in turn_rows(conv, meta):
                f.write(json.dumps(row) + "\n")
                stats["rows"] += 1
            stats["episodes"] += 1
            stats["by_policy"][policy] = stats["by_policy"].get(policy, 0) + 1
            stats["vsc_by_policy"][policy] = stats["vsc_by_policy"].get(policy, 0) + rec["vsc"]
    out.with_suffix(".stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
