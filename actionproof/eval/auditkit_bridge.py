"""ActionProof episodes -> AuditKIT agent evaluation (toolkit chain, spec §15/§17).

Each episode record becomes an AuditKIT `AgentEpisode` (OpenAI-style tool calls rebuilt from the env
trace; final answer = the submit_report payload). `final_state` carries the DB-verified verdict from
our verifier, and each task becomes an `AgentCase` whose outcome oracle is AuditKIT's own
`FinalStateAssertion("vsc", equals=1)`, so AuditKIT's "verified state wins over text" outcome logic
scores the run and never sees an `unknown` (every episode has a snapshot). Trace diagnostics come
from AuditKIT scorers: tool_call_validity, redundant_tool_calls, agent_loop_detection, tool_permission.

    python -m actionproof.eval.auditkit_bridge results/eval/r3/grpo_tiny_v2_pcstr_d_s0__neutral__test_iid.jsonl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from actionproof.bench.generator import make_split

SCORERS = ["tool_call_validity", "redundant_tool_calls", "agent_loop_detection", "tool_permission"]


def to_messages(rec: dict) -> list[dict]:
    msgs = [{"role": "user", "content": rec["conversation"][1]["content"]}]
    for s in rec["trace"]:
        cid = f"c{s['idx']}"
        msgs.append({"role": "assistant", "content": None, "tool_calls": [
            {"id": cid, "type": "function", "function": {"name": s["tool"], "arguments": json.dumps(s["args"])}}]})
        msgs.append({"role": "tool", "tool_call_id": cid, "content": json.dumps(s["obs"])})
    return msgs


def build(path: str):
    from auditkit.agent_eval import AgentCase, FinalStateAssertion, episode_from_openai_messages
    recs = [json.loads(l) for l in open(path)]
    split = recs[0]["task_id"].rsplit("_", 1)[0]
    tasks = {t.task_id: t for t in make_split(split)}
    episodes, cases = [], []
    for r in recs:
        t = tasks[r["task_id"]]
        report = json.dumps(r["report"]) if r.get("report") else ""
        ep = episode_from_openai_messages(to_messages(r), case_id=r["task_id"], final_answer=report)
        v = r["verdict"]
        ep.final_state = {"vsc": r["vsc"], "G": v["G"], "goals_met_frac": v["P"], "invariant_violations": v["I"],
                          "duplicate_side_effects": v["D"], "reported": v["reported"], "expected": v["expected"]}
        episodes.append(ep)
        tools = sorted({s["tool"] for s in r["trace"]} | {"submit_report"})
        from actionproof.rollout.agenttune_rollout import make_env
        allowed = list(make_env(t, "ak").META)
        cases.append(AgentCase(id=t.task_id, task=t.prompt, category="+".join(t.meta["faults"]) or "benign",
                               allowed_tools=allowed, outcome=FinalStateAssertion("vsc", equals=1),
                               metadata={"faults": t.meta["faults"], "tools_used": tools}))
    return cases, episodes


def run(path: str, out_dir: str = "results/auditkit") -> str:
    from auditkit.agent_eval import AgentEvalRunner, AgentEvalSpec
    cases, episodes = build(path)
    result = AgentEvalRunner().run(AgentEvalSpec(cases=cases, mode="recorded", episodes=episodes,
                                                 scorers=SCORERS, trials=1))
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out = f"{out_dir}/{Path(path).stem}.auditkit.json"
    result.to_json(out)
    verdicts = [r.outcome["verdict"] for r in result.rows]
    assert "unknown" not in verdicts, "every ActionProof episode carries a verified final state"
    return result.summary()


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(f"== {p}\n{run(p)}\n")
