"""Model agents (vLLM) through the ProcureSim gateway in one mode; same loop, prompt and parser as `scripts/eval_proc.py`.

Agent level (the agent's own report) and system level (the gateway's disposition replaces the report) are scored by
the truth verifier and kept separate. Never run on a confirmatory split (refused).
    python scripts/eval_proc_gateway.py --model CohereLabs/aya-expanse-32b --split proc_dev --mode enforce \
        --out results/eval/proc_gateway/aya32b__enforce__proc_dev.jsonl
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

from actionproof.bench.generator_proc import load_tasks
from actionproof.integrity.contracts import Contract
from actionproof.integrity.gateway import MODES, ProcGateway
from actionproof.integrity.sim.prompts import PROC_PROMPTS
from actionproof.integrity.sim.runtime import episode_record_proc, make_env_proc
from actionproof.integrity.sim.truth import score

ROOT = Path(__file__).resolve().parents[1]
CONFIRMATORY = {"proc_conf", "proc_combo", "proc_stress"}


def _eval_proc():
    spec = importlib.util.spec_from_file_location("eval_proc", ROOT / "scripts/eval_proc.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--split", required=True)
    ap.add_argument("--mode", choices=MODES, required=True)
    ap.add_argument("--prompt", default="neutral", choices=list(PROC_PROMPTS))
    ap.add_argument("--n-tasks", type=int, default=None)
    ap.add_argument("--max-turns", type=int, default=16)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu-mem", type=float, default=0.88)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.split in CONFIRMATORY:
        raise SystemExit("confirmatory splits are reserved for the frozen A12 models")
    ep = _eval_proc()
    gws = {}

    def make_env(task, rid, seed=None):  # every episode gets its own gateway in the requested mode
        env = make_env_proc(task, rid)
        gw = ProcGateway(env, Contract.for_task(task), a.mode)
        gws[id(gw)] = gw
        return gw

    def record(task, gw):
        rec = episode_record_proc(task, gw.env)
        d = gw.disposition() if a.mode != "passthrough" else {"status": (gw.env.final_report or {}).get("status")}
        rec["system_level"] = score({**rec, "report": {"status": d["status"], "summary": ""}})
        rec["gateway"] = {"mode": a.mode, "disposition": d, "decisions": gw.decisions,
                          "refusals": sum(1 for v in gw.agent_view if v.get("refused"))}
        return rec

    ep.make_env_proc, ep.episode_record_proc = make_env, record
    from vllm import LLM
    from actionproof.train.merge import merge
    weights = merge(a.adapter, a.model) if a.adapter else a.model
    llm = LLM(weights, tensor_parallel_size=a.tp, gpu_memory_utilization=a.gpu_mem, max_model_len=8192,
              dtype="bfloat16", seed=0)
    tasks = load_tasks(ROOT / f"data/bench/proc/{a.split}.jsonl")[: a.n_tasks]
    t0 = time.time()
    recs = ep.run(llm, tasks, PROC_PROMPTS[a.prompt], 1, 0.0, a.max_turns, 256)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in recs:
            f.write(json.dumps({**r, "model": a.model, "adapter": a.adapter, "mode": a.mode}) + "\n")
    n = len(recs)
    s = {"n": n, "mode": a.mode, "model": a.model, "split": a.split, "seconds": round(time.time() - t0),
         "agent_level": {k: sum(r["score"][k] for r in recs) / n for k in ("vsc", "false_success", "goal_met")},
         "system_level": {k: sum(r["system_level"][k] for r in recs) / n for k in ("vsc", "false_success", "goal_met")},
         "duplicate": sum(r["score"]["duplicate_irreversible"] for r in recs) / n,
         "refusals_per_episode": sum(r["gateway"]["refusals"] for r in recs) / n,
         "tool_call_validity": 1 - sum(r["invalid"] for r in recs) / max(sum(r["n_model_calls"] for r in recs), 1)}
    out.with_suffix(".summary.json").write_text(json.dumps(s, indent=2))
    print(json.dumps(s))


if __name__ == "__main__":
    main()
