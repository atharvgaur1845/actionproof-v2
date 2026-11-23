"""R1-T: τ-bench retail with Tiny Aya agents (PREREGISTRATION_R1_TAU.md §3).

τ-bench (sierra-research/tau-bench @ 59a200c, MIT) runs unchanged:
- `get_env("retail", "llm", ..., task_split="test")`;
- its `ChatReActAgent` (ReAct, temperature 0, 30 steps);
- its database-state reward.
`litellm` is replaced by `actionproof/retention/tau_shim/litellm.py`, which posts to two local vLLM servers:
- provider "agent": the Tiny Aya checkpoint under test, 8,192-token context;
- provider "user": the simulated user (Aya Expanse 32B, server-default sampling), shared by every model.

An episode whose agent prompt no longer fits the context ends with reward 0 and `overflow: true`. Any other
exception ends it with reward 0 and `error` set. Both are counted separately in the summary.

    PYTHONPATH=actionproof/retention/tau_shim:/scratch/pabitra/ext/tau-bench \
      python -m actionproof.retention.tau_run --tag base --trials 3
"""
from __future__ import annotations

import argparse
import json
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def run_one(task_index: int, trial: int, user_model: str) -> dict:
    from litellm import ContextOverflow
    from tau_bench.agents.chat_react_agent import ChatReActAgent
    from tau_bench.envs import get_env

    t0 = time.time()
    env = get_env("retail", user_strategy="llm", user_model=user_model, task_split="test", user_provider="user",
                  task_index=task_index)
    agent = ChatReActAgent(tools_info=env.tools_info, wiki=env.wiki, model="agent", provider="agent",
                           use_reasoning=True, temperature=0.0)
    row = {"task_index": task_index, "trial": trial, "reward": 0.0, "overflow": False, "error": None}
    try:
        res = agent.solve(env, task_index=task_index, max_num_steps=30)
        row.update(reward=float(res.reward), messages=res.messages,
                   reward_info=res.info.get("reward_info") if isinstance(res.info, dict) else None)
    except ContextOverflow as e:
        row.update(overflow=True, error=f"context: {str(e)[:200]}")
    except Exception as e:  # noqa: BLE001 - recorded, counted, never silently dropped
        row.update(error=f"{type(e).__name__}: {str(e)[:200]}", trace=traceback.format_exc()[-800:])
    row["seconds"] = round(time.time() - t0, 1)
    row["n_messages"] = len(row.get("messages") or [])
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="pilot: first N test tasks")
    ap.add_argument("--user-model", default="user")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out-dir", default="results/r1_tau")
    a = ap.parse_args()
    from tau_bench.envs.retail.tasks_test import TASKS_TEST
    n = a.limit or len(TASKS_TEST)
    out = Path(a.out_dir) / f"{a.tag}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {(r["trial"], r["task_index"]) for r in map(json.loads, out.read_text().splitlines())}
    jobs = [(i, t) for t in range(a.trials) for i in range(n) if (t, i) not in done]
    lock = threading.Lock()

    def work(job):
        r = run_one(job[0], job[1], a.user_model)
        r["tag"] = a.tag
        with lock, out.open("a") as fh:
            fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(work, jobs))
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    s = {"tag": a.tag, "n": len(rows), "pass1": sum(r["reward"] for r in rows) / len(rows),
         "overflow_rate": sum(r["overflow"] for r in rows) / len(rows),
         "error_rate": sum(bool(r["error"]) and not r["overflow"] for r in rows) / len(rows)}
    (Path(a.out_dir) / f"{a.tag}.summary.json").write_text(json.dumps(s, indent=2))
    print(json.dumps(s))


if __name__ == "__main__":
    main()
