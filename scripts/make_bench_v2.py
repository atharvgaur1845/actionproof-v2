"""ActionProof-Bench v2 evaluation split (env v2): test_iid tasks run in the v2 environment. Every task with a timeout
fault appears twice: timeout after commit (as in v1) and timeout before commit (no commit, same error). CPU.

    python scripts/make_bench_v2.py   # -> data/bench/v2/test_iid_v2.jsonl
"""
import json
from pathlib import Path

from actionproof.bench.generator import make_split
from actionproof.env.faults import Fault

out = []
for t in make_split("test_iid"):
    d = t.to_json()
    timeouts = [[x["tool"], x["k"]] for x in d["schedule"] if x["fault"] == Fault.TIMEOUT.value] \
        if d["schedule"] and isinstance(d["schedule"][0], dict) and "tool" in d["schedule"][0] else []
    d["meta"] = {**d["meta"], "env_version": 2, "precommit_timeouts": []}
    out.append(d)
    if timeouts:
        e = json.loads(json.dumps(d))
        e["task_id"] += "_pre"
        e["meta"]["precommit_timeouts"] = timeouts
        out.append(e)
Path("data/bench/v2").mkdir(parents=True, exist_ok=True)
with open("data/bench/v2/test_iid_v2.jsonl", "w") as f:
    for d in out:
        f.write(json.dumps(d, sort_keys=True) + "\n")
print(len(out), "tasks;", sum(1 for d in out if d["meta"]["precommit_timeouts"]), "timeout-before-commit")
