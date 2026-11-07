"""Append one run record to results/a4/runs.jsonl: id, kind, command, seed, config, adapter / data sha256, code commit."""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

from actionproof.eval.run_eval_a4 import file_sha

ap = argparse.ArgumentParser()
ap.add_argument("--id", required=True)
ap.add_argument("--kind", required=True)
ap.add_argument("--cmd", required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--adapter", default=None)
ap.add_argument("--data", default=None)
ap.add_argument("--out", default=None)
a = ap.parse_args()
commit = Path("results/a4/CODE_COMMIT").read_text().strip() if Path("results/a4/CODE_COMMIT").exists() else None
rec = {"id": a.id, "kind": a.kind, "time": datetime.datetime.now().isoformat(timespec="seconds"), "cmd": a.cmd,
       "seed": a.seed, "adapter": a.adapter, "adapter_sha256": file_sha(a.adapter), "data": a.data,
       "data_sha256": file_sha(a.data), "out": a.out, "out_sha256": file_sha(a.out), "code_commit": commit}
with open("results/a4/runs.jsonl", "a") as f:
    f.write(json.dumps(rec) + "\n")
print(json.dumps(rec))
