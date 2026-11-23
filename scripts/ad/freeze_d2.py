"""Freeze AD-D2 (results/ad_hashes_d2.json) and write its task files. --check compares; --tasks writes the tasks."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TASKS = [(1212, "nostd_ema", 0, "1209_ad_awrbc30_s0", True), (1213, "awrbn_ema", 0, "1211_ad_awrbnc30_s0", False),
         (1214, "awrbn_ema", 1, "1212_ad_nostd_ema_s0", False), (1215, "nostd_ema", 1, "1213_ad_awrbn_ema_s0", False)]
FILES = ["docs/research/algorithm_discovery/EXPERIMENT_PROTOCOL_D2.md", "actionproof/ad/scale.py",
         "actionproof/ad/train_ema.py", "tests/ad/test_scale.py", "slurm/ad_arm_d2.sh", "scripts/ad/freeze_d2.py",
         "actionproof/ad/worlds.py", "actionproof/ad/rewards.py", "actionproof/ad/rollout.py",
         "actionproof/ad/evaluate.py", "data/bench/proc/cw_train.jsonl", "data/bench/proc/cw_dev.jsonl"]
HEAD = """# minutes: 300
# after: {after}
# Algorithm discovery AD-D2 (docs/research/algorithm_discovery/EXPERIMENT_PROTOCOL_D2.md): ad_{est}_s{seed}.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
export MASTER_PORT={port}
source slurm/ad_arm_d2.sh {est} {seed} 200{smoke}
"""


def task_files() -> list[str]:
    return [f"slurm/tasks/{t}_ad_{e}_s{s}.sh" for t, e, s, _, _ in TASKS]


def derive() -> dict:
    return {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in FILES + task_files()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--tasks", action="store_true")
    a = ap.parse_args()
    if a.tasks:
        for t, e, s, after, smoke in TASKS:
            (ROOT / f"slurm/tasks/{t}_ad_{e}_s{s}.sh").write_text(HEAD.format(after=after, est=e, seed=s,
                                                                             port=29000 + t % 1000,
                                                                             smoke=" smoke" if smoke else ""))
        print("\n".join(task_files()))
        return
    ref = ROOT / "results/ad_hashes_d2.json"
    cur = derive()
    if a.check:
        old = json.loads(ref.read_text())["files"]
        bad = [f for f in old if cur.get(f) != old[f]]
        print("OK: AD-D2 matches the registration" if not bad else "MISMATCH: " + ", ".join(bad))
        sys.exit(1 if bad else 0)
    if ref.exists():
        sys.exit("results/ad_hashes_d2.json exists; already registered")
    ref.write_text(json.dumps({"files": cur, "registered": datetime.datetime.now().isoformat(timespec="minutes")}, indent=2))
    print(json.dumps(cur, indent=2))


if __name__ == "__main__":
    main()
