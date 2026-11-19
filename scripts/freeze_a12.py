"""Freeze Amendment 12 (PREREGISTRATION_A12): ProcureSim, its splits, rewards, warm-start recipe, gate and evaluation.

    python scripts/freeze_a12.py           # register (refuses to overwrite results/a12_hashes.json)
    python scripts/freeze_a12.py --check   # compare
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ["PREREGISTRATION_A12.md", "docs/stage2/VERIFIER_SPEC.md",
         "actionproof/integrity/sim/env.py", "actionproof/integrity/sim/truth.py",
         "actionproof/integrity/sim/runtime.py", "actionproof/integrity/sim/agents.py",
         "actionproof/integrity/sim/prompts.py", "actionproof/bench/generator_proc.py",
         "actionproof/rewards/proc_rewards.py", "actionproof/train/grpo_proc.py", "actionproof/train/proc_demos.py",
         "scripts/eval_proc.py", "scripts/a12_gate.py", "scripts/a12_reward_truth.py",
         "results/a12/reward_truth_table.md", "scripts/reference_verifier_proc.py", "tests/golden/proc_golden.jsonl"]
FILES += sorted(str(p.relative_to(ROOT)) for p in (ROOT / "data/bench/proc").glob("*.jsonl"))


def derive() -> dict:
    return {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in FILES}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    ref = ROOT / "results/a12_hashes.json"
    cur = derive()
    if a.check:
        if not ref.exists():
            sys.exit("Amendment 12 is not registered")
        old = json.loads(ref.read_text())["files"]
        bad = [f for f in old if cur.get(f) != old[f]]
        print("OK: Amendment 12 files match the registration" if not bad else "MISMATCH: " + ", ".join(bad))
        sys.exit(1 if bad else 0)
    if ref.exists():
        sys.exit("results/a12_hashes.json exists; Amendment 12 is already registered (deviations go in docs/stage2/DECISIONS.md)")
    ref.write_text(json.dumps({"files": cur, "registered": datetime.datetime.now().isoformat(timespec="minutes")},
                              indent=2))
    print(len(cur), "files hashed")


if __name__ == "__main__":
    main()
