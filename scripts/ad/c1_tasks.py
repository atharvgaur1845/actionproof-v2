"""Write AD-C1 training task files (PREREGISTRATION_AD_C1.md §2). Two serial chains so that at most two GPUs are used.

    python scripts/ad/c1_tasks.py --arms grpo [nostd awrb ...] --after-a 1214_ad_awrbn_ema_s1 --after-b 1215_ad_nostd_ema_s1 --first 1220
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HEAD = """# minutes: 300
# after: {after}
# AD-C1 (PREREGISTRATION_AD_C1.md): ad_{est}_s{seed}, training only.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
export MASTER_PORT={port}
source slurm/ad_arm_c1.sh {est} {seed}
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--after-a", required=True)
    ap.add_argument("--after-b", required=True)
    ap.add_argument("--first", type=int, required=True)
    a = ap.parse_args()
    jobs = [(e, s) for e in a.arms for s in (2, 3, 4)]
    last = {0: a.after_a, 1: a.after_b}
    tid = a.first
    for i, (e, s) in enumerate(jobs):
        chain = i % 2
        name = f"{tid}_ad_{e}_s{s}"
        (ROOT / "slurm/tasks" / f"{name}.sh").write_text(HEAD.format(after=last[chain], est=e, seed=s, port=29000 + tid % 1000))
        print(f"chain {'AB'[chain]}: {name} after {last[chain]}")
        last[chain] = name
        tid += 1
    print("chain ends:", last[0], last[1])


if __name__ == "__main__":
    main()
