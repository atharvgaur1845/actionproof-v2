"""CI guard for frozen files: runs every scripts/freeze_*.py --check. A mismatch passes only if every mismatched file is
listed in results/known_deviations.json with its current sha256 (a deviation logged when it happened).
    python scripts/ci_freeze_check.py
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    known = json.loads((ROOT / "results/known_deviations.json").read_text())
    bad = []
    # freeze_models.py and freeze_k1_models.py check trained adapters, which live on the cluster and HF, not in git;
    # they run inside each job
    skip = {"freeze_models.py", "freeze_k1_models.py"}
    for f in sorted(x for x in (ROOT / "scripts").glob("freeze_*.py") if x.name not in skip):
        p = subprocess.run([sys.executable, str(f), "--check"], cwd=ROOT, capture_output=True, text=True)
        last = (p.stdout.strip().splitlines() or [""])[-1]
        if p.returncode == 0 and last.startswith("OK"):
            print(f"{f.name}: {last}")
            continue
        paths = re.findall(r"[\w./-]+\.(?:py|md|jsonl|json|txt)", last)
        unexplained = [q for q in paths if q not in known or
                       hashlib.sha256((ROOT / q).read_bytes()).hexdigest() != known[q]["sha256"]]
        if paths and not unexplained:
            print(f"{f.name}: mismatch explained by logged deviations: {', '.join(known[q]['deviation'].split(' (')[0] for q in paths)}")
        else:
            bad.append(f.name)
            print(f"{f.name}: FAIL {last} {p.stderr.strip()[-300:]}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
