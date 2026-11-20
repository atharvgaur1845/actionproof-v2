"""A12 addendum 2 §2: record the step-200 adapter of every finished A12 run in results/a12/frozen_models.json
(append-only: an existing entry is never changed). `final` must be byte-identical to `checkpoint-200`.
    python scripts/freeze_models.py            # adds every finished run not yet frozen
    python scripts/freeze_models.py --check    # every frozen adapter still has its recorded sha256
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "results/a12/frozen_models.json"
PROTOCOL = {0: "original protocol; R_A stop rule fired, continued as deviation D-A12-1"}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    frozen = json.loads(REF.read_text()) if REF.exists() else {}
    if a.check:
        bad = [k for k, v in frozen.items() if sha(ROOT / v["adapter"] / "adapter_model.safetensors") != v["sha256"]]
        print("OK: frozen models match" if not bad else "MISMATCH: " + ", ".join(bad))
        sys.exit(1 if bad else 0)
    for run in sorted((ROOT / "runs").glob("a12_proc_r?_s*")):
        tag = run.name
        if tag in frozen or not (run / "final/adapter_model.safetensors").exists():
            continue
        h = sha(run / "final/adapter_model.safetensors")
        c200 = run / "checkpoint-200/adapter_model.safetensors"
        if not c200.exists() or sha(c200) != h:
            sys.exit(f"{tag}: final differs from checkpoint-200 or checkpoint-200 missing; not frozen")
        seed = int(tag.rsplit("_s", 1)[1])
        frozen[tag] = {"adapter": f"runs/{tag}/final", "sha256": h, "step": 200,
                       "protocol": PROTOCOL.get(seed, "addendum 2"),
                       "frozen": datetime.datetime.now().isoformat(timespec="minutes")}
        print("frozen", tag, h)
    REF.write_text(json.dumps(frozen, indent=2))


if __name__ == "__main__":
    main()
