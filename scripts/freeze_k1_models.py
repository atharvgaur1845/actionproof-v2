"""K1 (PREREGISTRATION_K1 §6): record the step-200 adapter of every finished K1 run in results/k1/frozen_models.json
(append-only), in the format scripts/conf_eval.py reads. `final` must be byte-identical to `checkpoint-200`.
    python scripts/freeze_k1_models.py            # adds every finished K1 run not yet frozen
    python scripts/freeze_k1_models.py --check --tags k1_k1t_ra_s0
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "results/k1/frozen_models.json"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--tags", default=None)
    a = ap.parse_args()
    frozen = json.loads(REF.read_text()) if REF.exists() else {}
    if a.check:
        tags = a.tags.split(",") if a.tags else list(frozen)
        bad = [k for k in tags if k not in frozen or
               sha(ROOT / frozen[k]["adapter"] / "adapter_model.safetensors") != frozen[k]["sha256"]]
        print("OK: K1 frozen models match" if not bad else "MISMATCH: " + ", ".join(bad))
        sys.exit(1 if bad else 0)
    for run in sorted((ROOT / "runs").glob("k1_k1?_ra_s*")):
        tag = run.name
        if tag in frozen or not (run / "final/adapter_model.safetensors").exists():
            continue
        h = sha(run / "final/adapter_model.safetensors")
        c200 = run / "checkpoint-200/adapter_model.safetensors"
        if not c200.exists() or sha(c200) != h:
            sys.exit(f"{tag}: final differs from checkpoint-200 or checkpoint-200 missing; not frozen")
        frozen[tag] = {"adapter": f"runs/{tag}/final", "sha256": h, "step": 200, "protocol": "K1",
                       "frozen": datetime.datetime.now().isoformat(timespec="minutes")}
        print("frozen", tag, h)
    REF.parent.mkdir(parents=True, exist_ok=True)
    REF.write_text(json.dumps(frozen, indent=2))


if __name__ == "__main__":
    main()
