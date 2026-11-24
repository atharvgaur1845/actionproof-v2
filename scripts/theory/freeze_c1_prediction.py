"""Freeze the AD-C1 theory prediction note (docs/research/AD_C1_THEORY_PREDICTION.md). --check compares."""
from __future__ import annotations

import datetime
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
F = "docs/research/AD_C1_THEORY_PREDICTION.md"
REF = ROOT / "results/ad_c1_theory_prediction_hash.json"

h = hashlib.sha256((ROOT / F).read_bytes()).hexdigest()
if "--check" in sys.argv:
    ok = json.loads(REF.read_text())["files"][F] == h
    print("OK" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)
if REF.exists():
    sys.exit("refusing to overwrite " + str(REF))
REF.write_text(json.dumps({"registered": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                           "files": {F: h}}, indent=1))
print("wrote", REF)
