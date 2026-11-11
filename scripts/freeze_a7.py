"""Freeze Amendment 7 splits and code (PREREGISTRATION_A7). Writes data/bench/a7/*.jsonl and results/a7_hashes.json.
Never touches results/prereg_hashes.json or data/bench/*.jsonl (Amendments 0-3).

    python scripts/freeze_a6.py           # first freeze (refuses to overwrite an existing registration)
    python scripts/freeze_a6.py --check   # re-derive and compare
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from actionproof.bench.generator_a7 import SPLITS as A7_SPLITS, make_a7_split, split_bytes

ROOT = Path(__file__).resolve().parents[1]
CODE = ["actionproof/bench/generator_a7.py", "actionproof/rewards/a7_rewards.py", "actionproof/train/a7_demos.py",
        "actionproof/eval/forced_prefix.py", "slurm/a7_arm.sh", "PREREGISTRATION_A7.md"]
FILES = ["data/bench/a7/forced_prefixes.jsonl"]  # built once from A6 eval traces (forced_prefix build); hashed as data


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def derive() -> dict:
    return {"splits": {s: {"n": len(make_a7_split(s)), "sha256": sha(split_bytes(s))} for s in A7_SPLITS},
            "code": {f: sha((ROOT / f).read_bytes()) for f in CODE},
            "files": {f: sha((ROOT / f).read_bytes()) for f in FILES}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    ref = ROOT / "results/a7_hashes.json"
    cur = derive()
    if a.check:
        old = json.loads(ref.read_text())
        bad = [f"split {s}" for s in old["splits"] if cur["splits"].get(s) != old["splits"][s]] + \
              [f"code {f}" for f in old["code"] if cur["code"].get(f) != old["code"][f]] + \
              [f"data {f}" for f in old["files"] if cur["files"].get(f) != old["files"][f]]
        on_disk = [s for s in A7_SPLITS if sha((ROOT / "data/bench/a7" / f"{s}.jsonl").read_bytes()) != old["splits"][s]["sha256"]]
        bad += [f"file {s}" for s in on_disk]
        print("OK: Amendment 7 splits and code match the registration" if not bad else "MISMATCH: " + ", ".join(bad))
        sys.exit(1 if bad else 0)
    if ref.exists():
        sys.exit("results/a7_hashes.json exists; Amendment 7 is already registered (deviations go in SPEC §27)")
    out = ROOT / "data/bench/a7"
    out.mkdir(parents=True, exist_ok=True)
    for s in A7_SPLITS:
        (out / f"{s}.jsonl").write_bytes(split_bytes(s))
    ref.write_text(json.dumps({**cur, "registered": __import__("datetime").datetime.now().isoformat(timespec="minutes")},
                              indent=2))
    print(json.dumps(cur, indent=2))


if __name__ == "__main__":
    main()
