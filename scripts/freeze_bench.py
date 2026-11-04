"""Freeze ActionProof-Bench for pre-registration (spec §7, gate G1).

Writes every split as canonical JSONL (sorted keys, one task per line) under data/bench/ and
records SHA-256 of each file plus of the code that defines scoring (verifier, rewards, stats,
generator). PREREGISTRATION.md quotes these hashes; `--check` re-derives them and fails on any
mismatch, so a silent change to a split or to the metric code is detectable.

    python scripts/freeze_bench.py            # write data/bench/* and results/prereg_hashes.json
    python scripts/freeze_bench.py --check    # verify against results/prereg_hashes.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from actionproof.bench.generator import SPLITS, make_split

ROOT = Path(__file__).resolve().parents[1]
CODE = ["actionproof/env/verifier.py", "actionproof/env/tools_ecom.py", "actionproof/env/faults.py",
        "actionproof/rewards/pcstr.py", "actionproof/rewards/baselines.py", "actionproof/eval/stats.py",
        "actionproof/eval/run_eval.py", "actionproof/bench/generator.py", "actionproof/bench/prompts.py"]


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def split_bytes(name: str) -> bytes:
    return "".join(json.dumps(t.to_json(), sort_keys=True) + "\n" for t in make_split(name)).encode()


def derive() -> dict:
    return {"splits": {s: {"n": SPLITS[s]["n"], "sha256": sha(split_bytes(s))} for s in SPLITS},
            "code": {f: sha((ROOT / f).read_bytes()) for f in CODE}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    ref = ROOT / "results/prereg_hashes.json"
    cur = derive()
    if a.check:
        old = json.loads(ref.read_text())
        bad = [f"split {s}" for s in old["splits"] if cur["splits"].get(s) != old["splits"][s]] + \
              [f"code {f}" for f in old["code"] if cur["code"].get(f) != old["code"][f]]
        print("OK: bench and scoring code match the pre-registration" if not bad else "MISMATCH: " + ", ".join(bad))
        sys.exit(1 if bad else 0)
    out = ROOT / "data/bench"
    out.mkdir(parents=True, exist_ok=True)
    for s in SPLITS:
        (out / f"{s}.jsonl").write_bytes(split_bytes(s))
    ref.parent.mkdir(exist_ok=True)
    ref.write_text(json.dumps(cur, indent=2))
    print(json.dumps(cur, indent=2))


if __name__ == "__main__":
    main()
