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
from actionproof.bench.generator_bank import BANK_SPLITS

ROOT = Path(__file__).resolve().parents[1]
CODE = ["actionproof/env/verifier.py", "actionproof/env/tools_ecom.py", "actionproof/env/faults.py",
        "actionproof/rewards/pcstr.py", "actionproof/rewards/baselines.py", "actionproof/eval/stats.py",
        "actionproof/eval/run_eval.py", "actionproof/bench/generator.py", "actionproof/bench/prompts.py",
        "actionproof/env/verifier_bank.py", "actionproof/env/tools_bank.py", "actionproof/bench/generator_bank.py"]
ALL_SPLITS = {**{s: c["n"] for s, c in SPLITS.items()}, **{s: c["n"] for s, c in BANK_SPLITS.items()}}


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def split_bytes(name: str) -> bytes:
    return "".join(json.dumps(t.to_json(), sort_keys=True) + "\n" for t in make_split(name)).encode()


def derive() -> dict:
    return {"splits": {s: {"n": n, "sha256": sha(split_bytes(s))} for s, n in ALL_SPLITS.items()},
            "code": {f: sha((ROOT / f).read_bytes()) for f in CODE}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--amend", default=None, help="reason; required to change an existing registration")
    a = ap.parse_args()
    ref = ROOT / "results/prereg_hashes.json"
    cur = derive()
    if a.check:
        old = json.loads(ref.read_text())
        bad = [f"split {s}" for s in old["splits"] if cur["splits"].get(s) != old["splits"][s]] + \
              [f"code {f}" for f in old["code"] if cur["code"].get(f) != old["code"][f]]
        print("OK: bench and scoring code match the pre-registration" if not bad else "MISMATCH: " + ", ".join(bad))
        sys.exit(1 if bad else 0)
    if ref.exists():
        old = json.loads(ref.read_text())
        changed = {f"{k}:{n}": {"from": old[k].get(n), "to": cur[k][n]} for k in ("splits", "code") for n in cur[k]
                   if old[k].get(n) != cur[k][n]}
        if any(v["from"] is not None for v in changed.values()) and not a.amend:
            sys.exit("registered hashes would change: pass --amend '<reason>' (recorded in the amendment log)")
        cur["amendments"] = old.get("amendments", []) + (
            [{"date": __import__("datetime").datetime.now().isoformat(timespec="minutes"), "reason": a.amend,
              "changed": changed}] if changed else [])
    out = ROOT / "data/bench"
    out.mkdir(parents=True, exist_ok=True)
    for s in ALL_SPLITS:
        (out / f"{s}.jsonl").write_bytes(split_bytes(s))
    ref.parent.mkdir(exist_ok=True)
    ref.write_text(json.dumps(cur, indent=2))
    print(json.dumps(cur, indent=2))


if __name__ == "__main__":
    main()
