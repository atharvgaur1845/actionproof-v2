"""Build the TXN v1 splits (docs/txn/TXN_SPEC.md §10): data/bench/proc/txn_{dev,val,test}.jsonl and
data/bench/proc/txn_manifest.json (per-task template id and sha256, per-file sha256). Refuses to overwrite.

Asserts: task ids, PO ids and prompt text disjoint from every other data/bench/proc/*.jsonl (PO ids and prompts are
shared only inside a quadruple); OOD workflows are not P4 training compositions; OOD fault combinations are absent
from p1_train_* and proc_combo; the alias/schema surfaces share no name with SR1, S4 or the canonical API.

    python scripts/txn_make_splits.py           # write (once)
    python scripts/txn_make_splits.py --check   # files match the manifest, regenerate byte-identically, and are
                                                # still disjoint from every other split now in data/bench/proc
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from actionproof.bench.generator_proc import DATA, load_tasks  # noqa: E402
from actionproof.txn.tasks import SEEDS, check_disjoint, make_split, task_line, write_splits  # noqa: E402


def check() -> int:
    man = json.loads((DATA / "txn_manifest.json").read_text())
    bad = []
    for s in SEEDS:
        text = (DATA / f"{s}.jsonl").read_text()
        if hashlib.sha256(text.encode()).hexdigest() != man["files"][f"{s}.jsonl"]["sha256"]:
            bad.append(f"{s}: sha256 differs from the manifest")
        if text != "".join(task_line(t) for t in make_split(s)):
            bad.append(f"{s}: does not regenerate byte-identically")
    try:
        check_disjoint({s: load_tasks(DATA / f"{s}.jsonl") for s in SEEDS})
    except AssertionError as e:
        bad.append(f"disjointness: {e}")
    print("OK: TXN splits match the manifest and are disjoint" if not bad else "MISMATCH: " + "; ".join(bad))
    for s in SEEDS:
        print(s, man["files"][f"{s}.jsonl"])
    return 1 if bad else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    if ap.parse_args().check:
        sys.exit(check())
    out = write_splits()
    print(json.dumps(out, indent=1))
    for s in out:
        print(s, dict(Counter(t.meta["txn"]["group"] for t in make_split(s))))
