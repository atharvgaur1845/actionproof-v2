"""Build the TXN v1 splits (docs/txn/TXN_SPEC.md §10): data/bench/proc/txn_{dev,val,test}.jsonl and
data/bench/proc/txn_manifest.json (per-task template id and sha256, per-file sha256). Refuses to overwrite.

Asserts: task ids, PO ids and prompt text disjoint from every other data/bench/proc/*.jsonl (PO ids and prompts are
shared only inside a quadruple); OOD workflows are not P4 training compositions; OOD fault combinations are absent
from p1_train_* and proc_combo; the alias/schema surfaces share no name with SR1, S4 or the canonical API.

    python scripts/txn_make_splits.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from actionproof.txn.tasks import make_split, write_splits  # noqa: E402

if __name__ == "__main__":
    out = write_splits()
    print(json.dumps(out, indent=1))
    for s in out:
        print(s, dict(Counter(t.meta["txn"]["group"] for t in make_split(s))))
