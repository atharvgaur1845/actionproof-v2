"""S3 addendum 4 (PREREGISTRATION_S3_ADDENDUM4): the acknowledgement-only recovery-start pool.

Each "read" recovery start of `s3_rec_train_half` becomes the "ack" start of the same task: the scripted prefix keeps
every call up to and including the faulted write's acknowledgement and drops the authoritative read that follows it.
Same tasks, worlds, faults, ratios, steps and order. "done" starts (every step written and read back; no faulted
write) are kept unchanged.
    python scripts/s3_make_ackonly.py
"""
from __future__ import annotations

import copy
import json
from collections import Counter

from actionproof.bench.generator_proc import DATA, load_tasks
from actionproof.stage3.tasks import prefix

if __name__ == "__main__":
    out = []
    for t in load_tasks(DATA / "s3_rec_train_half.jsonl"):
        m = copy.deepcopy(t)
        if m.meta.get("start_type") == "read":
            m.meta["start_prefix"] = prefix(m, m.meta["start_step"], False)
            m.meta["start_type"] = "ack"
            m.meta["fault_class"] = m.meta["fault_class"].rsplit(":", 1)[0] + ":ack"
            assert t.meta["start_prefix"][:-1] == m.meta["start_prefix"], t.task_id
        out.append(m)
    (DATA / "s3_rec_train_half_ackonly.jsonl").write_text(
        "".join(json.dumps(t.to_json(), sort_keys=True) + "\n" for t in out))
    print(len(out), Counter(t.meta.get("start_type") for t in out))
