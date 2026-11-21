"""S3 addendum 1 (PREREGISTRATION_S3_ADDENDUM1): the class-matched full-episode pool for the start-state control.

RS-GRPO's pool = `s3_trainb_half` (200 proc_train_b tasks) + `s3_rec_train_half` (200 recovery starts). The matched
pool keeps the same 200 proc_train_b tasks and replaces each recovery start with a FULL episode carrying the same
fault on the same step (same kind, same partial ratio, never-commit on the same tool; "complete" and "done" starts
become clean episodes). Prompts, worlds and faults are the recovery tasks' own; only `start_prefix` is removed, so the
two pools differ in where episodes start and in nothing else.
    python scripts/s3_make_matched.py
"""
from __future__ import annotations

import copy
import json

from actionproof.bench.generator_proc import DATA, load_tasks

if __name__ == "__main__":
    out = []
    for t in load_tasks(DATA / "s3_rec_train_half.jsonl"):
        m = copy.deepcopy(t)
        for k in ("start_prefix", "start_type", "start_step"):
            m.meta.pop(k, None)
        m.meta["fault_class"] = "s3m:" + t.meta["fault_class"].split(":")[1]
        m.task_id = t.task_id.replace("s3_rec_train", "s3_matched")
        out.append(m)
    (DATA / "s3_matched_full_half.jsonl").write_text("".join(json.dumps(t.to_json(), sort_keys=True) + "\n" for t in out))
    from collections import Counter
    print(len(out), Counter(t.meta["fault_class"] for t in out))
