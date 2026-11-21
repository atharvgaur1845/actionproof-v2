"""Stage 3 transfer splits (PREREGISTRATION_S3): fresh instances of the OpsSim templates from seeds no earlier split
uses (it_ops 231, care_admin 232), task ids prefixed `s3-`. The registered A17 splits are not reused.
    python scripts/s3_make_opsim.py
"""
from __future__ import annotations

from actionproof.integrity.opsim import DATA, load_tasks, write_tasks
from actionproof.integrity.opsim import care_admin as C
from actionproof.integrity.opsim import it_ops as I

if __name__ == "__main__":
    seen = {t.task_id for f in DATA.glob("*.jsonl") for t in load_tasks(f)}
    for name, tasks in (("s3_it_ops", I.make_split(231)), ("s3_care_admin", C.make_dataset(232))):
        for t in tasks:
            t.task_id, t.split = f"s3-{t.task_id}", name
        assert not {t.task_id for t in tasks} & seen
        write_tasks(DATA / f"{name}.jsonl", tasks)
        print(name, len(tasks))
