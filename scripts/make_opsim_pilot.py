"""A17 GPU-pilot splits (development only, never analysed): 12 tasks per domain from seeds not used by the
registered splits (it_ops 131, care_admin 132), task ids prefixed `pilot-`.
    python scripts/make_opsim_pilot.py
"""
from __future__ import annotations

from actionproof.integrity.opsim import DATA, write_tasks
from actionproof.integrity.opsim import care_admin as C
from actionproof.integrity.opsim import it_ops as I

if __name__ == "__main__":
    for name, tasks in (("it_ops_pilot", I.make_split(131)), ("care_admin_pilot", C.make_dataset(132))):
        picked = tasks[::10][:12]
        for t in picked:
            t.task_id, t.split = f"pilot-{t.task_id}", name
        write_tasks(DATA / f"{name}.jsonl", picked)
        print(name, len(picked), sorted({t.meta["fault_class"] for t in picked}))
