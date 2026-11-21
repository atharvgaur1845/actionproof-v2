"""S3 addendum 2 pools and targeted splits: only the workflow of partial payments changes; splits are balanced."""
import importlib.util
import json
from collections import Counter
from pathlib import Path

from actionproof.bench.generator_proc import DATA, load_tasks

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("rb", ROOT / "scripts/s3_make_rebal.py")
rb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rb)


def pp(ts):
    return Counter(t.workflow for t in ts if rb.is_pp(t))


def test_rebalanced_pools_keep_everything_but_workflow():
    for old, new in (("s3_trainb_half", "s3_trainb_half_rb"), ("s3_rec_train_half", "s3_rec_train_half_rb")):
        a, b = load_tasks(DATA / f"{old}.jsonl"), load_tasks(DATA / f"{new}.jsonl")
        assert len(a) == len(b)
        assert Counter(t.meta["fault_class"] for t in a) == Counter(t.meta["fault_class"] for t in b)
        key = lambda t: (json.dumps(t.meta.get("faults"), sort_keys=True), t.meta.get("start_type"))
        assert Counter(map(key, a)) == Counter(map(key, b))
        changed = [(x, y) for x, y in zip(a, b) if x.task_id != y.task_id]
        assert changed and all(rb.is_pp(x) and y.workflow == "receive_pay" for x, y in changed)
        assert max(pp(b).values()) - min(pp(b).values()) <= 2


def test_targeted_splits_balanced_and_disjoint():
    seen = set()
    for f in DATA.glob("*.jsonl"):
        if f.stem not in ("s3_pp_full", "s3_pp_rec", "s3_trainb_half_rb", "s3_rec_train_half_rb", "s3_matched_full_half_rb"):
            seen |= {p["po_id"] for t in load_tasks(f) for p in t.initial_state["pos"]}
    for sp in ("s3_pp_full", "s3_pp_rec"):
        ts = load_tasks(DATA / f"{sp}.jsonl")
        assert pp(ts) == Counter({"order_prepay": 50, "pay_post": 50, "receive_pay": 50})
        assert not {p["po_id"] for t in ts for p in t.initial_state["pos"]} & seen
        if sp == "s3_pp_rec":
            assert all(t.meta["start_step"] == "pay" for t in ts)
            assert abs(Counter(t.meta["start_type"] for t in ts)["ack"] - 75) <= 6
