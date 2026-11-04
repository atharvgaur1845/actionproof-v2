"""Language / pressure variants change only the task text; worlds, faults and goals are paired."""
from __future__ import annotations

from actionproof.bench.generator import make_split
from actionproof.bench.variants import VARIANT_SPLITS


def test_variants_pair_with_base_and_change_only_text():
    for name, (base, lang, pressure) in VARIANT_SPLITS.items():
        b, v = make_split(base, 20), make_split(name, 20)
        for t, u in zip(b, v):
            assert u.initial_state == t.initial_state and u.schedule.to_json() == t.schedule.to_json()
            assert u.goal.postconditions == t.goal.postconditions and u.goal.task_id == u.task_id
            assert u.meta["base_task"] == t.task_id and u.meta["faults"] == t.meta["faults"]
            ids = t.meta["ids"]
            assert all(ids[k] in u.prompt for k in ("ticket", "cust", "order"))
            assert (u.prompt != t.prompt) and (("Customer message" in u.prompt or "ग्राहक का संदेश" in u.prompt) == pressure)


def test_prompts_unique_for_lookup():
    for name in VARIANT_SPLITS:
        ts = make_split(name)
        assert len({t.prompt for t in ts}) == len(ts)
