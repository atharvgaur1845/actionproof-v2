"""Amendment 14 harness checks (PREREGISTRATION_A14 §6a): branch state, prefix identity, evidence wording,
scripted-reference calibration, split properties. Needs `transformers` for the system prompt; skipped without it."""
import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("transformers")

from actionproof.bench.generator_proc import make_split  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("cef", ROOT / "scripts/cef_eval.py")
cef = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cef)

TASKS = [t for t in make_split("proc_dev", 60) if t.meta["requested"][-1] == "post"][:12]


@pytest.mark.parametrize("rep", ["agent", "runtime"])
def test_prefix_identical_within_family_until_evidence(rep):
    for t in TASKS:
        convs = {b: cef.build(t, b, rep)[1] for b in cef.BRANCHES}
        for fam in (("C", "A", "P"), ("L", "T")):
            if rep == "agent":  # everything before the forced read's result is identical; the result differs
                heads = {b: convs[b][:-1] for b in fam}
                assert all(h == heads[fam[0]] for h in heads.values()), t.task_id
                assert len({convs[b][-1]["content"] for b in fam}) == len(fam), t.task_id
            else:  # the pay result is identical; the appended runtime block differs
                acks = {b: convs[b][-1]["content"].split("\n\nRuntime check")[0] for b in fam}
                assert len(set(acks.values())) == 1, t.task_id
        ack = -3 if rep == "agent" else -1  # the pay result: ok in C, timeout in L
        assert convs["C"][ack]["content"].split("\n\nRuntime")[0] != convs["L"][ack]["content"].split("\n\nRuntime")[0]
        if rep == "agent":  # C and L show the same committed payment; only the acknowledgement differs
            assert convs["C"][-1]["content"] == convs["L"][-1]["content"]


def test_evidence_has_no_verdict_words():
    for t in TASKS:
        for b in cef.BRANCHES:
            evidence = cef.build(t, b, "agent")[1][-1]["content"].lower()
            assert not any(w in evidence for w in cef.VERDICT_WORDS), (t.task_id, b, evidence)


def test_scripted_references_calibrate():
    recs = cef.run_scripted(TASKS[:6], "agent")
    by = {p: cef.summary([r for r in recs if r["policy"] == p]) for p in ("oracle", "naive", "fixed_retry")}
    assert by["oracle"]["bca_success_family"] == 1.0 and by["oracle"]["bca_timeout_family"] == 1.0
    assert by["oracle"]["correct_P"] == 1.0 and by["oracle"]["signed_S"] == 1.0
    for p in ("naive", "fixed_retry"):
        assert by[p]["bca_success_family"] == 0.5 and by[p]["signed_S"] == 0.0


def test_cf_split_properties():
    tasks = cef.make_cf_split(40)
    assert all(t.meta["requested"][-1] == "post" and "pay" in t.meta["requested"] for t in tasks)
    assert all(t.meta["faults"] == [] and t.meta["never_commit"] is None for t in tasks)
    assert len({t.task_id for t in tasks}) == 40 and all(t.task_id.startswith("proc_cf-") for t in tasks)
