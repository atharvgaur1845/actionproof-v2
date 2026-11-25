# TXN1 addendum 1: invariant checker false positive (D-TXN1-1)

Registered 2026-11-26, **after** the scripted policies (B0–B4, A1–A6) were run on `txn_test` (task 1704) and
**before** any model's `txn_test` output was read. Model evaluations 1706 (W, A, Dw) was running at this time;
its output had not been opened.

## What was found

The 1704 policy table reported 3 invariant violations for B4 (and A6, which equals B4) on `txn_test`, against 0 on
`txn_dev`. All 3 are `replay_after_partial` on `ood_combo` tasks with the fault composition
`receive.partial+pay.concurrent` (`txn_test-0383`, `-0389`, `-0395`). In each trace B4 tops up the partial goods
receipt and then pays. A colleague's concurrent payment lands just before B4's payment. B4 reads the payments,
refunds its own duplicate and posts the journal entry. The evaluator scores the episode VTC = STR = 1, DMR = 0.

The check that flags "a step re-executed after a partial commit of another step" looked for earlier full commits of
the same tool among **all** attempts, including those the environment attributes to the colleague (`by:
"colleague"`). A colleague's write is not the agent replaying its own workflow, and the TXN spec (§8, "no whole-
workflow replay after a partial commit") is about the agent's actions only. This is a checker bug, not a policy error.

## Change

`actionproof/txn/invariants.py::replay_after_partial`: count only the agent's own earlier commits (one filter,
`a.get("by", "agent") == "agent"`). Nothing else changed. A regression test
(`tests/txn/test_policies.py::test_colleague_commit_is_not_an_agent_replay_after_partial`) checks that B4 has no
violation on those tasks, and that the same record with the colleague's payment re-attributed to the agent is still
flagged. After the change: `pytest tests/txn` 143 passed; mutation testing 25/25 mutants killed.

## Effect on the registered analysis

- **None on H1–H5.** `scripts/txn_stats.py` does not read invariants; H3 uses DMR and URR from `actionproof/txn/truth.py`,
  which is unchanged. No split, task, metric, comparator, threshold or model list changes.
- **Invariant counts are descriptive.** Records keep their trace and attempts, so `scripts/txn_invariants_recheck.py`
  re-applies the corrected checker to every stored episode of every tag. Both counts are published per tag
  (`results/txn/invariants_recheck__txn_test.json`): the count under the checker frozen in TXN1 (the `invariants`
  field written at run time) and the corrected count. Reported invariant numbers use the corrected checker and cite
  this addendum.
- Evaluations that started before this addendum wrote run-time `invariants` with the frozen checker; those that start
  after it write the corrected one. The recheck file makes every tag comparable either way.

## Registration

`python scripts/freeze_txn1.py --addendum1` writes `results/txn1_addendum1_hashes.json` with the hashes of this file,
`actionproof/txn/invariants.py`, `scripts/txn_invariants_recheck.py` and `tests/txn/test_policies.py`.
`freeze_txn1.py --check` accepts the original TXN1 hashes, with those files replaced by their addendum-1 hashes.
