# A14 pilot (PREREGISTRATION_A14 §6)

- (a) Scripted references on 20 proc_dev tasks x 5 branches, both evidence representations (CPU, cluster login node,
  2026-11-20 08:45): oracle BCA 1.0 (success and timeout families), P correct 1.0; naive and fixed_retry BCA 0.5,
  signed S 0.0. An earlier attempt showed oracle BCA 0.95, caused by a harness bug (a source task's `never_commit`
  survived the branch override, so branch C did not commit); fixed, and the harness now asserts that the committed
  state after the prefix equals the branch. Tests: tests/integrity/test_a14.py (5 passed on the cluster).
- (b) A12 R_V seed 0 step-50 adapter (not analysed in A14) on 10 proc_dev tasks x 5 branches, HF job
  6affbc0fefaf5e14f4b28b3d (L40S): 50 episodes, all ended by a report, 0 invalid calls, 0 invalid or missing first
  actions, 102 s. Only these validity and timing fields were read; the metric file in this folder was not analysed.
