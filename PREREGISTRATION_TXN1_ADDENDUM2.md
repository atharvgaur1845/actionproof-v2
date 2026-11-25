# TXN1 addendum 2: H3 bootstrap restricted to the in-distribution tasks (D-TXN1-2)

Registered 2026-11-26, after the registered analysis task 1739 stopped with an assertion error, **before any H3
number was computed or printed**. Every model's `txn_test` output existed at that point. 1739 printed only the
registration check; its traceback is in `queue/h100/logs/1739_txn_stats.log` on the cluster.

## What happened

TXN1 registers H3 on the in-distribution groups (core and quadruples) with a template-clustered bootstrap (B = 10,000,
seed 20261126). In `scripts/txn_stats.py`, `h3()` filtered the episodes to the in-distribution groups, but it reused
the bootstrap built in `main()`, which spans every task common to all loaded tags (OOD groups included). The paired
check (`paired()`: "paired comparison needs every task in both arms") then failed, because the OOD tasks were no
longer in either arm. The synthetic test covered `txn_dev`, which has no OOD group, so it did not reach this path.

## Change

`h3()` now builds its bootstrap over the in-distribution tasks common to D_s0–2, B3 and B1, with the same B, seed and
template clusters (four lines). Nothing else changed. The hypotheses, family, metrics, comparators, Holm correction,
non-inferiority margin and decision rule are as registered. A regression test
(`tests/txn/test_scripts.py::test_h3_restricted_to_in_distribution_when_ood_groups_present`) relabels a third of
the `txn_dev` tasks as OOD and checks that H3 runs on the rest. `pytest tests/txn`: 144 passed.

The resamples differ from those of a bootstrap over all 440 tasks. That is the registered design (H3 is
in-distribution only), not a choice made after seeing a result: no H3 output existed under any bootstrap.

## Registration

`python scripts/freeze_txn1.py --addendum2` writes `results/txn1_addendum2_hashes.json` (this file,
`scripts/txn_stats.py`, `tests/txn/test_scripts.py`). `--check` applies addendum 1 and then addendum 2.
