# P5 addendum 1: unhashable idempotency keys in the behaviour metrics (D-P5-1)

Registered 2026-11-26, after task 1722 (tiny-aya-earth) stopped on its first evaluation (base model,
`p1_test_full`), **before any Earth result was written**. Fire and Water had finished. Expanse 8B was partway
through the Dw arm's evaluation and had not hit the error.

## What happened

`scripts/p1_eval.py` (frozen by P1 and reused by P5 unchanged) computes descriptive behaviour metrics after each
episode. `episode_metrics()` collects every `pay_supplier` idempotency key in a Python set. The base Earth model
passed the key as a JSON object, so `used.add(k)` raised `TypeError: unhashable type: 'dict'` and the whole run
stopped. The episode itself, the environment and the deterministic verifier had already handled the call.

## Change

`scripts/p5_eval.py` now wraps P1's helper: the helper receives a copy of the trace in which a non-scalar key is
replaced by its canonical JSON text (`json.dumps(sort_keys=True)`). `p1_eval.py` is not edited, so no other
registration's hashes change.

- The stored trace, the episode loop, the verifier and VSC (the P5 outcome) are unchanged.
- Scalar and missing keys pass through unchanged, so Fire, Water and Expanse 8B are unaffected (regression test
  `tests/p5/test_p5_eval.py`).
- A reused object key counts as a reused key, as a reused string would.

The hypothesis, models, splits, samples, statistics (`scripts/p5_stats.py`), Holm family and decision rule are
as registered.

## Registration

`python scripts/freeze_p5.py --addendum1` writes `results/p5_addendum1_hashes.json` (this file,
`scripts/p5_eval.py`, `tests/p5/test_p5_eval.py`); `--check` applies it over `results/p5_hashes.json`.
