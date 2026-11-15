# Amendment 10 addendum: existing policies in ActionProof-Bench v2

Registered 2026-11-15 18:40 IST, before any evaluation in the v2 environment. Hashed in `results/a10b_hashes.json`.

**Environment v2** (`actionproof/env/tools_ecom_v2.py`; v1 unchanged and hash-frozen): idempotency scoped to (operation, entity, key), fixing D-ENV-1; a timeout can come before the commit (no write, same error message). Tests: `tests/test_env_v2.py`.

**Split** `data/bench/v2/test_iid_v2.jsonl` (`scripts/make_bench_v2.py`): the 200 test_iid tasks in the v2 environment, plus a second copy of each of the 34 timeout tasks with the timeout before the commit (234 tasks).

**Models** (no retraining): v2 warm start, tool_trust s0, ack_outcome s0, outcome s0, PCSTR-d s0. Greedy, neutral prompt, 12 turns (the `run_eval` default used for the v1 test_iid evaluations), `run_eval_a4 --tasks`. Corrected from 10 before any v2 evaluation ran.

| ID | Prediction |
|---|---|
| H-A10e | On the 200 v1-equivalent tasks, each model's VSC is within 3 pp of its v1 test_iid VSC (the environment change does not move existing results) |
| H-A10f | On the 34 timeout-before-commit tasks, PCSTR-d and outcome VSC ≥ 0.75 (they read after the error and retry when nothing landed) |
| H-A10g (two-sided, descriptive) | On those 34 tasks, the acknowledgement arms (tool_trust, ack_outcome) may also succeed, because a blind keyed retry after an error now lands. No direction is predicted |

Results after 20:00 IST (the freeze set by the user on 2026-11-15) are reported as not finished.
