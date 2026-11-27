**Figure 1. One request, one call, two possible replies, four hidden outcomes.** The counterfactual quadruples of
txn_test hold the request and the agent-visible reply fixed and vary only what the system of record holds. The example
request and replies are quadruple txn_test-q00 (tasks txn_test-0201, txn_test-0200, txn_test-0202, txn_test-0203), taken
from B4's recorded first call.

- **Sources:** `results/txn/{B0,B3,B4,D_s0,D_s1,D_s2}__txn_test.jsonl` (per-episode `txn` flags);
  `actionproof/txn/tasks.py` (REALITIES, quadruple generator); `data/bench/proc/txn_test.jsonl`.
- **Split and denominators:** the 15 standard-regime quadruples of txn_test (60 tasks; 15 per cell). Scripted policies:
  1 episode per task (15 per cell). D: 3 seeds x 3 episodes per task (135 per cell).
- **Uncertainty:** counts only; with 15 tasks per cell, one task is 6.7 points.
- **Unresolvable regime** (same 15 quadruples with unreadable records; hand-offs among 60 tasks per scripted policy,
  540 D episodes): B0 trust the reply: 0/60; B3 retry + idempotency key: 0/60; B4 reconciliation (reference): 60/60; D SFT+GRPO, 3 seeds: 15/540.

| Reality | Arm | Verified | False completion | Duplicate | Episodes |
|---|---|---|---|---|---|
| ok_committed | B0 trust the reply | 15 | 0 | 0 | 15 |
| ok_committed | B3 retry + idempotency key | 15 | 0 | 0 | 15 |
| ok_committed | B4 reconciliation (reference) | 15 | 0 | 0 | 15 |
| ok_committed | D SFT+GRPO, 3 seeds | 134 | 0 | 0 | 135 |
| ok_phantom | B0 trust the reply | 0 | 15 | 0 | 15 |
| ok_phantom | B3 retry + idempotency key | 0 | 15 | 0 | 15 |
| ok_phantom | B4 reconciliation (reference) | 15 | 0 | 0 | 15 |
| ok_phantom | D SFT+GRPO, 3 seeds | 124 | 9 | 0 | 135 |
| timeout_committed | B0 trust the reply | 0 | 15 | 15 | 15 |
| timeout_committed | B3 retry + idempotency key | 15 | 0 | 0 | 15 |
| timeout_committed | B4 reconciliation (reference) | 15 | 0 | 0 | 15 |
| timeout_committed | D SFT+GRPO, 3 seeds | 135 | 0 | 0 | 135 |
| timeout_uncommitted | B0 trust the reply | 15 | 0 | 0 | 15 |
| timeout_uncommitted | B3 retry + idempotency key | 15 | 0 | 0 | 15 |
| timeout_uncommitted | B4 reconciliation (reference) | 15 | 0 | 0 | 15 |
| timeout_uncommitted | D SFT+GRPO, 3 seeds | 135 | 0 | 0 | 135 |
