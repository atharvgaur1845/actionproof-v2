# Pre-registration, Amendment 9: a grounded judge, as an audit and as a reward

Registered 2026-11-15 10:45 IST, before any run below had started. The commit time of this file is the registration time. Proposed by a cold external review (Fable 5.2, 2026-11-15): the project diagnoses that transcript judges accept false completions (AuditKIT TaskCompletion on Aya Expanse 32B: 95.2%, n = 476) but does not test a fix. Every earlier registration is unchanged. New files are hashed in `results/a9_hashes.json` by `scripts/freeze_a9.py`.

## 1. Intervention: the receipt ledger in the judge's input

The judge prompts are unchanged except for one block after the agent's report (`actionproof/rewards/llm_judge.py:render_ledger`): one line per write attempt, `op(entity): COMMITTED | PARTIALLY committed (only the first change applied) | NOT committed`, headed as the database's own ledger. It is built from the episode's recorded attempts, the same commit record the verifier reads. With no ledger, the transcript judge's prompt is byte-identical to the one audited before. For AuditKIT TaskCompletion the block is appended to the trajectory text; its prompt, system prompt and parser are otherwise unchanged.

The ledger shows whether writes landed. It does not show policy, eligibility or amounts, so it cannot reveal policy violations (F5, F7) or duplicate effects; those failures stay judged from the transcript.

## 2. Audit (task 420)

Same pool (`results/judge/pool.jsonl`, 476 G = 0 and 339 G = 1 claimed-"completed" episodes), same judges, greedy. Runs: Aya Expanse 32B (FP8) without the ledger (re-run, tag `aya32b_t`) and with it (`aya32b_grounded`); Aya Expanse 8B with it (`aya8b_grounded`, the judge model the reward uses). Episodes are split by `ledger_fail`: some write did not commit or committed partially.

| ID | Prediction (AuditKIT TaskCompletion unless stated) |
|---|---|
| H-A9a (primary) | 32B, grounded: false acceptance on `ledger_fail` episodes ≤ 0.20, against the re-run transcript-only rate on the same episodes |
| H-A9b | 32B, grounded: true acceptance (G = 1) ≥ 0.80 (the ledger does not make it reject everything) |
| H-A9c | 32B, grounded: overall false acceptance < 0.60 (the non-ledger failures stay accepted) |
| H-A9d | 8B, grounded, our judge prompt: false acceptance on `ledger_fail` episodes ≤ 0.50 |

## 3. Reward (task 421): `grounded_judge`

The `llm_judge` arm of round v2 (task 157: Aya Expanse 8B, likelihood of COMPLETE, warm start `rft_tiny_v2`, band `data/curriculum/band_tiny_v2.jsonl`, 200 steps, 8 × 4, seed 0) with the ledger added to the judge prompt. Nothing else changes. Evaluated on val, test_iid and test_ood_fault, greedy, neutral prompt.

| ID | Prediction |
|---|---|
| H-A9e | test_iid VSC ≥ 0.71 (the transcript-judge arm: 0.61) |
| H-A9f | Post-ack refund check on test_iid phantom tasks > 0.50 (transcript-judge arm: 0.00) |

If the judge reward saturates as before (zero-variance groups near 100%), the result is reported as "no learning signal", not as a negative for grounding. One seed; no between-seed claim. Results after the experiment freeze (2026-11-15 18:00 IST) are reported as not finished.
