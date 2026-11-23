# PREREGISTRATION P1: a practical recovery agent (four controlled arms)

Registered 2026-11-23, before any P1 training or evaluation output. Hashes: `results/p1_hashes.json` (`scripts/freeze_p1.py`). Branch `actionproof-research-upgrade`.

**Not a new RL algorithm.** This study tests training recipes and an inference-time aid built from known components.

## 1. Question

Can the existing best agent (recovery-start GRPO, RSC) be turned into an agent that repairs correctly from the evidence it can see? The motivating evidence is registered (`docs/research/E1E2_RESULT.md`):
- agents read the shortfall but often do not act on it;
- a reconciliation line computed from visible values raises correct first repairs after a goods receipt by +0.48 (E2 H2);
- re-paying with the original idempotency key is a common ineffective retry.

## 2. Business semantics (fixed before generating any target)

The verifier is unchanged (`actionproof/integrity/sim/truth.py`). The agent's stated policy is "Pay suppliers only the purchase order amount, once".

**Correct partial-payment repair:**
- refund the partial payment, then re-pay the full amount with a **new** idempotency key;
- then verify, then continue.

**Rejected as a repair:**
- **Paying only the outstanding remainder (top-up)** leaves two live payments, which the verifier marks DUPLICATE, consistent with "once". It is not a target.
- **Re-paying with the original key** is deduplicated by the payment service, so no new payment happens. It is not a target.

**Duplicate delivery:** refund every extra live payment. **Partial goods receipt:** record the missing quantity. **Never-commit:** report "failed" after the retry limit.

## 3. Data (`actionproof/p1/build.py`; splits and demonstrations hashed here)

**Fresh splits.** Each split uses new seeds, and task and PO ids are disjoint from every existing split, E1/E2 included.

| Split | Size | Use |
|---|---|---|
| p1_train_full | 240 | Train: full episodes, 80 per workflow |
| p1_train_pp | 90 | Train: partial payment at ratios 0.25/0.5/0.75 |
| p1_train_rec | 300 | Train: recovery starts |
| p1_dev_full | 90 | Development |
| p1_dev_rec | 90 | Development |
| p1_test_full | 120 | Frozen test |
| p1_test_pp | 90 | Frozen test: **unseen ratios 0.4 and 0.6** |
| p1_test_rec | 120 | Frozen test |
| p1_test_wf4 | 60 | Frozen test: **unseen workflow** order→receive→pay→post |
| p1_test_combo | 60 | Frozen test: two faults |

**Teacher.** The scripted oracle. It decides only from read results and the request; the simulator's fault log is never read, and hidden state is used only to validate.
- Episodes are replayed in the exact training/evaluation conversation format; only post-prefix turns become targets.
- Kept only if the verifier confirms the correct outcome: goal met, or an honest "failed" report on never-commit.
- Every post-refund re-pay must use a new key.

**Result:** 630 verified episodes. CuratorKIT exact + MinHash (0.95) deduplication removed 0; a balance cap of 12 per (workflow, fault class, masked action pattern) removed 76. That leaves **554 episodes, 3,810 SFT turns, 178 distinct action patterns** (`data/rft/p1_recovery_rows.jsonl`).

## 4. Arms (Tiny Aya Global, LoRA r16; same verifier, tasks and decoding at evaluation)

| Arm | Model | Extra training |
|---|---|---|
| **A** | RSC seeds 0–2 (`runs/s3_rv_mix_s{0,1,2}`), the existing best agent | — |
| **B** | A, plus the reconciliation line at inference on every `list_payments` result with 0 < live < total | — (inference aid) |
| **C** | A, then recovery SFT: 1 epoch, lr 5e-5, LoRA r16, seed k, on the merged RSC seed-k model | about 240 SFT steps |
| **D** | New warm start: original warm-start demos plus recovery demos, with `rft_proc`'s exact settings (2 epochs, lr 1e-4, seed 0). Then the **RSC recipe unchanged** (R_V; `s3_trainb_half` + `s3_rec_train_half`; 200 steps; 4 × 8 rollouts; lr 2e-5; β 0.02), seeds 0–2 | Warm-start data larger (budget difference reported) |

**About the line in B.**
- Its values are all visible to the policy: the request states the total, and the line sums the same result's unrefunded payments.
- Product reading: a runtime reconciliation gateway.
- Arm B is paired with A by seed (same weights).

**Budget.**
- A and D have the same RL budget; their warm-start data differ.
- C adds SFT on top of A.
- Training time and steps are reported per arm.

## 5. Evaluation (`scripts/p1_eval.py`; E1's episode loop)

- One greedy run plus 4 sampled runs (T = 1, seeded per task, sample and turn).
- 16 turns, 256 tokens, context 8,192.
- **Development** (p1_dev_full, p1_dev_rec): every arm and seed.
- **Frozen test**: all five test splits, every arm and seed, **run once**; the script refuses to overwrite.

## 6. Metrics (all from the deterministic verifier and the trace; no LLM judge decides any outcome)

- VSC, false success, unrefunded duplicates, honest failure.
- Partial-payment tasks repaired (VSC on tasks with a partial payment fault).
- Share of post-refund re-pays using a new idempotency key.
- Reads, and redundant reads (the same read repeated with no write in between), per episode.
- Policy tool calls per successful task.
- Episodes ended by the context limit.
- Each metric greedy and sampled, per split and per workflow.

**AuditKIT (secondary, descriptive).** Its TaskCompletion metric is scored on a sample of recorded test trajectories and reported next to the verifier. It never replaces it.

## 7. Hypotheses and analysis (`scripts/p1_stats.py`, frozen with this file)

| | Hypothesis | Test |
|---|---|---|
| **H1** | B > A | Sampled VSC, test union (450 tasks) |
| **H2** | C > A | Same |
| **H3** | D > A | Same |

For each test: a paired task bootstrap over the seed-averaged per-task scores (B = 10,000) with a one-sided p, Holm over the three. **Supported** iff Holm p < 0.05 **and** the gain is positive in all three seed pairs.

Three seeds per arm cannot establish seed-level significance; the seed-consistency condition is the stated guard.

**Development winner.** The arm among B, C, D with the highest dev sampled VSC, if it beats A by ≥ 0.03. It labels the demonstration agent only; every arm runs the test.

**Descriptive.** Every §6 metric per arm and split; per-workflow repair (pay-first vs receive-then-pay vs order-then-pay); greedy vs sampled.

## 8. Rules

- No test split is used for any choice.
- No hyperparameter is tuned after any output.
- A failed arm is reported with a diagnosis.
- Deviations are logged in `docs/stage2/DECISIONS.md`.
