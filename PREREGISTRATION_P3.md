# PREREGISTRATION P3: does AD-C1's confirmed fix carry into the practical recovery agent?

Registered 2026-11-25, before any P3 model is trained. The commit time of this file and of `results/p3_hashes.json` (`scripts/freeze_p3.py`) is the registration time.

## 1. Question and motivation

AD-C1 (registered, reported 2026-11-25) confirmed that removing the group standard deviation from the GRPO advantage (Dr. GRPO, prior work) raises verified completion on the costed-verification world: +0.120, all five seeds. The Phase II theorem predicts why. With a shared hidden state, std-normalised updates are stake-blind; centred updates follow the action-value gap.

P1's best agent, D, was trained with standard std-normalised GRPO (recovery-SFT warm start, then the recovery-start recipe). **P3 asks whether the same normaliser change improves D**, with nothing else changed.

This is a test of an existing method in a new setting. No new algorithm is claimed, whatever the outcome.

## 2. Arms

| Arm | Training | Seeds |
|---|---|---|
| D | P1 arm D, already trained and evaluated (`runs/p1_d_s{0,1,2}`; frozen test outputs `results/p1/test/D_s*__*`). Reused unchanged | 0, 1, 2 |
| **Dn** | P1 arm D's exact command run through `actionproof/p3/train_nostd.py`, which sets TRL `scale_rewards="none"`. Same warm start (`runs/p1_dwarm/final`), pools (`s3_trainb_half` + `s3_rec_train_half`), reward R_V, 200 steps, LoRA r16, lr 2e-5, β 0.02, group size 8, 4 prompts per step, vLLM colocate | 0, 1, 2 |

Matched:
- initialisation, training tasks, fault coverage, reward, rollout count, steps and seeds;
- the evaluator, `scripts/p1_eval.py`, unchanged: 4 samples at T = 1 plus greedy.

The only difference is the advantage normaliser.

## 3. Evaluation

- **Development:** p1_dev_full and p1_dev_rec, for monitoring only. They are used for no choice: Dn runs the test regardless.
- **Test:** the P1 frozen test union (450 tasks), once per model.
- **Disclosure:** D's per-task test outcomes were inspected during the P1 diagnosis. The P3 design (the normaliser) comes from the AD-C1 registration and the theory, not from P1 test outcomes, and no tuning is done.

## 4. Hypothesis and analysis (`scripts/p3_stats.py`, frozen with this file)

**H1.** Dn > D on sampled VSC over the test union.

- **Statistic:** a model's task score is its mean over 4 samples; an arm's task score is the mean over its 3 seeds. Δ = mean over 450 tasks of (Dn − D).
- **Test:** paired task bootstrap (10,000 resamples, seed 20261125), one-sided. There is one primary hypothesis, so no multiplicity correction.
- **Supported** iff p < 0.05 **and** all three seed-paired differences (Dn s_k − D s_k) are positive.

**Descriptive** (no tests): greedy VSC; per-split VSC; false success; reads per episode; partial-payment repair; the unseen workflow.

**Prediction recorded now.** From the theorem, Dn should read at least as often as D on faulted tasks. The direction of H1 is uncertain, because D already verifies (its recovery demonstrations put reading in the warm start), so the vote may not be binding for D.

## 5. Rules

- No test split is used for any choice.
- No hyperparameter is tuned.
- A failed or crashed seed is reported, never replaced silently.
- Deviations are logged in `docs/stage2/DECISIONS.md`.
