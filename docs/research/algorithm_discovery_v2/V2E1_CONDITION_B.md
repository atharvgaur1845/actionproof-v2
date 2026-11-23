# V2-E1: is the shared hidden world causally necessary for GRPO's verification collapse? (LLM, development)

Registered 2026-11-23 22:10 IST, before any V2-E1 output exists. Branch `research/algorithm-discovery-v2`.

**What had been seen at registration:**
- all first-stage D1 development outputs on cw_dev and proc_dev (grpo, nostd, awrb, awrbn at seeds 0–1; the
  c = 0.30 runs);
- their training rollouts (first-stage decomposition reproduced by `scripts/ad2/decompose.py`).

**Not seen:** no D2, D3 or C1 output; nothing on cw_conf.

**Hash file:** `results/ad2/hashes_V2E1_CONDITION_B.json` (`scripts/ad2/freeze.py V2E1_CONDITION_B`).

## 1. Question and design

The first-stage diagnosis attributes GRPO's collapse to group-wise standardisation acting on groups that share one
hidden world. The shared world is necessary for that account: within a group, reading then changes only the cost
(clean groups and failed-repair groups) or the outcome. The registered estimator comparison (grpo vs nostd)
varies only the normalisation. It cannot show that the shared world matters.

**Design.** A 2 × 2 factorial on the LLM testbed, with everything else fixed:

| | Shared world (A, first stage) | Resampled world (B, new) |
|---|---|---|
| group std | `grpo` (D1 s0, s1: on file) | **`grpo_rs`** |
| no std | `nostd` (D1 s0, s1: on file) | **`nostd_rs`** |

**Condition B** (`actionproof/ad2/rollout_rs.py`) gives each rollout an independent world from the registered cw1
prior, keyed by its rollout id. Prompts, the initial state, tools, observations, the reward
(R_V − 0.05·reads), the pool (cw_train), the warm start, 200 steps, 4 × 8 rollouts, lr, β, LoRA, temperature and
evaluation are identical to D1. Each arm runs seeds 0 and 1.

Evaluation is the first-stage development evaluation: `actionproof.ad.evaluate` on cw_dev (150 tasks, greedy plus
4 samples) and proc_dev, with each task's fixed pool world.

## 2. Predictions (registered; development, two seeds, descriptive)

The theory is in `GRPO_FAILURE_DERIVATION.md`. With i.i.d. worlds, the within-group std is the task's
world-marginal std σ_x, a per-task constant shared by the cost and benefit terms. Std scaling then rescales each
task's gradient without changing the sign of the read decision.

- **E1-P1 (primary).** `grpo_rs` does not collapse reading. In each seed, the training read rate in the last 20
  steps is at least half the rate in the first 20 steps. For comparison, grpo went 0.59 → 0.03 (s0) and
  0.56 → 0.07 within 25 steps (s1).
- **E1-P2.** On cw_dev, greedy, in each seed, `grpo_rs` makes > 0.3 reads per episode and has J_c > 0.717
  (grpo + 0.01; grpo is 0.707 with 0 reads in both seeds).
- **E1-P3.** The fraction of zero-std groups in `grpo_rs` training over steps 180–199 is below 0.5 (grpo: 0.86).
- **E1-P4 (mechanism).** In `grpo_rs` training rollouts from steps 0–39, the summed group-std advantage of reading
  rollouts (`scripts/ad2/decompose.py`, TRL denominator) is above −30 in each seed. grpo gave −134.3 (s0) and
  −119.3 (s1), while its centring-only sums were −0.36 and −0.19. A sign-match criterion was rejected before
  freezing: centring-only sums near 0 make a sign comparison uninformative.
- **E1-P5 (descriptive, no sign predicted).** `nostd_rs` vs `nostd`. Resampling removes the within-group
  common-random-number pairing of the shared world. Theory predicts the same expected direction and higher
  variance (Sakib 2026, arXiv 2609.24144).

**Falsifier.** If `grpo_rs` collapses in both seeds (last-20-step reads < 0.1 per episode, and greedy cw_dev reads
< 0.1), the shared world is *not* necessary for the collapse. That would refute the account in
`GRPO_FAILURE_DERIVATION.md` §4, and the failure analysis would say so.

## 3. Analysis

`scripts/ad2/e1_analyze.py`, frozen here, reads only `results/ad2/{dev,train_state,rollouts}`. It writes
`results/ad2/e1_summary.json`, with P1–P5 marked true, false or incomplete. No statistical test is made; there are
two seeds.

## 4. Runs and compute

- **Pool:** a100 only.
- **Chain:** one serial chain, at most one GPU:
  - 1300 grpo_rs s0;
  - 1301 grpo_rs s1;
  - 1302 nostd_rs s0;
  - 1303 nostd_rs s1.
- **Estimate:** about 2–2.5 h each on the A100.
- **First-stage tasks:** none is touched.

FILES:
- actionproof/ad2/rollout_rs.py
- actionproof/ad2/train.py
- slurm/ad2_arm.sh
- scripts/ad2/freeze.py
- scripts/ad2/decompose.py
- scripts/ad2/e1_analyze.py
- tests/ad2/test_rollout_rs.py
- actionproof/ad/train.py
- actionproof/ad/rollout.py
- actionproof/ad/worlds.py
- actionproof/ad/rewards.py
- actionproof/ad/evaluate.py
- data/bench/proc/cw_train.jsonl
- data/bench/proc/cw_dev.jsonl

## 5. Deviations

(none yet)
