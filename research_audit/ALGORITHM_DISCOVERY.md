# Algorithm discovery and RL-method audit

2026-11-25.

- **Code read:** `actionproof/rewards/*.py`, `actionproof/integrity/sim/truth.py`, `actionproof/stage3/rewards.py` and `actionproof/ad/rewards.py`.
- **Prior art:** the entries verified at their source in `docs/stage3/LITERATURE.md` (section letters below) and `docs/research/NOVELTY_AUDIT.md`. No new web search was possible in this session, so a novelty statement here is only as current as those logs (Nov 21–24).

## 1. Reward audit (from the implementations)

| Reward | Definition (as implemented) | Information used | Range | Report-sensitive | Penalises harm | Honest failure ≠ false success | Known failure |
|---|---|---|---|---|---|---|---|
| tool_trust (Stage 1) | share of required writes whose last reply was a success × 1[claimed completed] | replies + report | [0,1] | yes | no | no (both 0) | Rewards believing "ok" by construction |
| outcome (Stage 1) | 1 iff every postcondition and invariant holds in the final database | state | {0,1} | no | via invariants | no | Silence = honest failure |
| ack_outcome (A4 H10) | 1 iff every required write was acknowledged and no invariant broke | replies + state invariants | {0,1} | no | via invariants | no | The source-only ablation of outcome |
| PCSTR | G + 0.2P + 0.1V − 0.5I − 0.3F_over − 0.1F_under − 0.2D | state + report | ≈[−1.1, 1.3] | yes | yes (I, D) | yes | Silence loophole: no report scored like honest failure (report rate fell 1.00 → 0.91, seed 0) |
| PCSTR-d | PCSTR with F_over charged on any non-disclosed failure | state + report | same | yes | yes | yes | Fixes the silence loophole. Collapses under pre-commit timeouts (VSC 0.176) |
| **R_V** (A12, all later RL) | 1 iff every requested step verifies in the final state | state | {0,1} | **no** | **no** (collateral-blind) | no | Report-blind, so false success is not penalised directly |
| **R_A** (A12) | 1 iff the last call of each requested step got status ok | replies only | {0,1} | no | no | no | Cannot see faults by construction |
| safe (Stage 3, candidate A) | −1 on an irreversible violation; 1 for goal met and reported completed, or never-commit honestly reported; 0.25 for an honest failure on a reachable task; else 0 | state + report | {−1, 0, 0.25, 1} | yes | yes | yes | No development benefit (§3) |
| lag (Stage 3, candidate D) | task reward − λ·violation; λ ← max(0, λ + 2(violation rate − 0.02)), λ ≤ 5 | state + report | unbounded below | yes | yes | yes | No development benefit. Sparse violations make λ noisy |
| rc / awrb / cvar (AD) | R_V − c·reads; awrb replaces R_V with E[R_V \| trajectory] over aliased worlds | state + reads | [−c·n, 1] | no | no | no | awrb over-verifies. Its own contribution failed AD-C1 (−0.088 vs no-std) |

**Zero-variance groups.** Measured at the A12 gate: the share of groups with nonzero reward variance was 0.48 (R_V) and 0.44 (R_A) under addendum 1, against 0.15 and 0.17 at the original gate (`results/a12b/gate.json`, `results/a12/gate.json`). Per-run training-time shares are in the rollout logs (`results/research/llm_mechanism.json`, `zero_variance_share`).

**R_V and R_A have the same functional form:** a 0/1 indicator over the requested steps. They differ only in their information source, which is what A12's identical-form design requires.

## 2. Matched-comparison matrix (existing RL comparisons)

| Comparison | Init | Budget (steps × rollouts) | Training tasks | Fault coverage | Reward | Seeds | Mismatch |
|---|---|---|---|---|---|---|---|
| A12: R_V vs R_A | same warm start | same | same | same | **differs (the factor)** | 5 vs 5, paired | none |
| S3: RSC vs A12 R_V | same | same | different pool (adds recovery starts) | **RSC adds fault classes** | same | 5 vs 5 | **A recipe comparison**: coverage confounds start state. The duplicate gain is explained by coverage (S3-A1) |
| S3 add. 2: RSC vs class-matched control | same | same | matched classes | matched | same | 5 vs 5 | none for the start-state factor |
| P1: C, D vs A | C, D start from recovery SFT | D same GRPO budget as A | same pools | same | same | 3 vs 3 | D's warm start differs (the factor) |
| P2: E vs D | ledger-formatted SFT | same | same | same | same | 3 vs 3 | Rollout text differs (the factor) |
| AD-C1: estimators | same warm start | same | same | same | same objective; estimator differs | 5 vs 5 | none |
| **P3: Dn vs D (registered today)** | same | same | same | same | same | 3 vs 3 | **only the normaliser** |
| Stage 3 pilots: safe / lag / rv mix | same | 100 steps (pilots) | same mix | same | differs | 1 seed | Development only, single seed |

**Audit finding.** On `proc_dev`, the three 50%-mix pilot arms (rv, safe, lag) produce **identical greedy tool sequences on all 150 tasks**, though they differ on `s3_rec_dev`. Same hash dbd1c40441; the warm start's is ba0c552831.
- **Ruled out:** the evaluation command loads each arm's own adapter (`slurm/s3_arm.sh`), so this is not a pipeline error.
- **Consistent explanation:** under group-std normalisation, rewards that agree up to a per-group affine map give identical advantages. The three rewards differ only on rare events (violations, honest failures), so their updates coincide where those events are absent. This is the per-state invariance of the Phase II theorem, seen in practice. It is plausible, not proven.

## 3. Measured failure modes (what any method must fix)

1. **GRPO's stake-blind vote under shared hidden state.** The normaliser removes reading (AD-C1: all GRPO seeds read 0 times). Fix confirmed: no std (+0.120, 5/5 seeds).
2. **No gradient on rare recovery states.** All rollouts in a group fail. Addressed by recovery starts (S3 +0.21) and demonstrations (P1 +0.306).
3. **Reading without acting.** A14 and E2: the agent sees the shortfall and posts anyway. Demonstrations fix most of it (P1 diagnosis: "read, not acted" 0.117 → 0.019).
4. **Procedure replay** on unseen workflows. Every arm skips the new goods-receipt step (P1 wf4). Not fixed by any method tried.
5. **Development evidence on the recovery-start mixture.** 25% recovery starts underperform 50% on `s3_rec_dev`: 0.553 and 0.667 against 0.807 and 0.807 (2 seeds each, `results/s3/dev`).

## 4. Candidates from the brief, assessed

| | A: adaptive recovery-state sampling | B: constrained multi-objective GRPO | C: recovery-advantage resampling |
|---|---|---|---|
| **Closest prior art** | DAPO dynamic sampling (A2); No Prompt Left Behind (A6); reverse curriculum (C3, C5); Go-Explore (C4); failure-prefix conditioning (C7); Fission-GRPO (NOVELTY_AUDIT §2.1) | CPO (D1); RCPO (D2); Safe RLHF (D3); lexicographic MORL (D6); constrained GRPO (D7) | DAPO dynamic sampling (A2); No Prompt Left Behind (A6). **No adaptive group-size paper was verified in our logs**, so a literature search is required before any novelty claim |
| **Difference we could claim** | Allocation by measured recovery-stratum informativeness with an exposure floor: a curriculum variant | Constraint normalisation per component inside GRPO, which D7 already does | Budget-matched group-size adaptation per state class |
| **What falsifies novelty** | Any adaptive curriculum over start states. A6, C5 and C7 come close | D7 | Unknown until that search |
| **Existing evidence** | 50% mix > 25% mix on development (2 seeds) | **Already piloted: no benefit** (Stage 3 §6). safe_mix and lag_mix were below rv_mix on `s3_rec_dev` (0.540 / 0.560 vs 0.633, seed 0, 100 steps) | None |
| **Minimum experiment** | 5 arms × ≥3 seeds × 200 steps at an equal rollout budget, plus a registered test: about 15 GPU-runs of ~3.5 h | Already done at development scale | 3 arms × 3 seeds at an equal model-call budget |
| **Expected information gain** | Moderate: could improve recovery, but the mechanism is known | Low: already negative | Low to moderate |
| **Deadline feasibility (Nov 28; H200 draining; one A100 and one H100 lane)** | **No.** It needs ~50 GPU-hours before a registered test | — | Marginal |

## 5. Selection: criteria fixed before choosing, and decision

**Criteria, in order:**
1. Tests a mechanism that already has registered support.
2. Can falsify a published claim.
3. Changes one factor only.
4. Fits the remaining GPU budget with a registered confirmatory test.
5. Novelty is **not** a criterion. A curriculum variant would not be claimed as a new optimiser.

**Decision: P3**, registered today (`PREREGISTRATION_P3.md`, commit 7074e8d) before any training.
- **Question:** does AD-C1's confirmed fix, removing the group std, improve P1's best agent D, with nothing else changed? 3 seeds; frozen P1 test.
- **Why it wins:** it scores on criteria 1–4. Candidates A and C fail criterion 4, and B is already negative.
- **What it can falsify:** "the normaliser is what limits verification learning in the practical agent". The registration predicts the direction is uncertain, because D's warm start already verifies.

**Not done:**
- No new optimiser is proposed.
- No candidate was launched without its criteria fixed in advance.
- No confirmatory output was used to choose.

## 6. Answers

- **Does a new algorithm outperform RS-GRPO?** None was built. The only registered improvements over RS-GRPO come from **data**: recovery demonstrations (P1 C, D). On the estimator side, AD-C1 supports **Dr. GRPO's existing** estimator, in the costed world. P3 tests it on top of D.
- **Strongest algorithmic contribution:** an explanation, not an optimiser. Std normalisation is stake-blind under shared hidden state (theorem, plus registered TH1 and AD-C1).
