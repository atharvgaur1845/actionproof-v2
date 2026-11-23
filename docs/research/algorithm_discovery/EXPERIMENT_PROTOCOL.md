# Algorithm discovery: experiment protocol (development phase AD-D1)

Registered 2026-11-23 11:27 IST, before any GPU run of this programme. Branch `research/algorithm-discovery`
(cut from `actionproof-research-upgrade` at 637dcc8). Hash-frozen by `scripts/ad/freeze.py` into
`results/ad_hashes.json`; every Slurm task checks the hashes before training. Deviations go to §8 with a date; this
section above §8 is never edited after registration.

## 1. Question

ActionProof's earlier finding that "state reward makes read-back the default" (A12, read-back 1.00 in 5/5 seeds) was
measured with **free reads** and **dense faults** (about 79% of training episodes carry a fault). The assignment's
central question is about allocating a *costly* interaction budget between execution and verification. Two facts of
the training pipeline matter:

1. **Every rollout of a GRPO group shares one task, and so one hidden fault world** (`meta.faults` is fixed per task).
2. **TRL standardises rewards within the group** (`scale_rewards="group"`).

Derivation (docs/research/algorithm_discovery/FINAL_ALGORITHM.md §2; toy check `results/ad/t0/toy_bandit.json`):
with one aliased read-or-proceed decision, read cost 0 < c < 1 and perfect repair, the expected group-standardised
advantage of "read" is +sqrt((1-ρ)/ρ) in a faulted group and -sqrt((1-ρ)/ρ) in a clean group, **independent of c**.
GRPO's expected update on reading therefore has the sign of 2p - 1. GRPO learns to verify iff the fault probability
exceeds 1/2, whatever the cost. The value-of-information rule is "verify iff p·s > c". Without std scaling (Dr. GRPO),
the sign is p·s - c: correct in expectation.

**Candidate A, AWRB (aliased-world Rao-Blackwellised group advantages).** The simulator knows the fault prior. Each
sampled trajectory's return is replaced by E[R | trajectory]: the prior-weighted mean of R_V over every world in
which the agent's exact calls would have produced the exact same observations (computed by deterministic replay,
`actionproof/ad/worlds.py:marginalise`). The objective and its gradient in expectation are unchanged (tests:
`tests/ad/test_ad.py`). World luck is removed from the returns before group standardisation, so the per-group sign of
the read advantage becomes p·s - c.

## 2. Testbed CW (costed verification), fixed

- Environment: ProcureSim Stage 3 environment, plus call-indexed ids (`CWEnv`). Ids remain pseudo-random strings.
- World prior `cw1` (`actionproof/ad/worlds.py`):
  - P(clean) = 0.6.
  - Otherwise one fault on attempt 0 of one requested write. The site is uniform over the requested steps other
    than post.
  - Kinds by weight: phantom 0.35, partial 0.25 (ratio 1/4, 1/2 or 3/4), dup_delivery 0.15, lost_ack 0.125,
    precommit_timeout 0.125. Each site is restricted to the kinds that are repairable there (order: phantom and the
    two timeouts; receive: no dup_delivery), then renormalised.
- Reward for every arm: R_c = R_V − c · reads, with **c = 0.05**. Reads are the agent's read-tool calls, including
  get_purchase_order.
- Pools (`actionproof/ad/pools.py`; ids checked against every existing split):

  | Pool | Tasks | Seed | Use |
  |---|---|---|---|
  | `cw_train` | 400 | 1201 | training |
  | `cw_dev` | 150 | 1202 | development |
  | `cw_conf` | 300 | 1203 | **locked** until a confirmatory registration |

- Everything else as RSC/A12:
  - warm start `runs/rft_proc`;
  - 200 steps, 4 prompts × 8 rollouts;
  - lr 2e-5, β 0.02, T 1;
  - LoRA r 16;
  - 16 turns, 256 tokens;
  - the system prompt is unchanged, so the agent is never told that reads cost anything.
- Scripted frontier on cw_dev (`results/ad/scripted/cw_dev_frontier.json`, prior-expected):

  | Rule | J_c | VSC |
  |---|---|---|
  | never verify | 0.704 | 0.71 |
  | timeout-only | 0.706 | 0.71 |
  | random 0.5 | 0.778 | 0.88 |
  | always | 0.848 | 1.00 |
  | pay-only | 0.845 | 0.92 |
  | skip-post (verify every faultable write; the VoI rule here) | **0.898** | 1.00 |

## 3. Arms (AD-D1, development)

The arms differ only in the advantage estimator. The reward definition, pool, steps, rollouts, seeds and warm start
are identical.

| Arm | Return used | scale_rewards | Role |
|---|---|---|---|
| `grpo` | R_V − c·reads | group | the standard ActionProof/TRL estimator |
| `awrb` | E[R_V \| traj] − c·reads | group | candidate A |
| `nostd` | R_V − c·reads | none | candidate B: the known fix (Dr. GRPO's std removal only; loss aggregation unchanged) |

Seeds 0 and 1. Tasks `slurm/tasks/1201`–`1206`, after the 2-step smoke test `1200`, which is not analysed.

Two lanes, at most two GPUs at once:
- chain h100: 1201 awrb s0 → 1203 nostd s0 → 1205 grpo s1;
- chain h200: 1210 baselines → 1202 grpo s0 → 1204 awrb s1 → 1206 nostd s1.

Task 1210 is evaluation only (no training): the warm start, A12 R_V s0 and RSC s0 on cw_dev. The smoke test 1200
runs on the h100 lane before both chains.

## 4. Development gates (per candidate vs `grpo`, cw_dev, greedy, mean of seeds 0–1)

| Gate | Requirement |
|---|---|
| G1 | VSC difference ≥ **+0.03**, and positive in each seed |
| G2 | J_c (R_V − 0.05·reads) not lower than grpo's |
| G3 | Duplicate rate no more than grpo's + 0.02 |
| G4 | KL at step 100 no more than 10× grpo's (from `log_history`) |

A candidate that meets G1–G4 proceeds to a separate confirmatory registration:
- `cw_conf`;
- at least 5 seeds per arm;
- baselines: grpo, nostd, the scripted rules, the warm start, A12 R_V and RSC;
- the primary endpoint and the test fixed in advance.

AWRB vs nostd is descriptive at this stage. It becomes a confirmatory hypothesis only if both pass.

## 5. Registered predictions (mechanism; descriptive, two seeds)

- **P1.** grpo's verify-after-pay rate on cw_dev ends below the warm start's. awrb's ends above it.
- **P2.** Verify-after-post falls in awrb and nostd: it has no value under cw1.
- **P3.** nostd has the sign of awrb. Its per-step reward-signal noise (std of R across groups at a fixed policy) is
  higher.
- **Falsifier of the theory in the LLM setting:** grpo keeps or raises its verify-after-pay rate over 200 steps.

## 6. Secondary measurements (descriptive)

- Sampled evaluation: 4 samples at T = 1 per cw_dev task.
- Reads per episode.
- PO reads.
- Per-class VSC.
- proc_dev VSC: the standard free-read split, a regression check against a different objective.
- Training curves: m_reads, m_vsc, m_r_bar.

## 7. Budget and stopping

- At most 8 training runs per candidate version in development.
- At most 2 GPUs at once.
- Task numbers 1200–1299.
- No HF spend without logging it in `docs/stage2/HF_CREDITS.md` first.
- A candidate that fails G1 is diagnosed (FAILURE_ANALYSIS.md), not re-tuned on cw_dev. A revised version is a new
  registration with its own hash.

## 8. Deviations

### Amendment 1 (2026-11-23 11:39 IST, before any GPU run; no task had started, none was claimed)

**What was wrong.** §1's toy derivation and AD-T0 modelled AWRB's return for a *read* rollout as the expectation over
all worlds. That is a forked-continuation Q-value. The implemented estimator (`marginalise`) conditions on the
observations, and a read reveals the world. So a read rollout keeps its own revealed-world return; only rollouts that
never observe the world get the prior mean. The CPU dry run on the login node (`scripts/ad/cpu_dryrun.py`) made this
visible: a read that reveals an unrepaired fault scores 0 while a no-read rollout in the same group scores 0.69.

**Corrected analysis** (AD-T0b, `scripts/ad/toy_verify_bandit_v2.py`, `results/ad/t0/toy_bandit_v2.json`).

| Estimator (return / scaling) | Behaviour in the toy |
|---|---|
| raw/group (grpo) | Never reads whenever reading is optimal. This includes p = 0.6 with imperfect repair (s = 0.6). |
| rb/group (awrb as registered) | Reads even when reading is worthless (s = 0; or c = 0.15 > p·s = 0.12). This is the mirror-image bias: group std scaling is still wrong when the group shares the world. |
| raw/none (nostd) | Correct sign in every setting. |
| rb/none (new arm `awrbn`) | Correct sign in every setting. Higher gradient signal-to-noise than raw/none when repair works (10.7 vs 1.08 at s = 1; 1.09 vs 0.84 at s = 0.6). Lower when repair never works (s = 0), because the raw within-group comparison is already paired by the shared world. |
| raw/batch | Unreliable with 4 groups per batch. Not run. |

**Changes (before any output).**

1. **Arms at c = 0.05,** seeds 0 and 1 each: grpo, nostd, awrb (rb/group, kept as registered), and **awrbn**
   (rb/none; `ESTIMATORS["awrbn"] = ("r_bar", "none")`).
2. **Verification-cost intervention at c = 0.30,** seed 0: awrb and awrbn.
   - On the cw_dev frontier at c = 0.30, never-verify (J 0.704) beats skip-post (J 0.39) and always (J 0.09).
   - **Registered prediction:** awrb verifies after writes at a higher rate than awrbn, and below its own c = 0.05
     rate by less than awrbn does. This falsifies or supports the mirror-bias prediction in the LLM.
3. **Order,** two serial chains, so at most two GPUs:
   - A: smoke → grpo s0 → awrbn s0 → awrb s1 → nostd s1 → awrb c30 s0;
   - B: smoke → baselines → nostd s0 → awrb s0 → grpo s1 → awrbn s1 → awrbn c30 s0.

   Task ids 1200–1211; the never-run D1 task files 1201–1206 are replaced.
4. **Gates.** §4 applies to each of awrb, awrbn and nostd against grpo. AWRB's contribution beyond the known fix is
   tested descriptively here as awrbn vs nostd on cw_dev:
   - VSC;
   - J_c;
   - the across-step std of the training reward signal;
   - the step at which m_reads stabilises.

   It becomes a confirmatory hypothesis only if awrbn passes G1–G4.
5. **Hashes:** `results/ad_hashes_a1.json`. The original `results/ad_hashes.json` is kept as the record of the
   first registration.
