# PREREGISTRATION LAB1: counterfactual decision-value credit (CDVO family) in the exact costed-verification testbed

Registered 2026-11-24 (IST), before any registered LAB1 output. The registration time and file hashes are in
`results/lab1_hashes.json` (`python scripts/lab/freeze_lab1.py`). Branch `research/lab-final`.

**What was seen before registration (disclosed).** (i) A timing dry-run of the 14 L1 arms on one setting
(rich multi, c 0.05, L_f 1) with seed 99, which printed each arm's regret. (ii) An analysis smoke test on seeds 98–99
(written outside `results/`) for two L1 settings, the L2N cells with p = 0.4 and all of L3G, which exercised
`lab1_analyze.py` end to end and printed its gates. Every arm, budget, hypothesis and the whole analysis code were
written before (i) and (ii) and were not changed after them. No registered seed (0–4) has been run. Seeds 98–99 are
not used anywhere in the registered analysis.

## 1. Question

Does credit computed from paired forks of the simulator state at verification decisions (verify branch vs
continue branch, each continued on-policy, common random numbers, same latent fault state), the "decision-value"
signal Δ(h) = Q^π(h, verify) − Q^π(h, continue), give better net utility than the strongest simple baselines when
both get the same number of environment episodes? Is allocating forks to decisions whose Δ sign is uncertain better
than allocating them at random?

**Prior art (docs/research_lab/RELATED_WORK_AND_NOVELTY.md).** Paired forks with CRN are TRPO's vine estimator
(Schulman et al. 2015 §5.2); allocating rollouts to states whose best action is uncertain is RSPI (Dimitrakakis and
Lagoudakis 2008); forking on-policy trajectories at selected decision points to replace advantages is Counterfactual
Rollout Replay (arXiv 2609.33875); compute-optimal branch placement is EPIG-Tree (arXiv 2609.20004). LAB1 is
therefore a **test of known estimator families in this setting**, not of a new method.

## 2. Testbed

The v2 exact tabular testbed (`actionproof/ad2/tabular/`, frozen in `results/ad2/hashes_V2TB_PROTOCOL.json`),
unchanged. New estimators are in `actionproof/lab/cdvo.py`, which reproduces v2's trainer bit-exactly for every
shared estimator (`tests/lab/test_cdvo.py::test_reproduces_v2_trainer_exactly`). Policy: per-decision softmax over
observable features only (tested: `test_policy_inputs_never_contain_the_hidden_world`). The hidden fault plan is used
only by the simulator (fork targets), never as an actor or critic input.

## 3. Families

| Family | Content | Seeds |
|---|---|---|
| L1 | The 12 TB2 settings (mode {single, multi} × c {0.02, 0.05, 0.1} × L_f {0, 1}; D 0.5; three workflows trained jointly). Two episode budgets: E1 = 6,400 and E4 = 25,600 episodes. | 0–4 |
| L2N | Normalisation intervention on the core workflow: p {0.1, 0.2, 0.4, 0.6} × c {0.01, 0.05, 0.2} × s {0.6, 1.0}, L_f 0; arms grpo, nostd, indep_world_std, indep_task_std (n_ind = 16); 200 steps | 0–2 |
| L2G | Gradient bias and variance at a fixed policy (`scripts/lab/grad_snr.py`, N = 300 batches, θ ∈ {warm start ρ 0.3, undecided ρ 0.5}, rich multi c 0.05 L_f 1) | — |
| L3G | Generalisation (v2 TB3 tasks, feature policy): nostd, rloo, igc_v1, ppo, critic_q at 1,600 steps (51,200 episodes); cdvo_adapt@0.25, cdvo_ia@0.25, fork_act at a 51,200-episode budget; evaluated on train, test_iid, test_shift (unseen p, s, c) and test_long (unseen workflow length) | 0–4 |

**L1 arms.**
- Step-budget baselines (200 steps = E1, 800 steps = E4): grpo (E1 only), nostd (Dr. GRPO), rloo, critic_v
  (learned value baseline), critic_q (learned Q, all-action), ppo (learned value baseline, clipped surrogate, 4 epochs,
  clip 0.2, no advantage normalisation), ppo_norm (E1 only; per-batch advantage normalisation), igc_v1 (exact −c on
  uninformative first reads; needs the prior's support), awrb (E1 only), awrbn.
- Episode-budget CDVO arms (training stops at the first step with episodes_used ≥ budget; forks charged pro rata,
  as in v2): fork_act (forks at every verification decision = "full branching"), fork_act_nocrn (no common random
  numbers), cdvo_fixed@0.25 (random 25% of verification decisions), cdvo_adapt@0.25 and cdvo_adapt@0.10 (same expected
  count, allocated by lowest |mean Δ̂| / se over the context's last 20 fork contrasts from earlier steps),
  cdvo_adapt_critic@0.25 (unforked decisions in contexts with |z| ≥ 2 use the stored fork mean as a learned critic),
  cdvo_ia@0.25 (igc_v1 gate on uninformative first reads; adaptive forks among informative decisions).
- References computed exactly: the Bayes-optimal policy (expectimax; regret 0 by definition) and the scripted policy
  (verify every write with p > 0, repair on a fault flag, never re-verify, verify after a timeout, retry only when the
  read shows "not applied"), reported by `scripts/lab/lab1_refs.py`.

## 4. Hypotheses (decided now)

| ID | Statement | Rule |
|---|---|---|
| H1 (primary) | cdvo_adapt@0.25 beats the strongest non-privileged simple baseline SFB_np = argmin regret over {nostd, rloo, critic_v, critic_q, ppo} at the same episode budget | supported iff at E4: T1 (seed-mean regret ≥ 20% below SFB_np and lower in ≥ 4 of 5 seeds), T2 (never-faulted-op verify ≤ SFB + 0.02), T3 (duplicate loss ≤ SFB + 0.01) all pass **and** Holm-adjusted one-sided paired sign-flip p < 0.05 over the 60 (setting, seed) units |
| H2 (primary) | cdvo_ia@0.25 beats SFB_all = argmin over SFB_np ∪ {igc_v1} | same rule at E4 |
| — | Holm family | {H1, H2} × {E1, E4} (4 tests); E1 is reported, the decision is at E4 |
| H3 | Common random numbers reduce the variance of the fork gradient on the risky verify logits (order, receive, pay) | Var(fork_act) / Var(fork_act_nocrn) < 1 for all three entries at both θ in L2G |
| H4 | The hidden-state-conditional scale, not finite-sample sharing, is the causal source | indep_world_std wrong-decision rate ≥ 0.30 **and** indep_task_std ≤ 0.15 (TB1 definition: c ≥ 0.01, \|VOI\| > 0.02, decision = verify prob > 0.5 vs sign VOI) |
| H5 | Generalisation of a supported candidate (T4) | test_shift and test_long regret ≤ the best of its SFB pool in L3G. Read descriptively; only relevant if H1 or H2 is supported |

Predictions recorded now (from v2 V-F6 and the derivation): H1 not supported, H2 not supported, H3 supported,
H4 supported.

## 5. What the go/no-go rule (author's Part IX) requires of CDVO, and where each item is decided

| Item | Where |
|---|---|
| Defensible difference from nearest prior art | RELATED_WORK_AND_NOVELTY.md §4 (decided before results: none found) |
| Estimator justified | docs/research_lab/CDVO_THEORY.md |
| Implementation tests | tests/lab/ |
| Testbed supports it, beats strongest simple baseline at matched budget, net utility, no unnecessary verification, no duplicate regression, several seeds | H1/H2 with T1–T3 |
| Generalises | H5 |
| LLM evidence | Only if H1 or H2 is supported **and** the novelty item passes. Otherwise no GPU run is registered. |

## 6. Analysis

`scripts/lab/lab1_analyze.py` (frozen with this file) → `results/lab/lab1_summary.json`, run once after L1, L2N and
L3G finish; `scripts/lab/grad_snr.py` → `results/lab/grad_snr.json`; `scripts/lab/lab1_refs.py` →
`results/lab/lab1_refs.json`. Descriptive comparisons (adapt vs fixed, CRN vs none, critic variant, b = 0.1 vs 0.25,
ppo vs ppo_norm, cdvo_ia vs igc_v1) carry no tests.

## 7. Deviations

(none yet)

FILES:
- PREREGISTRATION_LAB1.md
- actionproof/lab/__init__.py
- actionproof/lab/cdvo.py
- actionproof/lab/exact_vote.py
- scripts/lab/lab1_sweep.py
- scripts/lab/lab1_analyze.py
- scripts/lab/grad_snr.py
- scripts/lab/lab1_refs.py
- tests/lab/test_cdvo.py
- tests/lab/test_exact_vote.py
