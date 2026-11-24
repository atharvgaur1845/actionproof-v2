# Experimental audit (research/latent-state-theory, 2026-11-24)

## 1. Baseline re-verified (local, seconds)

| Check | Result |
|---|---|
| `scripts/s2_numcheck.py` | 406 numbers checked, 7 documents, 0 failures |
| `python -m actionproof.reporting.canonical` | 54 claims, 0 mismatches (P1, P2 and AD-C1 then incomplete) |
| `pytest tests/reporting tests/p2 tests/product tests/ad2 tests/lab` | 89 passed |
| New: `pytest tests/theory` | 44 passed |
| `scripts/lab/freeze_j2.py --check` | OK |
| `scripts/theory/freeze_th1.py --check` (cluster copy) | OK before the held-out launch |

## 2. Evidence ledger of this phase

| Item | Level | File | Registered before output? |
|---|---|---|---|
| Exact identities (Props 2.1–4.4, Thm 3.1, Cors 3.2–3.3) | math | `docs/theory/PROOF_AND_ASSUMPTIONS.md`, `tests/theory` | n/a |
| Analytical scan | math (exact enumeration) | `results/theory/scan.json` | n/a (not a test) |
| TH1 held-out testbed | analytical simulation, registered | `results/theory/th1_heldout_summary.json` | **yes** (`results/th1_hashes.json`) |
| TH1 dev | development | `results/theory/th1_dev_summary.json` | used to fix strata only |
| LLM mechanism diagnostics | development, descriptive | `results/research/llm_mechanism.json` | no |
| AD-C1 theory predictions | prediction | `docs/research/AD_C1_THEORY_PREDICTION.md` | frozen before any cw_conf output |
| J2 | exploratory (registered) | `results/j2/j2_summary.json` | yes (J2) |
| P1 (lead engineer) | confirmatory | `results/p1/p1_summary.json`: H2 +0.259, H3 +0.306 supported; H1 −0.030 not supported | yes (P1); not analysed here |

## 3. Integrity checks on the existing experiments used here

- **Group sharing in training:** confirmed in code and in logs (800/800 groups).
- **TB1 recount:**
  - GRPO 319/495 and Dr. GRPO 28/495 runs wrong reproduce.
  - 494/495 is a run count from the exact enumerated sign, not from the p·s > 1/2 rule.
  - TB1 is development evidence for the theorem (it motivated it).
- **The "nostd" arm** is std-free GRPO with TRL's default DAPO token aggregation. That is not literally Dr. GRPO's
  constant normaliser; the difference is one positive scalar per step.
- **The c = 0.30 interventions** exist only for awrb (std + RB) and awrbn (centred + RB). There is no plain-GRPO run
  at another cost, so GRPO's cost-blindness was never tested on the LLM. Only the counterfactual re-weighting of fixed
  rollouts (MECHANISM_DIAGNOSTICS §3.2) bears on it.
- **Development greedy seeds:** grpo s0/s1 and nostd s0/s1 give identical cw_dev greedy summaries
  (`results/ad/d1_summary.json`), so they are not independent evidence. Their sampled VSC differ.
- **V2-E1 (iid states on the LLM, tasks 1300–1303):** never ran (GROUPS bug, verified in the previous lab). Under
  Cor. 3.3 it would test "interior read rate", not "no collapse". It is not re-queued: no GPU time after Nov 25 06:00.
- **The G audit** (`results/quality/g_mismatch_s0_ck100.json`): importance ESS 1.01 of 322 hinted episodes. Any
  estimator that reweights hinted rollouts on-policy is unusable. This rules out off-policy correction variants of
  the interventions considered in §5.

## 4. Statistical conventions

- **TH1:** criteria are prediction-accuracy rates with 95% bootstrap intervals over cells, seeds averaged within a
  cell. There are no p-values and no multiplicity correction; all criteria are reported.
- **Mechanism diagnostics:** group bootstrap, 2000 draws, seed 20261124.
- **AD-C1:** the registered permutation test and Holm (lead engineer). This branch only compares the outcome with the
  frozen predictions.

## 5. Interventions considered (go/no-go), none pursued

| Candidate | Failure mode it targets | Expected update | Already solved by | Verdict |
|---|---|---|---|---|
| Decorrelate hidden states within groups (independent resampling) | stake-free vote | Cor. 3.3: still vote-governed near always-verify; regret Θ(1/G) | — | **no-go**: provably partial; also changes the per-prompt baseline |
| State/task-family-level scale (EMA, batch std) | group-local scale depends on hidden state | sign ΔQ in expectation | Dr. GRPO, RLOO (no scale at all) | no-go: no advantage over removing the std; EMA arms collapsed in D2 |
| Observation-measurable return (AWRB) | stake-free vote | sign ΔQ only for s = d = 1 and state-independent cost; over-verifies otherwise | Dr. GRPO | no-go (prior result; counterexamples 5, 6) |
| Per-state weighting by estimated stakes | lost severity | requires state labels at training time | Dr. GRPO | no-go: needs hidden-state information unavailable to a deployed sampler, and Dr. GRPO already weighs stakes |

**No new algorithm is recommended.** The practical recommendation is to drop the group std, or to use any scale
independent of the group's hidden states, when groups share environment state and the task involves costed
information actions.
