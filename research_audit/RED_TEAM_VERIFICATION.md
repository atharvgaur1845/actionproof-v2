# Red-team verification (addendum §3)

2026-11-25. The hostile reviews are treated as hypotheses. Each core criticism gets a recomputation, a verdict and the wording consequence.

Reviews read:
- `docs/RESEARCH_RED_TEAM_REVIEW.md`: five reviewers, 2026-11-22, branch `stage4` at 198bea0;
- `docs/stage3/ACADEMIC_REVIEW_POST.md`: post-S3 review;
- `docs/CONSISTENCY_AUDIT.md`: rounds 1–2, 35 items, all marked fixed;
- `docs/research/FINAL_REPORT.md`: the Stage 4 report;
- `docs/final/FINAL_REVIEW.md`.

Recomputations were run on Slurm as `scripts/audit/redteam_recompute.py` (job 469877, output `results/audit/redteam_recompute.json`) and `scripts/audit/p1_decomposition.py` (job 469876).

Many criticisms predate later registered studies, so the verdict column also records which later study answered each one.

## 1. Reviewer 1: RL algorithms

| # | Criticism | Recomputation | Verdict | Wording consequence |
|---|---|---|---|---|
| R1.1 | "The update is plain policy gradient": `num_iterations = 1`, clip never entered | `clip_ratio/region_mean` max is 0.0 over every logged step of **all 120** GRPO runs that log it (`train_logs.*.clip_region_max`). The TRL 1.7.1 source and the saved configs agree (`OPTIMIZER_AUDIT.md` §1–2) | **Confirmed**, with one nuance. This is also the original GRPO setting (Shao et al. 2024: "a single update following each exploration stage"), so "GRPO" is not a misnomer if the single update is stated | **Fixed** (commit 7d06fe8): README, WRITEUP §17, paper v3 |
| R1.2 | Near-deterministic policy (entropy about 0.03) | RSC mean token entropy 0.029–0.033; A12 R_V 0.019–0.021; P1 D 0.008–0.009 (`train_logs.*.entropy_mean`) | **Confirmed**, and stronger for P1 D | none needed. Disclosed in OPTIMIZER_AUDIT §3 |
| R1.3 | Group scaling: one success in eight gets an advantage of +2.47 | (1 − 0.125) / (0.3536 + 10⁻⁴) = 2.474 (`advantage_one_success_in_8`) | **Confirmed** | — |
| R1.3b | "No run used the unscaled alternative" | Answered later: AD-C1 trained no-std GRPO (5 seeds). H_nostd supported, +0.120 [0.088, 0.153] (`results/ad/c1_summary.json` `results.H_nostd`). P3 applies it to P1 D (running) | **Resolved by AD-C1** | — |
| R1.4 | Zero-variance groups: RSC 0.48/0.44; dynamic sampling not used | RSC 0.44–0.51 over 5 seeds; R_A 0.78–0.81; **P1 D 0.78–0.82; ID1 0.98** (`frac_zero_std_mean`). `epsilon_high` null; no group filtering in the config | **Confirmed**. "DAPO" here is only the loss aggregation | OPTIMIZER_AUDIT §4 says so. No document calls any arm DAPO |
| R1.5 | A12 is close to true by construction (R_A never pays for reading; saturated) | R_A zero-variance share 0.78–0.81 (`train_logs.a12_proc_ra_s*`); R_A greedy calls equal the warm start's on 500/500 tasks (`results/s2/a12_identity.json`, claim A12-IDENTITY) | **Confirmed** | **Still open in wording.** WRITEUP §21 item 1 keeps "Reward observability selects verification". The red-team rewording ("a reward that depends on hidden state that one free read reveals …") is not adopted. The incentive-dissociation run was never done. **Main session: adopt the narrowed wording** |
| R1.6 | FCA's gradient argument fails; forced branches weighted below e⁻¹⁰ in about 90% of steps | FCA seeds 0/1: log-prob gap > 10 nats in 184 and 180 of 200 steps; mean IS weight 0.784/0.788, against 0.957–0.990 for RSC (`train_logs.s4_fca_s*`, `s3_rv_mix_s*`). Code: forced tokens logged with sampling log-prob 0 (`actionproof/stage4/rollout_fca.py` docstring) | **Confirmed** | **Not fixed in paper v1.** `paper/appendix.tex` l. 79 still says the IS weight is π_old(a), "giving the all-action gradient term". `paper/main.tex` l. 109 says "the all-action estimator". v1 is superseded by v3, which has no FCA text. **If v1 or its appendix ships, correct both sentences** to "weights a forced branch by the probability of its exact text, below e⁻¹⁰ for at least one branch in about 90% of steps" |
| R1.7 | Weak baselines: no SFT on oracle recoveries, no dynamic sampling, no unscaled advantages, no prompted baseline | SFT on oracle recoveries: **answered by P1** (C, D; +0.259 / +0.306, registered). Unscaled: answered by AD-C1. Dynamic sampling: still untested. Prompted baseline: still none (no ProcureSim run of a prompted or larger policy, red team R2.4) | **Partly resolved.** The reviewer's top suggested baseline became the strongest result | Note the two missing baselines in Limitations |

## 2. Reviewer 2: tool agents

| # | Criticism | Recomputation | Verdict | Consequence |
|---|---|---|---|---|
| R2.1 | "Verification" is one unconditional read (2.70 reads per clean episode) | claim S3-R: clean episodes 6.40 calls, 2.70 reads for every R_V-family arm (`results/s3/s3_review_numbers.json`). The AD costed testbed answers the costly-read version: GRPO stops reading entirely at c = 0.05, no-std reads 0.93–2.02 (`results/ad/c1_summary.json`) | **Confirmed** for free reads. The costed-read follow-up exists | Keep "read-back", not "calibrated verification", for the ProcureSim agents |
| R2.2 | A runtime wrapper already recovers every phantom write | claim H7: verify-before-retry recovers all; keys alone none (`results/a11/h7_matrix.json`). P2's guard *lowers* every arm (H4 −0.122), because rule R1 blocks the repair (`results/quality/p2_guard_diagnosis.json`) | **Confirmed for phantoms**, and the guard result shows a wrapper is not free | Already stated (P2-DIAG) |
| R2.4 | No other model on ProcureSim | still none | **Confirmed, open** | Limitation |
| R2.5 | Greedy evaluation only | answered later: E1 (sampled, 8 samples), P1/P2/P3 (4 samples, T = 1). Greedy vs sampled per-run ρ 0.92 (claim E1E2) | **Resolved** for the later studies | — |

## 3. Reviewer 3: interpretability

Recomputed for `INTERPRETABILITY_AUDIT.md` from `results/a16/a16_summary.json`:
- probe AUROC is 1.00 in every model, including the warm start;
- patching does no better than a wrong-task donor in every model;
- the ablation also hits Q0 (the absent-payment branch) in every model;
- R_V seed 4 meets the ablation rule without repair behaviour.

**Verdict: confirmed.** The README, write-up and paper v3 now say "decodable; ablation lowers the margin, more after R_V; patching did not meet its rule" (INTERPRETABILITY_AUDIT §2).

**Residual:** paper v1 l. 77 still leads with the AUROC result. Superseded.

## 4. Reviewer 4: benchmarks and systems

| # | Criticism | Recomputation | Verdict | Consequence |
|---|---|---|---|---|
| R4.2 | Constant training env seed | disclosed (CONSISTENCY_AUDIT #17) | Confirmed; no validity effect found by the reviewer | — |
| R4.3 | **Test splits are 28/47 cells, not 300 tasks; task CIs overstate precision** | Cell-clustered bootstrap (same method, B = 2,000, seed 0). Below. Full table: `results/audit/redteam_recompute.json` `s3`, `p1` | **Confirmed in kind**; **the conclusions change for one contrast** (below) | Report cell-clustered intervals next to task intervals for S3. **Main session: add to the write-up's S3 rows** |
| R4.4 | The verifier is the reward; second verifier by LLM agents from the same spec | `CLAIM_VERIFICATION.md`: 95,100 episodes recomputed, final-state agreement 100%, by a reference verifier written from the spec | **Confirmed**: a consistency check, not ground truth | Wording already "consistency check" (CONSISTENCY_AUDIT #33) |
| R4.5 | S3 rollout logs not in the repo | still true; P1/P2 traces (~760 MB) also unpublished | Confirmed, open | Author decision (SUBMISSION_CHECKLIST) |
| R4.6 | Public site was the Stage 1 build | Space redeployed Nov 24 with sections 09–10 (main session) | **Resolved** | — |

### Cell-clustered vs task intervals (seed-averaged VSC difference)

| Contrast | Split | Δ | Task 95% CI | Cell 95% CI (cells) | Red team, 3 control seeds |
|---|---|---|---|---|---|
| RSC − A12 R_V (5 v 5) | s3_conf_full | +0.209 | [0.173, 0.250] | [0.060, 0.365] (28) | [0.063, 0.353] (≈; same method, different resample draws) |
| RSC − A12 R_V | s3_conf_rec | +0.206 | [0.165, 0.249] | [0.098, 0.322] (47) | [0.098, 0.322] (matches) |
| **RSC − control (5 v 5)** | s3_conf_full | +0.102 | [0.081, 0.125] | **[0.014, 0.200]** | [−0.023, 0.209] with 3 seeds |
| RSC − control (5 v 5) | s3_conf_rec | +0.107 | [0.085, 0.131] | [0.040, 0.183] | [0.007, 0.175] |
| RSC − ACK (5 v 5) | s3_conf_full | +0.027 | [0.012, 0.042] | **[−0.033, 0.085]** | — |
| RSC − ACK | s3_conf_rec | +0.027 | [0.013, 0.041] | **[−0.008, 0.062]** | — |
| RSC − control, partial class | s3_conf_full | +0.384 | [0.325, 0.444] | [0.056, 0.513] (4 cells; crude) | — |
| P1 D − A | test union | +0.306 | [0.276, 0.337] | [0.203, 0.396] (116) | — |
| P1 C − A | test union | +0.259 | [0.231, 0.285] | [0.169, 0.343] | — |
| P1 D − C (exploratory) | test union | +0.047 | [0.029, 0.066] | [0.024, 0.071] | — |
| P1 A − M (exploratory) | test union | +0.034 | [0.021, 0.046] | [0.012, 0.054] | — |

Reading:
1. With five control seeds, **the full-episode RSC − control interval now excludes 0 under cell clustering.** The red team's 3-seed objection is answered on the mean difference.
2. The registered seed-level statistic still fails for the partial class (S3-A2 H4′p, Holm 0.111). The Fisher test on runs that learned partial repair gives **5/5 vs 2/5, one-sided p = 0.083** (`fisher`). That is exactly the red team's pre-stated power calculation: "5 of 5 vs 2 of 5 gives 0.083".
3. **RSC vs ACK shows no cell-robust difference** (both intervals include 0). Learners: 5/5 vs 4/5, Fisher p = 0.5. This agrees with registered S3-A4 "inconclusive". The scripted read in the recovery start is not shown to matter.
4. Every P1 headline survives cell clustering.

## 5. Reviewer 5: judge's view, and the review's top-5 problems (§5) as of today

| Problem (red team §5) | Status 2026-11-25 |
|---|---|
| 1. A12 headline close to true by construction | **Unchanged.** ID1 then showed that from a non-reading warm start R_V never samples a read (0/164 sampled, training read rate 0.00 per block; `results/id1/id1_summary.json`). That supports "selection from the repertoire" and also confirms the reviewer's mechanism: no sampled read, no gradient. Zero-variance share 0.98 (`train_logs.id1_rv_s*`) |
| 2. S3 start-state decomposition unresolved | **Partly resolved:** H3′ supported with 5 seeds (Holm 0.036); full-episode mean difference now cell-robust; partial class still not supported at seed level; ACK arm inconclusive. SELF starts (the reviewer's design) were never run. Addendum 3 was withdrawn as the reviewer advised |
| 3. Construct validity (one simulator, free reads, no prompted baseline) | Unchanged except the costed-read testbed (AD) |
| 4. A16 and FCA method sections | A16 wording narrowed. FCA: v3 drops it; v1 appendix still wrong (R1.6) |
| 5. Wrong public build | Resolved |

## 6. Reviewer wording errors (red team §7), current state

| # | Where | Fixed? |
|---|---|---|
| 1, 2 | wf4 and renamed-API sentences (arm means stated as per-model) | Not present in README, write-up or paper v3. `docs/research/FINAL_REPORT.md` l. 52–53 still say "13–35% of episodes" (arm means). Historical Stage 4 report; low priority |
| 3 | FCA appendix derivation | **Not fixed in `paper/appendix.tex` l. 79** (v1, superseded) |
| 4 | A16 abstract item | v3 fine; v1 l. 77 leads with AUROC (superseded) |
| 5 | Addendum 3 | Withdrawn (`docs/research/ADDENDUM3_WITHDRAWN.md`) |
| 6 | Stage 1 site | Resolved |

## 7. New issues found by this verification (not in any review)

1. **P1's C and D are not nested.** C is not D's warm start, so 0.772 → 0.819 is not GRPO's contribution; D's own warm start (`p1_dwarm`) was never evaluated (`P1_DECOMPOSITION.md` §1, §4). README l. 57 and WRITEUP §21 item 8 list the two numbers together without this note.
2. **ID1's null is mechanical.** No read was ever sampled (0/164 at T = 1), so R_V had nothing to reinforce. "R_V does not create read-back" (README l. 15, WRITEUP l. 23, 51) is the registered reading, but its scope is "not discovered within 200 steps × 32 rollouts at T = 1 from this warm start" (`PREREGISTRATION_ID1.md` §scope already limits it to one base model and one demonstration set).
3. **S3 registered statistic vs the write-up table.** WRITEUP §6 (registered-hypotheses table), row S3, says "Supported: +0.21 [0.17, 0.25], 5/5, seed permutation p = 0.004". The registered rule was:
   - ≥ 4/5 seeds positive and the interval excluding 0;
   - exact sign-flip one-sided p 1/32 (`results/s3/s3_summary.json`).

   The p = 0.004 unpaired permutation is the post-hoc review statistic (claim S3-R). Label it.
4. **Seed degeneracy under greedy decoding.** Identical greedy scores across seeds are common:
   - AD GRPO 0.687 in all 5 seeds (c1);
   - `ad_grpo_s0` = `s1` on cw_dev;
   - `s3_rv_mix_s0` = `s1` on `proc_dev`;
   - three pilot arms identical on `proc_dev` (`results/s3/dev`).

   Seed counts overstate independent evidence when the RL update barely moves the greedy policy. That is likely where zero-variance shares are high. The sampled evaluations (E1, P1–P3) are less affected.
5. **J2 has no ledger claim of its own** (MASTER_EXPERIMENT_LEDGER, "Gap"). The Aya 32B baseline accepts 0.573 of false completions even with the database snapshot (`results/j2/j2_summary.json` `main_vs_verifier.transcript_state.aya.false_acceptance`), against ≤ 0.013 for the four J1 judges. This is the strongest judge-side number in the repository, and it is not in the ledger.

WHAT WE LEARNED → every quantitative red-team criticism that could be recomputed reproduces (clip 0, +2.47, zero-variance shares, FCA weights, cell intervals). Two were answered by later registered work: unscaled advantages (AD-C1) and SFT on oracle recoveries (P1). One reverses with more seeds: the full-episode RSC − control mean is now cell-robust.
WHAT THIS RULES OUT → describing the optimizer as clipped PPO; describing FCA's negative as a test of counterfactual credit in general; claiming that the scripted read in recovery starts matters (RSC − ACK not cell-robust).
WHAT REMAINS UNCERTAIN → A12's observation-vs-incentive identification (no dissociation run); the start-state mechanism (SELF starts never run); any prompted or larger-model baseline on ProcureSim.
NEXT EXPERIMENT → none on GPU before Nov 28 from this list. The cheapest open item is the `p1_dwarm` evaluation (`P1_DECOMPOSITION.md` §5).
