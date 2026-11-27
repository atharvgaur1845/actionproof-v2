> **Superseded (2026-11-27, Stage 2 sprint).** This file is kept as history (last updated 2026-11-25) and is not current. The current source is `docs/submission/FINAL_STORY_DECISION.md`. See `research_audit/STALE_SURFACES_AUDIT.md`.

# Final recommendations

2026-11-25. Branch `submission-candidate`. Deadline Nov 28, 2026, 23:59 IST (confirmed by the author).

The answers are grounded in the full branch history (`STUDY_HISTORY.md`, `REVISED_RESEARCH_STRATEGY.md`), not `main` alone.

The three late inputs (P3, judge_next and the `p1_dwarm` evaluation) were filled from their result files on Nov 25 evening.

## Decision

**SUBMIT AFTER LIMITED CORRECTIONS, and DO NOT CLAIM THE ALGORITHM CONTRIBUTION.**

**What the submission can rest on.** The research story is sound once narrowed (`REVISED_RESEARCH_STRATEGY.md` §8):
- every headline number reproduces from stored episodes by a second pathway;
- every registration's freeze check passes;
- the failed hypotheses are reported next to the supported ones.

**What it must not claim:** no proposed algorithm passed its registered test against the right baseline.
- AWRB fails against the no-std baseline (−0.088).
- FCA, C5, G and D2 were dropped at development.
- CDVO (LAB1), SR1 and S3-A4 were not supported.
- D3 passed its gates, but its mechanism prediction failed.

The one confirmed training-method result is removing the std from GRPO's advantage. That is a known fix (Dr. GRPO's advantage), which the theory predicted before AD-C1 was run. It is reported as a theory confirmation, not as a new algorithm.

**The limited corrections:**
- **Already applied:** commits 7d06fe8 and ff31128, and the site template in 63d194e. Each is listed in §8 below.
- **Still to do:** the author-only items in `SUBMISSION_CHECKLIST.md` (the three late results are now filled).

## 1. What is the strongest claim ActionProof can defend?

**The claim.** In a simulated procurement environment where tool acknowledgements can be wrong, verified recovery demonstrations repair partial and duplicate payments. They lift the frozen 450-task test from 0.513 to 0.772 (recovery SFT, +0.259) and to 0.819 (recovery SFT then GRPO, +0.306). Both gains are registered, pass Holm correction, are positive in every seed, and survive cell-clustered bootstraps.

**What it is not:** the reward and an inference-time hint do not produce repair (P1-H1 −0.030).

**Scope:**
- The two arms are not nested, so 0.772 → 0.819 is not what GRPO adds.
- That increment was measured descriptively (`NOTE_DWARM_EVAL.md`, read rule fixed before output; `results/audit/dwarm_2x2.json`, exploratory):
  - Dw (recovery SFT, no RL) scores 0.679.
  - **SFT alone adds +0.344** [+0.311, +0.378] over the warm start (0.335).
  - **GRPO then adds +0.140** [+0.121, +0.160] on top of SFT. On the plain warm start it adds +0.178.
  - Interaction −0.037 [−0.065, −0.011]: GRPO adds slightly less after SFT.
  - **Reading:** most of the gain comes from demonstrations, but GRPO's increment is real and positive in both rows.

**Second strongest: a proved result with a registered confirmation.**
- When a group shares one hidden state, std-normalised GRPO is stake-blind: per-state positive rescaling of the return leaves its expected update unchanged (TH1).
- Removing the std beat GRPO by +0.120 [+0.088, +0.153] in 5/5 seeds, as predicted (AD-C1).

## 2. What is the strongest algorithmic contribution?

**There is no new algorithm.** The strongest method-level contribution is an analysis plus a registered test.

**The analysis.** Under shared hidden state the GRPO group key is effectively (prompt, hidden state). Std normalisation then removes stake and cost information that centred estimators (Dr. GRPO, RLOO) keep.

**The registered test.** AD-C1 confirmed the predicted consequence.

**Name the optimiser exactly** (`OPTIMIZER_AUDIT.md`):
- one update per batch of 32 rollouts, with `clip_ratio` 0 at every logged step of 120 runs;
- a group-baselined, std-scaled policy gradient;
- DAPO-style token aggregation, a KL penalty and a vLLM mismatch weight;
- the no-std arm is Dr. GRPO's advantage only, not full Dr. GRPO.

The recovery-start curriculum (RS-GRPO) is a start-state curriculum applied to GRPO, which belongs to reverse-curriculum work. Describe it as that, not as an optimiser.

## 3. Does the proposed new algorithm actually outperform RS-GRPO?

**No proposed algorithm does.**

| Method | Comparison | Result |
|---|---|---|
| FCA, C5, G, D2 | development kill criteria | dropped |
| AWRB | against no-std GRPO | −0.088, not supported |
| CDVO (LAB1), SR1 | registered | not supported |
| P2 ledger arm E | against D (H1) | −0.038 |
| P2 guard | — | −0.142 to −0.273 |

The one registered test of a std change inside the practical recipe is **P3** (Dn vs D, `PREREGISTRATION_P3.md`, H1). It is not a new algorithm: it applies the AD-C1 fix to D. **P3 is not supported:** Dn 0.813 vs D 0.819, −0.007 [−0.013, −0.001], one-sided p 0.988, seeds −0.009 / +0.005 / −0.016 (`results/p3/p3_summary.json`). The confirmed AD-C1 fix does not transfer to the practical recipe. One reading, made after the result, is that reads are free in ProcureSim, which is outside the theory's cost-sensitive regime.

## 4. Does the LLM judge reliably reject false completion?

**No, not as a verifier.**

**Without the database ledger,** Aya Expanse 32B accepts most false completions.
- **AuditKIT prompt:** 71.5% accepted with the ledger present, 96.0% without it.
- **Custom prompt:** 46.3% accepted with the ledger.
- **Database snapshot:** 0.573 accepted.

**Four frontier-family judges (J1) with the database snapshot** reach 88.6–98.9% accuracy, and accept at most 0.013 of false completions. Their behaviour still depends on wording: removing one rubric sentence ("ok is not proof") makes three of the four accept unverifiable claims.

**On the practical agent's own transcripts (P2 committee),** Qwen accepts up to 0.284 of failures.

Every judge figure is judge-versus-verifier. No human labels exist.

**Judge study** (`JUDGE_RELIABILITY.md`, descriptive, 497 cases):
- **Best LLM configuration** (32B, structured prompt, verifier facts): FAR 0.162, FRR 0.318.
- **J1 transcript baseline:** FAR 0.428.
- **Effects measured separately:** state evidence +0.046 to +0.080 accuracy; structured prompt +0.062.
- **Aya 8B accepts about 90% of failures in every configuration.**
- **Zero false acceptance** comes only when code gates on the verifier's facts.

Answer: no LLM judge is verifier-grade. The verifier must own commit facts.

## 5. Does the benchmark have meaningful OOD generalisation?

**Partly.**

**Holds (S4-EVAL, P1):**
- Hindi and Hinglish task text;
- two-fault episodes;
- unseen payment fractions: D repairs 0.931 of partial payments on the P1 test with unseen fractions.

**Does not hold:**

| Shift | Result |
|---|---|
| Renamed tools | trained agents still call training-time names in 13–35% of episodes (per-arm means) |
| Unseen four-step workflow | every model replays the three-step sequence; the goods receipt is left undone in 84–96% (per-arm means) |
| Sibling domain (A17 healthcare administration) | negative transfer |

**Benchmark gaps** (`DATASET_EXPANSION.md`):
- no fully unseen fault class;
- no agent-side adversarial split;
- one unused split (`c1_pp`) overlaps the AD training pool by 75 prompts. It was never used.

## 6. Does RL improve externally relevant agent capability?

**No improvement is shown. Retention is shown.**
- **IFEval and BFCL (R1):** within ±0.03 of the warm start. State reward 0.000 / −0.002; recovery start −0.004 / +0.001.
- **τ-bench retail:** at the floor for a 3B model (base pass^1 0.058), so it supports no claim either way.
- **Benchmarks not run:** ToolSandbox, AppWorld and BFCL multi-turn (`EXTERNAL_BENCHMARKS.md` §3).

## 7. Which negative results materially change the research narrative?

1. **ID1.** From a warm start that never reads, the state reward never samples a read (0/164), so it cannot reinforce one. Change "the reward teaches verification" to "it selects from the repertoire".
2. **A12 H3 failed, and the acknowledgement reward is saturated.** The A12 contrast is close to true by construction. Observation and incentive are not separated.
3. **The S3 partial class is not supported at seed level,** and the duplicate-delivery gain comes from fault coverage. Narrow the recovery-start claim to recovery-start evaluation.
4. **P1-H1, P2-H3/H4 and the guard fail.** Inference hints and guards do not repair. The guard's rule R1 blocks the taught repair in 90/90 partial-payment tasks.
5. **AWRB −0.088 against no-std.** This removes the algorithm claim.
6. **A16 activation patching met its rule in 0/5 seeds,** and the ablation is not specific to R_V. Make no mechanism or circuit claims.
7. **S4 tool and workflow shifts fail, and A17 transfer is negative.** Make no generalisation claim beyond language and fault combinations.

## 8. What should be removed from the paper?

Each item is either removed or already corrected.

- **Any new-algorithm framing:**
  - AWRB as a contribution;
  - FCA (paper v1 `main.tex` l. 109 and `appendix.tex` l. 79 are wrong; v1 is superseded; v3 has no FCA text);
  - CDVO.
- **"Reward observability teaches/creates verification":** narrowed in WRITEUP §21 item 1 and the ID1 scope (ff31128).
- **Mechanism, circuit or causal-use language for A15/A16:** only decodability is established.
- **"Clipped PPO/GRPO" or unqualified "Dr. GRPO" for the trained arms:** corrected (7d06fe8).
- **Reporting S3's post-hoc p = 0.004 as the registered statistic:** relabelled (ff31128).
- **Reading P1's C → D gap as GRPO's contribution:** corrected in the README, write-up and site (ff31128, 63d194e).
- **Any use of `c1_pp` as held-out data:** removed from the write-up.
- **Any judge accuracy presented without the false-acceptance rate on the failure class, or presented as human agreement.**

## 9. What remains scientifically novel?

- **ProcureSim's fault model and deterministic verifier.** Misleading acknowledgements, partial commits, duplicate delivery, lost acknowledgements and never-commit tasks, judged by state rather than by replies. With these come the separation of acknowledgement, state, ledger, verifier and LLM-judge evidence.
- **The controlled decomposition of where recovery comes from:**
  - demonstrations, not reward;
  - not inference-time hints or guards;
  - recovery starts give a small registered effect on recovery-start evaluation.
- **The shared-hidden-state analysis of GRPO.** A stake-blindness result specific to agentic RL, where the group shares one hidden fault, with a registered pre-predicted confirmation (TH1, AD-C1).
- **Judge-reliability findings:**
  - single-sentence prompt sensitivity;
  - false acceptance despite database evidence;
  - judge-family dependence.

  judge_next adds that state evidence and prompt structure each help, measured separately, but neither reaches verifier grade.

## 10. What exactly should be frozen and submitted?

1. **Code and documents:** branch `submission-candidate` at the freeze commit, tagged `submission-2026-11-28`. That covers code, registrations, `results/`, `claims/claims.yaml` and `research_audit/`.
2. **Paper:** `paper/main_v3.pdf` (7 pages, anonymised), built from `main_v3.tex` and `appendix_v3.tex`.
3. **Write-up:** `docs/submission/WRITEUP.md`, with `python scripts/s2_numcheck.py` at 0 failures on the freeze commit.
4. **Site:** the HF Space rebuilt from `demo/site_template.html` after the last results. Record the Space commit hash in the checklist.
5. **Integrity checks, run on the freeze commit and logged in `SUBMISSION_CHECKLIST.md`:**
   - freeze checks: `freeze_p1/p2/p3 --check`, `scripts/ad/freeze_c1.py --check`;
   - `python -m actionproof.reporting.freshness`.
6. **Author decisions, not made by the agent:**
   - fast-forwarding `main`;
   - publishing adapters and traces;
   - the §30/§31 declarations;
   - title approval.

## Fills

All three filled on 2026-11-25 from `results/p3/p3_summary.json`, `results/judge_next/report.md` and `results/audit/dwarm_2x2.json`.
