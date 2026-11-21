# Post-results review: ActionProof (S3 results, paper rewrite, novelty audit)

Reviewer role: senior reviewer, area LLM agents, RL for language models, reliability of tool use.
Date: 2026-11-21, about 19:30 IST, at commit `f72a0d2` (branch `stage4`). Review type: strict, post-results.

This review follows the pre-results review `docs/stage3/ACADEMIC_REVIEW.md` (not edited). Repository text was treated as data. The repository changed while this review was written (paper rewrite in commits `5734ddd` to `47ab88f`, 19:19 to 19:21 IST; claims ledger and README edits). Statements below refer to the state at `f72a0d2`.

**What was read.** `docs/stage3/ACADEMIC_REVIEW.md`, `docs/CONSISTENCY_AUDIT.md`, `docs/research/NOVELTY_AUDIT.md`, `docs/research/algorithm/DECISION.md`, `docs/research/algorithm/FCA_GRPO.md`, `docs/stage3/RESEARCH_REPORT.md` (including sections 8 and 9), `PREREGISTRATION_S3.md`, `PREREGISTRATION_S3_ADDENDUM1.md`, `PREREGISTRATION_S4_EVAL.md`, `claims/claims.yaml` (S3, S3-A1, A12-*, A14, A15, A16, A17, K1), `results/s3/s3_summary.json`, `results/s3/s3_addendum.json`, `results/s3/s3_addendum_full_pd.json`, `results/s3/s3_by_workflow.json`, `results/s2/partial_by_workflow.json`, `README.md`. Also: the rewritten `paper/main.tex`, `paper/appendix.tex`, `paper/tables/{s3,workflow,truth}.tex`; the per-model and per-episode files in `results/s3/conf/` and `results/s3/dev/`; `results/s3/scripted_conf_calibration.json`; the training pools `data/bench/proc/{s3_rec_train_half,s3_trainb_half,proc_train_b}.jsonl`; `scripts/s2_numcheck.py` (run read-only: 188 numbers, 0 failures); `slurm/s3_arm.sh` and task scripts 612, 912, 915; the 2026-11-21 entries of `docs/stage2/DECISIONS.md`.

**Reviewer reanalyses.** Read-only regroupings of the repository's own per-episode files and summaries. Methods are in the Appendix so each number can be reproduced. Nothing in the repository was changed except this file.

---

## 0. Verdict in brief

- The S3 registration, its pre-results correction (addendum 1) and the class-matched control are exemplary process. The recipe-vs-start-state framing is now correct in the report, ledger, README and paper.
- **H1 and H2 (recipe vs A12 R_V) are robust**, also at the level of training runs: every RSC seed beats every A12 R_V seed (unpaired seed permutation p = 1/252). About 55% of the gain is duplicate delivery, a class A12 never trained on.
- **H3 and H4 (start state vs class-matched control) pass the registered rule but are weak as science.** Three seeds; seed-level p = 0.05 and 0.10; the effect is entirely on partial payments (none on duplicates, where the control is equal or better); and it rests on two of three control seeds failing to learn partial repair. Control seed 0 and RSC seed 0 make identical tool calls on 283 of 300 full-episode tasks.
- **The workflow failure is not fixed.** Partial payments in the receive-then-pay workflow are repaired in 5 of 60 RSC full episodes (0 in four of five seeds) and 0 of 96 in the other arms. The RSC training pool contains 6 such tasks out of 400.
- **Transfer is negative, not null.** In care_admin, RSC is below A12 R_V on VSC in 5 of 5 seed pairs (0.047 vs 0.080; warm start 0.158) and above it on collateral writes in 5 of 5.
- Novelty: after the novelty audit, no method is new. The defensible contribution is the setting plus registered measurements. The paper's rewrite says this.
- Scores: Soundness 3, Excitement 2.5, Overall 2.5, Confidence 4. A workshop fits now; Findings is reachable after the fixes in Section 4.

---

## 1. What changed since the pre-results review

### 1.1 Checklist items (pre-results review, Section 6)

| # | Item | Status | Evidence | What remains |
|---|---|---|---|---|
| 1 | Stratify evidence-use results by workflow | **Partial** | `results/s2/partial_by_workflow.json` (A15, A12 partial class); `results/s3/s3_by_workflow.json`; paper section 5 (main.tex line 73) and Table 3 (`tables/workflow.tex`); README line 15 | No A14 per-workflow table; A16 margins not stratified; no workflow- or skeleton-clustered intervals; ledger A15 still says "generalises, not pattern-specific" (claims.yaml, entry A15); paper section 5 omits the order-then-pay stratum (R_V 0.05 to 0.55 in A15 seeds 0 to 2) and the abstract says "only when payment is the first step", which seed 1 contradicts |
| 2 | Identification story for A12 (W1) | **Partial** | Paper section 4 (line 62): "this contrast is between a state-trained agent and one that training did not move"; Limitations (line 114); ledger A12-IDENTITY; README line 14 | No identification experiment (items 6, 7). The Discussion (line 110) still states the general causal claim ("The information a reward observes decides whether an agent checks") |
| 3 | S3 addendum before reading results | **Addressed** | `PREREGISTRATION_S3_ADDENDUM1.md` committed 12:46 (`1eb2bf8`); results committed 16:25 (`c62ee1b`) | Control has 3 seeds, not the 4-of-5 I proposed; per-class decomposition registered but not reported (Section 2.3 below does it) |
| 4 | Integrate A15, A16, A17, K1 with corrected wording | **Mostly addressed in the paper** | main.tex sections 4 to 6, 8; A16 non-specificity and empty-record collateral now stated (line 77) | K1 "stopped declining" (line 62) is wrong; A17 decomposition missing (line 106); "No model refunds a duplicate delivery" (line 73) contradicts section 7 |
| 5 | Anonymise; current template | **Open** | Author block main.tex lines 15 to 17; GitHub URL appendix line 85; "hackathon's toolkits" line 87; "Track 3 metrics" appendix line 21; `emnlp2023` style line 3 | All |
| 6 | Warm-start repertoire experiment | **Open** | No run, registration or result found | All |
| 7 | Observability or gradient control | **Open** | Same | All |
| 8 | Statistics | **Partial** | Holm-adjusted 0.0625 now in main text (line 62); S3 report section 9 and ledger S3-A1 state 3 seeds, 1/8 resolution, conditional task bootstraps | No seed-level analysis for S3; new error in `PREREGISTRATION_S4_EVAL.md` (Section 4, item 5) |
| 9 | A16 | **Mostly addressed in the paper** | Line 77 | No greedy behavioural readout; ledger A16 statement unchanged ("grows with training", "causal rule met", no R_A outcome); README line 16 |
| 10 | A17 | **Partial** | Ledger A17 includes R_V vs warm +0.15 [-0.06, 0.39] | Paper line 106 and README line 17 still read as positive transfer |
| 11 | Construct-validity section | **Partial** | Limitations (line 114) | Fault prevalence, byte-identical acknowledgements, free and unconditional reading, greedy-only evaluation, one warm start, verifier provenance |
| 12 | Related work | **Addressed in docs, partial in paper** | `NOVELTY_AUDIT.md` (84 verified entries); refs.bib +13 entries (`4924b21`); paper section 2 | Paper no longer cites tau-bench, LIMBO, proxy-state rewards (`chuang2026proxystate`), Yue et al., Thaman, NoisyAgent, R3L, failure-prefix conditioning; several are already in refs.bib |
| 13 | Sampled evaluation | **Open** | `results/eval/a12/init__neutral__proc_train_T1x4.jsonl` is untracked and on the training split | pass@k on `proc_val` |
| 14 | Verify `smyth2026overclaim` | **Addressed** | CONSISTENCY_AUDIT item 16 | |
| 15 | A14 oracle `goal_met` 0.0 | **Open** | Not found in any document | |
| 16 | "Turn cost" in truth table | **Addressed** | `tables/truth.tex` has no cost term | New caption is inaccurate (Section 6, item 15) |
| 17 | Figure paths | **Moot** | The rewrite has no figures | |
| 18 | Ledger wording K1, A16, A17 | **Open** (K1, A16), **partial** (A17) | claims.yaml entries K1, A16 unchanged | |
| 19 | S3 selection checkpoint; Stage B disclosure | **Partial** | Stage B disclosed (PREREGISTRATION_S3 section 1; appendix line 59) | Which A12 R_V checkpoint the 0.05 constraint used is still unstated |
| 20 | Rename `track3.tex` | **Open** | Appendix section "Track 3 metrics" | |
| 21 | Generated tokens per run | **Open** | `results/run_manifest.json` has steps and rollouts, no tokens | |

### 1.2 The four items required before reading S3 results

| Item | Status | Evidence |
|---|---|---|
| (a) Correct "the only change is the start-state distribution" | Addressed | ADDENDUM1 section 1; RESEARCH_REPORT section 9; ledger S3 ("compares training RECIPES") |
| (b) Pre-specify per-class and per-workflow decompositions | Registered; per-workflow reported; per-class not reported | ADDENDUM1 section 3; `s3_by_workflow.json` |
| (c) Name the matched-class baseline and its rule | Addressed with 3 seeds and a 3-of-3 rule | ADDENDUM1 sections 2 and 3 |
| (d) Seeds 2 to 4 sensitivity | Registered; reported only as "agree" (DECISIONS 16:30) without numbers or a file | Computed here: H1 per-seed mean +0.163 (seeds 0 to 1) vs +0.234 (seeds 2 to 4); H2 +0.158 vs +0.243. The seeds trained after registration show larger gains, so there is no sign of selection inflation. The larger gains come from weaker A12 R_V seeds 3 and 4 |

### 1.3 New since the pre-results review

Positive:
- `docs/CONSISTENCY_AUDIT.md`: 18 dated corrections, including two disclosures a reviewer would otherwise have found: the training environment seed is always 0 (item 17), and the loss is TRL's DAPO-style token aggregation (item 18).
- `docs/research/NOVELTY_AUDIT.md` and `DECISION.md`: an honest conclusion that no candidate is novel as a mechanism.
- The rewritten paper: a tempered title, an explicit non-contributions sentence (line 35), the RSC family acknowledged as prior art (line 41), Holm in the main text, A16 non-specificity, an LLM-use statement and the H4 appendix filled.
- `PREREGISTRATION_S4_EVAL.md`: Hindi, Hinglish, paraphrase, API-surface, four-step-workflow and fault-combination shifts. This is the first real language dimension in the project.

New problems:
- `PREREGISTRATION_S4_EVAL.md` section 3 says G1 to G4 "can all survive only if each is 5 of 5". That is arithmetically wrong. With five seeds the smallest one-sided sign-flip p is 1/32 = 0.031. Holm's first step over four tests compares it with 0.05/4 = 0.0125, so **no G can be rejected at 0.05 even at 5 of 5** (the smallest adjusted p is 4/32 = 0.125). Fix it by addendum before any S4 output is read (Section 4, item 5).
- The H4 interval is written [0.16, 0.29] in RESEARCH_REPORT line 163, ledger S3-A1 and DECISIONS 16:30. The file value is [0.1545, 0.2879], which rounds to [0.15, 0.29]. The generated `tables/s3.tex` has it right. The ledger does not check this bound, so numcheck cannot catch it.
- The numcheck allow-list contains "0.21" (a package version). That equals the headline S3 effect, so an unlisted "0.21" passes the checker silently. CI bounds are mostly unchecked.
- The paper has a "[Pending]" section (line 104), and contribution (iv) (line 35) has no results.

---

## 2. Assessment of the S3 results as registered

### 2.1 Framing

The corrected framing is right: RSC vs A12 R_V (H1, H2) compares recipes, and RSC vs the class-matched control (H3, H4) compares start states. Two residual issues:
- Several sentences still make the curriculum the grammatical subject of the recipe effect. Examples: README line 22, "A recovery-start curriculum for unchanged GRPO raises VSC by +0.21", and the same in abstract line 26. The subject should be the recipe.
- **The recipe comparison also changes the code path.** A12 R_V was trained through `slurm/a12_arm.sh`, with the Stage 2 trainer and rollout. RSC and the control were trained through `actionproof.stage3.train` (`slurm/s3_arm.sh`), with the Stage 3 rollout and the ratio environment. PREREGISTRATION_S3 section 2 says the trainer is "unchanged", but no equivalence test is cited. The control shares RSC's path, which is one more reason H3 and H4 are the cleaner contrasts.

### 2.2 H1 and H2 (RSC vs A12 R_V, 5 seeds)

All values were checked against `results/s3/s3_summary.json`.

| Split | RSC per seed | A12 R_V per seed | Difference per seed | Seed-averaged [95% task CI] |
|---|---|---|---|---|
| `s3_conf_rec` (H1) | 0.830, 0.830, 0.813, 0.830, 0.803 | 0.643, 0.690, 0.633, 0.567, 0.543 | +0.187, +0.140, +0.180, +0.263, +0.260 | +0.206 [0.168, 0.247] |
| `s3_conf_full` (H2) | 0.830, 0.820, 0.840, 0.840, 0.833 | 0.643, 0.690, 0.637, 0.577, 0.570 | +0.187, +0.130, +0.203, +0.263, +0.263 | +0.209 [0.171, 0.249] |

- **Registered verdicts.** H1 is supported (5 of 5, interval excludes 0). H2 is supported and is also superior (lower bound 0.171, far above -0.05).
- **Seed-level robustness, added here.** Pairing by seed index carries no shared randomness across different pools, so I also ran an unpaired permutation test over seeds. Every RSC seed exceeds every A12 R_V seed on both splits (p = 1/252 = 0.004 one-sided). A seed-level t interval on the per-seed differences gives [0.14, 0.27] (H1) and [0.14, 0.28] (H2). **H1 and H2 survive inference across training runs.** This is the strongest S3 result and should be stated this way in the paper.
- **Spread across seeds.** The between-seed SD of VSC is 0.012 and 0.008 for RSC, against 0.060 and 0.050 for A12 R_V. Caveat: seeds are highly redundant. On `s3_conf_full`, pairs of RSC seeds make identical tool-call sequences on 263 to 291 of 300 tasks (A12 R_V pairs: 258 to 295).
- Calls per episode: 7.89 (RSC) vs 7.27 (R_V) on full episodes, so the report's "+0.6" is correct.
- A12 R_A seeds 0 to 2 equal the warm start on every S3 split (rec 0.473, full 0.363, stress 0.500). The A12 identity finding replicates on fresh splits.

### 2.3 Decomposition by fault class (pre-specified in ADDENDUM1 section 3; not reported until now)

From `by_class` in `results/s3/conf/*.summary.json`. Each entry is the class's contribution to the VSC difference in units of the whole 300-task split.

| Contrast | Split | Duplicate delivery | Partial | All other classes |
|---|---|---|---|---|
| RSC - A12 R_V (5 seeds) | rec | +0.112 | +0.093 | +0.001 |
| RSC - A12 R_V (5 seeds) | full | +0.116 | +0.093 | 0.000 |
| control - A12 R_V (seeds 0 to 2) | rec | +0.114 | -0.031 | 0.000 |
| control - A12 R_V (seeds 0 to 2) | full | +0.122 | -0.030 | 0.000 |
| RSC - control (seeds 0 to 2) | rec | -0.006 | +0.090 | +0.001 |
| RSC - control (seeds 0 to 2) | full | -0.011 | +0.092 | 0.000 |

- **About 55% of H1 and H2 is duplicate delivery.** A12 R_V never trained on this class and solves 0 of 46 (full) and 0 of 39 (rec) in every seed.
- **Every other class is identical across trained arms.** Clean, phantom, lost acknowledgement and timeout-before-commit are at 1.00. Never-commit is 0.00 (unreachable), with honest failure 1.00 in every trained model except A12 R_V seed 3 (0.89 to 0.96) and RSC seed 1 on ack starts (0.95).
- The report's "two roughly equal parts" (section 8, R_V 0.655, control 0.739, RSC 0.824 on rec, seeds 0 to 2) is arithmetically right but hides the structure. **Coverage explains duplicate delivery; start states explain partial payments.** The coverage part also contains a small loss on partial payments (-0.03).

### 2.4 H3 and H4 (RSC vs class-matched control, 3 seeds)

All values were checked against `s3_addendum.json` and `s3_addendum_full_pd.json`.

| Test | RSC per seed | Control per seed | Difference per seed | Seed-averaged [95% task CI] | Seed-level (added here) |
|---|---|---|---|---|---|
| H3, `s3_conf_rec` (300) | 0.830, 0.830, 0.813 | 0.810, 0.737, 0.670 | +0.020, +0.093, +0.143 | +0.086 [0.060, 0.111] | unpaired p = 1/20 = 0.05; t interval [-0.07, 0.24] |
| H4, full partial + duplicate (110) | 0.764, 0.736, 0.791 | 0.755, 0.400, 0.473 | +0.009, +0.336, +0.318 | +0.221 [0.155, 0.288] | unpaired p = 2/20 = 0.10; t interval [-0.24, 0.68] |
| Full, all classes (300) | 0.830, 0.820, 0.840 | 0.827, 0.697, 0.723 | +0.003, +0.123, +0.117 | +0.081 [0.054, 0.110] | p = 0.10 |

Assessment:
1. **The registered rule is met, so "H3 and H4 supported" is a correct statement of the registered test.** The rule is weak, however: a 3-of-3 sign pattern has p = 1/8, and the task-bootstrap interval conditions on three trained models. With the smallest possible seed-level p at 0.05 (3 vs 3), the data cannot establish a start-state effect across training runs.
2. **The effect is entirely on partial payments.** On duplicate delivery the control is equal or better, including on duplicate recovery starts, which are in-distribution for RSC only (duplicate "read" starts: control 0.84, 0.84, 0.95; RSC 0.79, 0.74, 0.84). "Start states help recovery" is false for duplicates.
3. **The partial effect is a seed-reliability effect.** Partial-class VSC on full episodes (n = 64):
   - A12 R_V: 0.44, 0.66, 0.41, 0.13, 0.09;
   - control: 0.80, 0.16, 0.13;
   - RSC: 0.78, 0.81, 0.78, 0.88, 0.66.

   Partial repair is learned (VSC of 0.40 or more) in 5 of 5 RSC runs, against 4 of 8 full-episode runs (3 of 5 A12 R_V, 1 of 3 control). Permutation p on partial VSC, RSC (5) vs control (3): 4/56 = 0.07. Control seed 0 learned partial repair as well as RSC, and the two policies make identical tool calls on 283 of 300 full-episode tasks. **The finding is "recovery starts made partial-payment repair reliable across training runs", and with 3 control seeds that is suggestive, not established.**
4. **Mechanism not measured.** The paper (line 81) says "GRPO rarely samples the repair from a full episode", and the report (section 2) argues zero-variance groups. That is the natural hypothesis, but S3 never measured it. `slurm/s3_arm.sh` writes rollout logs (`results/s3/rollouts/<tag>.jsonl.gz`), so the share of partial-fault groups with non-zero reward variance can be computed per arm without a GPU (A12-SIGNAL already does this for A12).

### 2.5 The violation trade-off vs the matched control

- **Every violation in every S3 ProcureSim split is a duplicate payment left in place, and every one is in the duplicate-delivery class.** `violation` equals `duplicate` in every per-seed and seed-averaged number. No arm ever makes a collateral or wrong-amount write.
- Against the control, the size is about 3 more unrefunded duplicates per 110 tasks per seed (-2, +5, +6). Seed-averaged it is +0.027 [0.003, 0.052] on the subset and +0.010 [0.001, 0.019] on all full episodes. On recovery starts it is +0.007 [-0.003, 0.018], not established. On stress it is +0.040 [0.013, 0.070] (per seed -2, +4, +10 of 100).
- Location: mostly duplicates in the pay-first workflow (control 29 of 51, RSC 20 of 51, seeds 0 to 2 pooled).
- **Interpretation.** This is the same fact as the duplicate-class deficit in Section 2.3, not a separate safety cost of starting at recovery states. Calling it "irreversible violations" (abstract line 26) is wrong: ProcureSim lets a duplicate be refunded, and the arms differ precisely in whether they refund it. Use "duplicates left unrefunded at episode end".
- Against A12 R_V, the -0.12 violation change (-0.118 [-0.154, -0.085]) is entirely duplicate coverage.

### 2.6 Stress

- `s3_conf_stress` is 50 duplicate-delivery tasks plus 50 lagging-read tasks.
- A12 R_V scores VSC 0.500 in every seed, with exactly 7.34 calls per episode in every seed. That VSC equals the warm start's, R_A's and the never-read script's.
- RSC scores 0.85 to 0.96 and the control 0.85 to 0.99. RSC minus control is -0.040 [-0.070, -0.013] (permutation p = 0.85 for RSC above the control).
- For RSC and the control, the duplicate half is a trained class, so this split is a coverage test, not a stress test. No arm fails the lagging-read half.
- The authors' attribution to coverage is correct. The paper should also stop calling it a stress result for these two arms.

### 2.7 Transfer (OpsSim)

| care_admin (120 tasks) | VSC | Collateral |
|---|---|---|
| warm start | 0.158 | 0.183 |
| A12 R_A s0 to s2 | 0.175, 0.133, 0.117 | 0.125, 0.117, 0.133 |
| A12 R_V s0 to s4 | 0.058, 0.075, 0.083, 0.075, 0.108 (mean 0.080) | 0.300, 0.267, 0.267, 0.267, 0.258 (mean 0.272) |
| RSC s0 to s4 | 0.042, 0.033, 0.050, 0.050, 0.058 (mean 0.047) | 0.333, 0.358, 0.308, 0.342, 0.308 (mean 0.330) |
| control s0 to s2 | 0.033, 0.025, 0.042 | 0.400, 0.358, 0.325 |

- **RSC is below A12 R_V on VSC in 5 of 5 seed pairs and above it on collateral in 5 of 5.** The ordering warm start > R_A > R_V > RSC suggests that more ProcureSim-specific RL makes care_admin outcomes worse. This is negative transfer, and "no gain" (README line 24; paper line 106) understates it.
- In it_ops every model scores 0.15 to 0.22 and the warm start 0.200. That is null.
- Both domains are a floor regime for all models, and they are sibling simulators. The result is informative only as a warning against claiming generality.

### 2.8 Per-workflow numbers

`s3_by_workflow.json` matches the documents (full-episode partial and duplicate tasks, seeds 0 to 2 pooled):

| Arm | order then pay | receive then pay | pay first |
|---|---|---|---|
| A12 R_V | 17/111 (0.15) | 16/102 (0.16) | 63/117 (0.54) |
| control | 62/111 (0.56) | 63/102 (0.62) | 54/117 (0.46) |
| RSC | 109/111 (0.98) | 61/102 (0.60) | 82/117 (0.70) |

- **RSC minus control by workflow:** order then pay +0.42, pay first +0.24, receive then pay -0.02. The paper's "concentrated in the order-then-pay workflow" (line 85) omits pay-first and hides the null cell. On recovery starts the pattern is the same: +0.32, +0.33, +0.02.
- **The absolute spread across workflows is unchanged:** 0.54 - 0.15 = 0.38 for A12 R_V and 0.98 - 0.60 = 0.38 for RSC. RESEARCH_REPORT section 8, "This removes most of the workflow dependence the review found in R_V", is wrong in absolute terms. Section 9 does not retract it.

**Per cell (workflow x faulted tool; my regrouping of `results/s3/conf/*.jsonl`):**

| Cell (n per model) | A12 R_V s0 to s4 | control s0 to s2 | RSC s0 to s4 |
|---|---|---|---|
| receive then pay, partial **payment**, full (12) | 0, 0, 0, 0, 0 | 0, 0, 0 | 0, 0, 0, 5, 0 |
| same cell, recovery starts (6 ack + 6 read) | 0 in every seed | 0 in every seed | 2, 0, 0, 2, 0 |
| receive then pay, partial goods receipt, full (6) | 6, 5, 5, 6, 6 | 6, 6, 6 | 6 in every seed |
| order then pay, partial payment, full (24) | 1, 15, 1, 0, 0 | 24, 2, 0 | 24, 24, 24, 24, 14 |
| pay first, partial payment, full (22) | 21, 22, 20, 2, 0 | 21, 2, 2 | 20, 22, 20, 21, 22 |

- **Partial-payment repair after a goods receipt remains unsolved by every arm.** RSC solves 5 of 60 full episodes and 4 of 60 recovery starts placed right after the faulted payment. RSC's 0.60 in receive-then-pay is made of duplicates and partial goods receipts, which every trained model solves.
- A representative trace (`s3_rv_mix_s0`, `s3_conf_full`, receive then pay, partial at 1/4): the agent reads the payment record showing 62,892 of 251,570 cents, then posts the journal entry and reports "completed". This is acquisition without use, the A14 pattern, persisting under RSC.
- **Training-frequency confound.** The RSC and control pools contain 6 receive-then-pay partial-payment tasks out of 400 (4 recovery starts and 2 full episodes), against 32 for order then pay and 29 for pay first. `proc_train_b` (A12) has 6, 20 and 19. Frequency is not a sufficient explanation, because A12 R_V had 20 order-then-pay tasks and still failed in four of five seeds. It is a confound that must be removed before any "workflow position" or "context" interpretation of W2.

### 2.9 Against the pre-results decision table

| Row | Met? |
|---|---|
| "No": gains confined to duplicates or pay-first | Not this row. Partial gains appear in order then pay and pay first |
| "Yes, modest": beats the matched-class control on partial + duplicate, CI excluding 0, at least 4 of 5 seeds | Partially. 3 of 3 seeds, one of them +0.009; task CI excludes 0; seed-level p = 0.10 |
| "Yes, solid": plus gains in order then pay **and** receive then pay, no regression on stress, holds on seeds 2 to 4 | No. Receive then pay is unchanged (0 of 36 vs 0 of 36, seeds 0 to 2); stress regresses vs the control; only seed 2 exists among seeds 2 to 4 for H3 and H4 |
| "Yes, strong": beats oversampling, dynamic sampling, SFT on recoveries | Not tested |

By the criteria fixed before the results, S3 is at most a modest empirical result about partial-payment repair, and it depends on more control seeds.

### 2.10 Other observations

- **On clean full episodes, every R_V-family model in every seed makes 6.40 calls and 2.70 reads per episode** (warm start: 3.70 and 0.00). The curriculum changes nothing outside the faulted classes, and reading is unconditional.
- **"Complete" and "done" starts are solved by the untrained warm start (1.00).** The report's claim that F "covers failure 4 ... and repairs the A17 fork failure mode in-domain" (RESEARCH_REPORT section 4.F) has no in-domain test: there was nothing to repair.
- **No model in any arm ever escalates** (escalation 0.00), so the escalate path is untested.
- **Headroom is small.** Never-commit tasks are unreachable, so the oracle ceiling is 0.897 (rec) and 0.917 (full). RSC averages 0.821 and 0.833. Of RSC seed 0's 20 recoverable failures on `s3_conf_rec`, 10 are receive-then-pay partial payments and 6 are pay-first duplicates.
- The honest-failure "no collapse" check is satisfied at ceiling. On recovery starts, RSC has slightly less honest failure than A12 R_V (-0.017 [-0.033, -0.004]) because it converts honest failures on partial starts into successes. That is not a concern, but it should be reported next to "unchanged".

---

## 3. Novelty after the novelty audit

**What the paper can claim (with the stated conditions).**
1. **A setting.** RL training of tool agents when write acknowledgements are byte-identical across committed, partial, duplicate and uncommitted outcomes, with retries that can duplicate. The audit (section 10) supports a narrow form only: "among the papers we checked, no training paper combines silent write commits with duplicate risk". NoisyAgent and Bench2Robust train with silent output corruption, and Bench2Robust assumes retries cannot duplicate. LIMBO, RetryBench and Evidence-Guided Recovery evaluate duplicates or ambiguous outcomes without training. **Condition:** resolve the UNVERIFIED Research Square item "Did It Happen? Counterfactual Evaluation of LLM Agent Recovery from Ambiguous Tool Outcomes" (audit, UNVERIFIED item 1), which may describe observation-equivalent hidden-commit pairs. Until then, no setting-level novelty claim should be made about forced-evidence evaluation (A14).
2. **Registered measurements in that setting.** A12: a state-observing reward made read-back the default while an acknowledgement reward left the warm start unchanged. A14 and A15: acquisition and use dissociate, and use is workflow-bound. A16: the evidence is decodable everywhere, and the ablation is not R_V-specific. A17 and S3: transfer is absent or negative. S3: a per-class decomposition of a known curriculum's gain into coverage (duplicates) and start state (partial payments, 3 seeds).
3. **A process contribution.** Hashed registrations, a dated deviation log, one-shot confirmatory access and a number-checked claims ledger. This is not a scientific finding, but reviewers value it. It is only credible if the checker's blind spots (Section 1.3) are closed.
4. **The authors' own Stage 1** found the same direction in a sibling environment. The paper now discloses it (appendix line 87). In an anonymous submission it must be cited as anonymised prior work, and the main text must say that A12 is a replication of that direction.

**What the paper cannot claim.**
- **A new algorithm or training method.** RSC is a reset-distribution curriculum in the family of Fission-GRPO (ACL 2026), R3L, failure-prefix conditioning, ReflectPO and Depth Before Breadth (audit section 8, "Stage 3 recovery-start curriculum").
- **A new credit estimator.** FCA is the all-action / counterfactual-baseline family (TRPO vine, COMA, BiPACE, RTMC, CRR, BPO, SURE-Act; audit section 8 "Candidate A"). The paper says this (line 35; appendix line 70).
- **That outcome-RL agents skip verification or claim false success** (Reward Hacking Benchmark, Confident Closing).
- **That RL selects behaviours already in the base distribution** (Yue et al.).
- **That tool calls can be decorative or acquisition can occur without use** (Proof-of-Use, EMNLP 2026). Cite it as the read-only analogue.
- **The audit's most promising residual idea**, curriculum-induced covariate shift under observation-equivalent hidden faults (SAMPLR's CICS). This cannot be claimed from current data. Reading is free and already unconditional, and clean-episode behaviour is identical across A12 R_V, the control and RSC (6.40 calls, 2.70 reads), so a CICS effect on verification is unobservable. Testing it needs a read cost or another unnecessary-action cost. That is a design change, not a reanalysis.

---

## 4. Remaining weaknesses, ranked by severity, with fixes feasible in about 2 days

Budget assumed: a 2xH200 Slurm lease (2 GPUs per user, 24 h wall), almost no cloud credit. Timings come from the repository: a 200-step GRPO run is about 1 h 50 min on one H200 including development evaluations (appendix line 14; `docs/stage2/HF_CREDITS.md`), and an SFT warm start about 16.5 min. A 300-task greedy confirmatory evaluation reports 22 to 29 s of rollout time (`seconds` in the S3 summaries), plus model loading. **Compute is not the binding constraint. Registration discipline and analysis time are.**

**1. Critical: the start-state claim (H3, H4) is under-powered and class-specific.**
- Evidence: Section 2.4. 3 seeds; seed-level p = 0.05 and 0.10; zero effect on duplicates; partial effect driven by control seeds 1 and 2.
- Fix:
  - Register addendum 2 before training. It should extend H3 and H4 to control seeds 3 and 4, use the unpaired seed-level permutation test or a two-level (seed then task) bootstrap as the decision statistic, make the partial-payment and duplicate classes separate co-primary contrasts, and report the 3-seed and 5-seed results side by side.
  - Train `s3_matched_s3` and `s3_matched_s4` (2 runs, about 4 GPU-h on the two lanes in parallel) and evaluate once on the S3 splits.
  - With no GPU, compute the zero-variance share of partial-fault groups per arm from the S3 rollout logs. This tests the claimed mechanism.

**2. Critical: the paper still overstates in places, contradicts itself, and has unrun content.**
- Evidence: Section 6 lists each instance. "[Pending]" section at line 104; contribution (iv) with no results; line 73 vs section 7 on duplicate refunds; Discussion line 110 vs section 4 line 62.
- Fix (no GPU): apply Section 6. Either run S4-EVAL (item 5; under about 1 GPU-h of inference) or remove contribution (iv) and section 8's pending text. Report FCA only if its kill criteria are evaluated.

**3. Critical for any recovery or evidence-use claim: partial-payment repair after a goods receipt is unsolved in every arm, with a 5x training-frequency confound.**
- Evidence: Section 2.8.
- Fix (no GPU): add the per-cell table to the paper.
- Fix (GPU): register and run a rebalanced-pool experiment. Use RSC and the control, 2 seeds each (4 runs, about 8 GPU-h), with equal partial-payment counts per workflow (for example 30 each) and everything else unchanged.
  - Evaluate on a fresh targeted split of about 150 partial-payment tasks balanced by workflow (the generator exists). The existing cell has only 12 tasks per split.
  - Pre-register both outcomes. If receive-then-pay repair rises to 0.5 or more, the W2 workflow effect is mostly pool frequency. If it stays near 0, it is a property of the evidence context (read after a goods receipt), which is a more interesting finding about evidence use.

**4. Major: A12's identification gap (pre-results W1) is unchanged, and the Discussion restates the general causal claim.**
- Fix (text, mandatory): restate the Discussion as in Section 6, item 1.
- Fix (GPU, if a lane is free after items 1 and 3): the warm-start repertoire test. SFT with oracle share 0 (about 17 min), then R_V for 200 steps with 2 seeds (about 4 GPU-h), evaluated on `proc_dev` and `proc_val`. Register it first. "Selection, not invention" predicts zero read-back. The pre-results review rated this the largest Excitement gain per GPU-hour, and that still holds.

**5. Major: statistics.**
- Evidence: (a) S3 has no seed-level inference, although the data support it for H1 and H2 (p = 0.004) and not for H3 and H4. (b) The S4-EVAL Holm statement makes G1 to G4 unpassable at 0.05 (Section 1.3). (c) Seeds are highly redundant: RSC pairs are identical on 263 to 291 of 300 tasks, and control pairs on 229 to 264. (d) The H4 bound is mis-rounded in three documents.
- Fix (no GPU):
  - Before any S4 output is read, file an S4-EVAL addendum that either makes the Holm-adjusted sign-flip p descriptive, with the "supported if" column as the decision rule, or switches the family to the unpaired seed permutation (smallest p 1/252; Holm over four tests is then attainable).
  - Add a two-level bootstrap and the identical-sequence fractions to S3 and A12.
  - Correct [0.16, 0.29] to [0.15, 0.29], and add every printed CI bound to the ledger.
  - Remove "0.21" from the numcheck allow-list, or scope the allow-list to version strings.

**6. Major: transfer is negative and is reported as null.**
- Evidence: Section 2.7.
- Fix (no GPU): one table with the warm start, R_A, R_V, RSC and the control on VSC and collateral for both domains, and the sentence in Section 6, item 6.

**7. Major: construct validity and generality.**
- Evidence: one model and one warm start; sibling simulators; fault prevalence far above any deployment (a 25% phantom share in `proc_conf`); byte-identical acknowledgements as a maximally adversarial design; reads always free and authoritative, so "verification" is one unconditional read (6.40 calls on clean episodes vs 3.70 for the warm start); greedy decoding only.
- Verifier provenance: the "independently written" reference verifier and 91 golden traces (appendix line 4) need a who-and-how statement, given the LLM-use declaration (appendix line 89). If the same assistant wrote both from one spec, call it a consistency check.
- Fix: text. Optionally, the sampled evaluation (T = 1, k = 8 on `proc_val` for the warm start, A12 R_V, R_A, RSC and the control; inference only, under 1 GPU-h) makes the repertoire claims quantitative.

**8. Moderate: the recipe comparison includes a code-path change** (Section 2.1).
- Fix: a CPU replay test showing that the Stage 2 and Stage 3 rollout paths produce identical environment replies and rewards on `proc_train_b` tasks for fixed action sequences. Or, if a lane is idle, one Stage 3 path run on the A12 pool (about 2 GPU-h).

**9. Moderate: FCA has low expected value now.**
- Evidence: RSC's remaining headroom on recovery starts is about 0.075. Half of it is one cell (item 3) that is probably a pool-frequency problem. The kill criterion (+0.03 on `s3_rec_dev`) needs FCA to fix about 40% of all remaining recoverable failures in two seeds. FCA is also not novel (Section 3).
- Fix: run it after items 1 and 3. Alternatively, before training, register cell-level kill criteria (receive-then-pay partial payments; pay-first duplicates) that match where the failures are.

**10. Moderate: related-work and novelty hygiene.**
- Fix: resolve the unverified "Did It Happen?" item. Re-cite tau-bench, LIMBO, `chuang2026proxystate` (R_A's practical motivation), `yue2025rlreasoning`, `thaman2026rhb`, NoisyAgent, R3L and failure-prefix conditioning.

**11. Minor.**
- Anonymisation and the current ACL template (item 5 in Section 1.1).
- Ledger statements A15, A16 and K1 are not updated. README lines 15 to 17 overstate A14/A15, A16 and A17.
- The truth-table caption is inaccurate (Section 6, item 15).
- Appendix line 85 says HF Jobs only, but S3 ran on the Slurm H200 lease.
- S3 statistics (`s3_stats.py`) are not described in appendix section C.
- Still open: the selection checkpoint, token counts and A14 oracle `goal_met`.

**Suggested 2-day schedule (two lanes, no cloud credit).**

| When | Lane A | Lane B | No-GPU work |
|---|---|---|---|
| Day 1 morning | | | Section 6 text fixes; per-class and per-cell tables; seed-level statistics; addendum 2 (H3/H4 to 5 seeds; rebalanced pool); S4-EVAL Holm addendum; zero-variance analysis from S3 rollout logs |
| Day 1 | `s3_matched_s3` (about 2 h), then rebalanced control s0 | `s3_matched_s4`, then rebalanced RSC s0 | Evaluate each model once as it finishes |
| Day 1 evening | rebalanced control s1 | rebalanced RSC s1 | Generate the fresh targeted partial-payment split before any rebalanced model is evaluated |
| Day 2 | S4-EVAL inference for all frozen adapters (about 1 h), then warm-start SFT at oracle share 0 and R_V s0 | R_V s1 on the oracle-0 warm start; sampled evaluation | Write up; update the ledger; run numcheck |

Total: about 9 GRPO runs and 1 SFT, roughly 19 GPU-h, which leaves margin for queue waits on a shared lease. If the queue stalls, drop the oracle-0 runs (item 4) first and the S4-EVAL inference last. S4-EVAL is the cheapest item and gives the paper its only language dimension.

---

## 5. Scores (ACL ARR-style)

The fields and labels follow my recollection of the ARR form: Soundness, Excitement, Overall Assessment and Reviewer Confidence, each 1 to 5 with half points. **The schema was not re-verified against the live ARR form.**

| Field | Score | Rationale |
|---|---|---|
| Soundness | **3 / 5** (Acceptable) | The registered claims hold as narrowly stated, and H1/H2 hold across training runs. The start-state claim rests on 3 seeds and one class. Several sentences outrun the evidence (Section 6), the paper has a pending section, and two open identification gaps remain (A12 W1, workflow frequency) |
| Excitement | **2.5 / 5** | Careful and honest, with a useful setting. But the method is known, one model and one warm start are used, transfer is negative, and the headline A12 contrast compares a trained agent with an untrained one |
| Overall assessment | **2.5 / 5** (borderline Findings) | Not main-track material. Expect 3 (Findings) if items 1 to 3 and 5 in Section 4 are done and the text follows Section 6. 3.5 would need the warm-start identification result and a second model, which do not fit in 2 days |
| Reviewer confidence | **4 / 5** | I re-read every S3 result file, regrouped the per-episode records and pools myself, and re-ran the read-only number checker. I did not re-run training or the bootstrap scripts |

**Recommendation.** The project's strongest assets are its experimental hygiene and the clarity of its negative findings. The rewritten paper now states the non-contributions honestly. What it can support is narrower than its abstract and Discussion:
- a state-observing reward made read-back the default where an acknowledgement reward left the warm start unchanged;
- reading and acting on evidence dissociate, and repair is bound to workflow context;
- a known recovery-start recipe gives a robust gain over A12 R_V, about half of it from adding a fault class, while its start-state component is plausible for partial payments only and is not yet established across training runs;
- nothing transfers to sibling domains, and care_admin gets worse.

That is a solid **workshop** paper today, on agent reliability or failure modes, RL for LLM agents, or evaluation of interactive agents. Those venues value a careful negative and decompositional study in a well-specified simulator over methodological novelty. It is a plausible **Findings** paper after the 2-day plan in Section 4: a 5-seed start-state test, the rebalanced-pool test of the workflow failure, S4-EVAL with Hindi and Hinglish (the only result that makes it an NLP-venue paper rather than an RL-for-agents paper), and the Section 6 corrections. It is **not main-track** material: there is no new method, a single model and warm start, and the main effects are partly set by the simulator's fault design. If the authors prefer an ML venue over an ACL venue, the same reasoning points to a workshop there.

---

## 6. Claims that would be inflated if written in the paper, with the correct wording

| # | Inflated claim (where) | Problem | Correct wording |
|---|---|---|---|
| 1 | "The information a reward observes decides whether an agent checks ... it replicates across five seeds" (Discussion, main.tex line 110) | R_A is a saturated arm equal to the warm start (line 62; Limitations). The causal channel is not identified. Seeds of one warm start make near-identical policies | "In this simulator, training with the state reward made read-back the default, and training with the acknowledgement reward left the warm start's behaviour unchanged, in five of five seeds from one warm start." |
| 2 | "two rewards that differ only in what they observe" (line 35) | They also differ in what they pay for and in saturation (mean reward 0.97 vs 0.63 in A12) | "two rewards of identical binary form that differ in what they observe, and therefore in what they pay for and how much gradient they give" |
| 3 | "only state-trained agents refund a partial payment, and only when payment is the first step of the workflow" (abstract) | A15 R_V seed 1 repairs 0.55 in order then pay; S3 A12 R_V seed 1 repairs 15 of 24 there | "... and almost only when payment is the first step: 0.99 to 1.00 when payment comes first, 0.00 after a goods receipt, and 0.05 to 0.55 after an order (seeds 0 to 2)" |
| 4 | "A recovery-start curriculum raises VSC by +0.21" (abstract; README line 22) | The recipe also adds fault types; 55% of the gain is duplicate delivery | "The recovery-start training recipe (recovery starts plus duplicate-delivery and extra partial-ratio training data) raises VSC by 0.21 over A12 R_V in five of five seeds; about 0.11 of this is duplicate delivery, a class A12 never trained on." |
| 5 | "starting at recovery states adds +0.09 ... and +0.22 on full-episode partial and duplicate tasks" (line 85; README line 23); "helps beyond seeing the faults" (line 110) | Effect is entirely on partial payments; none on duplicates; 3 seeds; seed-level p 0.05 and 0.10; control seed 0 matches RSC | "Against a full-episode control trained on the same faults, recovery starts improved partial-payment repair (+0.09 VSC on full episodes) and made it consistent across seeds: learned in 5 of 5 recovery-start runs vs 1 of 3 control runs. Duplicate handling did not benefit (the control was equal or better). Three control seeds; not established across training runs." |
| 6 | "Gains do not transfer" (abstract); "neither R_V nor RSC improves" (line 106); "no transfer to the OpsSim domains" (README line 24) | care_admin is worse than both the warm start and A12 R_V | "In care_admin, task outcomes got worse with ProcureSim training: VSC 0.158 (warm start), 0.080 (A12 R_V, mean of 5 seeds), 0.047 (RSC; below A12 R_V in 5 of 5 seed pairs); collateral writes 0.18, 0.27, 0.33. it_ops was unchanged." |
| 7 | "The R_V-over-R_A read-back gap transfers to healthcare administration (+0.37 [0.18, 0.58])" (line 106); "Read-back transfer to healthcare administration" (README line 17) | R_V vs warm start +0.15 [-0.06, 0.39] is not established; most of the gap is R_A reading less than the warm start | "In care_admin, R_V reads back after phantom writes more than R_A (+0.37 [0.18, 0.58]), mainly because R_A reads less than the warm start; R_V vs warm start is not established." |
| 8 | "slightly more irreversible violations" (abstract) | All violations are duplicates left unrefunded, which ProcureSim can refund; not established on recovery starts | "slightly more duplicate payments left unrefunded (about 3 more per 110 partial and duplicate tasks per seed)" |
| 9 | "lowers ... violations by 0.12" vs R_V (line 83) | Entirely duplicate coverage | "leaves fewer duplicate payments unrefunded (-0.12), because the pool includes duplicate delivery" |
| 10 | "This removes most of the workflow dependence" (RESEARCH_REPORT section 8); "concentrated in the order-then-pay workflow" (line 85) | Absolute spread unchanged (0.38 vs 0.38); start-state effect +0.42 order then pay, +0.24 pay first, -0.02 receive then pay; receive-then-pay partial payments 5 of 60 | "RSC raises partial and duplicate VSC in every workflow on average, but partial payments after a goods receipt remain unrepaired by every arm (RSC 5 of 60 episodes; 0 in four of five seeds)." |
| 11 | "No model refunds a duplicate delivery" (line 73) | RSC and the control do (duplicate-class VSC 0.63 to 0.97) | "No A12 model refunds a duplicate delivery." |
| 12 | "the sampled read rate stopped declining" (K1, line 62); ledger K1 "but not under targeted" | It declined less (about 0.21 to 0.16) than under the placebo (about 0.13 to 0.06) | "the sampled read rate declined less than under a random placebo pool" |
| 13 | RESEARCH_REPORT section 4.F: F "covers failure 4 ... and repairs the A17 fork failure mode in-domain" | The warm start already scores 1.00 on complete and done starts; no in-domain failure existed | "Complete and done starts were solved by the warm start, so the effect of recovery starts on stopping was not tested in ProcureSim." |
| 14 | Contribution (iv) and section 8 (lines 35, 104), as long as S4-EVAL and FCA have no results | Claims a contribution with no evidence | Remove until results exist; if FCA is run, "an adaptation of published counterfactual-baseline estimators, tested under pre-set kill criteria" with its outcome |
| 15 | Truth-table caption: "The rewards disagree exactly where reading changes the outcome" (line 56) | They also disagree on partial and never-commit under every behaviour, where reading does not change R_V | "R_V and R_A disagree on phantom, lost-acknowledgement, partial and never-commit behaviours; reading changes R_V only after a phantom write." |
| 16 | H4 interval [0.16, 0.29] (RESEARCH_REPORT line 163; ledger S3-A1; DECISIONS 16:30) | The file gives [0.1545, 0.2879] | [0.15, 0.29] |
| 17 | Ledger A15 "generalises, not pattern-specific"; README line 15 "pay-first workflows only" | Generalises across ratios, not across workflows; "only" is too strong | "generalises across ratios within the pay-first workflow; near 0 after a goods receipt; 0.05 to 0.55 after an order (seeds 0 to 2)" |
| 18 | "a state-only verifier checked against an independent reference implementation" (line 35; appendix line 4) | Independence is unstated given LLM-assisted coding | State who or what wrote each. If one assistant wrote both from one spec: "a second implementation written from the specification alone" |
| 19 | "RSC is more reliable / more stable" (if written) | Between-seed SD 0.01 vs 0.05 to 0.07, but seeds are highly redundant (RSC pairs identical on 263 to 291 of 300 tasks) | "In this run set, RSC seeds varied less (SD 0.01 vs 0.05 to 0.07)." Report the identical-sequence fractions next to it |

---

## Appendix: reviewer reanalyses (read-only; reproducible)

1. **Seed-level tests (Sections 2.2 and 2.4).** Per-seed VSC from `splits.<split>.models` in `s3_summary.json`, `s3_addendum.json` and `s3_addendum_full_pd.json`. The unpaired permutation test enumerates all label assignments of the pooled per-seed values (5 vs 5: 252; 3 vs 3: 20; 5 vs 3: 56), with the difference of means as the statistic, one-sided. The seed-level t interval uses the per-seed paired differences with t(4) = 2.776 or t(2) = 4.303.
2. **Class decomposition (Section 2.3).** For each model, the sum over classes of `n x vsc` from `by_class` in `results/s3/conf/<model>__<split>.summary.json`, divided by 300, averaged over seeds. Groups: duplicate = `dup_delivery` (rec: `s3:dup_delivery:ack`, `:read`); partial = `partial` (rec: `s3:partial:ack`, `:read`); other = the remaining classes.
3. **Per-cell table (Section 2.8).** In `results/s3/conf/<model>__s3_conf_full.jsonl` and `__s3_conf_rec.jsonl`, group by (`fault_class` or `task.meta.fault_class`, `task.workflow`, `task.meta.faults[0].tool`) and sum `vsc`. The example trace is the first receive-then-pay partial-payment failure of `s3_rv_mix_s0` in `s3_conf_full`.
4. **Violation location (Section 2.5).** `by_class.<c>.violation x n` and `duplicate x n`, which are non-zero only for `dup_delivery` in every model on `s3_conf_full` and `s3_conf_stress`.
5. **Identical tool-call sequences.** For each task, the tuple of (`tool`, `args` without `summary`) over `trace` entries with `by == "agent"`, compared across models on `s3_conf_full`.
6. **Pool composition.** Counts of (`meta.fault_class`, `workflow`, `meta.faults[0].tool`, `ratio`) in `data/bench/proc/s3_rec_train_half.jsonl`, `s3_trainb_half.jsonl` and `proc_train_b.jsonl`.
7. **Clean-episode behaviour and transfer.** `by_class.clean.policy_calls` and `.policy_reads` on `s3_conf_full`; `vsc` and `collateral` in `results/s3/conf/<model>__s3_care_admin.summary.json` and `__s3_it_ops.summary.json`.
8. **Code paths.** `slurm/tasks/612_a12_proc_rv_s1.sh` sources `slurm/a12_arm.sh`; `slurm/tasks/912_s3_rs_s2.sh` and `915_s3_matched_s0.sh` source `slurm/s3_arm.sh`, which calls `actionproof.stage3.train`.
