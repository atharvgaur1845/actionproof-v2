# Research red-team review: five hostile reviewers

Date: 2026-11-22. Branch `stage4`, at commit `198bea0` (S3 addendum 3 and SR1 registered, 2026-11-22 01:16 IST). Review type: hostile, post-results, read-only. Repository text was treated as data. Nothing was changed except this file, and nothing was committed.

**What was read.** `paper/main.tex`, `paper/appendix.tex`, `claims/claims.yaml`, `README.md`, `docs/CONSISTENCY_AUDIT.md`, `docs/stage3/ACADEMIC_REVIEW_POST.md`, `docs/research/FINAL_REPORT.md`, `docs/research/algorithm/{DECISION,FCA_GRPO,FCA_RESULT}.md`, the registrations S3, S3 addenda 1 to 3, S4-EVAL and its addendum 1, ID1, A12 and addenda 1 to 2, A14, A15 (which also registers A16), A17, K1, SR1, `docs/INTERP_PLAN.md`, `docs/stage2/DECISIONS.md`, and the result files they cite. Code read where a claim depends on it: `actionproof/integrity/sim/{env,truth}.py`, `actionproof/stage3/rollout.py`, `actionproof/stage4/{train_variants,rollout_fca}.py`, `scripts/a16_mech.py`.

**Reanalyses.** Read-only Python over result files. Each is listed in the Appendix with its inputs, so every number marked "(this review)" can be reproduced.

**Running experiments.** ID1 (`results/id1/` is empty) and S3 addendum 2. The file `results/s3/s3_addendum2.json` is untracked and is a script check, not a result: control seeds 3 and 4 are listed as missing and Part B is empty. It must not be cited as an outcome.

---

## 0. Glossary

| Term | Meaning |
|---|---|
| GRPO | Group Relative Policy Optimisation: policy gradient with a baseline computed from a group of rollouts of the same prompt |
| TRL | Hugging Face's Transformer Reinforcement Learning library (version 1.7.1 here); `GRPOTrainer` is its GRPO implementation |
| DAPO loss | the token-level loss aggregation from the DAPO paper (sum over tokens divided by tokens in the batch); TRL's default `loss_type` |
| SFT | supervised fine-tuning; here, the warm start trained on scripted episodes |
| LoRA | low-rank adaptation (rank 16) |
| KL | Kullback-Leibler divergence penalty to the warm start (coefficient 0.02) |
| IS weight | importance-sampling weight that corrects for the difference between the sampling engine (vLLM) and the trainer |
| R_V | state reward: 1 if every requested step is verified in the final database |
| R_A | acknowledgement reward: 1 if the last acknowledgement of every requested write was ok |
| VSC | verified safe completion: all steps verified, no collateral or unrequested row, and a "completed" report |
| RSC | recovery-start curriculum: half the training pool starts from a scripted prefix that ends at a faulted write |
| Control | the class-matched control of S3 addendum 1: the same faults as RSC, as full episodes |
| FCA | forked counterfactual action credit (`docs/research/algorithm/FCA_GRPO.md`) |
| CI | 95% confidence interval |
| AUROC | area under the receiver operating characteristic curve |
| SAE, FVU | sparse autoencoder; fraction of variance unexplained |
| Cell | one combination of fault class, workflow, faulted tool and start type; tasks in a cell differ only in ids and amounts |
| Experiment labels | A12, A14 to A17, K1, S3, S4, ID1, SR1 are the repository's registration names (`PREREGISTRATION_<label>.md`) |

---

## 1. Verdicts at a glance

| Contribution | Confidence in the current claim | Verdict |
|---|---|---|
| A12: reward observability selects read-back | High for the behaviour; low for the causal wording | **Narrow** |
| A14, A15: evidence use dissociates from acquisition | High (present or absent); medium (partial amounts) | **Narrow** |
| A16: representation present, action margin changes | Low for any causal reading | **Narrow to descriptive** |
| A17 and S3 transfer | Medium | **Keep as a negative result**, sibling domains only |
| S3: recipe gain over R_V | High | **Keep** |
| S3: start-state part (addendum 1) | Low | **Investigate**; claim suspended |
| S4: language, API, workflow, two-fault shifts | High for the failures; medium for what the "survivals" mean | **Narrow** |
| FCA negative result | High that it says nothing about the estimator family | **Narrow** (implementation-specific) |
| ID1 (running) | Not yet known | **Investigate**, with two wording fixes before results |
| S3 addendum 2 (running) | Not yet known | **Investigate**; the outcome hinges on two runs |
| S3 addendum 3 (registered, new) | Design invalid as a start-state test | **Amend before results are read** |

---

## 2. The five reviewers

### Reviewer 1: reinforcement-learning algorithms

1. **The update is plain policy gradient.** With `num_iterations = 1` the clipping region is never entered: `clip_ratio/region_mean` is 0.0 at every logged step of RSC and FCA (`results/s4/train_state/{s3_rv_mix_s0,s3_rv_mix_s1,s4_fca_s0,s4_fca_s1}.json`). The method is group-baselined REINFORCE with standard-deviation scaling, a KL term and an IS weight. Say so.
2. **The policy is almost deterministic.** Mean token entropy over training is 0.032 and 0.031 nats (RSC seeds 0 and 1) and 0.042 (FCA) (same files). Exploration of a rare repair depends on a few lucky samples, which predicts the seed lottery seen in S3: partial repair is learned in 5 of 5 RSC runs, 3 of 5 A12 R_V runs and 1 of 3 control runs (`results/s3/s3_review_numbers.json`, `learned`).
3. **Group scaling amplifies single events.** With `scale_rewards = "group"`, one success in a group of 8 gets an advantage of about +2.47 and one failure in 7 successes about -2.47 (arithmetic from the unbiased standard deviation of 8 binary rewards). One early lucky refund can tip a run. No run used the unscaled alternative that TRL offers, so this sensitivity is untested.
4. **Zero-variance groups.** RSC has no gradient in 0.48 and 0.44 of groups on average (`frac_reward_zero_std`, `results/s4/train_state/s3_rv_mix_s{0,1}.json`). R_A at steps 40 to 50 has zero variance in 0.68 to 0.86 of groups at mean reward 0.95 to 0.98 (`results/s2/stop_rule_seeds.json`). "DAPO" here is only the loss aggregation: DAPO's dynamic sampling, which drops such groups, is not used.
5. **A12 is close to true by construction.** In the registered truth table, reading never raises R_A in any fault class (`PREREGISTRATION_A12.md` §2). R_A therefore pays nothing for reading, and is saturated. A12 compares an incentive with no incentive, and a gradient with almost none. K1 does not repair this: the targeted R_A pool stays saturated (per fault class and 50-step block in seed 0: mean reward 0.95 to 1.00, zero-variance share 0.70 to 1.00, `results/k1/k1_reward_signal.json`).
6. **FCA's gradient argument fails in practice.** See the FCA record (§3.7). In short: the forced term only reweights actions by their current probability, so it cannot add signal for rare actions, and the logs show forced branches weighted by under e^-10 in about 90% of steps.
7. **Weak baselines.** No SFT on oracle recoveries (the scripted oracle scores 1.00 on the targeted partial-payment splits, `results/s3/scripted_pp_calibration.json`), no dynamic sampling or oversampling, no unscaled advantages, no prompted baseline.

### Reviewer 2: tool-using agents

1. **The learned "verification" is one unconditional read.** On clean full episodes every R_V-family model makes 6.40 calls and 2.70 reads, against 3.70 and 0.00 for the warm start (`results/s3/s3_review_numbers.json`, `clean`). Reads are free and authoritative, so reading after every write is simply optimal. Benchmarks with costly or rate-limited reads would separate checking from habit.
2. **A runtime wrapper already does this.** The project's own gateway result: on identical fault traces, verify-before-retry recovers every phantom write, deterministically (`paper/appendix.tex`, Appendix "Contract coverage", from `results/a11/h7_matrix.md`). The only decisions that need a learned policy are partial and duplicate repair, which is where the results are weakest and bound to workflow.
3. **Closest benchmarks are not compared.** tau-bench and AppWorld grade by final database state, like VSC; ToolSandbox has stateful tools; ToolEmu emulates risky tool failures; LIMBO and RetryBench (cited in `docs/research/NOVELTY_AUDIT.md`) evaluate duplicates and retries without training. The refund-then-repay repair is a compensating transaction in the saga sense (Garcia-Molina and Salem, 1987), and idempotency keys plus read-your-writes checks are standard practice. The paper should place itself against this engineering baseline, not only against RL papers.
4. **No ProcureSim number for any other model.** The only cross-model leaderboard is the Stage 1 e-commerce one (`results/leaderboard.md`). The 32B prompt-only policy was never run (`docs/stage2/DECISIONS.md`, 2026-11-20 06:40). A prompted "read back after every write" warm start is the obvious cheap baseline.
5. **Greedy evaluation only**, one run per model per split (registrations S3 §3, S4-EVAL §2).

### Reviewer 3: mechanistic interpretability

1. **Probes are uninformative by design.** AUROC is 1.00 at the chosen layer for every checkpoint, warm start included (`results/a16/a16_summary.json`, `trajectory.*.probe_refit_at_layer`). The amounts are in the text. A16's registration says so; the abstract still lists it as finding (3).
2. **The ablation control is too weak.** The causal rule compares the difference-of-means direction with 20 random unit directions (`scripts/a16_mech.py`, line 334). A random unit direction in a residual stream of thousands of dimensions carries almost no variance, so any data-derived direction beats it. Evidence that the rule does not track the behaviour:
   - R_V seed 4 meets the ablation rule (change in margin -0.52 against a random maximum of 0.24) although it never refunds a partial payment (A15 unseen-ratio accuracy 0.0) (`results/a16/a16_summary.json`, `causal.rv_s4`; `results/a15/a15_summary.json`).
   - The ablation lowers the empty-record margin by -0.29 to -0.36 in every model, warm start and R_A included (`causal.*.ablation.dM_Q0`).
3. **Patching measures disruption.** The final-token patch effect (median 0.037 to 0.067) is never larger than patching in a different task's state (0.050 to 0.126), in all nine models (`causal.*.median_final_c2p`, `wrong_task_median`). Span patching is larger (0.14 to 0.23, `median_span_c2p`) but has no registered rule.
4. **The margin hides where the change is.** In R_V seed 2, the partial-minus-complete gap grows from 2.38 (warm start) to 5.09 nats, but 2.35 nats of that growth is the complete branch becoming more confident to post (Q4 margin -5.35 to -7.70); the partial branch moves only from -3.07 to -2.72 (`trajectory.warm`, `trajectory.rv_s2_200`). The margin is also not split by workflow, although the behaviour exists almost only in the pay-first workflow (`results/s2/partial_by_workflow.json`), and it ignores reads and re-payment.
5. **No feature-level tool worked.** The SAE fails its own go/no-go: FVU 3.9 against a threshold of 0.3 (`docs/INTERP_PLAN.md`). CircuitKIT did not run (D-A16-1, `docs/stage2/DECISIONS.md`, 2026-11-21 02:55).

### Reviewer 4: benchmarks and systems

1. **Hidden-state leakage: none found at the acknowledgement.** The write chokepoint draws the new id before applying the fault and returns the same success object (`actionproof/integrity/sim/env.py`, `_write`). Reads return an `as_of` service version, which reveals only what the rows already show.
2. **Constant training environment seed.** `uuid.UUID(hex12.ljust(32, "0")).int % 2**31` is 0 for every rollout (`actionproof/stage3/rollout.py`, line 57). With seed 0 every training episode sees the same id sequence, and a duplicate delivery draws one extra id, so later acknowledgements carry different ids than in a clean episode: a training-only cue. Evaluation uses per-task seeds, and no evaluated refund named an id the agent had not seen: 0 of 1,574 refund calls across 13 models on `s3_conf_full` and `s3_conf_rec` (`results/s3/conf/*.jsonl`, this review). No validity effect found; keep the disclosure.
3. **Test splits are 28 cells, not 300 tasks.** `s3_conf_full` has 28 cells and `s3_conf_rec` 47; RSC seed 0 has a constant outcome within 24 of the 28 full-episode cells (this review). Task-level bootstrap intervals therefore overstate precision. A cell-clustered bootstrap gives, for RSC minus control (seeds 0 to 2), +0.081 [-0.023, 0.209] on full episodes and +0.086 [0.007, 0.175] on recovery starts; for RSC minus A12 R_V (5 seeds), +0.209 [0.063, 0.353] and +0.206 [0.098, 0.322] (this review). The registered task intervals were [0.054, 0.110] and [0.060, 0.111] for the control contrast (`results/s3/s3_addendum.json`, as reported in `docs/stage3/ACADEMIC_REVIEW_POST.md` §2.4).
4. **The verifier is the reward.** R_V is computed by the same `truth.py` that scores VSC. The second verifier and golden traces were written by separate language-model agents from the same specification (`docs/stage2/BENCHMARK_CARD.md`, line 35): a consistency check. No human audit exists for ProcureSim; H4 audited the e-commerce environment (`paper/appendix.tex`, Appendix "Human audit").
5. **Reproducibility gaps.** S3 rollout logs are not in the repository (`results/s3/rollouts/` does not exist), so Part C of addendum 2 cannot be reproduced from a clone. ID1 and SR1 train on the A100 lane while their references trained on H200 (both registrations disclose this).
6. **The public website is the Stage 1 build.** `demo/static/index.html` states "built from commit bea455c" (2026-11-15 21:53 IST), never mentions ProcureSim or procurement, and mentions refunds 342 times. Its last edit is `3d89b7d` the same evening. The README deliverables table still points judges to it as the "interactive write-up". The Stage 2 product (`product/`) is local only (`docs/research/FINAL_REPORT.md` §7).

### Reviewer 5: hackathon judge

1. **Is there a better small Cohere model?** Inside this simulator, yes: on fresh full episodes the warm start scores VSC 0.363 with false success 0.633, and RSC seed 0 scores 0.830 with false success 0.083 (`results/s3/s3_seedlevel.json`). That is a convincing before/after. But no other Cohere model, prompted or larger, was run on ProcureSim (Reviewer 2, item 4), and the oracle ceiling is about 0.92 (`docs/stage3/ACADEMIC_REVIEW_POST.md` §2.10). The judge cannot tell whether a one-line prompt would close most of the gap.
2. **Toolkit use.** AgentTune is used substantively (the GRPO trainer wrapper and rollout parser, `actionproof/rollout/`). AuditKIT appears in the Stage 1 judge audit (an LLM judge with the receipt ledger still accepted 213 of 298 false completions, `claims/claims.yaml`, `grounded_far`). CuratorKIT is used only to deduplicate Stage 1 acknowledgement texts (`actionproof/bench/naturalistic_acks.py`). CircuitKIT failed to install (D-A16-1). In Stage 2 work, only AgentTune is load-bearing.
3. **One memorable thesis exists:** "Acknowledged is not done." The cleanest demonstration of it (a judge fooled by acknowledgements; an agent trained on acknowledgements never checks) should lead. The current story has 13 registered sub-studies; a judge will remember one.
4. **What a judge can see is the wrong build** (Reviewer 4, item 6).

---

## 3. Decision records per contribution

Each record: claim, strongest support, strongest objection, closest prior work, the most decisive falsification, confidence, verdict.

### 3.1 A12: reward observability

- **Claim** (`paper/main.tex` abstract item 1, §4): a state reward made read-back after phantom writes the default; an acknowledgement reward of identical binary form left the warm start unchanged.
- **Support.** Phantom read-back 1.00 (R_V) vs 0.00 (R_A) in each of 5 seeds, 153 of 153 tasks vs 0 of 153; VSC +0.313, +0.323, +0.313, +0.277, +0.273; sign-flip p = 1/32 (`results/s2/a12_summary.json`).
- **Objection.** The two arms differ in incentive and in gradient, not only in observation. Reading never raises R_A (truth table, `PREREGISTRATION_A12.md` §2); R_A is saturated (Reviewer 1, item 4) and its greedy tool calls equal the warm start's on 500 of 500 confirmatory tasks in every seed (`results/s2/a12_identity.json`). The result follows from the truth table plus a warm start that already samples reading (phantom read-back 0.309 at temperature 1, `PREREGISTRATION_A12_ADDENDUM1.md`).
- **Closest prior work.** Yue et al. 2025 (`yue2025rlreasoning`: reinforcement learning with verifiable rewards mostly re-weights what the base model samples); proxy-state rewards (`chuang2026proxystate`); the authors' Stage 1 result in the e-commerce environment (`paper/appendix.tex`, Appendix "Stage 1 results").
- **Decisive falsification.** Dissociate observation from incentive: train R_V restricted to the lost-acknowledgement and timeout-before-commit classes, where the truth table gives reading zero advantage under R_V. Same observation as R_V, no incentive. If read-back still rises, observation matters beyond incentive; if it does not, the claim is "incentive", and "observability" should go from the title. Two seeds, about 4 GPU-hours.
- **Confidence.** High for the behaviour; low for "what the reward observes decides".
- **Verdict: narrow.** Wording: "a reward that depends on hidden state that one free read reveals, starting from a warm start that samples that read, made the read unconditional; an acknowledgement reward that never pays for reading left the warm start unchanged."

### 3.2 A14 and A15: evidence use

- **Claim** (`paper/main.tex` §5): every model acts on present or absent payments; almost only R_V agents refund partial payments, at unseen ratios, mostly when payment is the first step.
- **Support.** Branch-balanced accuracy 1.00 for every policy including the warm start (`results/a14/a14_summary.json`). Unseen-ratio accuracy: R_V seeds 0.405, 0.58, 0.385, 0.247, 0.0; warm start 0.035; R_A 0.0 (`results/a15/a15_summary.json`). By workflow, R_V seeds 0 to 2 repair 0.99 to 1.00 in pay-first, 0 of 168 after a goods receipt, 0.05 to 0.55 after an order (`results/s2/partial_by_workflow.json`).
- **Objection.** The "ratio generalisation" is a workflow effect. The trained ratio is never repaired more often than the unseen ones (pattern gap -0.055 to 0.0 per seed, `a15_summary.json`, `across_seeds`), and unseen accuracy tracks the pay-first share of tasks. The forced prefix is written in the oracle's style (A14 §2), so the context is not the agent's own.
- **Closest prior work.** "Did It Happen?" (paired hidden-commit evaluation, cited by URL in the paper); Proof-of-Use (`ma2026proofofuse`).
- **Decisive falsification.** S3 addendum 2 Part B (running): if receive-then-pay repair stays at most 0.15 after rebalancing, the evidence context, not frequency, binds the behaviour. Add a forced-evidence run where the prefix is the agent's own sampled history (§4), to rule out oracle-style context.
- **Confidence.** High for the present-or-absent result and the workflow binding; medium for its explanation.
- **Verdict: narrow** to "amount-based repair was learned within the pay-first workflow."

### 3.3 A16: representation

- **Claim** (`paper/main.tex` §6): every model encodes partial vs complete; training changes how that evidence moves the next action; the ablation meets the causal rule in 5 of 5 R_V seeds but is not R_V-specific.
- **Support.** Probe AUROC 1.00 everywhere; margin gap at step 200: R_V 2.91, 3.91, 5.09, R_A 1.97 to 2.12, warm start 2.38 (`results/a16/a16_summary.json`).
- **Objection.** Reviewer 3, items 2 to 4: the ablation rule is met by a model with no repair behaviour (R_V seed 4), patching is no better than a wrong-task donor, and most of the margin growth in one seed is on the complete branch.
- **Closest prior work.** Control tasks for probes (Hewitt and Liang, 2019); amnesic probing (Elazar et al., 2021); nullspace projection (Ravfogel et al., 2020); difference-of-means directions (Marks and Tegmark, 2023; Arditi et al., 2024); activation-patching practice (Heimersheim and Nanda, 2024).
- **Decisive falsification.** Matched controls: ablate difference-of-means directions for irrelevant contrasts (workflow label, a random split of tasks, purchase-order id parity) with matched projected variance; stratify by workflow; and read out greedy behaviour, not only the margin. A specific mechanism predicts an effect only in repair-learning seeds and only in pay-first tasks. CPU-light, under 2 GPU-hours.
- **Confidence.** Low for any causal statement.
- **Verdict: narrow to descriptive**: "evidence is decodable in every model; training changes the action margin." Drop "meeting our causal criterion" from the main text.

### 3.4 A17 and S3 transfer to sibling domains

- **Claim** (`paper/main.tex` §7, "Sibling domains"): ProcureSim training makes healthcare administration worse; IT operations unchanged; R_V reads back more than R_A in healthcare administration, mainly because R_A reads less than the warm start.
- **Support.** Healthcare administration VSC 0.158 (warm start), 0.080 (R_V mean), 0.047 (RSC mean), RSC below R_V in 5 of 5 seed pairs; collateral 0.18, 0.27, 0.33 (`results/s3/s3_review_numbers.json`). R_V minus R_A read-back +0.37 [0.18, 0.58]; R_V minus warm start +0.15 [-0.06, 0.39] (`results/a17/a17_seedavg.json`).
- **Objection.** Both domains are written by the same team on the same code pattern, and every model is at a floor (VSC at most 0.22). A drop of 0.11 at a floor is a warning, not a measured transfer law.
- **Closest prior work.** Overfitting to training environments in reinforcement learning (Cobbe et al., 2019, Procgen; Kirk et al., 2023 survey); "SFT memorizes, RL generalizes" (Chu et al., 2025), which this result contradicts for tool procedures.
- **Decisive falsification.** Zero-shot evaluation on an externally written stateful benchmark (for example tau-bench retail with phantom and partial writes injected at its write layer). If R_V and RSC still fall below the warm start there, the negative transfer is general.
- **Confidence.** Medium.
- **Verdict: keep** as a negative result, scoped to sibling simulators.

### 3.5 S3: recipe gain and its decomposition

- **Claim** (`paper/main.tex` §7): RSC raises VSC by +0.21 over R_V in 5 of 5 runs; 0.12 is duplicate delivery; against the class-matched control, recovery starts help only partial-payment repair (3 seeds, suggestive).
- **Support.** RSC minus R_V: +0.206 [0.168, 0.247] on recovery starts, every RSC run above every R_V run, unpaired p = 0.004 (`results/s3/s3_seedlevel.json`). Duplicate delivery contributes +0.112 to +0.116 and partial payments +0.093 (`docs/stage3/ACADEMIC_REVIEW_POST.md` §2.3, from `results/s3/conf/*.summary.json`). Against the control: per-seed +0.020, +0.093, +0.143 on recovery starts (`results/s3/s3_addendum2.json`, `registered_3_seeds`, a recomputation of addendum 1).
- **Objection.**
  1. *Seed lottery.* Partial-class VSC per run: control 0.797, 0.156, 0.125; RSC 0.781 to 0.875 (`s3_addendum2.json`, `partial_vsc_full_per_run`). Learned 5 of 5 vs 1 of 3 has a one-sided Fisher exact p of 0.107 (this review).
  2. *Clustered uncertainty.* The full-episode start-state contrast is +0.081 [-0.023, 0.209] once cells are resampled (Reviewer 4, item 3).
  3. *Bundled factors.* See §4.
  4. *Evaluation favours RSC on recovery starts.* H3 is measured on scripted-prefix starts, a format only RSC trained on. Only full-episode tests are fair to both arms.
- **Closest prior work.** Restart distributions (Kakade and Langford, 2002); reverse curricula (Florensa et al., 2017); starting from demonstration states (Salimans and Chen, 2018); in the GRPO setting, Fission-GRPO, R3, ReflectPO and Depth Before Breadth (`paper/refs.bib`); curriculum-induced belief shift, SAMPLR (`jiang2022samplr`).
- **Decisive falsification.** The factorial in §4.
- **Confidence.** High for the recipe gain over R_V; low for the start-state part.
- **Verdict: keep** the recipe result (report cell-clustered intervals next to task intervals); **investigate** the start-state part and suspend the claim until §4 or addendum 2 resolves it.

### 3.6 S4: registered shifts

- **Claim** (`paper/main.tex` §7): the RSC gain survives Hindi, Hinglish and two-fault episodes, and vanishes under renamed tools and in an unseen workflow.
- **Support.** Hindi +0.127 [0.080, 0.176], Hinglish +0.072 [0.040, 0.109], two faults +0.072 [0.032, 0.118], Holm p = 0.016 each; renamed API -0.016 [-0.044, 0.011] (`results/s4/s4_summary.json`; `docs/research/FINAL_REPORT.md` §4).
- **Objection.**
  1. What "survives" is fault coverage. On Hindi, +0.077 of the gain is duplicate delivery and +0.049 partial payments, the same split as English (+0.076, +0.047); two-fault gains come from pairs containing a partial or a duplicate (`results/s4/eval/*.summary.json`, this review).
  2. The language shift touches only the user request; system prompt, tool names and tool outputs stay English. Hindi costs RSC 0.021 VSC and the warm start -0.013 (`results/s4/s4_diagnostics.json`, `language_cost_per_arm`): barely a shift.
  3. Two sentences are wrong at model level (§6, items 1 and 2).
  4. On the renamed API, the control (0.293) beats RSC (0.196) (`s4_diagnostics.json`, `vsc_per_arm`).
- **Closest prior work.** Robustness of tool learning to noisy or renamed tools (RoTBench, Ye et al., 2024); compositional generalisation (SCAN, Lake and Baroni, 2018).
- **Decisive falsification.** SR1 (registered) for the API surface. For the workflow, training with randomised step orders. For language, translate tool outputs and the system prompt too.
- **Confidence.** High for the API and workflow failures; medium for what the survivals mean.
- **Verdict: narrow.** "Coverage gains were not destroyed by a change of request language or by a second fault."

### 3.7 FCA: forked counterfactual action credit

- **Claim** (`paper/main.tex` §7; `paper/appendix.tex`, Appendix "Forked counterfactual credit"): forced branches, weighted by TRL's sequence-level IS correction, give the all-action gradient; FCA failed its kill criterion.
- **Support.** Development VSC on recovery starts -0.097 against RSC (0.640 and 0.780 vs 0.807 and 0.807) (`results/s4/fca_dev_summary.json`).
- **Objection.**
  1. *Mathematics.* In expectation the forced term equals what on-policy sampling already estimates: the gradient of an action's probability times its advantage. That term is proportional to the action's probability. It lowers variance; it cannot add signal for a repair the policy rarely takes, which is the problem `FCA_GRPO.md` §3 says FCA targets.
  2. *Weights.* The IS weight is the probability of the exact canonical token string, not of the action class. The largest gap between trainer and sampler log-probabilities averaged 17.6 and 17.3 nats per step and exceeded 10 nats in 184 and 180 of 200 steps (FCA seeds 0 and 1), against at most 2.45 for RSC (`results/s4/train_state/*.json`, `sampling/sampling_logp_difference/max`). At least one forced branch per step was therefore weighted by under e^-10. The mean IS weight fell to 0.79 (FCA) from 0.96 to 0.99 (RSC) (same files).
  3. *Little unique signal.* Groups where every on-policy rollout had the same reward but some forced class beat the on-policy mean: 35 of 382 (seed 0) and 24 of 382 (seed 1) (`results/s4/train_state/s4_fca_s{0,1}.fca_groups.jsonl`, this review).
  4. *Confound.* FCA replaced 4 of 8 on-policy rollouts at decision states, so the most likely cause of the loss is half the on-policy samples. Under R_V, "escalate then continue" is neither terminal nor penalised, so it beat the on-policy mean in 78 and 74 groups (same files).
- **Closest prior work.** COMA (`foerster2017coma`); expected policy gradients (Ciosek and Whiteson, 2018); mean actor-critic (Allen et al., 2017); VinePPO (Kazemnejad et al., 2024); GiGPO, BiPACE, CRR, SURE-Act (`paper/refs.bib`); bias of state-conditioned critics for history-conditioned policies (Baisero and Amato, cited in `docs/research/NOVELTY_AUDIT.md`).
- **Decisive falsification.** Add forced branches on top of 8 on-policy rollouts (do not substitute), weight forced tokens by the teacher-forced class probability or by 1, and log the per-branch IS weight. Low priority.
- **Confidence.** High that the negative result says nothing about counterfactual credit in general.
- **Verdict: narrow.** "An implementation in which forced branches received near-zero weight and halved on-policy samples did not help." Correct the appendix derivation.

### 3.8 ID1 (running)

- **Question.** Does R_V create read-back from a warm start that never demonstrates it (`PREREGISTRATION_ID1.md`)?
- **Strongest objection to the registered reading.** "Both seeds at most 0.10 means selection" conflates "cannot be created" with "not found in 200 steps of 32 rollouts by a policy with token entropy near 0.03". The no-verification warm start also loses every demonstrated repair, so any evidence-use comparison with A12 is confounded. Hardware differs (A100 vs H200).
- **Closest prior work.** Yue et al. 2025 (`yue2025rlreasoning`).
- **Decisive add-on (no new training).** Report the new warm start's sampled read rate after a write (the gate output of task 980) and the read rate per 50-step block in the training rollouts. If sampled reads are 0, a null is uninformative about discovery; if they are above 0 but do not grow, it is evidence for selection by the warm start's repertoire.
- **Confidence.** Not yet known.
- **Verdict: investigate.** Before reading results, file a dated note that a null will be worded "not discovered within this budget".

### 3.9 S3 addendum 2 (running)

- **Part A.** With control seeds 0 to 2 known, H4′p (partial class) can pass only if both new control seeds fail partial repair. If both score about 0.1, the unpaired p is 0.016; if one learns (about 0.8), it is 0.056 and fails before Holm (this review, from `partial_vsc_full_per_run`). Full-episode runs learned partial repair in 4 of 8 so far (`results/s3/s3_review_numbers.json`), so failure of H4′p is the more likely outcome, and either way two bistable runs decide it. Part A cannot separate mechanisms.
- **Part B.** Well designed. The interpretation table is fixed and the scripted calibration is clean (`results/s3/scripted_pp_calibration.json`).
- **Verdict: investigate**, and report Part A with the cell-clustered interval and the Fisher test on learned counts next to the registered statistic.

### 3.10 S3 addendum 3 (registered 2026-11-22 01:16; not requested, but it decides the S3 confound)

The STATE arm replays the scripted prefix into the simulator but hides it from the conversation (`actionproof/stage4/train_variants.py`, `hide_prefix`). This is **not a semantically valid start-state manipulation**:
1. **Keyed writes are answered by the idempotency store.** The scripted prefix uses `pay-<po>` and `order-<po>` keys (155 and 67 prefix calls in `data/bench/proc/s3_rec_train_half.jsonl`). In the four models checked (warm start, A12 R_V seed 0, RSC seed 0, control seed 1), the first payment uses the same key in 295 to 299 of 300 episodes and the first order in 100 of 100 (`results/s3/conf/*__s3_conf_full.jsonl`, this review). A hidden committed write (partial, duplicate, lost acknowledgement, complete) is therefore replayed to the policy as an ok acknowledgement with no new commit. The policy then reads, and meets the same decision as in a full episode. For partial and duplicate faults, STATE is almost the control. Hidden phantom and timeout-before-commit writes were never stored, so the policy's own write commits and the fault disappears.
2. **Unkeyed receipts become traps.** All 60 receive-then-pay recovery starts (30% of recovery starts, 14 of them with a partial fault) contain a goods receipt in the hidden prefix. The same four models record receipts without a key (93 of 93 first receipt calls each, same files), so the natural action, correct in the agent's context, doubles the received quantity. The verifier then marks the step as a duplicate (`actionproof/integrity/sim/truth.py`, receive branch), and R_V is 0 whatever the agent does about the payment.

Consequence: C1 (STATE minus control) is predicted near 0 by the mechanics, and C2 (RSC minus STATE) will absorb every bundled factor in §4. The registered reading "the scripted history is the active ingredient" would be unjustified. **Before any addendum 3 output is read:** file a dated note with these mechanics, log per rollout the idempotency-store hits on the first write and any duplicate receipt, analyse STATE only on the order-then-pay and pay-first workflows, and rename the arm "amnesic restart".

---

## 4. The S3 confound: where does the recovery-start gain come from?

### 4.1 What the evidence already settles

| Candidate source | Status | Evidence |
|---|---|---|
| Fault coverage | **Settled.** It explains the duplicate-delivery half of RSC minus R_V | Duplicates +0.112 to +0.116 of +0.21; the control matches or beats RSC on duplicates (`docs/stage3/ACADEMIC_REVIEW_POST.md` §2.3) |
| Turn budget (prefix turns not counted, `actionproof/stage3/rollout.py`) | **Not binding at evaluation** | In the seven models checked (RSC and control seeds 0 to 2, A12 R_V seed 0), every partial-class episode ends by report, at most 12 of 16 turns (`results/s3/conf/*__s3_conf_full.jsonl`, this review). In training it is unmeasured |
| Start position (the 8 rollouts of a group share an identical decision context; no loss on the prefix) | **Open** | Bundled with the next three rows |
| Scripted history in context (oracle-written keys, arguments and reads) | **Open** | The prefix is oracle-written (`paper/appendix.tex`, Appendix "Recovery-start curriculum") |
| Evidence handed in context (read starts include the authoritative record) | **Open, plausible** | 31 of the 54 partial-fault recovery starts are read starts that already show the partial record (`data/bench/proc/s3_rec_train_half.jsonl`). Early in training, a full-episode rollout sees that record only if the policy chooses to read |
| Recovery-decision frequency | **Matched by design, unmeasured in practice** | The pools carry the same faults (addendum 1). How often full-episode rollouts actually reach and read at the faulted payment is not logged |
| Seed lottery | **Not excluded** | Fisher p = 0.107 for 5 of 5 vs 1 of 3 (§3.5) |

### 4.2 Requirements for a semantically valid manipulation

1. The conversation must be a history this agent could have produced in this world, and the simulator state must be exactly what that history implies, including the idempotency store and attempt counters.
2. The hidden fault must stay unobservable in acknowledgements.
3. Only one factor changes between neighbouring arms; coverage (the class-matched pool), steps, rollouts, reward and code path stay fixed.
4. The primary test must use full episodes, which no arm has a format advantage on.

STATE (§3.10) fails requirement 1. A **self-history start** meets it: sample the current policy from the start of the episode until the faulted write's acknowledgement, rebuild the simulator by replaying those calls (exact, because the simulator is deterministic and the training seed is constant), then branch 8 rollouts from there. The prefix is the agent's own, true, and in its own style.

### 4.3 Factorial (all arms on the class-matched fault pool, R_V, the Stage 3 code path, 200 steps)

| Arm | Group starts at the decision | History author | Authoritative record in the start history | Status |
|---|---|---|---|---|
| FULL (control) | no | the policy | only if the policy reads | seeds 0 to 2 done; 3 to 4 running (addendum 2) |
| SELF (new) | yes | the policy (sampled, then branched) | only if the policy read before the branch point | new, seeds 0 to 4 |
| HIST (RSC) | yes | oracle script | in read starts (31 of 54 partial-fault starts) | seeds 0 to 4 done |
| HIST-ACK (new) | yes | oracle script | never (every read start converted to the ack start of the same task) | new, seeds 0 to 4 |

Fixed across new arms: no loss on prefix tokens; the 16-turn budget counts from the decision; per-step logs of decision visits, partial-fault groups with non-zero reward variance, and whether the payment record was in context at the first decision.

| Contrast | Isolates |
|---|---|
| SELF minus FULL | starting the group at the decision (with the agent's own history) |
| HIST minus SELF | oracle-written history instead of the agent's own |
| HIST minus HIST-ACK | evidence handed in the start history |
| logged visit counts, all arms | recovery-decision frequency (descriptive) |

Residual not separated: within SELF minus FULL, the shared-context group baseline and the removal of loss on pre-decision tokens stay bundled. An extra arm (SELF with the group advantage also applied to the prefix tokens) would separate them, if a lane is free.

---

## 5. The five most damaging problems, ranked

1. **The headline A12 contrast is close to true by construction.** Reading never pays under R_A, R_A is saturated (zero-variance 0.68 to 0.86, mean reward 0.95 to 0.98, `results/s2/stop_rule_seeds.json`), and the R_A agent is the untrained warm start on 500 of 500 tasks (`results/s2/a12_identity.json`). The title's "reward observability" is not identified; K1 cannot identify it because its pool stays saturated (`results/k1/k1_reward_signal.json`). Fix: the wording in §3.1 and the incentive-dissociation run.
2. **The S3 start-state decomposition, the only training-method finding, is unresolved, and the registered fix cannot resolve it.** Learned partial repair 5 of 5 vs 1 of 3 (Fisher p = 0.107, this review); the cell-clustered interval on full episodes includes 0 (+0.081 [-0.023, 0.209], this review); start position, oracle history and handed evidence are bundled; addendum 3's STATE arm reduces to the control for partial payments and turns all 60 receive-then-pay starts (30% of recovery starts) into traps (§3.10). Fix: §6 below.
3. **Construct validity.** One authored simulator; 28 effective cells in the full-episode test (this review); free authoritative reads, so "verification" is one unconditional read (2.70 reads per clean episode, `results/s3/s3_review_numbers.json`); a deterministic runtime check already recovers every phantom write (`results/a11/h7_matrix.md` via `paper/appendix.tex`); no prompted, larger-model or SFT-on-recoveries baseline; learned behaviour fails under renamed tools, a new step and a sibling domain (`results/s4/s4_summary.json`, `results/s3/s3_review_numbers.json`).
4. **Both "method" sections would be taken apart by a specialist.** A16's causal rule is met by a model that never repairs and by R_A seeds, and patching is no better than a wrong-task donor (`results/a16/a16_summary.json`). FCA's appendix derivation does not hold in the run: forced branches were weighted by under e^-10 in about 90% of steps (`results/s4/train_state/s4_fca_s{0,1}.json`).
5. **The judged deliverable shows the wrong project.** The public site is the Stage 1 e-commerce build (`demo/static/index.html`, "built from commit bea455c"); the Stage 2 product is local only; in Stage 2 only AgentTune is load-bearing (AuditKIT and CuratorKIT are Stage 1 uses, CircuitKIT failed); no other Cohere model was run on ProcureSim.

---

## 6. Highest-value experiment for about 36 hours (2 H200, 1 A100)

**Experiment: self-history recovery starts (addendum 4).** Train SELF and HIST-ACK (§4.3), 5 seeds each, and compare them with the existing FULL and HIST runs.

**Why this one.** It decides the paper's only claim about how to train (problem 2), repairs a registered design that would otherwise produce a misleading answer (§3.10), reuses existing code (deterministic replay as in `actionproof/stage4/forks.py`, the Stage 3 rollout and trainer) and existing frozen splits, and fits the budget with room for the queued runs.

**Registration (before any code runs on a GPU).**
- Arms, contrasts and the residual as in §4.3.
- **Primary:** partial-payment VSC on `s3_pp_full` (150 full episodes, 50 per workflow; hashed by addendum 2), per run.
- **Secondary:** `s3_conf_full` partial (64) and duplicate (46) classes; `s3_conf_rec`; `s3_pp_rec`; learned repair (partial VSC at least 0.40) per run.
- **Statistic:** one-sided unpaired seed permutation, 5 vs 5 (252 assignments), Holm over the three contrasts; supported only if, in addition, the cell-clustered 95% interval excludes 0. Report Fisher exact on learned counts.
- **Power, stated in advance:** 5 of 5 vs 1 of 5 learners gives Fisher p = 0.024; 5 of 5 vs 2 of 5 gives 0.083 (this review). Only large differences are detectable; say so.

| Result | Reading fixed in advance |
|---|---|
| SELF about equal to HIST, and SELF above FULL | Starting the group at the decision is the active ingredient; the start-state claim stands |
| SELF about equal to FULL, HIST above SELF | The oracle-written history is the active ingredient: RSC acts as in-context demonstration, not as a start state |
| HIST above HIST-ACK | Handing the authoritative record in the start history matters: RSC acts as a forced-evidence curriculum |
| No contrast supported, learners spread across arms | The 3-seed start-state effect was a seed lottery; retract it |

**Schedule.**

| Hours | H200 lane 1 | H200 lane 2 | A100 | No GPU |
|---|---|---|---|---|
| 0 to 4 | queued addendum 2 Part A | queued addendum 2 Part A or B | ID1 | write addendum 4 and the §3.10 note; implement SELF (two-phase rollout, replay check, prefix mask, turn budget from decision, logging); unit tests; 5-step smoke run |
| 4 to 16 | SELF seeds 0 to 2, HIST-ACK seed 0 | HIST-ACK seeds 1 to 4 | ID1, then evaluations | zero-variance analysis of existing logs (addendum 2 Part C) |
| 16 to 20 | SELF seeds 3 to 4 | addendum 2 Part B or addendum 3 | evaluations: each new model once on the four splits | |
| 20 to 26 | spare | spare | spare | statistics, cell-clustered intervals, ledger, paper |
| 26 to 36 | buffer for crashes and queue waits | | | |

Cost: 10 runs at about 2.2 hours (prefix sampling adds about 10%), about 22 H200-hours, plus under 2 hours of evaluation. With addendum 2 (6 runs), addendum 3 (5), SR1 (3) and ID1 (2) also queued, total demand is about 57 GPU-hours against about 108 available. If the queue stalls, drop in this order: SR1, addendum 3, HIST-ACK seeds 3 to 4.

---

## 7. Wording errors found in this review (not covered by `docs/CONSISTENCY_AUDIT.md`)

| # | Where | Problem | Evidence | Correct wording |
|---|---|---|---|---|
| 1 | `paper/main.tex` line 105; `docs/research/FINAL_REPORT.md` §4 | "every trained model scores below the warm start (VSC 0.03 to 0.11 vs 0.12)" is false per model | `s3_matched_s1` 0.27 and `s3_rv_mix_s4` 0.14 vs warm start 0.12 (`results/s4/eval/*__s4_wf4.summary.json`); 0.03 to 0.11 are arm means (`results/s4/s4_diagnostics.json`) | "every trained arm's mean is below the warm start (0.03 to 0.11 vs 0.12); two of 13 trained runs score above it" |
| 2 | `paper/main.tex` line 105; FINAL_REPORT §4 | "call training-time tool names in 13 to 35% of episodes" are arm means | per run 0.07 to 0.47 (`s4_diagnostics.json`, `per_model`) | "13 to 35% on average per arm (7 to 47% per run)" |
| 3 | `paper/appendix.tex`, Appendix "Forked counterfactual credit" | "weights a forced branch by π_old(a), giving the all-action gradient term" | §3.7, item 2 | "by the probability of the exact forced text, which was below e^-10 for at least one branch in about 90% of steps" |
| 4 | `paper/main.tex` abstract item 3 | "every model linearly encodes ... and training changes how that evidence moves the next action" listed as a finding | §3.3 | move to "descriptive"; state the R_V seed 4 counterexample |
| 5 | `PREREGISTRATION_S3_ADDENDUM3.md` §2 (frozen; correct by dated note) | "the conversation stays truthful" and "a world where some requested steps already happened" | §3.10 | "the idempotency store replays hidden keyed writes; hidden unkeyed receipts are duplicated by the agent's natural action" |
| 6 | `README.md` deliverables table | presents the Stage 1 site as the interactive write-up | Reviewer 4, item 6 | label it "Stage 1 site" and link the Stage 2 product or a recording |

---

## Appendix: reanalysis methods

All read-only; Python 3 standard library.

1. **Cell-clustered bootstrap** (Reviewer 4, item 3). For each model and split, read `results/s3/conf/<model>__<split>.jsonl`. Cell = (`fault_class`, `task.workflow`, `task.meta.faults[0].tool`, `start_type`). Statistic: seed-averaged VSC difference over the selected tasks. Resample cells with replacement, 2,000 draws, random seed 0; 2.5th and 97.5th percentiles. Arms: `s3_rv_mix` seeds 0 to 4 vs `a12_rv` seeds 0 to 4; `s3_rv_mix` seeds 0 to 2 vs `s3_matched` seeds 0 to 2.
2. **Unseen refund ids** (Reviewer 4, item 2). For every agent `refund_payment` call in `results/s3/conf/*__s3_conf_{full,rec}.jsonl`, check whether `payment_id` appears in an earlier observation or in the start prefix.
3. **Turn-limit check** (§4.1). `ended_by` and `turns` for `fault_class == "partial"` in `results/s3/conf/{s3_rv_mix_s0..s2,s3_matched_s0..s2,a12_rv_s0}__s3_conf_full.jsonl`.
4. **FCA weights and signal** (§3.7). `log_history` in `results/s4/train_state/{s3_rv_mix_s0,s3_rv_mix_s1,s4_fca_s0,s4_fca_s1}.json`: steps with `sampling/sampling_logp_difference/max` above 10; means of `sampling/importance_sampling_ratio/mean`, `entropy`, `frac_reward_zero_std`, `clip_ratio/region_mean`. In `*.fca_groups.jsonl`, count groups with `on_zero_var` true and some `Q[a] > V`, and per class the groups with `Q[a] > V`.
5. **S4 class decomposition** (§3.6). For each split, sum over classes of `n x vsc / N` from `by_class` in `results/s4/eval/<model>__<split>.summary.json`, RSC minus A12 R_V, averaged over seeds 0 to 4.
6. **Keys and receipts** (§3.10). In `data/bench/proc/s3_rec_train_half.jsonl`, count `meta.start_prefix` keys with the purchase-order id masked, and prefixes containing `record_goods_receipt`. In `results/s3/conf/*__s3_conf_full.jsonl`, the key of each model's first `pay_supplier`, `place_supplier_order` and `record_goods_receipt` call.
7. **Seed-level tests** (§3.5, §3.9, §6). Exact enumeration of all 252 assignments of ten per-run values; hypergeometric Fisher exact test on learned counts.
8. **A16 controls** (Reviewer 3). Fields `causal.<model>.ablation.{dM_partial,dM_Q0,max_random_abs_dM_partial}`, `median_final_c2p`, `wrong_task_median`, `median_span_c2p`, and `trajectory.<model>.M_by_branch` in `results/a16/a16_summary.json`; behaviour from `results/a15/a15_summary.json`.
