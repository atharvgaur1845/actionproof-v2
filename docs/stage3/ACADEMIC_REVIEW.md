# Mock EMNLP main-track review: ActionProof

Reviewer role: senior reviewer, area LLM agents, RL for language models, reliability and safety of tool use.
Date: 2026-11-21. Review type: strict, pre-submission.

**What was read.** `paper/main.tex`, `paper/appendix.tex`, `paper/tables/*.tex`, `paper/refs.bib`; `PREREGISTRATION_A12.md` with addenda 1 and 2, `PREREGISTRATION_A14.md`, `_A15.md`, `_A17.md`, `_K1.md`, `_S3.md`; `docs/stage2/DECISIONS.md`; `claims/claims.yaml`; `docs/stage3/RESEARCH_REPORT.md`, `docs/stage3/LITERATURE.md`; `results/s2/a12_summary.json`, `results/a14/a14_summary.json`, `results/a15/a15_summary.json`, `results/a16/a16_summary.json`, `results/a17/a17_summary.json`, `results/a17/a17_seedavg.json`, `results/eval/k1/conf/*proc_conf.summary.json`, `results/s3/dev/*.summary.json`. Supporting files consulted: `results/s2/stop_rule_seeds.json`, `results/s2/reward_signal_summary.json`, `results/k1/k1_reward_signal.json`, `results/s3/scripted_conf_calibration.json`, `docs/stage2/HF_CREDITS.md`, the split files in `data/bench/proc/`, and `actionproof/stage3/tasks.py`.

**Reviewer reanalyses.** Several findings below come from read-only regrouping of the repository's own per-episode files (A12, A14, A15, S3 development). The method is described in the Appendix at the end so the authors can reproduce each number. No code or data in the repository was changed.

**Scope note.** The paper draft predates A15, A16, A17, K1 and S3. The review assesses the draft plus the registered evidence now in the repository, as a reviewer would if these results were added in revision. S3 confirmatory results are not available; Section 3.3 reviews its design and development evidence and states what result would or would not make it a contribution.

---

## 1. Summary of the claimed contributions

The project studies whether the information channel of an RL reward decides whether a tool-using LLM agent checks the effects of its own writes.

1. **Matched reward comparison (A12, registered).** A 3.35B Tiny Aya Global policy (LoRA) is warm-started by SFT on scripted demonstrations (30% from a verifying oracle, 70% from an acknowledgement-trusting policy) and then trained with GRPO under two binary, report-blind rewards: R_V (every requested step verified in the final authoritative state) and R_A (the last acknowledgement of every requested write was `ok`). The environment, ProcureSim, returns byte-identical success replies for committed, phantom and partial writes. On 300 untouched confirmatory tasks, R_V gives phantom read-back 1.00 and R_A 0.00 in each of five paired seeds (153/153 vs 0/153 tasks), and verified completion (VSC) rises by +0.27 to +0.32. The registered mechanism test (H3) fails, so the registered composite claim C1 is not made.
2. **"Selection, not invention."** The warm start samples read-back (0.35 at the gate) but never reads at greedy decoding. Every R_A model issues exactly the warm start's tool calls on all 500 confirmatory tasks; R_V makes read-back the default. An exploratory reward-signal analysis locates where each reward supplies gradient.
3. **Forced-evidence forks (A14, registered).** For 250 tasks the real payment is placed in one of five execution states behind identical acknowledgements, and the authoritative read is forced into context. Every policy, trained or not, re-pays when the read shows no payment and proceeds when it shows one (BCA 1.00). Only R_V refunds a partial payment (0.35 to 0.48). When the same read is attributed to the runtime, R_A falls to BCA 0.83.
4. **Newer results (not yet in the paper).** A15: R_V's partial response is similar at unseen ratios 1/4 and 3/4 in seeds 0 to 3 ("generalises"), absent in seed 4. A16: probes decode partial vs complete everywhere (AUROC 1.00); ablating a dev-fitted direction at layer 18 meets the registered rule in 5/5 R_V seeds; activation patching fails in 0/5. A17: zero-shot transfer to two new simulated domains: care_admin R_V minus R_A +0.37 [0.18, 0.58], R_V minus warm start not established, it_ops null. K1: R_A trained on a pool without the classes that penalise reading, or on a random pool of the same size, stays at 0/153.
5. **Proposed algorithm (S3, registered, running).** "RS-GRPO": the A12 R_V trainer with half the training pool replaced by recovery starts (scripted prefixes that end in a faulted state). H1: superiority in VSC on recovery starts; H2: non-inferiority (margin 0.05) on full episodes.
6. **Process.** Pre-registrations hashed before runs, a dated deviation log, one-shot confirmatory access logs, a claim ledger with an automatic number checker, a reference verifier and golden traces.

---

## 2. Strengths

- **S1. Research hygiene well above the venue norm.** Registrations precede runs and are hashed; every deviation is logged with time and reason (D-A12-1, D-A12-2, D-A13-1, D-K1-1, D-A17-1, D-A16-1); confirmatory splits are accessed once with an access log; failed registered tests are reported as failures (A12 H3, A11 interaction, A16 patching). The A13 pseudoreplication was caught and the p-value withdrawn (D-A13-1). The claim ledger ties each published number to a file and key. Few submissions in this area meet this standard.
- **S2. A clean experimental unit.** Same warm start, trainer, hyperparameters, tasks and rollout budget for both arms; only the reward function differs. The reward-truth table computed before training (`paper/tables/truth.tex`) is a useful device that other reward-design papers should copy.
- **S3. Large, consistent primary effect.** 153/153 vs 0/153 tasks in every seed; per-seed VSC intervals exclude zero (+0.27 to +0.32, `results/s2/a12_summary.json`); the effect holds on unseen fault pairs (0.97 to 0.98 vs 0.00).
- **S4. The forced-evidence fork protocol is a good idea.** It separates acquiring evidence from acting on it, uses scripted references to calibrate the scale, and the runtime-attribution ablation (R_A 0.83) is informative about interface sensitivity.
- **S5. Negative and null results are reported with equal weight.** K1 null, it_ops null, A16 patching null, S3 pilot negatives (safety-first and Lagrangian rewards), and the A12-IDENTITY correction of an earlier wording that implied R_A changed greedy behaviour.
- **S6. Practical relevance.** Proxy-state rewards that update on success replies (chuang2026proxystate) are R_A-like. The work gives a concrete, reproducible warning and a testbed.
- **S7. K1 is a well-designed placebo-controlled follow-up.** It cleanly attributes R_A's sampled drift away from reading to pool composition (placebo read rate 0.13 to 0.06; targeted 0.21 to 0.16, `results/k1/k1_reward_signal.json`).

---

## 3. Weaknesses, ranked by severity

### W1 (Critical). Identification: the R_A arm is a saturated, near-null training arm, and the size of the headline effect is set by the fault mix

**Evidence.**
- R_A is saturated from the start. Mean reward over steps 40 to 50 is 0.974 (seed 0) and 0.977 (seed 1); the zero-variance group share is 0.864 and 0.818, so the registered stop rule fired for seed 0 (D-A12-1) and would have fired for seed 1. R_V: mean reward 0.625, zero-variance share 0.545 (`results/s2/stop_rule_seeds.json`).
- Consequently, at greedy decoding every R_A model issues the warm start's tool calls on 500/500 confirmatory tasks in all five seeds (`results/s2/a12_identity.json`), and six further R_A runs in K1 reproduce the warm start exactly: VSC 0.47, false success 0.53, phantom read-back 0/153 and benign read-back 0/606 (`results/eval/k1/conf/*proc_conf.summary.json`). R_A seeds also give identical VSC (0.470) in every seed.
- At greedy decoding, A12 is therefore a comparison of "R_V-trained" against "untrained". The paper's causal reading ("the reward's information channel decided whether the agent acquired evidence", main.tex line 118) needs an argument that a channel-blind reward which is *not* saturated would also leave read-back at zero. That is plausible from the truth table, but it is not tested.
- K1 does not separate the two accounts. Under the targeted pool, R_A pays nothing for reading in any remaining class, so both the "channel" account and a "no reading gradient anywhere" account predict 0/153; K1 section 4 predicts exactly that. K1 is informative about the *sampled* decline, not about acquisition.
- The VSC effect size is a design parameter. On `proc_conf`, R_V solves clean, phantom, lost-ack and precommit-timeout tasks at 1.00, partial at 0.17 to 0.50, never-commit at 0 (unreachable); the warm start and R_A fail every phantom task (`docs/stage3/RESEARCH_REPORT.md` section 1). The +0.27 to +0.32 VSC gain is approximately the phantom share chosen by the generator (25%) plus a small partial contribution. It does not estimate a natural quantity.
- The paper acknowledges that "part of the phantom-write effect holds by construction" (Limitations, main.tex line 122), but the title and abstract state the general causal claim.

**Why it matters.** The registered H2 result is valid as stated. The interpretation that the *channel* is the operative cause, rather than the presence or absence of any usable gradient on this warm start, is the paper's main scientific claim, and it is not yet identified.

### W2 (Critical). Every "evidence use" result on partial payments is confined to one workflow; task-level intervals overstate precision

ProcureSim's confirmatory and fork splits contain only three workflows: `pay+post`, `order+pay+post`, `receive+pay+post` (`data/bench/proc/proc_conf.jsonl`, `proc_cf.jsonl`, `proc_mech.jsonl`). Regrouping the per-episode records by workflow (method in the Appendix) gives:

| Result | pay+post | order+pay+post | receive+pay+post |
|---|---|---|---|
| A14 partial branch correct, R_V s0 / s1 / s2 | 1.00 / 1.00 / 1.00 (n = 85) | 0.06 / 0.48 / 0.03 (n = 71) | 0.00 / 0.00 / 0.00 (n = 94) |
| A14 partial branch correct, warm start | 0.00 | 0.10 | 0.00 |
| A15 unseen ratios (Q1, Q3), R_V s0 / s1 / s2 / s3 / s4 | 0.99 / 1.00 / 0.99 / 0.63 / 0.00 | 0.11 / 0.60 / 0.06 / 0.04 / 0.00 | 0.00 in every seed |
| A12 `proc_conf` partial-class VSC, R_V s0 to s4 | 1.00, 1.00, 1.00, 0.08, 0.00 (n = 12) | 0.0 to 0.3 (n = 10) | 0.43 in every seed (n = 14) |

Within a seed, the refund decision is a property of the task, not of the ratio: in R_V s0, 74 of 200 A15 tasks refund at all three partial ratios and 114 refund at none; at Q2, refund rate is 0.99 when the forced prefix has 2 calls (pay is the first step) and 0.02 when it has 4 calls (an earlier step was written and read).

**Consequences.**
- "Only the state-trained agents refund a partial payment (35 to 48% of tasks)" (abstract, main.tex line 26; section A14, line 112) averages near-perfect behaviour in one workflow with near-zero behaviour in the other two.
- A15's registered verdict "generalises, not pattern-specific" is technically correct (ratio does not matter), but the behaviour does not generalise across workflow position. The natural reading, that R_V learned to compare the paid amount with the order amount, is not supported for two of three workflows.
- A16's teacher-forced margins and ablation effects average over these strata.
- Task bootstrap intervals in A12, A14 and A15 treat 200 to 300 tasks as independent although behaviour is set by three strata (and 132 to 191 prompt skeletons per split). A17 already uses a template-cluster bootstrap; the ProcureSim analyses do not.

### W3 (Critical for any S3 claim). S3 confounds start states with new fault classes, tests H1 in-distribution for one arm only, and is a curriculum change rather than an algorithm

See Section 3.3 for the full S3 assessment and the decision table.

### W4 (Major). Novelty relative to related work is modest

- **A12.** R_V is final-state grading as in τ-bench (LITERATURE F1), used as an RL reward. R_A resembles proxy-state trackers that update on success replies (chuang2026proxystate). Lang et al. (lang2024deceive) predict that partial-observation rewards pay for the appearance of success; Thaman (thaman2026rhb) finds agents skip verification when it is not rewarded; VeriGUI (zhang2026verigui) trains verification with an effect-reading reward; Zhang (zhang2026leaky) already uses a pre-registered matched two-arm GRPO design for verifier quality. Given the reward-truth table, the A12 outcome is the expected one. The contribution is a careful controlled confirmation, not a surprising finding.
- **The authors' own prior result.** Stage 1 already reported the same direction (sampled refund read-back 0.71 to 0.72 under state rewards vs 0.10 under the acknowledgement reward, claim S1-CHANNEL; A8 interaction +0.678, claim S1-A8). A12 is a replication in a sibling environment from the same codebase. This must be stated (anonymised) in the paper.
- **A14.** Oracle-state interventions (jia2026oraclestate) and Proof-of-Use evidence perturbation (ma2026proofofuse) are close; forced-evidence forks for stateful writes are a reasonable incremental extension.
- **"Selection, not invention"** restates Yue et al. (yue2025rlreasoning) and Gandhi et al. (gandhi2025cognitive) in an agentic setting.
- **S3** is a fixed-mix reset distribution (reverse curriculum C3, Go-Explore C4, R^3 C5, failure-prefix conditioning C7, AgentPRM resets B4, R³L H6). The dropped safety-first reward is TruthRL-like (E2, H11); the Lagrangian pilot scalarised before standardising, the known weakness that Constrained GRPO (D7) fixes.

### W5 (Major). Generality: one model, one warm start, one simulator family, greedy decoding

- One model (Tiny Aya Global 3.35B, LoRA rank 16). The Aya-only rule is a competition rule (`docs/stage2/DECISIONS.md`, 2026-11-19 22:40), not a scientific choice.
- **One warm start shared by every arm and seed** (A12 section 4). The paper's central explanations ("read-back was already in the warm start's sampled repertoire"; "what the agent does with evidence is largely inherited from its demonstrations") are about this single SFT run with a 30% oracle share, and its variance is not sampled at all.
- One environment family built by the same team: ProcureSim and both OpsSim domains share patterns and code (A17 section 2; main.tex line 43).
- Greedy decoding throughout; sampled behaviour differs (warm start 0.35 sampled read-back vs 0.00 greedy).
- Seeds carry little independent information: between-seed SD of read-back is 0.0; RS-GRPO seeds 0 and 1 produce identical tool-call sequences on 146/150 `proc_dev` tasks (Appendix). Five seeds sample optimiser noise only.

### W6 (Major). Statistics

- **Seed-level inference is at its floor and the registered primary family cannot reach 0.05.** With five seeds the exact sign-flip test has minimum p 1/32 one-sided and 2/32 two-sided. Holm over the two primary metrics gives 0.0625 for both (`results/s2/a12_summary.json`, key `holm_sign_flip`). At five seeds, a Holm-corrected two-metric family can never reach 0.05. The main text reports "1/32 one-sided" (main.tex line 77) without the Holm value; the appendix names Holm without the number (appendix.tex line 19).
- **Per-seed McNemar p-values (1.75e-46) and task bootstraps measure evaluation-sample noise for one fixed trained model.** They say nothing about replication across training runs, warm starts or models.
- **Degenerate intervals.** A12 read-back CIs [1.00, 1.00]; A14 BCA [1.00, 1.00] and every contrast 0.00 [0.00, 0.00]. A14's "equivalent within ±0.10" verdicts rest on zero-variance ceiling outcomes, so the primary endpoint cannot discriminate between policies. Phrase these as "identical on all 250 tasks", not as equivalence tests.
- **Clustering is applied inconsistently** (W2): template-cluster bootstrap in A17, task bootstrap in A12, A14 and A15.
- **A17's positive criterion depends on a post-hoc script.** The seed-averaged interval was computed by `scripts/a17_seedavg.py`, written after per-seed results were seen (D-A17-1). This is disclosed, but the paper must say so next to the number.
- **Independence of validation.** The reference verifier was "written from the spec only, by a separate author" (A12 section 10). In a single-author project with LLM-assisted coding (appendix.tex line 66), state who or what that separate author was. If an AI assistant wrote both from the same spec, agreement is a consistency check, not independent validation.

### W7 (Major). The A16 mechanism claim is weaker than the ledger states

- **Not specific to R_V.** The registered ablation rule is also met by R_A seeds 0 and 2: dM_partial -0.22 and -0.20 against a maximum random |dM| of 0.19 and 0.15 (`results/a16/a16_summary.json`, key `causal`). These models never refund (A14, A15 partial accuracy 0.00). The warm start nearly meets it (-0.29 vs 0.30). The direction is a generic "payment row content" feature present before RL, not the mechanism R_V learned.
- **Collateral on the no-payment branch.** Ablation shifts the Q0 margin by -0.29 to -0.36 in every model, comparable to the partial-branch effect in R_A and the warm start. The registered collateral check covered only Q4.
- **Small effects, no behavioural readout.** dM_partial is -0.49 to -1.14 nats against partial-branch margins of about -2 to -3 nats. No greedy action change under ablation is reported.
- **Patching fails.** Final-token effects (0.037 to 0.056) do not exceed the random 95th percentile in any R_V seed (0/5). Span effects (0.14 to 0.23) are similar in R_A and the warm start (0.19 to 0.23).
- **"Margin grows with training" holds for two of three seeds.** R_V s1 3.43 to 3.91 and s2 3.63 to 5.09, but s0 2.92, 2.51, 2.77, 2.91 at steps 50, 100, 150, 200.
- Probes (AUROC 1.00 everywhere) are uninformative, as the registration anticipated.
- **What the evidence supports.** A linearly decodable partial-vs-complete direction exists at layer 18 in all models and modestly modulates the refund-vs-post margin. A16 did not localise what R_V changed. Margins should also be stratified by workflow (W2).

### W8 (Major). A17 transfer claims: the care_admin gap is mostly R_A reading less than the warm start, and outcomes do not transfer

- **care_admin phantom read-back** (`results/a17/a17_summary.json`): warm start 0.43 (12/28); R_V 0.48 to 0.67; R_A 0.13 to 0.28. Seed-averaged R_V minus R_A +0.37 [0.18, 0.58]; R_V minus warm +0.15 [-0.06, 0.39] (`a17_seedavg.json`). About 0.23 of the 0.37 gap is R_A falling *below* the warm start. The robust signal in this domain is R_A suppression, which ProcureSim greedy evaluation could not show. This is interesting in itself. It also contradicts "visible only in sampled behaviour" (main.tex line 88) and "behaves exactly like its warm start on every untouched task" (abstract).
- **it_ops is null.** The warm start already reads back 0.71; R_V minus R_A -0.03 [-0.07, 0.01]; 2/5 seeds positive.
- **Outcomes do not transfer.** In care_admin, VSC is R_V 0.075 to 0.117, R_A 0.15 to 0.175, warm start 0.125; collateral is R_V 0.24 to 0.33, R_A 0.12 to 0.13. R_V's false success on transfer classes is lower (0.15 to 0.25 vs 0.33 to 0.41). In it_ops every model's VSC is 0.12 to 0.19 against an oracle 0.83: the transfer test runs in a regime where all policies mostly fail.
- **Forks.** No model meets the credit rule. In it_ops every model behaves like the fixed-retry reference (q < 1: 0.90 to 1.00; q = 1: 0.10 to 0.23). In care_admin at q = 1, R_V redoes completed work more often than R_A and the warm start (correct 0.00 to 0.42 vs 0.58 and 0.50).
- **Small samples.** 24 templates x 5 instances per domain; 26 to 32 phantom writes per model in care_admin; forks use 12 base tasks per q.
- **Sibling simulators.** Both domains were built in one evening with ProcureSim's patterns (DECISIONS 2026-11-21 01:05). "Cross-domain" means a sibling simulator, not an independent benchmark.

### W9 (Major). Construct validity of the simulator and of "verification"

- **Adversarial acknowledgement design.** Byte-identical success replies, with ids drawn before the fault (`actionproof/integrity/sim/env.py` line 164), are the maximally adversarial case. This is legitimate as a stress test, but the practical framing (main.tex line 31) needs the caveat.
- **Fault prevalence far above any deployment.** Phantom writes are 25% of `proc_conf` and 35% of the training pool; never-commit 15%. The VSC gain scales with this choice (W1).
- **Reads are always authoritative and consistent**, except in the replica-lag stress split. Verification cannot fail or mislead, so "verification" reduces to issuing one read call.
- **The learned behaviour is an unconditional habit, not calibrated verification.** R_V benign read-back equals phantom read-back (1.0). Calls per episode rise from 3.9 to 7.3 and p50 latency from 2.41 s to 3.62 s (`paper/tables/track3.tex`). With identical acknowledgements this is optimal, but it should not be called "verification" without qualification.
- **Reading without acting.** On `proc_stress`, R_V reads after every write yet leaves the duplicate in place as often as R_A (0.58 in all seeds; claim STRESS-DUP).
- **R_V is report-blind.** False success remains 0.05 to 0.10 on `proc_conf`; claimed success 0.84 vs verified 0.74 to 0.79.
- **Narrow task space.** Rewards, verifier and simulator share a codebase (main.tex line 43). There are 3 workflows, 12 tools, templated prompts, one policy prompt, no user simulator and no linguistic ambiguity. For an EMNLP venue the language component is thin. The paper should say what is linguistic about the problem, or position itself for an RL or agents venue.

### W10 (Moderate). Places where the wording outruns the evidence

| Location | Current wording (paraphrased or quoted under 15 words) | Problem | Suggested fix |
|---|---|---|---|
| Title (main.tex line 13) | "What the Reward Sees Decides What the Agent Checks" | General causal claim from one model, one warm start, one simulator; registered C1 not made; it_ops null | Name the setting and the contrast, for example "A state-observing reward, unlike an acknowledgement reward, makes read-back the default in a fault-injected procurement simulator" |
| Abstract, line 26 | rewards "differ only in this information channel" | They also differ in induced incentives (R_A pays a blind same-key retry) and in saturation (mean reward 0.97 vs 0.63) | "differ in information channel, and therefore in what they pay for and how much gradient they give" |
| Abstract | R_A agent "behaves exactly like its warm start on every untouched task" | False in care_admin (read-back 0.43 to 0.13 to 0.28) | Restrict to ProcureSim |
| Abstract; line 112 | partial refunds "35 to 48% of tasks" | Bimodal by workflow (W2) | Report per workflow |
| Line 88 | R_A suppression "visible only in sampled behaviour" | A17 shows greedy suppression out of domain | Revise with A17 |
| Line 103 | "once the evidence is in context, the next action follows it" | Only in the agent's own format; runtime-attributed R_A 0.83; A17 forks are evidence-insensitive | Add the conditions |
| Discussion, line 118 | showing evidence "is a more reliable lever than training an agent to look" | From Stage 1 e-commerce with a privileged gateway (warm start 0.87 vs trained without gateway 0.62 to 0.73); PCSTR-d with Enforce reached 0.94 (`paper/tables/a11.tex`); A11's registered interaction claim was not made | Remove, or restate as a single observation in a different environment |
| Ledger A16 | margin "grows with training"; "causal rule met" | Seed 0 flat; rule also met by R_A s0 and s2 | "In seeds 1 and 2"; add the R_A outcome |
| Ledger K1 | sampled decline "not under targeted (about 0.21 to 0.16)" | That is a decline, only smaller | "a smaller decline" |
| Ledger A17 | "positive criterion met" | Correct, but the gap is driven by R_A below warm start | Add the decomposition |
| `PREREGISTRATION_S3.md` section 2 | "The only change is the start-state distribution" | The pool also adds duplicate delivery and partial ratios 1/4 and 3/4 (Section 3.3) | Correct the description in an addendum |
| `RESEARCH_REPORT.md` section 1 | "Verification is already solved" on classes where reading suffices | In one simulator, one warm start, at greedy | Qualify |

### W11 (Minor). Presentation and compliance

- **Not anonymised.** Author block (main.tex lines 15 to 17), GitHub URL (appendix.tex line 62), "Stage 1/Stage 2", "hackathon", "Track 3", and "AgentTune and AuditKIT are the hackathon's toolkits" (appendix.tex line 64).
- **Outdated style file.** The paper uses `emnlp2023.sty`; use the current ACL template (not re-verified against the current call).
- **Stale placeholder.** Appendix H4 still says "To be completed" (appendix.tex line 52), although `claims/claims.yaml` records H4 as supported (8/10 vs 1/10).
- Figures are loaded from `../docs/paper/figs/`, outside the paper directory.
- **Citation verification conflict.** `smyth2026overclaim` is in `refs.bib` under a "fetched" header, but `docs/stage3/LITERATURE.md` lists the same arXiv id (2609.20812) as UNVERIFIED (search snippet only). Re-verify before submission.
- **Unexplained oracle result in A14.** The scripted oracle has `goal_met` 0.0 on branches A, T and P while `correct` is 1.0 (`results/a14/a14_summary.json`). Explain this (scripted continuation?) or fix it.
- **Undefined term.** The truth table refers to "minus turn cost", but the rewards are binary. Define the term or remove it.

### 3.3 S3 (RS-GRPO): design, registration, development evidence, and what would count

**Design facts.**
- **Pools.** RS-GRPO trains on 200 `proc_train_b` tasks (`s3_trainb_half`) plus 200 recovery starts (`s3_rec_train_half`). The baseline, A12 R_V, trained on 400 `proc_train_b` tasks.
- **Fault-class coverage differs.** `s3_rec_train_half` has 35 duplicate-delivery starts (15 ack, 20 read) and 54 partial starts at ratios 1/4, 1/2 and 3/4. `proc_train_b` has **0** duplicate-delivery tasks and 52 partial tasks at ratio 1/2 only (Appendix). `s3_conf_full` has 46/300 duplicate-delivery and 64/300 partial tasks across three ratios; `s3_conf_rec` has 39 duplicate and 80 partial starts.
- **H1 is in-distribution for RS-GRPO only.** `s3_conf_rec` uses "same start mix as training" (S3 section 3); for the baseline the same split is out of distribution.
- **Privileged prefixes.** Recovery-start prefixes are generated by the scripted oracle, including its write-then-read convention for earlier steps (`actionproof/stage3/tasks.py`, `prefix`). That is in-context demonstration content the baseline never sees.
- **Budget is matched only in steps, prompts and rollouts.** RS-GRPO sees half as many full-episode prompts (about 400 vs 800 draws). Its rollouts are shorter (about 5.2 vs 7.5 policy calls per episode on dev), so it generates fewer tokens. It also sees more fault classes and oracle prefixes. Steps and rollouts are matched; information and compute are not.

**Development evidence (`results/s3/dev`).**
- `s3_rec_dev` VSC 0.807 (RS seeds 0 and 1) vs 0.600 (A12 R_V s0). By start class, the gain is duplicate delivery 0/23 to 19/23 (+19 tasks) and partial 13/34 to 25/34 (+12 tasks), with no change elsewhere. **About 61% of the gain (19 of 31 tasks) is on a fault class the baseline never trained on.**
- **Full episodes.** RS seed 0 and A12 R_V s0 produce identical tool-call sequences on 145/150 `proc_dev` tasks (report text ignored). VSC 0.860 vs 0.847 is two partial tasks (`proc_dev-0047`, `-0098`); `proc_dev` has only 12 partial tasks and no duplicate delivery.
- **Rewards made no difference to full episodes.** The three 100-step pilots with different rewards (R_V, safety-first, Lagrangian) produced identical tool-call sequences on all 150 `proc_dev` tasks. The safety-first and Lagrangian variants did not change full-episode greedy behaviour at all. Consistent with the report's own warning, the Lagrangian multiplier probably never saw enough violations to matter.

**Registration and selection.**
- The selection rule (RESEARCH_REPORT section 6) was written for a four-arm, 100-step pilot. F's pilot `proc_dev` VSC was 0.793; A12 R_V s0 at step 200 is 0.847, so the "no more than 0.05 below" constraint (0.797 or above) is missed by 0.004 if the step-200 checkpoint was the reference. Addendum 2 reports R_V s0 dev VSC from 0.795 to 0.845 across checkpoints. State which checkpoint was used.
- Stage B (mix ratio 25% vs 50%, 200 steps, seeds 0 and 1) is not in section 6. The seeds used to select the mix ratio are also confirmatory seeds 0 and 1. Report H1 and H2 on seeds 2 to 4 as a sensitivity analysis.
- **Name and framing.** The report defines RS-GRPO as "recovery-start, safety-first GRPO"; the registered algorithm has no safety-first component. Rename it, for example "recovery-start training", and frame it as a reset-distribution curriculum with the optimiser and reward unchanged (the report's own words in section 4.F).
- **Positive points.** Treating H1 and H2 as an intersection-union test without a multiplicity correction is valid. Disclosing that seeds 0 and 1 were trained before registration is good practice.

**Is S3 an algorithmic contribution?** No. It changes the training data and start-state distribution with a fixed mix and no adaptive schedule. Relative to reverse curriculum (C3), Go-Explore (C4), R^3 (C5), failure-prefix conditioning (C7) and AgentPRM resets (B4), the novelty is the instantiation: faulted execution states replayed in a fault-injecting simulator. It could still be a useful *empirical* contribution if the result below holds.

**Decision table: what result would or would not make S3 a contribution.**

| Outcome | Would it be a contribution? |
|---|---|
| H1 holds, H2 holds only as non-inferiority, and the H1 gain is concentrated in duplicate-delivery starts | **No.** It shows that training on a class and its states improves that class and those states, which is expected. |
| H1 and H2 hold, but full-episode (`s3_conf_full`) gains are confined to duplicate delivery, or to the `pay+post` workflow | **No.** Explained by class coverage or by the W2 workflow effect. |
| H1 and H2 hold, **and** RS-GRPO beats a matched-class full-episode R_V baseline on `s3_conf_full` partial + duplicate classes (seed-averaged CI excluding 0, at least 4 of 5 seeds). The baseline uses the same 400-task budget and a fault mix with duplicate delivery and partial ratios 1/4, 1/2, 3/4 at the recovery-start frequencies. | **Yes, modest.** Evidence that start states, not just class coverage, matter. |
| As above, **plus** the gain appears in `order+pay+post` and `receive+pay+post` (where A12 R_V is near 0 on partial), with no regression on phantom, lost-ack, never-commit honesty or `s3_conf_stress`, and holds on seeds 2 to 4 alone | **Yes, solid.** A clear empirical finding about reset distributions for agentic GRPO. |
| As above, **plus** it beats equal-rollout simpler fixes: oversampling partial and duplicate tasks, DAPO-style dynamic sampling (A2), or SFT on oracle recoveries from the same starts | **Yes, strong.** Shows that RL from recovery starts is needed. |
| H1 or H2 fails | Report as registered. Given the development evidence this would be informative (for example, start-state gains that do not survive in natural trajectories). |

**Required before reading the S3 confirmatory results.** Register an addendum that (a) corrects "the only change is the start-state distribution", (b) adds the per-class and per-workflow decomposition of `s3_conf_full` and `s3_conf_rec` as pre-specified secondary analyses, (c) names the matched-class baseline and its decision rule, and (d) pre-specifies the seeds 2 to 4 sensitivity analysis.

---

## 4. Questions for the authors

1. Why does R_V refund partial payments almost always in `pay+post` workflows and almost never in `receive+pay+post` (W2)? What is the distribution of partial faults by workflow and step in `proc_train_b`? Is the cause context length, position of the pay step, or the content of the preceding read?
2. For every seed, what are R_A's mean training reward and zero-variance share over all 200 steps? Do you have any channel-blind reward that is not saturated on this warm start?
3. What greedy read-back does R_V produce from a warm start whose demonstrations contain no reading (oracle share 0)? Your "selection, not invention" account predicts none.
4. Who wrote the reference verifier, the 91 golden traces and the OpsSim domains? If an AI assistant was involved, how was independence from the main implementation enforced?
5. A16: does ablating the layer-18 direction change any greedy action? Why does the ablation rule also hold for R_A seeds 0 and 2?
6. A17: do you agree the care_admin gap is mostly R_A reading below the warm start? Why does R_V redo completed referrals more often than R_A at q = 1?
7. S3: will you add the matched-class baseline? Which A12 R_V checkpoint was the 0.05 selection constraint compared against? Were seeds 0 and 1 of the confirmatory set the seeds used in Stage B?
8. What are the sampled-decoding (T = 1, k samples) read-back and VSC of the warm start, R_V and R_A on `proc_val`? pass@k would make "selection, not invention" quantitative.
9. How do the VSC results look as per-class conditional rates, independent of the 25% phantom prevalence?
10. Does any reward include a turn or call cost (the truth table mentions one)? Would a call cost change R_V's unconditional reading?
11. A14: why is the scripted oracle's `goal_met` 0.0 on branches A, T and P?
12. What makes this an NLP contribution rather than an RL-for-agents contribution? Is there any linguistic variation in the task prompts or tool outputs that matters for the results?

---

## 5. Scores (ACL ARR-style schema)

The labels below follow my recollection of the ARR review form (Soundness, Excitement, Overall Assessment and Reviewer Confidence, each on a 1 to 5 scale with half points). **The schema was not re-verified against the live ARR form.**

| Field | Score | Rationale |
|---|---|---|
| Soundness | **3 / 5** (Acceptable) | The registered primary claims (A12 H2, A14 primary) are supported as narrowly stated, with exemplary process. The causal interpretation (W1), the partial-evidence results (W2), A16 (W7) and A17 (W8) need more support or narrower wording. |
| Excitement | **2.5 / 5** | Careful and useful, but the A12 outcome follows largely from the reward-truth table and prior work; one model, one warm start, sibling simulators. |
| Overall assessment | **2.5 / 5** (borderline Findings) | Not ready for the main track as is. With the Critical items fixed and one identification experiment added, I would expect 3 (Findings). With a positive, well-identified S3 result and a second warm start or model, 3.5 is reachable. |
| Reviewer confidence | **4 / 5** | I read the registrations, deviation log, ledger and result files, and regrouped the per-episode data myself. I did not re-run training or the statistics scripts. |

---

## 6. Prioritised revision checklist

### Critical

1. **Stratify every partial or evidence-use result by workflow (W2).** Add per-workflow rows to `paper/tables/a14.tex` and Figure 4 (`fig4_a14_branches`), to the A15 results, the A12 partial class and the A16 margins. Replace "35 to 48% of tasks" (main.tex lines 26 and 112). Re-word the A15 "generalises" verdict in `claims/claims.yaml` (entry A15). Use workflow- or prompt-skeleton-clustered intervals, as A17 already does. Cost: no GPU.
2. **Fix the identification story (W1).** In section 3 (Rewards, main.tex line 57) and section 4, report R_A's saturation (mean reward 0.97, zero-variance share 0.82 to 0.86) and the warm-start identity of R_A and K1 models. Decompose the VSC gain by class. Temper the title and abstract. Add at least one identification experiment (items 6 and 7 below).
3. **S3: register an addendum before reading the confirmatory results (Section 3.3).** Correct the "only change" sentence; add the matched-class baseline and its decision rule; add per-class, per-start-type, per-workflow and seeds 2 to 4 analyses; rename RS-GRPO and frame it as a reset-distribution curriculum. Location: new `PREREGISTRATION_S3_ADDENDUM1.md`; RESEARCH_REPORT section 4.F.
4. **Integrate A15, A16, A17 and K1 into the paper with the corrected wording from W7, W8 and W10.** Fix main.tex line 88 ("visible only in sampled behaviour"), line 103 and line 118 (discussion). Fill or remove the H4 placeholder (appendix.tex line 52) using the H4 ledger entry.
5. **Anonymise for ARR and remove hackathon framing.** Author block (main.tex lines 15 to 17); GitHub link (appendix.tex line 62); "Stage 1/2", "Track 3", "hackathon's toolkits" (appendix.tex lines 21 and 64). Move to the current ACL template.

### Major

6. **Warm-start repertoire experiment.** This tests the main mechanism claim and samples warm-start variance (W1, W5). Train two more SFT warm starts with oracle share 0.0 and 0.10 (optionally a second SFT seed at 0.30), then R_V for 200 steps, 1 to 2 seeds each, and evaluate on `proc_dev` and `proc_val`. Register it first if `proc_conf` will be used. Prediction under "selection, not invention": oracle share 0 gives no read-back.
7. **Observability or gradient control (W1).** Option (a): an observability dose-response with an episode reward equal to R_V with probability p and R_A otherwise, p = 0.25 and 0.5, two seeds each. Option (b), cheaper: R_V on `proc_train_b` minus phantom and partial tasks, two seeds, predicting no read-back. Either shows that read-back tracks reward observability rather than generic RL dynamics.
8. **Statistics (W6).** State the Holm-adjusted 0.0625 in the main text, and that five seeds cannot reach 0.05 under Holm for two metrics. Separate evaluation noise from training replication. Rephrase degenerate-CI "equivalence" as "identical on all n tasks". Put D-A17-1 next to the A17 interval.
9. **A16 (W7).** Report the rule outcome for the warm start and R_A; report Q0 collateral; add a behavioural readout (greedy action change under ablation); add a specificity criterion (R_V effect larger than R_A effect); stratify by workflow. Soften the claim to "a partial-vs-complete direction present in all models modulates the margin".
10. **A17 (W8).** Report R_V minus warm and R_A minus warm per domain, VSC and collateral where R_V is worse, and the q = 1 fork result. Restate transfer as "the R_V/R_A divergence appears in care_admin, mainly because R_A reads less than the warm start; outcomes do not transfer".
11. **Construct validity section (W9).** Cover fault prevalence, byte-identical acknowledgements, consistent reads, three workflows, provenance of the reference verifier and golden traces, the cost of unconditional reading (calls 3.9 to 7.3; latency), and report-blindness.
12. **Related work (W4; table in Section 7).** Add τ-bench, the reset-distribution and curriculum line, TruthRL and Reinforced Hesitation, Constrained GRPO, Verified Tool Calls and Failure-Transparent Agents. Acknowledge Stage 1 as anonymised prior work.
13. **Sampled evaluation.** Report T = 1, k = 8 on `proc_val` for the warm start, R_V, R_A and K1 models (pass@k read-back), so the repertoire claims are not greedy-only.

### Minor

14. Re-verify `smyth2026overclaim` (UNVERIFIED in LITERATURE.md, present in refs.bib).
15. Explain the A14 scripted oracle `goal_met` 0.0 on A, T and P.
16. Define or remove "turn cost" in the truth table.
17. Move figures into `paper/` and fix the relative paths.
18. Ledger wording: K1 "smaller decline"; A16 "seeds 1 and 2"; A17 decomposition.
19. In S3, state the checkpoint used for the 0.05 selection constraint and that Stage B was added after section 6 was written.
20. Rename `track3.tex` and its caption; label latency hardware in the main text.
21. Report generated tokens per training run for A12 and S3 so that budget matching can be checked.

### Experiments that would raise the score most within about 2 days, about 30 USD and a 2xH200 Slurm lease

Costs come from `docs/stage2/HF_CREDITS.md`: about 9 USD per 200-step GRPO run on HF H200 (about 1 h 50 min including dev evaluations), about 1.4 USD per SFT warm start (16.5 min), about 0.35 USD per confirmatory evaluation on L40S. The Slurm lease is free in dollars but shared with the running S3 jobs.

| Priority | Experiment | Compute | Expected effect on review |
|---|---|---|---|
| P0 | Reanalyses: workflow stratification and clustered intervals (A12, A14, A15, A16, S3); R_A saturation table; A17 decomposition; S3 decomposition by start type, class and workflow | none, about half a day | Fixes W2, W6, W8 and most of W10. Soundness rises by about 0.5. |
| P1 | S3 matched-class baseline: R_V on a 400-task full-episode pool with duplicate delivery and partial 1/4, 1/2, 3/4 at recovery-start frequencies, 3 seeds, registered first; one evaluation each on `s3_conf_full` and `s3_conf_rec` | Slurm, about 6 GPU-h; evaluations about 2 USD | Decides whether S3 is a contribution (Section 3.3). Without it, S3 should not be claimed. |
| P2 | Warm-start repertoire: SFT at oracle share 0.0 and 0.10, then R_V with 1 to 2 seeds each, dev and val evaluation | SFT about 3 USD (HF); GRPO on Slurm, about 4 to 8 GPU-h | Directly tests "selection, not invention" and samples warm-start variance. The largest Excitement gain per GPU-hour. |
| P3 | Observability dose-response (p = 0.25, 0.5; 2 seeds each) or the R_V-without-phantom control (2 seeds) | Slurm about 4 to 8 GPU-h; spill-over to HF at most 2 runs (about 18 USD) | Closes the W1 identification gap. |
| P4 | Sampled evaluation (T = 1, k = 8) on `proc_val` for existing models | L40S, about 2 to 3 USD | Makes the repertoire claims quantitative. |
| Stretch | Second model (a larger Aya model), A12 R_V and R_A, 2 seeds each | Probably beyond 2 days with S3 on the lease | Largest single fix for W5, if the lease frees early. |

Suggested allocation: P0 immediately. P1 and P2 on the two Slurm lanes as soon as S3 seeds release them. HF credit for the P2 SFTs, P4 evaluations and at most two P3 runs (total about 25 to 30 USD).

---

## 7. Missing related work

Every entry is either verified in `docs/stage3/LITERATURE.md` (section id given) or already in `paper/refs.bib` (key given). Venues are those recorded in LITERATURE.md; where it marks a venue as unverified, cite the arXiv version.

| Title | Key contribution | Relevance to this paper | Where to cite |
|---|---|---|---|
| τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains (F1; arXiv, venue unverified) | Grades agents by final database state; pass^k reliability | R_V is τ-bench-style state grading used as a reward; pass^k suits the sampled analysis | Related work (reward observability); Setup (R_V) |
| Toward Scalable Verifiable Reward: Proxy State-Based Evaluation for Multi-turn Tool-Calling LLM Agents (`chuang2026proxystate`, in refs) | Proxy state trackers as scalable rewards | R_A is the failure case of such trackers; strengthen the link | Intro (already); Discussion |
| Identifying the Risks of LM Agents with an LM-Emulated Sandbox (F2, ToolEmu) | LM-emulated tools and safety evaluation | Contrast with a deterministic simulator; construct validity | Limitations; construct-validity section |
| Reverse Curriculum Generation for Reinforcement Learning (C3) | Start from near-goal states, select starts with intermediate success | S3 is a fixed-mix version of this idea | S3 section |
| Go-Explore (C4) | Return to archived states via simulator restore, then explore | Same reset mechanism as recovery starts | S3 section |
| Training LLMs for Reasoning through Reverse Curriculum RL, R^3 (C5) | Outcome RL from late points of a demonstration, sliding backward | Closest LLM analogue; S3 lacks the schedule | S3 section |
| Training Reasoning Models on Saturated Problems via Failure-Prefix Conditioning (C7) | Condition rollouts on failure prefixes to restore mixed groups | Nearly the same mechanism as recovery starts | S3 section |
| Process Reward Models for LLM Agents (B4, AgentPRM) | Includes exploration by resetting to states an optimal policy visits | Reset distribution for agents | S3 section |
| R³L: Reflect-then-Retry RL (H6) | Restarts from diagnosed failure points | Close to recovery starts | S3 section |
| Agent-R (H7) | Splices failed trajectories with correct continuations to build recovery data | SFT-on-recoveries baseline for S3 | S3 baselines |
| Group-in-Group Policy Optimization (A9) | Step-level advantages from shared anchor states | Alternative to recovery starts for step credit | S3 baselines; Discussion |
| DAPO (A2) | Dynamic sampling drops zero-variance groups | Relevant to R_A saturation and as an S3 baseline | A12 reward-signal analysis; S3 baselines |
| No Prompt Left Behind, RL-ZVP (A6) | Signal from zero-variance prompts | R_A's 0.82 to 0.86 zero-variance share | A12 reward-signal analysis |
| Understanding R1-Zero-Like Training, Dr. GRPO (A3) | Std normalisation inflates near-saturated groups | Interpreting gradient in a saturated arm | A12 reward-signal analysis |
| RAGEN (A7) | Diverse initial states and reward-variance collapse in multi-turn RL | Supports start-state diversity; collapse diagnostics | S3 section |
| TruthRL (E2) | Ternary reward: correct above abstain above hallucinate | R_V is report-blind; the dropped safety-first reward follows this design | Limitations (report-blindness); S3 pilots |
| Honesty over Accuracy: Reinforced Hesitation (H11) | Ternary reward with a risk parameter | Same as above | S3 pilots |
| Constrained Group Relative Policy Optimization (D7) | Standardise each reward and cost within the group before combining | Explains a likely flaw in the Lagrangian pilot | S3 pilots (exploratory) |
| Reward Constrained Policy Optimization (D2); Safe RLHF (D3) | Lagrangian constrained policy optimisation | Background for the Lagrangian candidate | S3 pilots |
| Learning When to Act or Refuse, MOSAIC (D8) | Act-or-refuse as first-class actions for safe multi-step tool use | Escalation and stopping; "done" starts | Related work; S3 |
| Agentic Abstention (H14) | Stopping as a sequential decision; agents stop too late or never | "Already complete" failure mode (A17 forks, "done" starts) | Discussion; S3 |
| Verified Tool Calls Improve LLM Agent Reliability Under Non-Atomic Failures (H17) | Wrapper with postconditions, verify-before-retry, idempotency keys | Non-learned baseline close to the gateway and to the H7 defence matrix | Related work (execution integrity); Appendix A13/H7 |
| Failure-Transparent Agents (H18) | Evidence contract cuts false-success reporting | Reporting accuracy; R_V's residual false success | Related work (evidence use); Limitations |
| From Confident Closing to Silent Failure (`advani2026confident`, in refs) | False-success rates and judge weakness | Already cited; link to Track 3 claimed-vs-verified table | Intro; Appendix |
| Where Does Exactly-Once Live? LIMBO (`li2026limbo`, in refs) | Model, harness and contract effects on duplicates | Closest environment; duplicate-delivery results | Related work (already); S3 duplicate class |
| UndoBench (`sah2026undobench`, in refs) | Separates task competence from recovery | Recovery framing for S3 | S3 section |
| Training Language Models to Self-Correct via RL, SCoRe (B6) | Multi-turn RL for self-correction | Recovery after own errors | Related work |
| Towards Best Practices of Activation Patching (G1); How to use and interpret activation patching (G2) | Metric and corruption choices change conclusions; denoising vs noising | A16 design and interpretation | A16 section |
| Have Faith in Faithfulness (G4) | Faithfulness over overlap for circuits | A16 claim standards | A16 section |
| RL Finetunes Small Subnetworks in LLMs (G5) | RL updates sparse but full-rank subsets | LoRA rank as a possible limit on what RL can change | A16; Limitations |
| Does RL Really Incentivize Reasoning Capacity Beyond the Base Model? (`yue2025rlreasoning`, in refs) | RL sharpens behaviours already in the base distribution | Already cited; tie to the warm-start repertoire experiment (P2) | A12 "Selection, not invention" |
| Acting Less is Reasoning More, OTC (E3); SMART (H13) | Penalise or reduce unnecessary tool calls | R_V's unconditional reading nearly doubles calls | Construct validity; Discussion |
| Penalizing side effects using stepwise relative reachability (H9) | Inaction baselines for side-effect penalties | S3 violation and collateral metrics | S3 metrics |
| Monitoring Reasoning Models for Misbehavior (H16) | Optimising against a monitor teaches obfuscation | Why no monitor or judge score is placed in the reward | Discussion |

---

## Appendix: reviewer reanalyses (read-only; reproducible)

1. **Workflow stratification (W2).** Join `task_id` in `results/eval/a14/<policy>__proc_cf__agent.jsonl` (records with `branch == "P"`, field `correct`) to `meta.requested` in `data/bench/proc/proc_cf.jsonl`. For A15 use `results/eval/a15/rv_s<k>__proc_mech.jsonl` (branches Q1 and Q3) with `data/bench/proc/proc_mech.jsonl`. For A12 use `results/eval/a12/conf/a12_proc_rv_s<k>__neutral__proc_conf.jsonl` (field `vsc`, tasks whose `meta.fault_class == "partial"`) with `data/bench/proc/proc_conf.jsonl`. The A15 prefix-length split uses the per-record field `n_prefix_calls` (2 vs 4).
2. **Per-task consistency (A15).** For each task, the tuple (first action is refund at Q1, Q2, Q3). R_V s0: (True, True, True) for 74 tasks, (False, False, False) for 114 tasks.
3. **Identical tool-call sequences (S3, A12).** For each task in `results/s3/dev/<run>.jsonl`, the list of (`tool`, `args` without `summary`) over `trace` entries with `by == "agent"`, compared across runs. Results: RS s0 vs s1 on `proc_dev`, 146/150; RS s0 vs A12 R_V s0 on `proc_dev`, 145/150; the three 100-step pilot arms, 150/150; A12 R_A s0 vs warm start, 150/150.
4. **S3 pool composition.** Counts of `meta.fault_class`, `meta.start_type` and `meta.faults[*].ratio` in `data/bench/proc/s3_rec_train_half.jsonl`, `s3_trainb_half.jsonl`, `proc_train_b.jsonl`, `s3_conf_rec.jsonl`, `s3_conf_full.jsonl`.
5. **S3 per-start-class gain.** `by_class` in `results/s3/dev/{a12_rv_s0,s3_rv_mix_s0}__s3_rec_dev.summary.json`: weighted sums of `vsc x n` per class.
