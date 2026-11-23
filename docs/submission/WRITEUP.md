# ActionProof: what reinforcement learning teaches tool agents when "ok" does not mean it happened

**One-line claim.** When a write tool's success reply cannot prove the write happened, a reward that reads the database makes a tool agent check its writes, and a reward that reads the reply does not (5 of 5 paired seeds). Checking, however, is not yet recovering, and a few hand-written checks still beat every trained agent we built.

| | |
|---|---|
| Track | Track 3: agentic fine-tuning (problem statement 9: Measure → Understand → Fix) |
| Team | Atharv Gaur (BITS Pilani); author contributions in §34 |
| Base model | Tiny Aya Global, 3.35B (CohereLabs, CC-BY-NC-4.0) |
| Method in one line | LoRA SFT warm start, then GRPO (AgentTune `TrlAgenticGrpo`) on ProcureSim, a SQLite procurement world with silent write faults; rewards: database-state (R_V) vs acknowledgement (R_A); recovery-start curriculum |
| Code, models, data | GitHub `atharvgaur1845/actionproof-v2`; HF collection `atharvgaur18/actionproof-procurement-study-6b0321cd11bad297aead6e81` (30 adapters, splits, traces); site https://atharvgaur18-actionproof.static.hf.space/index.html |

---

## 1. Abstract (under 500 words)

Tool-using agents increasingly write to systems of record: payments, goods receipts, journal entries. A write tool's reply ("ok") is usually treated as proof that the write happened. We built **ProcureSim**, a SQLite procurement environment in which the success reply is **byte-identical** whether the write stored everything, part of it, two copies, or nothing. A deterministic verifier reads the authoritative database, which the agent can query only through read tools.

**Same model, one change.** We trained Tiny Aya Global (3.35B) with GRPO from the same supervised warm start, changing only the reward's information source:
- The state reward, which reads the database, produced read-back after every phantom write in all five seeds (1.00).
- The acknowledgement reward produced read-back in none (0.00).
- Verified completion rose by +0.27 to +0.32 per seed.
- A warm start that never reads stayed at zero reads under the same reward (ID1). So the reward *selects* checking that the starting model already does; it does not create it.

**Checking is not recovering.** Agents that read a partial payment then posted the journal entry anyway. Training from faulted states (a recovery-start curriculum, a known technique) raised verified completion by +0.21 over the state-reward agent in five of five runs. Against a control trained on the same faults, the start-state effect held on recovery starts across five runs per arm (Holm p = 0.036). A partial-payment-specific effect was not established.

**The gains held** under Hindi, Hinglish and two-fault episodes. **They failed** under renamed tools and a new workflow step, where agents replay the trained procedure. Randomising tool names did not help (+0.00) and cost a quarter of canonical performance. Our own counterfactual credit method (FCA) failed its pre-set test.

**New in this round:**
1. Capability retention on BFCL and IFEval, registered before any output (R1): across 35 models, no trained arm loses measurable general ability. GRPO is non-inferior to its warm start within 0.03 on both benchmarks (all four Holm p < 0.001).
2. A four-family LLM judge committee measured against the database (J1). Told that "ok" is not proof, judges abstain rather than accept unverifiable failures, so they cannot certify outcomes. Without that one sentence, three of four judges accept nearly all of them (exploratory). With a database snapshot, two of four families stay near 0.89 accuracy, and majority voting (0.897) does not beat the best single judge (0.989).
3. A cost model built from measured per-fault outcome rates. Recovery-start GRPO cuts projected losses by about 80% against the untrained agent, but hand-written read-back-and-repair checks beat every trained agent. So we present ActionProof as fault-injection testing plus runtime verification, with training as a second line.

Every experiment was registered and hashed before it ran. A checker re-reads every reported number from its result file.

## 2. Before and after (same base model)

| Same model, `proc_conf` (300 confirmatory tasks) | Warm start (SFT) | After GRPO, acknowledgement reward | After GRPO, state reward |
|---|---|---|---|
| Read-back after a phantom write | sampled 31% at the gate (58/188) | 0.00 in each of 5 seeds | **1.00 in each of 5 seeds** |
| Verified completion, state − acknowledgement reward | — | — | **+0.27 to +0.32 per seed** |

Source: `results/s2/a12_summary.json` and `results/eval/a12/conf/*`. The two-metric Holm-adjusted sign-flip p = 0.0625, the minimum possible with five seeds, so "5 of 5" is a consistency statement.

## 3. One-minute summary for judges

1. **The problem is real and silent.** The reply looks the same whether the payment happened or not (§9).
2. **The reward's information source decides whether the agent checks.** Database reward: checks in 5/5 seeds; reply reward: never. Same model, same data, one change (§2).
3. **RL selects, it does not invent.** From a starting model that never reads, nothing changes (ID1).
4. **Checking is not recovering.** Agents read the shortfall and post anyway. Training from broken states helps (+0.21, 5/5), but partial payments after a goods receipt remain unsolved by every arm.
5. **The skill does not transfer** to renamed tools or new workflows. The known fix (tool-name randomisation) failed here.
6. **New:** capability retention (R1), a judge committee against database truth (J1), and a cost model whose honest answer is "write the checks; use ActionProof to find them and to train a second line".

## 4. Real use case

**Accounts payable and procurement agents.** An agent records a goods receipt, pays the supplier and posts the journal entry. When the payment service times out after committing, or commits only part of the amount, the agent receives "ok" either way. Industry surveys put duplicate *or erroneous* payments at about 0.8–2.0% of disbursements (APQC, via vendor summaries; `docs/upgrade/ROI.md` §3). The same pattern applies to refunds, payouts and IT change execution.

## 5. Research questions

- **RQ1.** Does the information source of the reward (database vs acknowledgement) change whether GRPO agents verify their writes?
- **RQ2.** Does verification translate into correct action on the evidence (repair, refund, escalate)?
- **RQ3.** Does training from faulted states improve recovery, and is it the start state or the added fault coverage?
- **RQ4.** Do the gains survive shifts in language, fault count, tool names and workflow?
- **RQ5 (new).** Does the training cost general capability?
- **RQ6 (new).** Can independent LLM judges detect silent write failures without database access?
- **RQ7 (new).** Is any new credit-assignment or sampling method better than the recovery-start curriculum?

## 6. Preregistered hypotheses (selection; all 33 registrations in `docs/upgrade/INVENTORY.md`)

| Registration | Hypothesis | Verdict |
|---|---|---|
| A12 | R_V selects phantom read-back; R_A does not; same sign in every seed | H2 supported (5/5); composite C1 not supported because H3 failed |
| S3 | Recovery-start curriculum beats the state-reward agent | Supported: +0.21 [0.17, 0.25], 5/5, seed permutation p = 0.004 |
| S3 addendum 2 | Start-state effect vs a class-matched control (five runs per arm) | Recovery starts supported (Holm 0.036); partial class not supported (Holm 0.111) |
| S3 addendum 4 | The handed evidence explains the effect | Inconclusive (Holm 0.206, 0.262) |
| S4-EVAL | Gains survive four shifts | Hindi, Hinglish and two-fault supported; renamed API and new workflow not |
| SR1 | Tool-name randomisation fixes renamed tools | Not supported (+0.00); canonical tools −0.248 |
| ID1 | The reward creates verification from a non-reading start | Reading "selection": 0/153 read-back |
| E1/E2 | Sampled decoding contrasts; decision-point probes | Sampled contrasts not supported; reconciliation line +0.48 |
| FCA (development) | Forked counterfactual credit beats RSC by ≥ 0.03 | Failed: −0.10; dropped |
| **R1 (new)** | H1: SFT costs capability. H2: GRPO is non-inferior within 3 points (primary). H3: GRPO lowers abstention | H2 **supported** (non-inferior on both benchmarks, both arms); H1 and H3 not supported |
| **J1 (new)** | J-H1: transcript-only judges are fooled on unverifiable failures. J-H2: a snapshot gives ≥ 0.90 accuracy | J-H1 **not supported** (≤ 1 of 120 accepted: judges abstain); J-H2 **not supported** (0.886–0.989) |
| **C5/G (new, development)** | Verified-progress turn credit; hint-guided sampling; kill criterion +0.03 over RSC | C5 **dropped** (−0.32 and −0.37; it penalises reads by construction, `docs/research/algorithm/C5_RESULT.md`); G pending (seed 1 re-run after an HF bundle failure) |

## 7. Threat model

We do not consider an adversary. The failure model is:
- **benign infrastructure faults**: phantom commit, partial commit, duplicate delivery, lost acknowledgement, pre-commit timeout, never-commit;
- **misleading text inside tool outputs**: Stage 1's naturalistic and injected acknowledgements.

The agent may only call tools. The verifier and its database are outside the agent's reach, and the reward reads them only in training.

## 8. Related work (full matrix in `docs/research/NOVELTY_MATRIX.md`, 84 verified entries in `docs/research/NOVELTY_AUDIT.md`)

- **Tool-use RL:** GRPO (DeepSeekMath), agentic GRPO variants, MT-GRPO, GiGPO, RewardFlow.
- **Training from failure states:** Fission-GRPO, R3L, ReflectPO, Depth Before Breadth. These are the same family as our recovery-start curriculum, so **we claim no novelty for it**.
- **Counterfactual credit:** COMA, BiPACE, GiGPO. FCA is an adaptation, and it failed.
- **Surface binding of tool agents:** "Can Agents Generalize to the Open World?". Our S4 result replicates it.
- **"Did It Happen?":** evaluation-only paired hidden-commit design.
- **Silent-commit benchmarks:** Bench2Robust assumes retries cannot duplicate. We found no training paper with silent write commits plus duplicate risk.

**Our contribution is the setting and the registered measurements, not an optimiser.**

## 9. Tool and task environments, and trajectories

**ProcureSim** (`actionproof/integrity/sim/env.py`). SQLite state with purchase orders, supplier orders, payments, receipts and journal.
- **Write tools:** `place_supplier_order`, `pay_supplier`, `record_goods_receipt`, `post_journal_entry`, `refund_payment`, `escalate_to_human`.
- **Authoritative read tools:** `list_payments`, `list_goods_receipts`, `get_journal_entries`, `find_supplier_order`, `get_purchase_order`.
- **Workflows:** receive→pay→post, order→prepay, pay→post.
- **Faults:** phantom, partial, duplicate delivery, lost acknowledgement, pre-commit timeout, never-commit.
- The acknowledgement for a write is byte-identical across outcomes (`tests/integrity/test_procsim.py`).

**Recorded lifecycles** (`docs/submission/TRACES.md`, generated from result files):
- **Failure** (`s3_conf_full-0014`): the agent pays, reads that 62,892 of 251,570 cents were stored, posts the full journal entry and reports "completed". Verifier: pay `PARTIAL`, goal not met, false success. The acknowledgement reward gave this episode full marks (R_A = 1).
- **Rare success** (`s3_conf_full-0052`, recovery-start seed 3): the agent reads 87,180 of 116,240 cents, refunds, re-pays with a new idempotency key, verifies, posts and reports. Verifier: all steps verified. This is one of 5 repaired tasks out of 60 in this cell.

Traces of all 30 procurement-study models are published in the HF collection.

## 10. Toolkit usage and integration (`docs/upgrade/TOOLKITS.md`)

| Kit | Use |
|---|---|
| **AgentTune** | Trains every GRPO policy (`TrlAgenticGrpo`); its parser and prompt are used in rollouts and evaluation |
| **AuditKIT** | Stage 1 judge audit (`TaskCompletion`, `AgentEvalRunner`). With the ledger shown, Aya 32B still accepted 71% (213/298) of false completions |
| **CuratorKIT** | Stage 1 deduplication of acknowledgement texts (`ExactDeduplicator`, `MinHashDeduplicator`) |
| **CircuitKIT** | Attempted; failed to run (D-A16-1); no claim |
| **AlignTune, SafeTune** | Not used |

## 11. Models and licences

| Model | Role | Licence |
|---|---|---|
| CohereLabs/tiny-aya-global (3.35B, `Cohere2ForCausalLM`, 8,192 context) | Policy (all training) | CC-BY-NC-4.0 + Cohere Labs acceptable-use policy |
| CohereLabs/aya-expanse-32b | Stage 1 judge; τ-bench simulated user (R1-T) | CC-BY-NC-4.0 + acceptable-use policy |
| google/gemma-4-31B-it | J1 judge | Gemma licence (model card) |
| Qwen/Qwen3-235B-A22B-Instruct-2507 | J1 judge | Apache-2.0 |
| openai/gpt-oss-120b | J1 judge | Apache-2.0 |
| deepseek-ai/DeepSeek-V3.2 | J1 judge | MIT |

**Our adapters inherit CC-BY-NC-4.0** and are research artefacts.

## 12. Datasets, sources, languages, splits, leak checks

**Sources:**
- All ProcureSim tasks are generated by our code (`actionproof/bench/generator_proc.py`), Apache-2.0.
- External benchmarks, used only for evaluation: BFCL v4 (Apache-2.0), IFEval (Apache-2.0), τ-bench (MIT).

**Languages:** English; Hindi and Hinglish shift splits (`s4_lang_hi`, `s4_lang_hinglish`, 150 tasks each), reviewed by a native speaker (the author).

**Splits:** train pools (`proc_train_b`, `s3_trainb_half`, `s3_rec_train_half`), development (`proc_dev`, `s3_rec_dev`) and confirmatory (`proc_conf`, `s3_conf_full`, `s3_conf_rec`, S4 shift splits, fresh `e1_*`/`e2_pp`, `c1_pp`). Each is hashed at registration (`results/*_hashes.json`), checked in CI (`scripts/ci_freeze_check.py`), and **run once**.

**Leak checks:**
- Task-id disjointness across splits: `actionproof/stage3/tasks.py` `seen_ids`.
- The byte-identical acknowledgement test.
- R1 benchmarks share no prompts or schemas with ProcureSim.
- J1 judges receive no hidden-state fields.

## 13. CuratorKIT filtering

Used in Stage 1 only, on the naturalistic acknowledgement corpus (`actionproof/bench/naturalistic_acks.py`): exact and MinHash near-duplicate removal before the acknowledgement templates were frozen. The procurement study generates its tasks programmatically and needs no curation.

## 14. Metrics and why

- **VSC (verified-success completion):** every requested step is verified in the database and the report is honest.
- **False success:** completion was claimed when it is not true.
- **Violation:** a duplicate payment left unrefunded.
- **Phantom read-back:** the agent read back after a phantom write.
- **Tool-call validity.**
- **Calls per episode.**
- **Latency** (`results/latency_single.json`).

We use VSC rather than reward or claimed success because claimed success is exactly what the failure inflates. Definitions: `docs/stage2/METRICS.md`.

## 15. Automatic judge validation (J1)

Full report: `docs/upgrade/JUDGES.md`, generated from `results/j1/j1_summary.json`.

| Judge (660 main cases) | Transcript only: accuracy / coverage | With database snapshot: accuracy | Prompt-paraphrase κ | $ per 1,000 |
|---|---|---|---|---|
| Gemma 4 31B | 0.576 / 0.58 | **0.989** | 0.96 | 0.17 |
| Qwen3 235B-A22B | 0.373 / 0.59 | 0.889 | 0.56 | 0.17 |
| gpt-oss-120b | 0.514 / 0.56 | 0.933 | 0.49 | 0.44 |
| DeepSeek V3.2 | 0.300 / 0.43 | 0.886 | 0.65 | 0.34 |
| Majority committee | 0.385 / 0.42 | 0.897 | — | — |

Accuracy counts an abstention as wrong.
- **Not fooled, but abstaining.** No judge accepted more than 1 of 120 unverifiable failures; judges confirmed only 29–67% of genuinely completed episodes.
- **One systematic bias.** In 4 snapshot cases all four judges contradict the verifier, and the verifier is right: a refunded duplicate nets to one payment.
- **Human calibration labels:** pending (one annotator).
- **Exploratory addendum 1** (registered after J1 was read; `results/j1/j1_addendum1.json`). Remove that one rubric sentence, and transcript-only judges accept unverifiable failures in 120 of 120 (Gemma), 120 of 120 (gpt-oss), 115 of 120 (Qwen) and 9 of 120 (DeepSeek) episodes. **A judge's caution is a property of the prompt, not of the model.**

Design (`PREREGISTRATION_J1.md`):
- four families, temperature 0, seeded option order;
- 76 stratified calibration cases, labelled by the author with the same rubric, and 660 main cases;
- two views: transcript only, and transcript plus database snapshot;
- a prompt-paraphrase variant.

Outputs: accuracy, coverage, precision, recall, F1, κ, Fleiss κ, abstention, cost and latency. The database remains ground truth.

## 16. Capability retention (R1, R1-T)

Full report: `docs/upgrade/CAPABILITY_RETENTION.md`, generated from `results/r1/r1_summary.json`.

| Same model, change vs its warm start (5 seeds; 95% paired task bootstrap) | IFEval prompt-strict (warm 0.771) | BFCL non-live AST (warm 0.493) | BFCL irrelevance (warm 0.062) |
|---|---|---|---|
| State-reward GRPO | +0.000 [−0.016, +0.017] | −0.002 [−0.009, +0.006] | −0.003 [−0.020, +0.015] |
| Recovery-start GRPO | −0.004 [−0.021, +0.012] | +0.001 [−0.008, +0.009] | +0.005 [−0.009, +0.021] |
| SFT warm start vs base (base 0.760 / 0.471 / 0.079) | +0.011 [−0.020, +0.043] | +0.022 [−0.004, +0.048] | −0.017 [−0.050, +0.017] |

- **No trained arm shows a measurable loss.** The most negative point estimate across all arms and endpoints is −0.012.
- **Low abstention is the base model's own property** (0.079).
- **τ-bench (R1-T):** pending. The feasibility pilot's first launch failed at server start (D-R1T-1) and has been re-queued.

Benchmark selection:
- BFCL and IFEval are primary.
- τ-bench retail is secondary, with a pre-set context-overflow cancellation rule.
- AgentBench is excluded: heavy infrastructure, floor scores for 3B models, little relevance.

Reasons: `docs/upgrade/PLAN.md` §2. Thirty procurement-study models plus the Stage 1 chain are compared with their own warm start and the base model.

## 17. Training configuration and hyperparameters

- **Warm start:** SFT with LoRA r16, 2 epochs, lr 1e-4, on `proc_demo` (`actionproof/train/rft.py`).
- **GRPO:**
  - 200 steps × 4 prompts × 8 rollouts;
  - lr 2e-5, β 0.02, temperature 1;
  - ≤ 16 turns, at most 256 completion tokens per turn;
  - LoRA r16;
  - batched colocated vLLM rollouts.
- **RSC** is the same, on the pools `s3_trainb_half` + `s3_rec_train_half`.
- **Hardware:** one H200 per run, about 1 h 55 min per run (`logs/*_s*.full.log`).
- **Library versions:** TRL 1.7.1, vLLM 0.23.0, AgentTune d4db4c1 (`requirements.lock`, `uv.lock`).

## 18. Seeds

Five independent training seeds per arm (0–4) for A12 R_V and R_A, RSC, the class-matched control, and acknowledgement-only starts; three for SR1; two for FCA, ID1, C5 and G (development). Every seed is reported separately in the result files.

## 19. Ablations

- A12: reward information source only.
- S3 addendum 1/2: fault coverage vs start state.
- Addendum 4: handed evidence.
- SR1: tool-surface randomisation.
- E2: forced evidence and a reconciliation line.
- A14/A15: forced evidence by workflow position.
- A11: gateway factorial.
- Stage 1: PCSTR reward components (no-report, no-ND, no-Funder).

## 20. Confidence intervals and statistical tests

- **Task-level:** 95% paired task bootstrap (10,000 resamples), holding the trained models fixed. This reflects task uncertainty only.
- **Seed-level:** unpaired seed permutation tests and sign-flip tests.
- **Multiplicity:** Holm correction within each registration.
- **Proportions:** Wilson intervals (J1).
- Stated with every result in `docs/final/FINAL_CLAIMS.md`.

## 21. Named findings

1. **Reward observability selects verification** (A12, ID1).
2. **Reading is not recovery** (A14/A15, trace §9).
3. **Start state matters, partial-specificity unproven** (S3 addenda).
4. **Procedure replay** (S4, SR1).
5. **Warned judges abstain; snapshot judges are not verifier-grade** (J1, `docs/upgrade/JUDGES.md`).
6. **No capability cost** (R1, `docs/upgrade/CAPABILITY_RETENTION.md`).
7. **Progress credit penalises evidence** (C5 diagnosis).

## 22. Negative results (kept and reported)

- A12's composite C1 is not supported (H3 failed).
- The A11 interaction claim was not made (one seed of two).
- The partial-specific start-state effect is not supported.
- Addendum 4 is inconclusive.
- E1 sampled contrasts are not supported.
- Renamed API and new workflow: no gain.
- SR1 hurts canonical tools by −0.248.
- Negative transfer to healthcare administration (VSC 0.158 → 0.047).
- FCA failed (−0.10).
- Addendum 3 was withdrawn before training as a degenerate design.
- The after-receipt partial cell remains unsolved, and rebalancing did not fix it (0.005).

## 23. Sample inputs and outputs

See `docs/submission/TRACES.md`: two complete recorded episodes with the database before and after, plus the verifier's verdicts. The site's sections 00 and 03 replay further recorded episodes from five models on the same task.

## 24. Compute and ROI

**Compute:**
- HF Jobs spend is logged per job in `docs/stage2/HF_CREDITS.md`.
- The cluster ran the BITS sharanga H200, H100 and A100 lease workers.
- J1 inference costs about $3 projected ($0.19 for calibration).

**ROI** (`docs/upgrade/ROI.md`, `results/roi/roi_model.json`). Measured per-fault outcome rates × cited assumptions. Moderate scenario, per 1,000 workflows:

| Arm | Projected cost |
|---|---|
| SFT agent | $10,261 |
| Recovery-start GRPO | $2,106 |
| Hand-written verify-and-repair checks | $120 |

- Inference is under $0.50 per 1,000 workflows.
- **Projections, not savings.**

## 25. Limitations

- One base model, 3.35B, from one family.
- One simulated domain for the main study.
- Task-level intervals hold the models fixed.
- Seed-level power is low (5 runs).
- Trained skills do not transfer to new workflows or tool surfaces.
- The policy licence is non-commercial.
- ROI rests on assumed fault rates and costs.
- One human annotator.
- No mechanistic causal claim (A16 is not specific; CircuitKIT failed).

## 26. Dual use and ethics

The environment injects benign faults; no real payment systems or personal data are involved. A policy trained to pass verification could in principle learn to game a verifier. Our verifier reads state the policy cannot write except through the audited tools. Judge models received only synthetic procurement transcripts. The models are released for research under their licences.

## 27. Future work

- A second model family and a commercially licensed base.
- Own-history recovery starts.
- A runtime gateway evaluated with trained agents on ProcureSim.
- Retention mitigations, if R1 shows a loss.
- The candidate algorithms from the separate discovery programme (`docs/research/algorithm_discovery/`), which go to a confirmatory test only if they pass development gates.

## 28. Artefacts

- The GitHub repository.
- The HF collection: 30 adapters with cards, ProcureSim splits and evaluation traces.
- The website.
- New this round: J1 cases, rubric and all judge replies; R1 frozen inputs and generations.

## 29. Reproduction

| Step | Command |
|---|---|
| Numbers | `python scripts/s2_numcheck.py` (290 numbers, 0 failures) |
| Inventory | `python scripts/build_inventory.py` |
| Registrations | `python scripts/freeze_<id>.py --check` |
| ROI | `python scripts/roi_model.py` |
| R1 scoring | `.venv-r1/bin/python scripts/r1_score.py && .venv-r1/bin/python scripts/r1_stats.py` |
| J1 | `python evaluation/llm_judge_committee/analyze.py` |
| Training | `slurm/tasks/*.sh` (one file per registered run) |

Command reference: `docs/stage2/COMMANDS.md`.

## 30. Declaration of work done outside the hackathon

- **AUTHOR TO CONFIRM:** whether any part predates Nov 1, 2026.
- **Stage 1** (Nov 1–15): SupportSim, PCSTR rewards, A4–A11.
- **Stage 2** (Nov 20–27): ProcureSim, A12–SR1, E1/E2, and this round's R1, J1, ROI and C5/G.
- The toolkits are the hackathon's own repositories, vendored at the stated revisions.

## 31. Author contributions

Atharv Gaur:
- research direction, hypotheses and the review of every registration;
- Hindi and Hinglish review;
- human calibration labels;
- the video.

Claude (Anthropic) implemented the code, ran the experiments and drafted the documents under the author's direction (§33).

## 32. Acknowledgements

Lexsi Labs for the hackathon, its toolkits and HF credits; BITS Pilani HPC (sharanga) for GPU time; Cohere Labs for the Aya models.

## 33. LLM usage statement

Claude (Anthropic) was used throughout, under human review of every registration and claim:
- writing the code;
- running and monitoring experiments;
- drafting registrations, documents and this write-up;
- literature search.

LLM judges (§15) are evaluation instruments whose outputs never replace the database verifier.

## 34. Changelog

- **Stage 1** (Nov 14): SupportSim study.
- **Stage 2** (Nov 20–23): ProcureSim A12–SR1, E1/E2.
- **Nov 23, this round:**
  - audit and inventory (C3 ledger sync);
  - R1, R1-T and J1 registered and run;
  - ROI model;
  - website sections 06–08;
  - Stage 5 development (C5, G);
  - a separate algorithm-discovery programme.
- **Deviations:** `docs/stage2/DECISIONS.md` (this round: D-R1-1).

## 35. Appendix pointers

| Topic | Location |
|---|---|
| Registrations | `PREREGISTRATION_*.md` |
| Claims | `claims/claims.yaml`, `docs/final/FINAL_CLAIMS.md` |
| Audit | `docs/upgrade/AUDIT.md` |
| Inventory | `docs/upgrade/INVENTORY.md` |
| ROI | `docs/upgrade/ROI.md` |
| Toolkits | `docs/upgrade/TOOLKITS.md` |
| Traces | `docs/submission/TRACES.md` |
| LaTeX paper | `paper/main.tex`, `paper/appendix.tex` |
