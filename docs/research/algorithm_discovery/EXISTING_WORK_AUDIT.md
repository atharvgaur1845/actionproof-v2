# Existing work audit (Phase I)

Written 2026-11-23 by the algorithm-discovery programme.
- **Branch:** `research/algorithm-discovery`, cut from `actionproof-research-upgrade` at 637dcc8. That branch contains
  `stage4` → `research/e1e2` → upgrade.
- **Sources read:** the code, `docs/research/*`, `docs/final/FINAL_CLAIMS.md`, the registrations, the result files,
  and the read-only research-mandate report (`../research_mandate/REPORT.md`, 2026-11-22).
- **Independent recomputation:** `scripts/ad/recompute_audit.py` → `results/ad/audit/recompute.json`. It shares no
  code with the project's statistics scripts.

**Labels:**
- **[M]** a measured result that I recomputed from raw per-episode files;
- **[R]** reported in a project file, not independently recomputed here;
- **[H]** a hypothesis;
- **[I]** an interpretation;
- **[U]** an unresolved question.

## 1. System

| Part | What it is | Where |
|---|---|---|
| Environment | ProcureSim: five services (oms, supplier, payments, inventory, accounting), append-only tables, one write chokepoint `_write` where a per-task fault plan applies | `actionproof/integrity/sim/env.py`, `actionproof/stage3/env.py` (partial ratios) |
| Hidden state | Fault kinds per (tool, attempt), listed below; `never_commit` | same |
| Observation | The JSON tool result of each call. The acknowledgement of a write is `{"status":"ok", <id>}` whatever happened, or a TIMEOUT error for the two timeout kinds. Reads return authoritative rows (stale for `replica_lag`). | `env.py` |
| Actions | One assistant turn: a JSON list of tool calls (in practice one call), parsed by AgentTune's parser. 5 read tools, 6 write tools (incl. `refund_payment`, `escalate_to_human`), `submit_report` | `actionproof/stage3/rollout.py` |
| Verifier | Final-state step verdicts (VERIFIED / ABSENT / PARTIAL / DUPLICATE / WRONG) per requested step, collateral rows, report status. It never reads acknowledgements or report text. | `actionproof/integrity/sim/truth.py` |
| Rewards | R_V = 1 iff every requested step verifies (report-blind, collateral-blind); R_A = 1 iff the last acknowledgement of every requested write was ok. Also `safe`, `lag` (Stage 3 pilots), PCSTR (Stage 1). | `truth.py:score`, `actionproof/stage3/rewards.py` |
| Primary metric | VSC = goal met (all steps verified, no collateral) and report "completed" | `truth.py` |
| Policy | Tiny Aya Global (`CohereLabs/tiny-aya-global`, Cohere2, 3.35B, 8,192-token context); LoRA r 16 on all projections | `actionproof/train/*`, `stage3/train.py` |
| Initialisation | SFT warm start `runs/rft_proc` (rejection-sampled and oracle demonstrations; samples read-back after 31% of phantom writes at T = 1, 58/188 [R]) | `actionproof/train/rft.py`, FINAL_CLAIMS row 2 |
| Trainer | TRL 1.7.1 GRPO via AgentTune `TrlAgenticGrpo`, as configured in `stage3/train.py` (TRL loss defaults otherwise) | `stage3/train.py` |

Trainer settings:
- 4 prompts × 8 rollouts per step, 200 steps;
- lr 2e-5, β (KL) 0.02, T = 1;
- 16 turns, 256 tokens per turn, max_grad_norm 0.5;
- vLLM colocated;
- `scale_rewards="group"` (TRL default);
- loss on policy tokens only (`env_mask`).

Fault kinds:

| Kind | Effect |
|---|---|
| ok | normal commit |
| phantom | acknowledged, nothing committed |
| partial (ratio) | acknowledged, a fraction committed |
| lost_ack | committed, TIMEOUT shown |
| precommit_timeout | not committed, TIMEOUT shown |
| dup_delivery | committed twice |
| replica_lag | the next read shows stale state |

Training objective: maximise E[R] by GRPO. Advantages are group-standardised sequence rewards, with a DAPO-style token
loss and β-weighted KL to the warm start.

**A structural fact no earlier document states, and the starting point of this programme.** Every rollout in a GRPO
group shares one task, and so one hidden fault world. `meta.faults` is fixed per task, and `make_env_s3` reads it.
Group standardisation therefore compares rollouts that all face the same hidden world.

## 2. Splits

The generators record the seeds.
- Stage 2: `actionproof/bench/generator_proc.py`, splits proc_*.
- Stage 3: `actionproof/stage3/tasks.py`, splits s3_*.
- S4: `actionproof/stage4/shift.py`, `surface.py`.
- E1/E2 splits.

Fault prevalence is high in training:
- `s3_trainb_half` carries a fault in 158/200 tasks;
- `s3_rec_train_half` carries a non-"complete" fault in 133/200.

Reads are free in every registered experiment.

## 3. Everything tried before, with status

| Id | What | Registration | Status | Key result | Label |
|---|---|---|---|---|---|
| Stage 1 (A4–A11) | E-commerce simulator; 7 reward arms (tool_trust, llm_judge, outcome, process, PCSTR…) | PREREGISTRATION_A4–A11 | done | Best-baseline margin +1.5 pp (small); the tool-trust reward contrast is larger | [R] |
| A12 | R_V vs R_A on ProcureSim, 5+5 seeds | PREREGISTRATION_A12 | done | Phantom read-back 1.00 (R_V) vs 0.00 (R_A) every seed; VSC +0.27 to +0.32 per seed | [R] (raw per-episode A12 files are on the cluster, not in this checkout) |
| ID1 | No-verification warm start, then R_V | PREREGISTRATION_ID1 | done | 0/153 read-back after training: the reward *selects* reading, it does not create it | [R] |
| A14/A15 | Forced evidence; workflow position | A14, A15 | done | Simple evidence is acted on (1.00); partial repair bound to workflow position (pay-first 0.99–1.00, after a goods receipt 0.00) | [R] |
| A16 | Probes and patching | A16 | done | Evidence decodable; action margin linked; patching causal but not specific; no circuit | [R] |
| A17 | Sibling domains | PREREGISTRATION_A17 | done | Transfer mostly absent; care_admin negative | [R] |
| S3 | Recovery-start curriculum (RSC) vs A12 R_V, 5+5 | PREREGISTRATION_S3 | done | s3_conf_rec **+0.206**; s3_conf_full **+0.209**; stress +0.388; 5/5 seeds each | **[M]** (recomputed exactly) |
| S3 add. 1/2 | RSC vs class-matched control | ADDENDUM1, ADDENDUM2 | done | 5 seeds: s3_conf_rec **+0.107**, 5/5 positive (Holm 0.036, supported); partial and duplicate class effects not supported | **[M]** (point estimate) |
| S3 add. 2B | Workflow-rebalanced pools | ADDENDUM2 | done | After-receipt repair 0.005: "not frequency" | [R] |
| S3 add. 3 | State-only starts | ADDENDUM3 | withdrawn before training (degenerate design) | — | [R] |
| S3 add. 4 | Acknowledgement-only recovery starts | ADDENDUM4 | done | D1 +0.203 (Holm 0.206), D2 +0.181 (Holm 0.262) on the 64-task partial-payment endpoint; inconclusive. Learners 5/4/2 of 5 | [R] (endpoint checked; whole-split s3_conf_rec: D1 +0.027, D2 +0.080 [M]) |
| FCA | Forked counterfactual action credit, 2 seeds | FCA_GRPO.md §7 | **dropped** | s3_rec_dev **−0.097** vs RSC (kill criterion: +0.03 needed) | **[M]** |
| S4-EVAL | 7 shift splits, 14+ frozen models | PREREGISTRATION_S4_EVAL + add. 1 | done | Renamed API **−0.016**; Hindi **+0.127**; Hinglish **+0.072**; unseen 4-step workflow **+0.016**; two faults **+0.072** (RSC − R_V) | **[M]** |
| SR1 | Tool-name randomisation in training, 3 seeds | PREREGISTRATION_SR1 | done | Renamed API **+0.000**; canonical tools **−0.248**; wf4 +0.11 (one run) | **[M]** |
| E1/E2 | Sampled evaluation; forced decision-point probes | PREREGISTRATION_E1E2 | done | Sampled RSC − control +0.098 (Holm 0.103), not supported; "threshold" reading (ρ = 0.92); reconciliation line +0.48 | [R] |
| C5, G | Verified-progress turn credit; realizable hint sampling | `docs/research/algorithm/C5G_DEV.md` | **running** (tasks 1010–1013) | — | [U] |
| R1 | Capability retention (IFEval, BFCL, τ-bench) | PREREGISTRATION_R1 (main session) | running/queued | — | [U] |

**Baselines already on file:**
- warm start;
- R_A (acknowledgement reward);
- R_V (state reward, A12);
- RSC;
- class-matched control;
- acknowledgement-only starts;
- SR1;
- FCA;
- scripted oracle / naive policies;
- Stage 1 reward arms (tool_trust, outcome, LLM judge, PCSTR);
- prompt-only verify instruction (PROC_VERIFY).

**Not on file:**
- a PPO/critic baseline;
- a no-std GRPO baseline;
- any cost-of-verification setting.

## 4. Recomputation notes (`results/ad/audit/recompute.json`)

- **Point estimates.** Every S3, S3-A2, S4, SR1 and FCA point estimate I recomputed from raw episodes equals the
  reported one to 3 decimals.
- **Intervals.** The reported 95% intervals ("seed_avg ci95") resample tasks with the trained runs held fixed. The
  ledger says so (FINAL_CLAIMS conventions). A two-level bootstrap (seeds, then tasks) is wider, and for three
  claims the change matters:

  | Claim | Reported (runs fixed) | Two-level |
  |---|---|---|
  | G3 (unseen workflow) | [0.004, 0.030] | [−0.024, 0.076] |
  | SR1 renamed API | [−0.04, 0.04] | [−0.067, 0.062] |
  | S3-A2 H3′ | [0.085, 0.131] | [0.028, 0.205] |

  The registered decisions used seed-level permutation tests, so no verdict changes. Any sentence that reads a
  runs-fixed interval as uncertainty over training runs would be wrong.
- **Addendum 4.** Its endpoint is the 64-task partial-payment subset of s3_conf_full, not the whole split. On the
  whole s3_conf_rec split the same contrasts are small (D1 +0.027, D2 +0.080) and cross zero in the two-level
  interval. The registered reading ("inconclusive") stands.

## 5. Which claims are supported, weakened or open

| Status | Claims |
|---|---|
| **Supported (registered)** | The R_V vs R_A observability contrast (A12). RSC > A12 R_V on recovery starts, full episodes, stress and the Hindi, Hinglish and two-fault shifts. RSC > class-matched control on recovery starts (5 seeds). |
| **Weakened or falsified** | A partial-payment-specific start-state effect (3 seeds → 5 seeds: not supported). The three-seed safety trade-off (reversed). FCA (killed). Renamed-API and unseen-workflow generalisation (not supported). SR1 as the fix (fails non-inferiority). Transfer (negative in care_admin). |
| **Interpretations** | "Selection, not invention" (ID1) [I, well supported]. "Reading is not recovery; the binding is in acting after reading" [I, from E2 exploratory analyses]. "Greedy decoding exaggerates run differences" [I]. |
| **Unresolved** | Whether any optimiser change beats RSC (C5/G pending). Whether verification survives a read cost [U, this programme]. Whether retention holds (R1 pending). The after-receipt partial cell. |

## 6. Rejected algorithmic ideas, and why

| Idea | Reason rejected | Source |
|---|---|---|
| Forked counterfactual action credit (FCA) | Kill criterion | `FCA_RESULT.md` |
| Recovery-aware constrained / Lagrangian GRPO | Published (Constrained GRPO, Safety as a Budget, GDPO); its pilots gave nothing | `DECISION.md` |
| Adaptive recovery-frontier scheduling | Published (PLR, MoPPS, PAC, LILO…) | `DECISION.md` |
| Matched-world CRN credit (the mandate's candidate A) | Equals GRPO in expectation; targets a decision A12 already solved with free reads | mandate §6 |
| Counterfactual-world negative reinforcement (mandate B) | Changes the objective; hacking mode | mandate §5 |
| Certified invariance + contrast regularisation (mandate D) | Recommended but never implemented. Novelty 4/10 | mandate §5 |

## 7. Bottlenecks the evidence points to

1. **The post-evidence decision.** It is bound to workflow position and tool surface (A15, S4, E2).
2. **No gradient on never-commit or all-fail groups.** 46.7% of RSC groups have zero variance.
3. **The second step of the repair chain is dropped** (mandate D5).
4. **Untested and central to the assignment: costly verification.**
   - Reads are free in every experiment, so "verification" has only ever meant one unconditional read. A12's
     read-back of 1.00 is the optimum only because reading costs nothing.
   - The prior report says: "with a read cost c, plain policy gradient already penalises redundant reads; no new
     estimator is needed" (mandate §6.5).
   - That statement was never tested. Group-standardised GRPO is not plain policy gradient, and its groups share
     the hidden world (§1).
