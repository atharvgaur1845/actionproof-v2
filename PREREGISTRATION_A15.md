# A15 and A16: amount-ratio evidence use, and what changes inside the model

Drafted 2026-11-21. The registration time is the commit time of this file and `results/a15_hashes.json`, which is before any model is run on `proc_mech`. Development data (`proc_mech_dev`) was used only by the CPU scripted pilot and the GPU pilot (task 709: one model on 10 tasks; A16 hook check, margins and a 3-pair layer sweep). Pilot outputs are kept in `results/a15/pilot` and `results/a16/pilot` and are not analysed. Every earlier registration is unchanged. The cross-domain arm of A15 and the A17 transfer evaluation are registered separately (`PREREGISTRATION_A17.md`).

## 1. Why

A14 found that every policy, trained or not, picks the correct next action when forced evidence shows a payment present or absent (BCA 1.00). Only on the partial branch did they differ: R_V 0.35–0.48, R_A 0, warm start 0.03. ProcureSim produces only one partial amount, half the order (`env.py`, `amount_cents // 2`). A15 asks whether R_V's partial response reflects comparing the paid amount with the order amount, or a learned response to that one pattern. A16 asks what differs inside the models at that decision point, and whether it is causally used.

## 2. A15 design

- **Split:** `proc_mech`, 200 tasks, seed 111, workflows with pay before post. PO ids and task ids are disjoint from every other ProcureSim split (asserted). Order amounts are restricted so the paid amounts at every ratio have the same number of digits. `proc_mech_dev` (60 tasks, seed 112) is for development only.
- **Branches:** the fault is on pay attempt 0, as in A14. The prefix is scripted and policy-independent, the acknowledgement is ok in every branch, and the forced `list_payments` result is in the agent's tool-call format. Branches differ only in the stored payment:

| Branch | Stored payment (q of order amount) | Correct first decisive action |
|---|---|---|
| Q0 | none (phantom) | re-pay (any key) |
| Q1 | 1/4 | refund that payment (its `payment_id`) or escalate |
| Q2 | 1/2 (the A14 pattern) | same |
| Q3 | 3/4 | same |
| Q4 | full | post the journal entry |

- **Implementation:** `scripts/a15_eval.py`. It imports the frozen A14 harness (`scripts/cef_eval.py`) unchanged and points it at `RatioProcEnv`, which equals ProcEnv at q = 1/2 (tested).
- **Policies:** warm start; R_V and R_A seeds 0–4 (step-200 adapters in `results/a12/frozen_models.json` and `frozen_models_later.json`); scripted oracle, never-read and fixed-retry references. Greedy decoding, up to 16 turns, one run per model.
- **Scoring:** the first decisive action uses the A14 classifier. In the partial branches, **a refund counts only if it names the stored payment's id** (strict rule; A14 accepted any refund). The A14 rule is reported alongside.

## 3. A15 endpoints and rules

- **Primary:** accuracy on the unseen ratios, the mean of Q1 and Q3, per R_V seed. Contrasts per seed: R_V minus warm start, and R_V minus R_A (same seed). Paired task bootstrap (10,000 resamples) and exact McNemar on discordant tasks.
- **Key secondary:** the per-ratio accuracy curve, with Q2 reported as the trained pattern. Also, Q2 compared with A14's partial result as a consistency check.
- **Secondary:** binary accuracy (mean of Q0 and Q4; expected near 1, replicating A14); accuracy under the A14 rule; end-to-end goal met, duplicates and report accuracy per branch; the teacher-forced margin from A16.
- **Rules (δ = 0.10, as in A14):**
  - *Generalises:* the lower 95% bound of unseen-ratio accuracy exceeds the warm start's accuracy by more than δ.
  - *Pattern-specific:* Q2 accuracy exceeds unseen-ratio accuracy by more than δ (paired 95% interval).
  - *Inconclusive:* neither, reported as such.
- Five seeds give an exact sign-flip resolution of 1/32 one-sided and 2/32 two-sided. Seeds are reported individually.

## 4. A16 design (`scripts/a16_mech.py`)

Decision points are the A15 branches. `proc_mech_dev` is used for fitting and layer choice. Test is `proc_mech`: the first 100 tasks for the trajectory and the causal tests.

- **Margin:** M = log P(refund or escalate) − log P(post), teacher-forced on the first tool name (`[{"tool_name": "` + tool + `",`), exact.
- **Hooks:** hidden-state index L is the output of decoder layer L−1. Before any measurement, every job checks that the hooks reproduce `output_hidden_states` and that a self-patch leaves M unchanged. Stage 1's interpretability run hooked one layer later than its probe; that bug is fixed here and tested.

**Analyses**, in order of evidential weight. Only items 3 and 4 can support a causal claim.

1. **Trajectory (descriptive).** Models: the warm start (step 0), and R_V and R_A seeds 0–2 at steps 50, 100, 150 and 200. Reported per checkpoint: M by branch.
2. **Probes (descriptive).** Last-prompt-token residual stream at every layer; label partial (Q1–Q3) vs complete (Q4); logistic regression (C = 1, standardised), fitted on dev and scored on test by AUROC. Two variants: a probe refitted per checkpoint, and the warm-start probe applied frozen. **Expected near 1.0 everywhere**, because the amount is in the text (Stage 1 found the same). No claim rests on decodability.
3. **Activation patching.**
   - *Layer choice (dev only, fixed rule):* the layer with the largest mean normalised final-token patch effect (complete-branch state written into the partial branch) for R_V seed 0, over 30 dev pairs. The rule runs before any test pair is touched.
   - *Measurements (test, chosen layer):* per pair, the normalised effect (M_patched − M_base)/(M_donor − M_base), complete→partial and partial→complete, at the final prompt token and at the token span where the evidence differs (token-aligned pairs only).
   - *Exclusion:* pairs with |M_donor − M_base| < 1 nat are excluded and counted.
   - *Controls (first 50 eligible pairs):* self-patch, 20 random same-norm displacements, and a donor from a different task.
4. **Direction ablation.** The difference of means (partial − complete) at the chosen layer, final token, fitted on dev for each model, is projected out from the final prompt token onwards on test. Reported: ΔM per branch. Controls: 20 random unit directions on the first 20 test tasks. Collateral: ΔM on Q0 and Q4.
- **Models for 3–4:** warm start, R_V seeds 0–4, R_A seeds 0–2.

## 5. A16 interpretation (fixed now)

- **Causal role at the chosen layer** (claimed only if all three hold for R_V):
  - the median final-token complete→partial effect exceeds the 95th percentile of the random-control effects;
  - the self-patch effect is within 0.05 of 0;
  - both hold in at least 4 of 5 R_V seeds.
- **Ablation effect:** the mean ΔM on Q1–Q3 is negative and larger in magnitude than the largest random-direction ΔM, in at least 4 of 5 R_V seeds, with |ΔM| on Q4 below half the partial-branch effect.
- **Models with no evidence-dependent action:** if a model's |M_complete − M_partial| < 1 nat on most pairs (expected for R_A and the warm start), this is reported as "no evidence-dependent action to localise", not as a null mechanism.
- **Otherwise:** not established, reported as such.
- **CircuitKIT:** one exploratory 45-minute smoke test (EAP-IG on 20 dev pairs, R_V seed 0). It is reported only if it runs, and is never used for a claim.

## 6. Protocol

Models are the frozen step-200 adapters and the intermediate checkpoints saved during A12 training (`save_steps` 25). Runs use HF Jobs and the Slurm H200 worker; the executor is logged. Deviations go to `docs/stage2/DECISIONS.md` with their time.
