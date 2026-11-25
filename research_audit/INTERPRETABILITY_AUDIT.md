# Interpretability audit (A15/A16)

2026-11-25. Sources:
- `PREREGISTRATION_A15.md` (frozen 2026-11-21, `results/a15_hashes.json`);
- `results/a16/a16_summary.json` and `causal__*.json`;
- `docs/stage2/DECISIONS.md` D-A16-1.

## 1. Design (as registered), checked

| Item | What the registration fixes | Audit |
|---|---|---|
| Layer indexing | Hidden-state index L is the output of decoder layer L − 1. Every job first checks that its hooks reproduce `output_hidden_states` | sound; checked in code before any measurement |
| Probe construction | Last-prompt-token residual at every layer; logistic regression (C = 1, standardised); partial (Q1–Q3) vs complete (Q4) | — |
| Train/test separation | Fit and layer choice on `proc_mech_dev`; tests on `proc_mech` (first 100 tasks) | sound: separate splits; the dev/test prompt-overlap check found no overlap |
| Patch selection | Layer chosen on dev only, by a fixed rule (largest mean normalised final-token effect for R_V) | chosen layer **18**, fixed before test |
| Controls | Self-patch; 20 random same-norm displacements; a donor from a different task; first 50 eligible pairs | present for every model |
| Checkpoints | Warm start, R_V seeds 0–4, R_A seeds 0–2; intermediate A12 checkpoints for the trajectory | present |
| Claim rules | Fixed in §5 before any test output | applied as written |

## 2. Results at the three levels

**Level 1: decodable. Established.** At layer 18 every checkpoint linearly separates partial from complete branches (probe accuracy 1.0, refit and frozen), from step 50 onward and in both reward arms. **The information is present in every model, including those that never act on it.**

**Level 2: influences output. Partly.** The action margin M differs between partial and complete branches by about 2 nats in every model, including R_A. Training changes the margin's size, not its existence.

**Level 3: causally used in the decision. Not established.** Values are R_V seeds 0–4, final token, layer 18 (`causal__*.json`).

| Test | Rule (registered) | Result |
|---|---|---|
| Patching (complete → partial) | median effect > 95th percentile of random controls, self-patch within 0.05 of 0, in ≥ 4/5 R_V seeds | **met in 0/5.** Medians 0.037–0.056 against random p95 0.037–0.059. Self-patch exactly 0 |
| Wrong-task donor | (control) | **Larger than the matched patch in every model**: 0.050–0.088 for R_V, 0.11–0.13 for R_A. The patch effect is not specific to the evidence |
| Direction ablation | mean ΔM on Q1–Q3 negative and larger than the largest random-direction ΔM, in ≥ 4/5 R_V seeds, with \|ΔM\| on Q4 below half the partial effect | **met in 5/5 R_V seeds** (ΔM −0.49 to −1.14 against random max 0.13–0.24) |
| Specificity of the ablation | (descriptive) | Also met in R_A seeds 0 and 2 (ΔM about −0.2). Every model, including the warm start, loses about 0.3 on Q0, the absent-payment (phantom) branch, which is outside the partial-vs-complete contrast the direction was fitted on |

**Reading.**
- The partial-vs-complete direction is decodable everywhere.
- Projecting it out lowers the partial-branch margin more in R_V models than in R_A models. That is a **difference in degree, not a mechanism unique to R_V**.
- The ablation also has a collateral effect on the absent-payment branch (Q0).
- Patching gives no causal evidence beyond the random and wrong-task controls.

**The publishable statement:** "information is decodable in every model; projecting out its direction reduces the trained action margin, more strongly after R_V training; activation patching did not meet its registered causal rule". This matches the README ("the ablation that lowers it is not specific to R_V").

## 3. CircuitKIT

EAP-IG (exploratory, 45-minute budget) **never ran**. Version 0.2.4 installs and imports, but its discovery path fails importing a transformers class (D-A16-1). No circuit claim exists or is made.

## 4. What should not be claimed

- "RL creates a representation of partial payments": every model, including the warm start, decodes it.
- Any "mechanism" or "circuit" language.
- A causal role from patching: the rule failed, and the wrong-task donor beats the matched donor.

## 5. No new interpretability method this sprint

The brief asks for none without a concrete hypothesis. The one concrete follow-up: does the ablation's effect on Q0 (absent payment) mean the direction encodes "payment not complete" generally rather than "partial"? Test it by fitting a Q0-vs-Q4 direction and comparing it with the partial direction (cosine, plus ablation cross-effects). Next command, after registering it as an A16 addendum: `scripts/a16_causal.py --control-direction q0q4` (to be written).
