# Final scientific review (deliverable L)

Draft of 2026-11-23, about 21:55 IST, on branch `actionproof-research-upgrade`.

- **Every row is backed by a ledger entry:** `claims/claims.yaml` (316 numbers re-read from their files, 0 failures).
- **Pending:** the algorithm-discovery confirmatory study (AD-C1) and P2 (structured ledger, action guard, arm E; `PREREGISTRATION_P2.md`). The P1 frozen test reported on 2026-11-24 at 05:58 IST (`results/p1/p1_summary.json`). The second-stage programme reported on 2026-11-23 at 23:35 IST and has been merged. This file is updated when the pending studies report.

## Established: registered, rule met

| Finding | Evidence | Strength and caveat |
|---|---|---|
| The reward's information source selects read-back: database reward 1.00 vs acknowledgement reward 0.00, in 5 of 5 paired seeds; +0.27 to +0.32 VSC per seed | A12 | One base model. R_A barely moves from the warm start. The seed test is at its 5-seed minimum (Holm 0.0625) |
| The reward selects reading; it does not create it | ID1 | Two seeds |
| Recovery-start curriculum: +0.21 VSC over the state-reward agent, 5 of 5 | S3 | Recipe comparison; adds a fault class |
| Start-state effect on recovery starts vs a class-matched control, five runs per arm | S3 addendum 2 (Holm 0.036) | Not partial-specific |
| Gains hold under Hindi, Hinglish and two-fault episodes | S4-EVAL | Task-level intervals |
| **No measurable capability loss**: GRPO non-inferior to its warm start within 0.03 on IFEval and BFCL, both arms | R1 H2 | Single-turn benchmarks; prompting mode |
| **A practical recovery agent** (P1 frozen test, 450 fresh tasks, sampled): recovery SFT +0.259 [+0.231, +0.287] and recovery SFT then the same GRPO +0.306 [+0.276, +0.337] over the recovery-start agent (0.513), Holm 0.0003, every seed positive | P1 H2, H3 | One base model, 3 seeds. Descriptive: partial payments 0.319 → 0.931, but the unseen workflow only 0.486. The demonstrations encode the registered repair rule, so this shows the rule can be taught, not discovered |
| Hand-written read-back-and-repair checks beat every trained agent in ProcureSim (0.917 VSC, 0 false success) | `results/roi/scripted_s3_conf_full.json` | The scripted checker knows the fault taxonomy |

## Not supported or failed (registered rules not met)

- Renamed tools and an unseen workflow: no gain (S4).
- SR1 tool-name randomisation: +0.00 on renamed tools, −0.248 on canonical tools.
- Partial-specific start-state effect; addendum 4 (handed evidence) inconclusive; E1 sampled contrasts.
- A11 interaction: not made. Negative transfer to healthcare administration.
- **FCA** (−0.10), **C5** (−0.32 and −0.37; it penalises reads by construction) and **G** (−0.11, −0.033): all dropped by criteria fixed before training.
- R1 H1 (SFT costs capability) and H3 (training lowers abstention): not supported. Abstention is low in the base model itself.
- **P1 H1**: a reconciliation line at inference lowers sampled VSC (−0.030 [−0.041, −0.019]); it induces same-key top-ups (`docs/research/P1_DEV_DIAGNOSIS.md`).
- J1 J-H1 (warned judges fooled) and J-H2 (snapshot judges ≥ 0.90 for 3 of 4): not supported.

## Exploratory: real measurements, registered after looking or descriptive

- **J1 addendum 1.** One rubric sentence decides whether judges are fooled. Without it: 120 of 120, 120 of 120, 115 of 120 and 9 of 120 unverifiable failures accepted; with it: 0–1.
- **J1 bias.** Every judge misreads a refunded duplicate as a double payment (4 snapshot cases).
- **τ-bench (R1-T).** Floor regime: base pass^1 0.058. SFT −0.020 [−0.038, −0.006]; GRPO no change.
- **C5 diagnosis.** Read turns receive negative normalised credit in every block (−0.5 to −0.7).
- **Algorithm discovery (development, 2 seeds).** At read cost 0.05, group-std GRPO stops reading in 2 of 2 seeds. AWRB 0.867 vs no-std 0.833 vs GRPO 0.707. No arm verifies selectively.
- **Algorithm discovery v2** (`docs/research/algorithm_discovery_v2/FINAL_ALGORITHM_REPORT.md`; derivation plus a CPU testbed with exact optima).
  - **The vote rule.** Under group std, a verification cost cancels in every group where verifying changes only cost, so GRPO verifies iff P(verifying changes the outcome) ≳ 1/2. The value-of-information threshold plays no role.
  - **Testbed.** GRPO gives the wrong decision in 64% of cells, against 5.7% for no-std.
  - **LLM logs.** The decomposition was reproduced exactly: faulted groups split +49.6 / −49.6.
  - **AWRB audit (negative).** Its VSC lead over no-std comes from over-verifying, and the unbiased variant is below no-std. It also needs the simulator and the prior.
  - **igc** (exact local credit for provably uninformative reads) passes development gates in the testbed only, as a sample-efficiency gain. It is not a new mechanism.
- **P1 development diagnosis** (`docs/research/P1_DEV_DIAGNOSIS.md`). The inference-time reconciliation line induces same-key top-ups on short payments: 12 of 14 false successes, against 1 of 14 without it.
- **ROI.** Projections from measured rates and cited assumptions, not savings.

## Untested or out of scope

- A second model family, and a commercially licensed base (the policy is CC-BY-NC-4.0).
- A runtime verification gateway with trained agents on ProcureSim.
- Real deployment data.
- A mechanistic causal claim: CircuitKIT failed, and A16 is not specific.
- Human calibration of the judges: labels pending (one annotator).

## Strongest defensible contribution

**The setting.** Silent write commits with byte-identical acknowledgements, partial and duplicate effects, and an authoritative verifier.

**Registered measurements in that setting:**
- reward observability selects verification;
- verification is not recovery;
- the gains bind to the trained procedure;
- no general-capability cost was measured.

**An honest map of what monitors and optimisers do there:**
- LLM judges are only as cautious as their prompt;
- state-progress credit penalises evidence;
- group-std GRPO makes the decision to verify blind to its cost whenever the advantage scale depends on the hidden state of the sampled rollouts. This is derived and shown in a controlled testbed and in the training logs. The 1/σ mechanism is Bereket & Leskovec's and the fix is Dr. GRPO's; the consequence for verification is ours.

**No new optimiser is claimed.** That changes only if AD-C1 or the second-stage programme passes its registered confirmatory test.
