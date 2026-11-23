# Statistical analysis

## 1. What is confirmatory and what is not

| Evidence | Status | Decision rule |
|---|---|---|
| AD-C1 (first stage, LLM, cw_conf, 5 vs 5 seeds) | **confirmatory**, registered before any cw_conf output | one-sided exact seed-level permutation (252 assignments), Holm over all C1 hypotheses; supported iff Holm p < 0.05 **and** the runs-fixed paired task-bootstrap 95% CI excludes 0 (`scripts/ad/c1_stats.py`) |
| V2 testbed TB1–TB6 | **development**, registered before output, CPU only | registered hypotheses H1a–H1g and gates T1–T3 (`scripts/ad2/tab_analyze.py`); not significance tests |
| TB2 secondary statistics | **descriptive** | paired (setting, seed) units; 10,000-resample bootstrap CI; Cohen's d_z; one-sided sign-flip permutation (20,000 draws); Holm over every reported pair (`scripts/ad2/tab_stats.py`) |
| First-stage cw_dev development runs | development, 2 seeds | first-stage gates G1–G4; no tests |
| derivation_numeric | numerical demonstration | none |

**No v2 algorithm has confirmatory evidence.** The only v2 LLM experiment (E1) was suspended before any output
(DV-2), and no v2 candidate was trained on the LLM.

## 2. Testbed statistics: units and pairing

- **TB1.** 216 cells × 18 arms × 3 seeds. Agreement and wrong-decision rates pool cells and seeds (n = 3 × cells
  with |VOI| > 0.02 and c ≥ 0.01). Regret is exact per run, so the only randomness is training.
- **TB2.** 12 settings (mode × c × L_f) × 5 seeds. Each run trains on the three rich workflows jointly. The unit of
  analysis is (setting, seed): 60 paired units per comparison. Pairing is by setting and seed index. Different
  estimators share the seed index but not their random streams, so the pairing controls the setting, not the
  sampling noise.
- **TB3.** 5 seeds; regret averaged over each split's tasks.
- **TB4.** 5 seeds on one setting.
- **TB6.** 9 cells × 3 seeds for each combination of factors.

## 3. Multiple comparisons

- **AD-C1:** Holm over every C1 hypothesis (registered).
- **TB2 secondary:** Holm over every (candidate, baseline) pair printed by `tab_stats.py`.
- **TB1 hypotheses:** each is reported separately with its registered threshold. Nothing is pooled into a
  significance claim.

## 4. Compute and wall-clock (actual)

| Item | Where | Wall-clock |
|---|---|---|
| TB1, 11,664 runs | 16-core laptop, 14 workers | 2,094 s |
| TB6, 1,944 runs | same | about 5 min |
| TB4, 135 runs | same | about 2 min |
| TB3, 75 runs | same | about 3 min |
| TB2, 2,220 runs | same | 1,914 s |
| derivation_numeric | same, 1 core | about 3 min |
| V2 GPU | A100, task 1300 | < 1 min (failed at start; DV-2) |
| First-stage AD runs | H100/H200 | about 1.5 h each (logs in `queue/*/logs` on sharanga) |

## 5. Results

The TB2 and AD-C1 results are reported in SELECTIVE_VERIFICATION_RESULTS.md and FINAL_ALGORITHM_REPORT.md, from
their summary files only.
