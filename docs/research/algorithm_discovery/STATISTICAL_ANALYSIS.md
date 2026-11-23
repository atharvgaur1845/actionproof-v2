# Statistical analysis

## 1. Levels of evidence used in this programme

| Level | Where | What may be claimed |
|---|---|---|
| Tabular derivation and exact enumeration | AD-T0b, `tests/ad/test_ad.py` | Sign statements P1–P5 for one aliased decision; exact unbiasedness and variance facts for AWRB (no sampling error: full enumeration) |
| Development (2 seeds, cw_dev) | D1, D2, D3 | Gate decisions only (G1–G4), plus descriptive mechanism readouts. **No significance claims.** |
| Mechanism on training data | `advantage_decomposition.py`, `analyze_d1.py` | Descriptive decompositions of the logged returns |
| Confirmatory (5 seeds, cw_conf, once) | AD-C1 + addendum 1 | Hypotheses H_c, H_AWRB, H_D3 under §2 |

## 2. Confirmatory procedure (registered in PREREGISTRATION_AD_C1.md before any output)

- **Unit.** The training run (seed). Five runs per arm, seeds 0–4. Seeds 0–1 are the frozen development runs; their
  cw_conf outputs had never been seen.
- **Statistic.** The seed-averaged difference in cw_conf greedy VSC (H_D3b: J_c).
- **Test.** A one-sided unpaired permutation test over the C(10,5) = 252 assignments of runs to arms. The smallest
  attainable p is 0.004.
- **Multiplicity.** Holm over every hypothesis tested.
- **Decision.** Supported iff Holm p < 0.05 and the seed-averaged paired task-bootstrap 95% interval excludes 0.
  That interval holds the runs fixed (the project's convention).
- **Also reported:**
  - a two-level bootstrap interval (seeds, then tasks), which reflects run-to-run variation;
  - per-seed differences;
  - the number of positive seeds.
- **Code.** `scripts/ad/c1_stats.py`, committed before any C1 output (b352c07).

## 3. Results

PENDING: from `results/ad/c1_summary.json` only.
