# AD-C1: registered outcome vs the theory's frozen predictions

**Sources:**
- **Registered analysis:** `scripts/ad/c1_stats.py`, run once by the lead engineer (Slurm 469851) after
  `freeze_c1.py --check` passed. Output: `results/ad/c1_summary.json` (copied from
  `/scratch/pabitra/actionproof/results/ad/c1_summary.json`).
- **Per-seed greedy evaluations:** `results/ad/conf_summaries/ad_<arm>_s<k>__cw_conf.summary.json` (copied).
- **Predictions:** `docs/research/AD_C1_THEORY_PREDICTION.md`, frozen at commit 2354929 before any cw_conf output.

Nothing was recomputed. The values below are read from these JSON files. cw_conf has 300 tasks, 121 of them
faulted.

**Evidence level: confirmatory** for the registered hypotheses. The T-prediction checks are a pre-stated comparison,
not additional tests.

## Registered result (c1_summary.json; 5 vs 5 seeds, Holm over 4 tests)

| H | Δ VSC (seed avg) | 95% CI (runs fixed) | Holm p | Supported |
|---|---|---|---|---|
| H_nostd (nostd > grpo) | +0.120 | [0.088, 0.153] | 0.016 | yes |
| H_awrb (awrb > grpo) | +0.192 | [0.149, 0.239] | 0.016 | yes |
| H_awrbn (awrbn > grpo) | +0.032 | [0.017, 0.048] | 0.048 | yes |
| H_AWRB (awrbn > nostd) | −0.088 | — | — | no |
| H_D3 | not run: seeds 2–4 never queued (note 3) | | | untestable |

## Per seed (greedy, cw_conf)

| Arm | seed | VSC | reads/ep | false success | faulted-task VSC | clean-task VSC | verify-after order / pay / post / receive |
|---|---|---|---|---|---|---|---|
| grpo | 0–4 (identical) | 0.687 | 0.000 | 0.313 | 0.223 | 1.000 | 0 / 0 / 0 / 0 |
| nostd | 0 | 0.827 | 1.803 | 0.173 | 0.570 | 1.000 | 0.80 / 0.57 / 0.55 / 0.97 |
| nostd | 1 | 0.797 | 1.570 | 0.200 | 0.496 | 1.000 | 0.60 / 0.50 / 0.48 / 0.98 |
| nostd | 2 | 0.823 | 1.803 | 0.173 | 0.562 | 1.000 | 0.81 / 0.58 / 0.56 / 0.98 |
| nostd | 3 | 0.743 | 0.927 | 0.253 | 0.364 | 1.000 | 0.00 / 0.30 / 0.29 / 0.98 |
| nostd | 4 | 0.843 | 2.017 | 0.157 | 0.612 | 1.000 | 0.97 / 0.64 / 0.62 / 0.98 |
| awrb | 0–2 | 0.880 | 2.70 / 2.69 / 2.70 | 0.120 | 0.702 | 1.000 | 0.97 / 0.95 / 0.94 / 0.98 |
| awrb | 3–4 | 0.877 | 2.69 | 0.120 | 0.694 | 1.000 | same |
| awrbn | 0 / 1 / 2 / 3 / 4 | 0.743 / 0.743 / 0.703 / 0.687 / 0.717 | 0.963 / 0.930 / 0.420 / 0.000 / 0.497 | 0.253 / 0.253 / 0.293 / 0.313 / 0.283 | 0.364 / 0.364 / 0.264 / 0.223 / 0.298 | 1.000 | awrbn s3 never reads |

## Frozen predictions vs outcome

| ID | Prediction | Observed | Met |
|---|---|---|---|
| T1 | H_nostd supported; Δ ∈ [+0.05, +0.20] | supported, +0.120 | **yes** |
| T2 | grpo reads ≤ 0.2 in every seed, verify-after 0 everywhere in ≥ 4/5; nostd ≥ 0.8 reads in ≥ 4/5 | grpo 0.000 reads, all sites 0, 5/5 seeds; nostd 0.927–2.017, 5/5 | **yes** |
| T3 | ≥ 70% of the gain on faulted tasks; clean \|Δ\| ≤ 0.03 | clean-task Δ = 0.000 in every seed. Faulted Δ (mean 0.521 − 0.223 = 0.298) × 121/300 = 0.120, i.e. 100% of the gain | **yes** |
| T4 | false-success gap ≈ VSC gap (± 0.05) | 0.313 − 0.191 = 0.122 vs 0.120 | **yes** |
| T5 | H_AWRB not supported | not supported (−0.088) | **yes** |
| T6 | awrb reads more than nostd in ≥ 4/5 seed pairs | 5/5 (2.69–2.70 vs 0.93–2.02) | **yes** |
| T7 | grpo seed VSC range ≤ 0.02 | range 0.000 (all 0.687) | **yes** |
| T8 | untestable (no GRPO run at another cost) | — | untestable |

## Interpretation and caveats

- All seven testable predictions are met. This is the confirmatory link between the vote mechanism and the language
  agent:
  - every GRPO seed converged to the no-read corner;
  - std-free training reads at every site;
  - the whole VSC gain sits on faulted tasks, as unread silent faults reported as completed.
- **Caveat 1: degenerate GRPO seeds.** All five GRPO seeds give identical greedy behaviour, so the unpaired seed
  permutation test sees no baseline variance. The effect is large relative to the candidates' seed spread, but the
  seed-level n is effectively the candidates' alone.
- **Caveat 2: AWRB is not evidence that the vote is good.** AWRB (std + observation-measurable return) is the best
  arm on VSC (+0.192), consistent with Prop. 4.2's prediction that it over-verifies (2.7 reads per episode, about 0.95
  at every site). It is reported, not recommended: its sign guarantee fails under state-dependent costs (scan
  counterexample 5). Net utility J_c is in the eval summaries but was not part of the registered test.
- **Caveat 3: centring is not sufficient for stability.** One awrbn seed (s3) collapsed to 0 reads, and nostd s3 read
  about half as often as the other nostd seeds.
- **Cost-blindness remains untested on the LLM (T8).** The confirmatory result supports the direction of the
  mechanism, not the cost-invariance claim, which rests on the counterfactual re-weighting of fixed rollouts
  (MECHANISM_DIAGNOSTICS §3.2).
