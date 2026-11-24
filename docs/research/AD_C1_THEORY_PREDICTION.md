# AD-C1: theory-derived predictions, frozen before any cw_conf output

Written 2026-11-24 ~08:00 IST on `research/latent-state-theory`. Hash: `results/ad_c1_theory_prediction_hash.json`
(`python scripts/theory/freeze_c1_prediction.py`). The commit time is the record.

**What was seen when this was written:**
- the AD-D1/D2 development summaries (`results/ad/d1_summary.json`, `d2_summary.json`; cw_dev);
- the training-rollout logs of the development runs, plus C1 training logs for grpo s2–s3 that were in progress
  (`results/research/llm_mechanism.json`; training-pool rollouts only).

**Not seen:** any cw_conf output. Tasks 1240/1241 have not run.

This note does **not** modify `PREREGISTRATION_AD_C1.md`, its addendum, its notes 2–3 or their hashes. The registered
statistic, test, Holm family and decision rule are unchanged, and the registered analysis (`scripts/ad/c1_stats.py`)
is run by the lead engineer. This note only states what the theory predicts. The comparison is written after the
result exists, in `docs/research/AD_C1_ANALYSIS.md`.

## Basis

- Theorem 3.1 and Cor. 3.2 (`docs/theory/PROOF_AND_ASSUMPTIONS.md`). Every C1 group shares one fault plan
  (800/800 groups per logged run).
- In GRPO's training groups, the early-estimated vote is negative in every run: −0.375 to −0.625 for the grpo and
  nostd arms (`llm_mechanism.json`, `effects_by_window`). This is because about 60% of tasks are clean (cw1 prior) and
  most clean groups vary only through reads (111/146 and 127/162 clean non-degenerate groups have constant R_V in
  grpo s0/s1).
- The vote predicts GRPO collapses reading at every site. Centred training follows ΔQ, which became positive after
  step 20 in every centred development run.

## Predictions

| ID | Prediction (cw_conf, greedy unless stated) | Basis | How it is checked |
|---|---|---|---|
| T1 | H_nostd (VSC nostd > grpo) is **supported** under the registered rule. Seed-averaged difference in **[+0.05, +0.20]** (dev: +0.127 per seed) | vote vs ΔQ | c1_stats output |
| T2 | grpo reads per episode **≤ 0.2 in every seed**, with verify-after-write 0 at every site in ≥ 4/5 seeds; nostd reads per episode **≥ 0.8 in ≥ 4/5 seeds** | Cor. 3.2 | `reads`, `verify_after` in the C1 eval summaries |
| T3 | The VSC gain of nostd over grpo is carried by faulted tasks: (faulted-task VSC difference × faulted share) ≥ 70% of the total VSC difference, and the clean-task VSC difference has \|Δ\| ≤ 0.03 | the vote only removes reads, so it changes outcomes only where reading has value | per-task VSC by fault class |
| T4 | grpo's false-success rate exceeds nostd's by roughly the VSC difference (within ±0.05) | unread silent faults are reported as completed | `false_success` |
| T5 | H_AWRB (awrbn > nostd) is **not supported** | both are centred estimators of the same objective; Prop. 2.2 gives no superiority | c1_stats |
| T6 | awrb (AWRB + group std) reads **more** than nostd in ≥ 4/5 seed pairs (over-verification) | Prop. 4.2: under AWRB the clean-state β is p − c > 0, so the vote is positive | `reads` |
| T7 | grpo seeds are near-identical: seed-level VSC range ≤ 0.02 | the vote drives every seed to the same no-read corner (dev greedy outputs were identical) | per-seed VSC |
| T8 | Not testable in C1: cost-blindness of GRPO (no GRPO run at another cost). The awrb/awrbn c = 0.30 development runs are std-normalised RB and centred RB, not plain GRPO | — | — |

**Failure meaning.**
- T1 or T2 failing would contradict the claim that the vote mechanism governs the language-agent outcome. It would
  not contradict the mathematics.
- T3 or T4 failing would mean the gain comes from something other than reading at faulted sites.
