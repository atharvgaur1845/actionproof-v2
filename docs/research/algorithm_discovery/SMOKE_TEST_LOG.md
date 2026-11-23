# Smoke and development test log

Entries are in time order. Every number is from the named file. All of them are exploratory development results:
no significance claims.

## 2026-11-23

**11:2x AD-T0, tabular toy v1** (`results/ad/t0/toy_bandit.json`).
- GRPO with group std never verifies when p < 1/2, e.g. p = 0.45, c = 0.05 → 0.002. It verifies at p = 0.55.
- Dr. GRPO-style no-std verifies correctly.
- *Later found wrong in one respect:* the "awrb" arm was modelled as a full-world Q (see AD-T0b).

**11:27 Registration AD-D1** (6fa46d8).

**11:3x CPU dry run on the login node** (`scripts/ad/cpu_dryrun.py`: fake LLM, real tokenizer, parser and
environment).
- The plumbing works.
- The read-rollout returns exposed the toy's modelling error: a read that reveals an unrepaired fault scores 0, while
  no-read scores 0.69.

**11:39 AD-T0b and Amendment 1** (ae50f4b). Corrected toy (`results/ad/t0/toy_bandit_v2.json`):
- AWRB with group std over-verifies: it reads at s = 0.
- raw/none and rb/none are correct.
- Added the awrbn arm and the c = 0.30 intervention, before any GPU run.

**12:18–12:22 Task 1200 smoke.** 2 steps each of awrb and nostd, plus a 10-task evaluation: SMOKE_OK.

**12:38–12:44 Task 1210 baselines on cw_dev** (`results/ad/dev/ad_base_*`).

| Model | VSC | J_c | reads | Verify after pay |
|---|---|---|---|---|
| warm start | 0.707 | 0.707 | 0.00 | 0.00 (greedy) |
| A12 R_V s0 | 0.913 | 0.769 | 2.88 | 0.96 |
| RSC s0 | 0.960 | 0.812 | 2.96 | 0.96 |

The free-read-trained models verify after every write, the journal included.

**12:22–15:36 D1 seed 0, all four arms** (cw_dev greedy, c = 0.05; `results/ad/dev/`).

| Arm | VSC | J_c | reads | Verify after order / pay / receive / post | Faulted-task VSC | proc_dev VSC |
|---|---|---|---|---|---|---|
| grpo | 0.707 | 0.707 | 0.00 | 0 / 0 / 0 / 0 | 0.254 | 0.487 |
| nostd | 0.833 | 0.740 | 1.87 | .80 / .59 / .90 / .58 | 0.576 | 0.667 |
| awrb | 0.867 | 0.736 | 2.61 | .89 / .92 / .92 / .91 | 0.661 | 0.793 |
| awrbn | 0.773 | 0.722 | 1.03 | 0 / .32 / .92 / .32 | 0.424 | 0.600 |

Training curves (T = 1 rollouts, `scripts/ad/analyze_d1.py`):
- **grpo:** reads 0.59 → 0.03; zero-std groups 19% → 86%.
- **nostd:** reads 0.69 → 1.56.
- **awrbn:** reads 0.77 → 1.35.
- **awrb:** reads 0.94 → 2.25 by step 40.
- **Selectivity (P2) fails in every arm.** Verify-after-post tracks verify-after-pay.

**16:17 Registration AD-D2** (7341825). The world-independent EMA advantage scale, after the seed-0 diagnosis that
no-std learning is slow at GRPO's learning rate.

(continued below as results land)

**16:27 D1 awrb seed 1.**
- cw_dev greedy is identical to seed 0 in every summary metric (VSC 0.867, J_c 0.736, reads 2.61).
- The traces are *not* identical: 54 of 150 call sequences differ (`scripts/ad/compare_traces.py`), but every
  task's VSC agrees, so greedy outcomes have saturated.
- KL₁₀₀ differs (0.009 vs 0.018). Not a seeding bug. (Correction, 16:53: an earlier version of this entry gave a sampled VSC of 0.860 for seed 1. That was read from a partial file holding 1 of the 4 samples. With all 4, sampled VSC is 0.788, against 0.787 for seed 0.)

**16:30 Read-information diagnostic** (`actionproof/ad/readinfo.py`, `results/ad/read_waste_cw_dev.json`).
- 42–49% of every reading model's reads are provably uninformative under cw1. This includes RSC (0.49) and A12 R_V
  (0.48).
- They cost 0.02–0.07 J_c.

**16:35 Registration AD-D3** (uninformative-read local credit) **and C1 addendum 1** (494ba06).

**16:37 D1 grpo seed 1, training** (live rollout log): reads 0.56 → 0.07 by step 25. This replicates seed 0's
collapse.

**16:48 World-luck share of within-group R_V variance** (training rollouts, seed-0 arms): ratio of Rao-Blackwellised
to raw variance 0.66–0.75.
