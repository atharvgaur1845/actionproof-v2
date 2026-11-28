# Evidence ledger for paper/main_v4.tex

Every quantitative statement in `main_v4.tex` and `appendix_v4.tex` (new sections), its source file, its registration
status and its evidence class. Values were recomputed or read from these files on 2026-11-28 (not copied from
`main_v3.tex`). Classes: **confirmatory** (a registered hypothesis, analysed once), **registered-descriptive** (a
registered secondary estimate), **descriptive** (post hoc or development; never a test), **math** (exact enumeration
or proof).

| Claim (paper) | Value | Source | Registration | Class |
|---|---|---|---|---|
| TH1 cells | 731 (600 + 131 twins), 0 missing | `results/theory/th1_heldout_summary.json` (`cells_*`) | PREREGISTRATION_TH1.md | confirmatory |
| H1 median abs. error GRPO / std-free / RLOO; p90 | 0.00026 / 0.00025 / 0.00022; ≤ 0.0073 | same, `H1.by_est` | TH1 | confirmatory |
| H2 GRPO decision predicted | 721/721 | same, `H2` | TH1 | confirmatory |
| H3 GRPO follows sign(vote); disagreement cells | 152/152; 77/77 | same, `H3` | TH1 | confirmatory |
| H4 endogenous homogeneity | 67/69, CI [0.928, 1.000] | same, `H4` | TH1 | confirmatory |
| H5 std-free follows sign(ΔQ) | 661/661 | same, `H5` | TH1 | confirmatory |
| H6 (not supported) | 53/75 = 0.707 < 0.90 | same, `H6`; diagnosis `results/theory/th1_heldout_posthoc_h6.json` | TH1 | confirmatory (miss); diagnosis descriptive |
| H7 (untestable) | 3 eligible pairs < 10 required; 3/3 in direction | same, `H7` | TH1 | confirmatory (untestable) |
| Bayes agreement / mean / median regret, 10 estimators; by law | table `tab:est`, Figure 1 | same, `descriptive` | TH1 descriptive block | registered-descriptive |
| Severity ×1000: GRPO −2.961 unchanged; centred −0.015 → +57.3 | — | `results/theory/scan.json` | none | math |
| Threshold 0.452 at c = 10⁻³; 0.499–0.5005; 0.654/0.667/0.706 | — | `results/theory/scan.json` | none | math |
| iid interior π* 0.597, regret 0.101; ρ* 0.211/0.149 (mix), 0.152→0.305 (dirmult) | — | `results/theory/scan.json` | none | math |
| 800/800 groups share a fault plan | — | `docs/research/MECHANISM_DIAGNOSTICS.md` | none | descriptive |
| AD-C1 std-free − GRPO | +0.120 [0.088, 0.153], Holm 0.016, 5/5 | `results/ad/c1_summary.json` (`H_nostd`) | PREREGISTRATION_AD_C1.md | confirmatory |
| AWRB − GRPO; AWRB-centred − std-free | +0.192; −0.088 (not supported) | same | AD-C1 | confirmatory |
| GRPO greedy reads / VSC / false success, all seeds; warm start identical | 0.000 / 0.687 / 0.313 | `results/ad/conf_summaries/ad_grpo_s*__cw_conf.summary.json`, `ad_base_warm__cw_conf.summary.json` | AD-C1 (evaluations) | registered-descriptive |
| std-free reads range | 0.927–2.017 | `results/ad/conf_summaries/ad_nostd_s*__cw_conf.summary.json` | AD-C1 | registered-descriptive |
| Clean-task Δ 0.000; faulted 0.223 vs 0.521; false-success gap 0.122 | — | `docs/research/AD_C1_ANALYSIS.md` (from the per-seed summaries) | AD-C1 T3/T4 | confirmatory (frozen predictions) |
| T1–T7 met; T8 untestable | 7/7 | `docs/research/AD_C1_ANALYSIS.md`, `AD_C1_THEORY_PREDICTION.md` | frozen before outputs | confirmatory |
| Seeds 0 and 3 identical conversations | 175/300 | computed on sharanga from `results/ad/conf/ad_grpo_s{0,3}__cw_conf.jsonl` (2026-11-28; episode files not in git) | none | descriptive (audit) |
| p = 0.016 is the 5v5 permutation floor | 1/252 × 4 (Holm) | `results/ad/c1_summary.json` (`p_perm_one_sided` 0.00397) | — | math |
| Sampled reads: warm 0.189; GRPO 0.000–0.001; std-free s0 1.051 | mean of 4 T=1 samples | `results/ad/conf_summaries/*.t1_s*.summary.json` | AD-C1 evaluations exist; the comparison is post hoc | descriptive |
| AWRB reads 2.7, 5/5 pairs (T6) | — | `results/ad/conf_summaries/ad_awrb_s*` | AD-C1 | confirmatory (T6) |
| Clean groups varying only through reads 111/146 … 127/163 | — | `docs/research/MECHANISM_DIAGNOSTICS.md` l.63 | none | descriptive |
| Vote direction 7/7 runs; read share 0.19–0.22 → 0.02–0.04 | — | same | none | descriptive |
| Cost re-score: −898.0 → −961.9 (+7%); −86.3 → −1162.8 | counterfactual | same, l.41 and l.66 | none | descriptive (not an intervention) |
| P3 null | 0.813 vs 0.819, −0.007 [−0.013, −0.001] | `results/p3/p3_summary.json` | PREREGISTRATION_P3.md | confirmatory (not supported) |
| Transfer of learned policies | languages/combos yes; renamed API −0.016, workflow +0.016 | `results/s4/s4_summary.json` | S4 registration | confirmatory |
| AD-C2 | pending | `results/ad/c2/c2_summary.json` | PREREGISTRATION_AD_C2.md (2026-11-28 17:50, before outputs) | confirmatory predictions P1–P3; P4–P5 secondary |
| Appendix practical study (P1, TXN, P4/P5, SEL1, RG1/RG2) | as in `paper/actionproof_main.tex` | `claims/claims.yaml` (899 numbers, `scripts/s2_numcheck.py`) | per study | as registered |
