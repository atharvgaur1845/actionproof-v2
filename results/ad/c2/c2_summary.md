# AD-C2 registered analysis (scripts/ad/c2_stats.py)

cw_c2: 300 sealed tasks (120 faulted). Seed 0 per arm; 5 episodes per task (greedy + 4 sampled).

| Arm | train cost | greedy reads/ep | greedy VSC | greedy false success | all-5 reads | all-5 VSC | net utility |
|---|---|---|---|---|---|---|---|
| grpo_c005 | 0.05 | 0.000 | 0.707 | 0.293 | 0.000 | 0.705 | 0.705 |
| nostd_c005 | 0.05 | 1.810 | 0.803 | 0.197 | 1.163 | 0.758 | 0.700 |
| grpo_c001 | 0.01 | 0.000 | 0.707 | 0.293 | 0.000 | 0.703 | 0.703 |
| nostd_c001 | 0.01 | 2.630 | 0.883 | 0.117 | 2.212 | 0.816 | 0.794 |
| grpo_c000 | 0.0 | 2.723 | 0.883 | 0.117 | 2.619 | 0.830 | 0.830 |

| Difference | estimate [95% CI] |
|---|---|
| D1_vsc_nostd_minus_grpo_c001 | +0.113 [+0.077, +0.152] |
| D2_reads_grpo_c001_minus_c005 | +0.000 [+0.000, +0.000] |
| D3_reads_nostd_c001_minus_c005 | +1.049 [+0.950, +1.145] |
| D4_reads_grpo_c000_minus_c001 | +2.619 [+2.533, +2.704] |
| D5_vsc_grpo_c000_minus_c001 | +0.127 [+0.087, +0.170] |

| Prediction | met |
|---|---|
| P1_grpo_c001_cost_blind (greedy reads <= 0.2) | yes |
| P2_nostd_c001_reads (greedy reads >= 0.8) | yes |
| P3_vsc_nostd_gt_grpo_at_c001 (D1 lower bound > 0) | yes |
| P4_boundary_grpo_c000_reads (greedy reads > 0.2) [secondary] | yes |
| P5_nostd_reads_not_lower_at_c001 (D3 estimate >= 0) [secondary] | yes |
