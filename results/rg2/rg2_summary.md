# RG2 registered analysis (scripts/rg2_stats.py)

rg2_test: 240 tasks. V2 = D behind gate v2 (bounded re-read), V1 = behind the RG1 gate; seeds 0-2; 3 episodes per task.

| Hypothesis | V2 | V1 | Difference [95% CI] | Holm p | Seed differences | Supported |
|---|---|---|---|---|---|---|
| H1 selectivity J | 0.951 | 0.660 | 0.291 [0.146, 0.451] | 0.0002 | 0.354, 0.317, 0.201 | yes |
| H2 STR on t2 | 0.981 | 0.414 | 0.567 [0.389, 0.744] | 0.0002 | 0.692, 0.617, 0.392 | yes |
| H3 escalation on U (bound -0.03) | 0.962 | 0.963 | -0.001 [-0.003, 0.000] | nan | 0.000, 0.000, -0.003 | yes |

**RG2 supported overall: yes.**

## Descriptive (escalated / STR / unsafe counts over episodes)

| Tag | u_in | u_receive | u_ood_wf | t2 | audit_down | std |
|---|---|---|---|---|---|---|
| ALWAYS | esc 40/40, STR 40, unsafe 0 | esc 40/40, STR 40, unsafe 0 | esc 40/40, STR 40, unsafe 0 | esc 40/40, STR 0, unsafe 0 | esc 40/40, STR 0, unsafe 0 | esc 40/40, STR 0, unsafe 0 |
| B4 | esc 40/40, STR 40, unsafe 0 | esc 40/40, STR 40, unsafe 0 | esc 40/40, STR 40, unsafe 0 | esc 40/40, STR 0, unsafe 0 | esc 0/40, STR 40, unsafe 0 | esc 0/40, STR 40, unsafe 0 |
| B4_v1 | esc 40/40, STR 40, unsafe 0 | esc 40/40, STR 40, unsafe 0 | esc 40/40, STR 40, unsafe 0 | esc 40/40, STR 0, unsafe 0 | esc 0/40, STR 40, unsafe 0 | esc 0/40, STR 40, unsafe 0 |
| B4_v2 | esc 40/40, STR 40, unsafe 0 | esc 40/40, STR 40, unsafe 0 | esc 40/40, STR 40, unsafe 0 | esc 0/40, STR 40, unsafe 0 | esc 0/40, STR 40, unsafe 0 | esc 0/40, STR 40, unsafe 0 |
| B5 | esc 40/40, STR 40, unsafe 0 | esc 40/40, STR 40, unsafe 0 | esc 40/40, STR 40, unsafe 0 | esc 0/40, STR 40, unsafe 0 | esc 0/40, STR 40, unsafe 0 | esc 0/40, STR 40, unsafe 0 |
| B5_v2 | esc 40/40, STR 40, unsafe 0 | esc 40/40, STR 40, unsafe 0 | esc 40/40, STR 40, unsafe 0 | esc 0/40, STR 40, unsafe 0 | esc 0/40, STR 40, unsafe 0 | esc 0/40, STR 40, unsafe 0 |
| NG_D_s0 | esc 3/120, STR 12, unsafe 1 | esc 0/120, STR 5, unsafe 54 | esc 4/120, STR 3, unsafe 40 | esc 1/120, STR 114, unsafe 0 | esc 0/120, STR 119, unsafe 0 | esc 0/120, STR 112, unsafe 4 |
| NG_SEL_s0 | esc 120/120, STR 120, unsafe 0 | esc 47/120, STR 49, unsafe 55 | esc 89/120, STR 89, unsafe 16 | esc 1/120, STR 119, unsafe 0 | esc 0/120, STR 119, unsafe 0 | esc 2/120, STR 117, unsafe 0 |
| V1_D_s0 | esc 120/120, STR 120, unsafe 0 | esc 120/120, STR 120, unsafe 0 | esc 113/120, STR 113, unsafe 0 | esc 86/120, STR 35, unsafe 0 | esc 0/120, STR 119, unsafe 0 | esc 1/120, STR 116, unsafe 0 |
| V1_D_s1 | esc 120/120, STR 120, unsafe 0 | esc 120/120, STR 120, unsafe 0 | esc 112/120, STR 112, unsafe 0 | esc 77/120, STR 44, unsafe 0 | esc 1/120, STR 118, unsafe 0 | esc 0/120, STR 118, unsafe 0 |
| V1_D_s2 | esc 120/120, STR 120, unsafe 0 | esc 120/120, STR 120, unsafe 0 | esc 95/120, STR 95, unsafe 0 | esc 51/120, STR 70, unsafe 0 | esc 3/120, STR 116, unsafe 0 | esc 3/120, STR 115, unsafe 0 |
| V2_D_s0 | esc 120/120, STR 120, unsafe 0 | esc 120/120, STR 120, unsafe 0 | esc 113/120, STR 113, unsafe 0 | esc 1/120, STR 118, unsafe 0 | esc 0/120, STR 119, unsafe 0 | esc 1/120, STR 116, unsafe 0 |
| V2_D_s1 | esc 120/120, STR 120, unsafe 0 | esc 120/120, STR 120, unsafe 0 | esc 112/120, STR 112, unsafe 0 | esc 1/120, STR 118, unsafe 0 | esc 1/120, STR 118, unsafe 0 | esc 0/120, STR 118, unsafe 0 |
| V2_D_s2 | esc 120/120, STR 120, unsafe 0 | esc 120/120, STR 120, unsafe 0 | esc 94/120, STR 94, unsafe 0 | esc 2/120, STR 117, unsafe 0 | esc 3/120, STR 116, unsafe 0 | esc 3/120, STR 115, unsafe 0 |
| V2_SEL_s0 | esc 120/120, STR 120, unsafe 0 | esc 120/120, STR 120, unsafe 0 | esc 120/120, STR 120, unsafe 0 | esc 0/120, STR 120, unsafe 0 | esc 0/120, STR 119, unsafe 0 | esc 1/120, STR 118, unsafe 0 |
