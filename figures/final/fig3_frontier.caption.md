**Figure 3. Completion versus harm on the locked txn_test.** Left: verified transaction completion (VTC) against
selective risk (unsafe episodes, i.e. false completion, duplicate, wrong amount or collateral write, among episodes the
agent did not escalate; `actionproof/txn/truth.py::summarise`). Right: false completion (FCR) and duplicate mutation
(DMR, all episodes) per arm.

- **Sources:** `results/txn/<tag>__txn_test.summary.json` and `results/sel1/<tag>__txn_test.summary.json`, block
  `txn.all`; built by `scripts/build_final_figures.py`; table view `figures/final/fig3_frontier.csv`.
- **Split and denominators:** txn_test, 440 locked tasks. Scripted policies: 1 deterministic episode per task (440).
  Models: 1 greedy + 2 sampled episodes per task, 1,320 per seed; 3-seed arms pool 3,960.
- **Seed scope:** D, Dn, Dw_aug, SEL and CTL have 3 seeds (small open marks); base, Dw and the scripted policies are
  single runs.
- **Uncertainty:** Wilson 95% intervals on pooled episodes, treating episodes as independent. Episodes of one task and
  seeds are correlated, so model intervals are too narrow; the registered analyses use task-cluster bootstrap.
- **Scope:** a simulator (ProcureSim). B4 and its ablations A1-A6 are scripted with the fault taxonomy written in; they
  are references, not learned agents. B0 and B2 coincide; A6 has B4's flags and coincides with it.

| Arm | Tags | VTC | Selective risk | FCR | DMR |
|---|---|---|---|---|---|
| B0 trust the acknowledgement | B0 | 60/440 = 0.136 [0.107, 0.172] | 380/440 = 0.864 [0.828, 0.893] | 380/440 = 0.864 [0.828, 0.893] | 270/440 = 0.614 [0.567, 0.658] |
| B1 final-state check only | B1 | 197/440 = 0.448 [0.402, 0.494] | 183/440 = 0.416 [0.371, 0.462] | 183/440 = 0.416 [0.371, 0.462] | 183/440 = 0.416 [0.371, 0.462] |
| B2 audit-log check only | B2 | 60/440 = 0.136 [0.107, 0.172] | 380/440 = 0.864 [0.828, 0.893] | 380/440 = 0.864 [0.828, 0.893] | 270/440 = 0.614 [0.567, 0.658] |
| B3 bounded retry + idempotency key | B3 | 216/440 = 0.491 [0.444, 0.537] | 224/440 = 0.509 [0.463, 0.556] | 224/440 = 0.509 [0.463, 0.556] | 107/440 = 0.243 [0.205, 0.285] |
| A1: B4 without read-back after ok | A1 | 211/440 = 0.480 [0.433, 0.526] | 199/410 = 0.485 [0.437, 0.534] | 199/440 = 0.452 [0.406, 0.499] | 82/440 = 0.186 [0.153, 0.225] |
| A2: B4 reading without consistent | A2 | 346/440 = 0.786 [0.746, 0.822] | 21/380 = 0.055 [0.036, 0.083] | 7/440 = 0.016 [0.008, 0.032] | 21/440 = 0.048 [0.031, 0.072] |
| A3: B4 with a new key on every retry | A3 | 366/440 = 0.832 [0.794, 0.864] | 14/380 = 0.037 [0.022, 0.061] | 0/440 = 0.000 [0.000, 0.009] | 14/440 = 0.032 [0.019, 0.053] |
| A4: B4 rewriting the full amount on partials | A4 | 333/440 = 0.757 [0.715, 0.795] | 23/380 = 0.061 [0.041, 0.089] | 0/440 = 0.000 [0.000, 0.009] | 23/440 = 0.052 [0.035, 0.077] |
| A5: B4 that never escalates | A5 | 425/440 = 0.966 [0.945, 0.979] | 15/440 = 0.034 [0.021, 0.055] | 15/440 = 0.034 [0.021, 0.055] | 0/440 = 0.000 [0.000, 0.009] |
| A6 (identical flags to B4) | A6 | 380/440 = 0.864 [0.828, 0.893] | 0/380 = 0.000 [0.000, 0.010] | 0/440 = 0.000 [0.000, 0.009] | 0/440 = 0.000 [0.000, 0.009] |
| B4 transaction-aware reconciliation (scripted reference) | B4 | 380/440 = 0.864 [0.828, 0.893] | 0/380 = 0.000 [0.000, 0.010] | 0/440 = 0.000 [0.000, 0.009] | 0/440 = 0.000 [0.000, 0.009] |
| Base Tiny Aya Global, prompted | tag_base | 19/1320 = 0.014 [0.009, 0.022] | 634/1121 = 0.566 [0.536, 0.594] | 215/1320 = 0.163 [0.144, 0.184] | 331/1320 = 0.251 [0.228, 0.275] |
| Dw recovery SFT warm start | Dw | 789/1320 = 0.598 [0.571, 0.624] | 360/1315 = 0.274 [0.250, 0.298] | 341/1320 = 0.258 [0.235, 0.283] | 195/1320 = 0.148 [0.130, 0.168] |
| Dw_aug randomised recovery SFT (3 seeds) | Dwaug_s0 Dwaug_s1 Dwaug_s2 | 2762/3960 = 0.697 [0.683, 0.712] | 898/3960 = 0.227 [0.214, 0.240] | 873/3960 = 0.220 [0.208, 0.234] | 502/3960 = 0.127 [0.117, 0.137] |
| D recovery SFT + GRPO (3 seeds) | D_s0 D_s1 D_s2 | 2864/3960 = 0.723 [0.709, 0.737] | 511/3941 = 0.130 [0.120, 0.141] | 462/3960 = 0.117 [0.107, 0.127] | 208/3960 = 0.053 [0.046, 0.060] |
| SEL1 SEL: escalation demonstrations (3 seeds) | sel1:SEL_s0 sel1:SEL_s1 sel1:SEL_s2 | 3285/3960 = 0.830 [0.818, 0.841] | 95/3422 = 0.028 [0.023, 0.034] | 79/3960 = 0.020 [0.016, 0.025] | 54/3960 = 0.014 [0.010, 0.018] |
| SEL1 CTL: matched replay (3 seeds) | sel1:CTL_s0 sel1:CTL_s1 sel1:CTL_s2 | 3451/3960 = 0.871 [0.861, 0.882] | 106/3959 = 0.027 [0.022, 0.032] | 88/3960 = 0.022 [0.018, 0.027] | 69/3960 = 0.017 [0.014, 0.022] |
