Split: heldout; cells complete 731/731.

### Registered criteria

| Criterion | Value | Threshold | Met |
|---|---|---|---|
| grpo median |sim - pred| | 0.0003 | <= 0.03 | True |
| nostd median |sim - pred| | 0.0003 | <= 0.03 | True |
| rloo median |sim - pred| | 0.0002 | <= 0.03 | True |
| grpo p90 |sim - pred| | 0.0073 | <= 0.1 | True |
| nostd p90 |sim - pred| | 0.0057 | <= 0.1 | True |
| rloo p90 |sim - pred| | 0.0056 | <= 0.1 | True |
| grpo decision agreement | 1.0000 | >= 0.95 | True |
| grpo moves with sign(vote) | 1.0000 | >= 0.95 | True |
| grpo pi_T < 0.95 | 0.9710 | >= 0.9 | True |
| nostd pi_T > grpo pi_T | 1.0000 | >= 0.9 | True |
| nostd moves with sign(delta_q) | 1.0000 | >= 0.95 | True |
| grpo regret <= nostd regret + 0.01 | 0.7067 | >= 0.9 | False |
| twin difference direction | 1.0000 | >= 0.9 | True |
| pairs with predicted difference >= 0.05 | 3 | >= 10 | False |

### Hypothesis detail

| H | Cells / rate [95% CI] |
|---|---|
| H2 GRPO decision | 721/721 = 1.000 [1.000, 1.000] |
| H3 vote direction | 152/152 = 1.000 [1.000, 1.000] |
| H3 subset where vote and ΔQ disagree | 77/77 = 1.000 [1.000, 1.000] |
| H4a GRPO π_T < 0.95 | 67/69 = 0.971 [0.928, 1.000] |
| H4b nostd π_T > GRPO π_T | 69/69 = 1.000 [1.000, 1.000] |
| H5 nostd follows sign ΔQ | 661/661 = 1.000 [1.000, 1.000] |
| H6 GRPO regret ≤ nostd + 0.01 | 53/75 = 0.707 [0.600, 0.800] |
| H7 twins (pairs 131, with predicted diff ≥ 0.05: 3) | 3/3 = 1.000 [1.000, 1.000] |

### H1: mean-field accuracy |sim − pred| of π_T

| Estimator | median | p90 | mean | max | n |
|---|---|---|---|---|---|
| grpo | 0.0003 | 0.0073 | 0.0031 | 0.1905 | 731 |
| nostd | 0.0003 | 0.0057 | 0.0034 | 0.2238 | 731 |
| rloo | 0.0002 | 0.0056 | 0.0041 | 0.3896 | 731 |
| awrb | 0.0001 | 0.0008 | 0.0011 | 0.0771 | 731 |
| gdpo | 0.0001 | 0.0004 | 0.0002 | 0.0108 | 731 |

### Descriptive: all estimators (seed-level means over cells)

| Estimator | Bayes-action agreement (cells) | mean regret | median cell regret | regret shared / iid / mix / dirmult | false completion | mean |grad| | grad sd | rollouts/run |
|---|---|---|---|---|---|---|---|---|
| grpo | 436/731 = 0.596 [0.561, 0.632] | 0.6096 | 0.0650 | 0.6761 / 0.3641 / 0.7637 / 0.5711 | 0.283 | 0.0378 | 0.0512 | 31400 |
| nostd | 709/731 = 0.970 [0.956, 0.982] | 0.0112 | 0.0091 | 0.0120 / 0.0116 / 0.0105 / 0.0107 | 0.208 | 0.0358 | 0.0715 | 31400 |
| rloo | 712/731 = 0.974 [0.962, 0.985] | 0.0080 | 0.0071 | 0.0084 / 0.0083 / 0.0077 / 0.0077 | 0.206 | 0.0395 | 0.0865 | 31400 |
| batchstd | 694/731 = 0.949 [0.934, 0.964] | 0.0429 | 0.0092 | 0.0453 / 0.0288 / 0.0508 / 0.0424 | 0.208 | 0.0450 | 0.0582 | 31400 |
| reinforce | 713/731 = 0.975 [0.964, 0.985] | 0.0082 | 0.0071 | 0.0089 / 0.0082 / 0.0079 / 0.0077 | 0.206 | 0.0478 | 0.0950 | 31400 |
| ppo | 692/731 = 0.947 [0.930, 0.962] | 0.0398 | 0.0035 | 0.0633 / 0.0151 / 0.0341 / 0.0318 | 0.205 | 1536.5271 | 26568.3463 | 31400 |
| dapo | 436/731 = 0.596 [0.561, 0.632] | 0.6010 | 0.0335 | 0.6683 / 0.3558 / 0.7549 / 0.5617 | 0.281 | 0.1109 | 0.0906 | 79170 |
| grpo_kl | 435/731 = 0.595 [0.562, 0.629] | 0.6099 | 0.0731 | 0.6758 / 0.3602 / 0.7673 / 0.5718 | 0.284 | 0.0392 | 0.0520 | 31400 |
| awrb | 660/731 = 0.903 [0.881, 0.925] | 0.0309 | 0.0041 | 0.0555 / 0.0153 / 0.0228 / 0.0199 | 0.211 | 0.0392 | 0.0633 | 31400 |
| gdpo | 162/731 = 0.222 [0.192, 0.252] | 1.9138 | 0.5087 | 1.5373 / 2.1971 / 2.1957 / 1.9898 | 0.490 | 0.0312 | 0.0479 | 31400 |

### Descriptive predictions

- DAPO follows GRPO's predicted decision: 720/721 = 0.999 [0.996, 1.000]
- PPO follows sign ΔQ: 641/661 = 0.970 [0.956, 0.982]
- batch-std follows sign ΔQ: 647/661 = 0.979 [0.967, 0.989]
- GRPO decision prediction, stratum broad: 331/331 = 1.000 [1.000, 1.000]
- GRPO decision prediction, stratum adversarial: 95/95 = 1.000 [1.000, 1.000]
