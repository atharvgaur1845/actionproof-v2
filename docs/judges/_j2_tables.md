### Q1 (did the writes take effect?) against the SQLite verifier, main set (n = 660)

| View | Judge | Coverage | Precision | Recall | F1 | False acceptance | False rejection | κ | Brier | ECE |
|---|---|---|---|---|---|---|---|---|---|---|
| transcript | gemma | 0.576 | 1.000 | 0.667 | 0.800 | 0.000 | 0.333 | 0.645 | 0.106 | 0.030 |
| transcript | qwen | 0.592 | 0.991 | 0.300 | 0.461 | 0.003 | 0.700 | 0.277 | 0.293 | 0.236 |
| transcript | gptoss | 0.564 | 1.000 | 0.575 | 0.730 | 0.000 | 0.425 | 0.552 | 0.150 | 0.087 |
| transcript | deepseek | 0.429 | 1.000 | 0.292 | 0.452 | 0.000 | 0.708 | 0.272 | 0.257 | 0.147 |
| transcript | aya | 0.930 | 0.600 | 0.875 | 0.712 | 0.700 | 0.125 | 0.183 | 0.338 | 0.317 |
| transcript_state | gemma | 1.000 | 1.000 | 0.981 | 0.990 | 0.000 | 0.019 | 0.979 | 0.011 | 0.011 |
| transcript_state | qwen | 1.000 | 0.986 | 0.808 | 0.889 | 0.013 | 0.192 | 0.781 | 0.102 | 0.080 |
| transcript_state | gptoss | 0.991 | 0.994 | 0.894 | 0.942 | 0.007 | 0.106 | 0.879 | 0.051 | 0.047 |
| transcript_state | deepseek | 1.000 | 1.000 | 0.792 | 0.884 | 0.000 | 0.208 | 0.776 | 0.105 | 0.106 |
| transcript_state | aya | 0.994 | 0.644 | 0.864 | 0.738 | 0.573 | 0.136 | 0.301 | 0.297 | 0.276 |

### Other questions (F1 against verifier-derived labels), main set

| View | Judge | Q2 claimed success | Q3 false success | Q4 unrefunded duplicate | Q5 read-back (proxy) |
|---|---|---|---|---|---|
| transcript | gemma | 0.992 | 0.500 | 0.206 | 0.899 |
| transcript | qwen | 0.998 | 0.345 | 0.288 | 0.431 |
| transcript | gptoss | 0.991 | 0.429 | 0.273 | 0.905 |
| transcript | deepseek | 1.000 | 0.346 | 0.289 | 0.521 |
| transcript | aya | 0.964 | 0.052 | 0.014 | 0.677 |
| transcript_state | gemma | 1.000 | 0.986 | 0.530 | 0.841 |
| transcript_state | qwen | 1.000 | 0.866 | 0.521 | 0.530 |
| transcript_state | gptoss | 0.996 | 0.916 | 0.462 | 0.891 |
| transcript_state | deepseek | 1.000 | 0.865 | 0.495 | 0.694 |
| transcript_state | aya | 0.997 | 0.379 | 0.184 | 0.625 |

### Fooled on unverifiable failures (transcript view, J-H1 strata)

| Judge | Fooled / n | Share | Wilson 95% |
|---|---|---|---|
| gemma | 0 / 120 | 0.000 | [-0.000, 0.031] |
| qwen | 1 / 120 | 0.008 | [0.001, 0.046] |
| gptoss | 0 / 120 | 0.000 | [-0.000, 0.031] |
| deepseek | 0 / 120 | 0.000 | [-0.000, 0.031] |
| aya | 105 / 120 | 0.875 | [0.804, 0.923] |

### Robustness on the 198-case subset (transcript_state, Q1)

| Judge | Order-reversal agreement | κ | Repeat agreement | κ |
|---|---|---|---|---|
| gemma | 0.990 | 0.980 | 0.970 | 0.939 |
| qwen | 0.949 | 0.898 | 0.949 | 0.898 |
| gptoss | 0.955 | 0.910 | 0.934 | 0.871 |
| deepseek | 0.990 | 0.979 | 0.990 | 0.979 |
| aya | 0.924 | 0.809 | 0.970 | 0.928 |

Aya paraphrase (v2 vs v1, transcript): agreement 0.813, κ 0.510, n 198.

### Aggregation (fit on the 76 calibration cases, evaluated on the 660 main cases)

| View / set | Method | Coverage | Precision | Recall | F1 | FPR | κ | Brier | ECE |
|---|---|---|---|---|---|---|---|---|---|
| transcript|J1 four | weighted | 1.000 | 0.996 | 0.708 | 0.828 | 0.003 | 0.685 | 0.131 | 0.125 |
| transcript|J1 four | majority | 0.526 | 0.994 | 0.467 | 0.635 | 0.003 | 0.440 | 0.236 | 0.295 |
| transcript|with aya | weighted | 1.000 | 1.000 | 0.692 | 0.818 | 0.000 | 0.671 | 0.122 | 0.106 |
| transcript|with aya | majority | 0.914 | 0.678 | 0.808 | 0.738 | 0.460 | 0.355 | 0.180 | 0.186 |
| transcript_state|J1 four | weighted | 1.000 | 1.000 | 0.928 | 0.963 | 0.000 | 0.921 | 0.026 | 0.033 |
| transcript_state|J1 four | majority | 0.939 | 1.000 | 0.811 | 0.896 | 0.000 | 0.796 | 0.044 | 0.069 |
| transcript_state|with aya | weighted | 1.000 | 1.000 | 0.925 | 0.961 | 0.000 | 0.918 | 0.027 | 0.035 |
| transcript_state|with aya | majority | 0.998 | 1.000 | 0.861 | 0.925 | 0.000 | 0.849 | 0.054 | 0.093 |

### Inter-model agreement (five judges, Q1)

| View | Split cases | Share | Fleiss κ |
|---|---|---|---|
| transcript | 228 | 0.345 | 0.421 |
| transcript_state | 263 | 0.398 | 0.631 |

Human labels available: none. J2 cost (token-priced estimate): $2.816.
