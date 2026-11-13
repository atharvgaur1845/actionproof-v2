| model | type | schema | n | target | P v1 | P v2 | greedy rate | next-tool argmax = greedy (v1 / v2) | report-status argmax = greedy status (v1 / v2; where greedy reports) |
|---|---|---|---|---|---|---|---|---|---|
| init | a | a4 | 50 | failed | 0.731 | 0.273 | 0.02 | 50/50 / 50/50 | 3/50 / 50/50 |
| init | a | plain | 50 | failed | 0.999 | 0.994 | 1.00 | 50/50 / 50/50 | 50/50 / 50/50 |
| init | b | a4 | 50 | failed | 0.000 | 0.000 | 0.00 | 50/50 / 50/50 | – (greedy never reports) |
| init | b | plain | 50 | failed | 0.000 | 0.000 | 0.00 | 50/50 / 50/50 | – (greedy never reports) |
| init | c | a4 | 50 | query_ledger | 0.000 | 0.000 | 0.00 | 50/50 / 50/50 | – (greedy never reports) |
| init | c | plain | 50 | query_ledger | 0.000 | 0.000 | 0.00 | 50/50 / 50/50 | – (greedy never reports) |
| init | d | a4 | 50 | query_ledger | 0.157 | 0.663 | 1.00 | 0/50 / 50/50 | – |
| init | d | plain | 50 | query_ledger | 0.046 | 0.537 | 1.00 | 0/50 / 42/50 | – ⚠ |
| init | e | a4 | 50 | apply_store_credit | 0.587 | 0.237 | 0.00 | 11/50 / 50/50 | – |
| a6_pcstr_d_noreport_s0 | a | a4 | 50 | failed | 0.964 | 0.690 | 0.92 | 50/50 / 50/50 | 46/50 / 50/50 |
| a6_pcstr_d_noreport_s0 | a | plain | 50 | failed | 1.000 | 0.998 | 1.00 | 50/50 / 50/50 | 50/50 / 50/50 |
| a6_pcstr_d_noreport_s0 | b | a4 | 50 | failed | 0.000 | 0.000 | 0.00 | 50/50 / 50/50 | – (greedy never reports) |
| a6_pcstr_d_noreport_s0 | b | plain | 50 | failed | 0.000 | 0.000 | 0.00 | 50/50 / 50/50 | – (greedy never reports) |
| a6_pcstr_d_noreport_s0 | c | a4 | 50 | query_ledger | 0.000 | 0.000 | 0.00 | 50/50 / 50/50 | – (greedy never reports) |
| a6_pcstr_d_noreport_s0 | c | plain | 50 | query_ledger | 0.000 | 0.000 | 0.00 | 50/50 / 50/50 | – (greedy never reports) |
| a6_pcstr_d_noreport_s0 | d | a4 | 50 | query_ledger | 0.565 | 0.911 | 1.00 | 48/50 / 50/50 | – |
| a6_pcstr_d_noreport_s0 | d | plain | 50 | query_ledger | 0.270 | 0.854 | 1.00 | 0/50 / 50/50 | – |
| a6_pcstr_d_noreport_s0 | e | a4 | 50 | apply_store_credit | 0.669 | 0.273 | 0.00 | 2/50 / 50/50 | – |
| a6_pcstr_d_s0 | a | a4 | 50 | failed | 0.545 | 0.135 | 0.00 | 50/50 / 50/50 | 19/50 / 50/50 |
| a6_pcstr_d_s0 | a | plain | 50 | failed | 0.999 | 0.992 | 1.00 | 50/50 / 50/50 | 50/50 / 50/50 |
| a6_pcstr_d_s0 | b | a4 | 50 | failed | 0.000 | 0.000 | 0.00 | 50/50 / 50/50 | – (greedy never reports) |
| a6_pcstr_d_s0 | b | plain | 50 | failed | 0.000 | 0.000 | 0.00 | 50/50 / 50/50 | – (greedy never reports) |
| a6_pcstr_d_s0 | c | a4 | 50 | query_ledger | 0.000 | 0.000 | 0.00 | 50/50 / 50/50 | – (greedy never reports) |
| a6_pcstr_d_s0 | c | plain | 50 | query_ledger | 0.000 | 0.000 | 0.00 | 50/50 / 50/50 | – (greedy never reports) |
| a6_pcstr_d_s0 | d | a4 | 50 | query_ledger | 0.632 | 0.907 | 1.00 | 49/50 / 50/50 | – |
| a6_pcstr_d_s0 | d | plain | 50 | query_ledger | 0.351 | 0.847 | 1.00 | 0/50 / 50/50 | – |
| a6_pcstr_d_s0 | e | a4 | 50 | apply_store_credit | 0.690 | 0.264 | 0.00 | 1/50 / 50/50 | – |
| a6_ack_outcome_s0 | a | a4 | 50 | failed | 0.673 | 0.199 | 0.00 | 50/50 / 50/50 | 5/50 / 50/50 |
| a6_ack_outcome_s0 | a | plain | 50 | failed | 0.999 | 0.994 | 1.00 | 50/50 / 50/50 | 50/50 / 50/50 |
| a6_ack_outcome_s0 | b | a4 | 50 | failed | 0.000 | 0.000 | 0.00 | 50/50 / 50/50 | – (greedy never reports) |
| a6_ack_outcome_s0 | b | plain | 50 | failed | 0.000 | 0.000 | 0.00 | 50/50 / 50/50 | – (greedy never reports) |
| a6_ack_outcome_s0 | c | a4 | 50 | query_ledger | 0.000 | 0.000 | 0.00 | 50/50 / 50/50 | – (greedy never reports) |
| a6_ack_outcome_s0 | c | plain | 50 | query_ledger | 0.000 | 0.000 | 0.00 | 50/50 / 50/50 | – (greedy never reports) |
| a6_ack_outcome_s0 | d | a4 | 50 | query_ledger | 0.064 | 0.536 | 1.00 | 0/50 / 40/50 | – ⚠ |
| a6_ack_outcome_s0 | d | plain | 50 | query_ledger | 0.017 | 0.380 | 0.00 | 50/50 / 50/50 | – |
| a6_ack_outcome_s0 | e | a4 | 50 | apply_store_credit | 0.412 | 0.151 | 0.00 | 41/50 / 50/50 | – |
| a7_init | a | a4 | 50 | failed | 0.762 | 0.282 | 0.08 | 50/50 / 50/50 | 8/50 / 47/50 |
| a7_init | a | plain | 50 | failed | 0.999 | 0.995 | 1.00 | 50/50 / 50/50 | 50/50 / 50/50 |
| a7_init | b | a4 | 50 | failed | 0.000 | 0.000 | 0.00 | 50/50 / 50/50 | – (greedy never reports) |
| a7_init | b | plain | 50 | failed | 0.000 | 0.000 | 0.00 | 50/50 / 50/50 | – (greedy never reports) |
| a7_init | c | a4 | 50 | query_ledger | 0.000 | 0.000 | 0.00 | 50/50 / 50/50 | – (greedy never reports) |
| a7_init | c | plain | 50 | query_ledger | 0.000 | 0.000 | 0.00 | 50/50 / 50/50 | – (greedy never reports) |
| a7_init | d | a4 | 50 | query_ledger | 0.604 | 0.911 | 1.00 | 50/50 / 50/50 | – |
| a7_init | d | plain | 50 | query_ledger | 0.139 | 0.518 | 1.00 | 0/50 / 26/50 | – ⚠ |
| a7_init | e | a4 | 50 | apply_store_credit | 0.945 | 0.997 | 1.00 | 50/50 / 50/50 | – |
| a7_pcstr_d_noreport_s0 | a | a4 | 50 | failed | 0.723 | 0.256 | 0.02 | 50/50 / 50/50 | 5/50 / 47/50 |
| a7_pcstr_d_noreport_s0 | a | plain | 50 | failed | 0.999 | 0.996 | 1.00 | 50/50 / 50/50 | 50/50 / 50/50 |
| a7_pcstr_d_noreport_s0 | b | a4 | 50 | failed | 0.000 | 0.000 | 0.00 | 50/50 / 50/50 | – (greedy never reports) |
| a7_pcstr_d_noreport_s0 | b | plain | 50 | failed | 0.000 | 0.000 | 0.00 | 50/50 / 50/50 | – (greedy never reports) |
| a7_pcstr_d_noreport_s0 | c | a4 | 50 | query_ledger | 0.000 | 0.000 | 0.00 | 50/50 / 50/50 | – (greedy never reports) |
| a7_pcstr_d_noreport_s0 | c | plain | 50 | query_ledger | 0.000 | 0.000 | 0.00 | 50/50 / 50/50 | – (greedy never reports) |
| a7_pcstr_d_noreport_s0 | d | a4 | 50 | query_ledger | 0.853 | 0.978 | 1.00 | 50/50 / 50/50 | – |
| a7_pcstr_d_noreport_s0 | d | plain | 50 | query_ledger | 0.428 | 0.813 | 1.00 | 18/50 / 50/50 | – |
| a7_pcstr_d_noreport_s0 | e | a4 | 50 | apply_store_credit | 0.961 | 0.998 | 1.00 | 50/50 / 50/50 | – |
| a7_ack_outcome_s0 | a | a4 | 50 | failed | 0.550 | 0.161 | 0.00 | 50/50 / 50/50 | 20/50 / 50/50 |
| a7_ack_outcome_s0 | a | plain | 50 | failed | 0.999 | 0.993 | 1.00 | 50/50 / 50/50 | 50/50 / 50/50 |
| a7_ack_outcome_s0 | b | a4 | 50 | failed | 0.000 | 0.000 | 0.00 | 50/50 / 50/50 | – (greedy never reports) |
| a7_ack_outcome_s0 | b | plain | 50 | failed | 0.000 | 0.000 | 0.00 | 50/50 / 50/50 | – (greedy never reports) |
| a7_ack_outcome_s0 | c | a4 | 50 | query_ledger | 0.000 | 0.000 | 0.00 | 50/50 / 50/50 | – (greedy never reports) |
| a7_ack_outcome_s0 | c | plain | 50 | query_ledger | 0.000 | 0.000 | 0.00 | 50/50 / 50/50 | – (greedy never reports) |
| a7_ack_outcome_s0 | d | a4 | 50 | query_ledger | 0.203 | 0.681 | 1.00 | 0/50 / 50/50 | – |
| a7_ack_outcome_s0 | d | plain | 50 | query_ledger | 0.018 | 0.212 | 0.00 | 50/50 / 50/50 | – |
| a7_ack_outcome_s0 | e | a4 | 50 | apply_store_credit | 0.955 | 0.998 | 1.00 | 50/50 / 50/50 | – |
| a6_pcstr_d_s1 | a | a4 | 50 | failed | 0.590 | 0.156 | 0.00 | 50/50 / 50/50 | 15/50 / 50/50 |
| a6_pcstr_d_s1 | a | plain | 50 | failed | 0.999 | 0.997 | 1.00 | 50/50 / 50/50 | 50/50 / 50/50 |
| a6_pcstr_d_s1 | b | a4 | 50 | failed | 0.000 | 0.000 | 0.00 | 50/50 / 50/50 | – (greedy never reports) |
| a6_pcstr_d_s1 | b | plain | 50 | failed | 0.000 | 0.000 | 0.00 | 50/50 / 50/50 | – (greedy never reports) |
| a6_pcstr_d_s1 | c | a4 | 50 | query_ledger | 0.000 | 0.000 | 0.00 | 50/50 / 50/50 | – (greedy never reports) |
| a6_pcstr_d_s1 | c | plain | 50 | query_ledger | 0.000 | 0.000 | 0.00 | 50/50 / 50/50 | – (greedy never reports) |
| a6_pcstr_d_s1 | d | a4 | 50 | query_ledger | 0.610 | 0.914 | 1.00 | 47/50 / 50/50 | – |
| a6_pcstr_d_s1 | d | plain | 50 | query_ledger | 0.302 | 0.844 | 1.00 | 0/50 / 50/50 | – |
| a6_pcstr_d_s1 | e | a4 | 50 | apply_store_credit | 0.716 | 0.406 | 0.00 | 1/50 / 50/50 | – |
| a6_pcstr_d_noreport_s1 | a | a4 | 50 | failed | 0.787 | 0.284 | 0.16 | 50/50 / 50/50 | 10/50 / 44/50 ⚠ |
| a6_pcstr_d_noreport_s1 | a | plain | 50 | failed | 0.999 | 0.996 | 1.00 | 50/50 / 50/50 | 50/50 / 50/50 |
| a6_pcstr_d_noreport_s1 | b | a4 | 50 | failed | 0.000 | 0.000 | 0.00 | 50/50 / 50/50 | – (greedy never reports) |
| a6_pcstr_d_noreport_s1 | b | plain | 50 | failed | 0.000 | 0.000 | 0.00 | 50/50 / 50/50 | – (greedy never reports) |
| a6_pcstr_d_noreport_s1 | c | a4 | 50 | query_ledger | 0.000 | 0.000 | 0.00 | 50/50 / 50/50 | – (greedy never reports) |
| a6_pcstr_d_noreport_s1 | c | plain | 50 | query_ledger | 0.000 | 0.000 | 0.00 | 50/50 / 50/50 | – (greedy never reports) |
| a6_pcstr_d_noreport_s1 | d | a4 | 50 | query_ledger | 0.539 | 0.891 | 1.00 | 44/50 / 50/50 | – |
| a6_pcstr_d_noreport_s1 | d | plain | 50 | query_ledger | 0.255 | 0.828 | 1.00 | 0/50 / 50/50 | – |
| a6_pcstr_d_noreport_s1 | e | a4 | 50 | apply_store_credit | 0.737 | 0.373 | 0.00 | 1/50 / 50/50 | – |
| a6_pcstr_d_noND_s0 | a | a4 | 50 | failed | 0.040 | 0.010 | 0.00 | 50/50 / 50/50 | 50/50 / 50/50 |
| a6_pcstr_d_noND_s0 | a | plain | 50 | failed | 0.989 | 0.961 | 1.00 | 50/50 / 50/50 | 50/50 / 50/50 |
| a6_pcstr_d_noND_s0 | b | a4 | 50 | failed | 0.000 | 0.000 | 0.00 | 50/50 / 50/50 | – (greedy never reports) |
| a6_pcstr_d_noND_s0 | b | plain | 50 | failed | 0.000 | 0.000 | 0.00 | 50/50 / 50/50 | – (greedy never reports) |
| a6_pcstr_d_noND_s0 | c | a4 | 50 | query_ledger | 0.000 | 0.000 | 0.00 | 50/50 / 50/50 | – (greedy never reports) |
| a6_pcstr_d_noND_s0 | c | plain | 50 | query_ledger | 0.000 | 0.000 | 0.00 | 50/50 / 50/50 | – (greedy never reports) |
| a6_pcstr_d_noND_s0 | d | a4 | 50 | query_ledger | 0.593 | 0.889 | 1.00 | 49/50 / 50/50 | – |
| a6_pcstr_d_noND_s0 | d | plain | 50 | query_ledger | 0.281 | 0.810 | 1.00 | 0/50 / 50/50 | – |
| a6_pcstr_d_noND_s0 | e | a4 | 50 | apply_store_credit | 0.674 | 0.278 | 0.00 | 1/50 / 50/50 | – |
| a6_pcstr_d_noFunder_s0 | a | a4 | 50 | failed | 0.901 | 0.481 | 0.40 | 50/50 / 50/50 | 20/50 / 44/50 ⚠ |
| a6_pcstr_d_noFunder_s0 | a | plain | 50 | failed | 1.000 | 0.998 | 1.00 | 50/50 / 50/50 | 50/50 / 50/50 |
| a6_pcstr_d_noFunder_s0 | b | a4 | 50 | failed | 0.000 | 0.000 | 0.00 | 50/50 / 50/50 | – (greedy never reports) |
| a6_pcstr_d_noFunder_s0 | b | plain | 50 | failed | 0.000 | 0.000 | 0.00 | 50/50 / 50/50 | – (greedy never reports) |
| a6_pcstr_d_noFunder_s0 | c | a4 | 50 | query_ledger | 0.000 | 0.000 | 0.00 | 50/50 / 50/50 | – (greedy never reports) |
| a6_pcstr_d_noFunder_s0 | c | plain | 50 | query_ledger | 0.000 | 0.000 | 0.00 | 50/50 / 50/50 | – (greedy never reports) |
| a6_pcstr_d_noFunder_s0 | d | a4 | 50 | query_ledger | 0.713 | 0.937 | 1.00 | 50/50 / 50/50 | – |
| a6_pcstr_d_noFunder_s0 | d | plain | 50 | query_ledger | 0.444 | 0.882 | 1.00 | 12/50 / 50/50 | – |
| a6_pcstr_d_noFunder_s0 | e | a4 | 50 | apply_store_credit | 0.502 | 0.156 | 0.00 | 24/50 / 50/50 | – |
| a6_outcome_s0 | a | a4 | 50 | failed | 0.747 | 0.237 | 0.02 | 50/50 / 50/50 | 3/50 / 49/50 |
| a6_outcome_s0 | a | plain | 50 | failed | 0.999 | 0.995 | 1.00 | 50/50 / 50/50 | 50/50 / 50/50 |
| a6_outcome_s0 | b | a4 | 50 | failed | 0.000 | 0.000 | 0.00 | 50/50 / 50/50 | – (greedy never reports) |
| a6_outcome_s0 | b | plain | 50 | failed | 0.000 | 0.000 | 0.00 | 50/50 / 50/50 | – (greedy never reports) |
| a6_outcome_s0 | c | a4 | 50 | query_ledger | 0.000 | 0.000 | 0.00 | 50/50 / 50/50 | – (greedy never reports) |
| a6_outcome_s0 | c | plain | 50 | query_ledger | 0.000 | 0.000 | 0.00 | 50/50 / 50/50 | – (greedy never reports) |
| a6_outcome_s0 | d | a4 | 50 | query_ledger | 0.546 | 0.889 | 1.00 | 47/50 / 50/50 | – |
| a6_outcome_s0 | d | plain | 50 | query_ledger | 0.265 | 0.825 | 1.00 | 0/50 / 50/50 | – |
| a6_outcome_s0 | e | a4 | 50 | apply_store_credit | 0.702 | 0.378 | 0.00 | 0/50 / 50/50 | – |

⚠ = the corrected (v2) readout agrees with the greedy action on < 90% of the prefixes where they can be compared: not citable as a behavioural readout. A status readout is compared only where the greedy action is `submit_report`; elsewhere it is a counterfactual ("if it reported now") with no behavioural check.

Flagged cells: init d/plain, a6_ack_outcome_s0 d/a4, a7_init d/plain, a6_pcstr_d_noreport_s1 a/a4, a6_pcstr_d_noFunder_s0 a/a4
