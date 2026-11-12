Mean probability over prefixes (renormalised over the candidate set). Types: a = 4 refunds, last read empty; b = same, last read shows the refund; c = after the first empty read; d = after the first refund ack; e = credit task after a read showing the refund present and `credits: []`.

| model | a/plain P(failed) | a/a4 P(failed) | b/plain P(failed) | b/a4 P(failed) | c/a4 P(issue_refund) | c/a4 P(submit_report) | d/a4 P(query_ledger) | d/plain P(query_ledger) | e/a4 P(apply_store_credit) | e/a4 P(issue_refund) |
|---|---|---|---|---|---|---|---|---|---|---|
| init | 0.999 | 0.731 | 0.000 | 0.000 | 0.999 | 0.001 | 0.157 | 0.046 | 0.587 | 0.409 |
| a6_pcstr_d_s0 | 0.999 | 0.545 | 0.000 | 0.000 | 0.999 | 0.000 | 0.632 | 0.351 | 0.690 | 0.306 |
| a6_pcstr_d_s1 | 0.999 | 0.590 | 0.000 | 0.000 | 0.999 | 0.000 | 0.610 | 0.302 | 0.716 | 0.278 |
| a6_pcstr_d_noreport_s0 | 1.000 | 0.964 | 0.000 | 0.000 | 0.999 | 0.000 | 0.565 | 0.270 | 0.669 | 0.326 |
| a6_pcstr_d_noreport_s1 | 0.999 | 0.787 | 0.000 | 0.000 | 0.999 | 0.000 | 0.539 | 0.255 | 0.737 | 0.259 |
| a6_pcstr_d_noND_s0 | 0.989 | 0.040 | 0.000 | 0.000 | 0.999 | 0.000 | 0.593 | 0.281 | 0.674 | 0.319 |
| a6_outcome_s0 | 0.999 | 0.747 | 0.000 | 0.000 | 0.999 | 0.001 | 0.546 | 0.265 | 0.702 | 0.294 |
| a6_ack_outcome_s0 | 0.999 | 0.673 | 0.000 | 0.000 | 0.999 | 0.000 | 0.064 | 0.017 | 0.412 | 0.584 |
| a6_pcstr_d_s0@50 | 0.999 | 0.511 | 0.000 | 0.000 | 0.999 | 0.000 | 0.540 | 0.238 | 0.651 | 0.345 |
| a6_pcstr_d_s0@100 | 0.998 | 0.399 | 0.000 | 0.000 | 0.999 | 0.000 | 0.647 | 0.369 | 0.672 | 0.324 |
| a6_pcstr_d_s0@150 | 0.998 | 0.452 | 0.000 | 0.000 | 0.999 | 0.000 | 0.610 | 0.304 | 0.670 | 0.326 |
| a6_pcstr_d_noreport_s0@50 | 0.999 | 0.741 | 0.000 | 0.000 | 0.999 | 0.000 | 0.158 | 0.047 | 0.624 | 0.373 |
| a6_pcstr_d_noreport_s0@100 | 1.000 | 0.874 | 0.000 | 0.000 | 0.999 | 0.000 | 0.375 | 0.137 | 0.665 | 0.330 |
| a6_pcstr_d_noreport_s0@150 | 1.000 | 0.962 | 0.000 | 0.000 | 0.999 | 0.000 | 0.523 | 0.242 | 0.683 | 0.312 |
| a6_pcstr_d_noND_s0@50 | 0.998 | 0.400 | 0.000 | 0.000 | 0.999 | 0.000 | 0.274 | 0.086 | 0.580 | 0.416 |
| a6_pcstr_d_noND_s0@100 | 0.997 | 0.187 | 0.000 | 0.000 | 0.999 | 0.000 | 0.466 | 0.191 | 0.669 | 0.325 |
| a6_pcstr_d_noND_s0@150 | 0.994 | 0.075 | 0.000 | 0.000 | 0.999 | 0.000 | 0.522 | 0.228 | 0.663 | 0.331 |

n per cell: 50 prefixes.
