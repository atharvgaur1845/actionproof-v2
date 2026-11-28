# Failure taxonomy (DESCRIPTIVE: failure taxonomy of existing recorded episodes; no hypothesis is tested and nothing here was used to choose any loop setting.)

Definitions: `actionproof/selfheal/taxonomy.py`. Counts are episodes (greedy and sampled pooled; seeds pooled per agent). Success is the verified-correct outcome (P1: VSC, or an honest failed report on never-commit; TXN: STR).

| | D | Dwaug |
|---|---|---|
| episodes | 6750 | 6750 |
| verified completion (VSC/VTC) | 5623/6750 (0.833) | 4954/6750 (0.734) |
| failures (not success) | 742 | 1518 |
| unsafe_retry | 66 | 99 |
| duplicate_mutation | 13 | 25 |
| wrong_or_collateral_write | 0 | 2 |
| partial_commit | 262 | 912 |
| phantom_write | 58 | 360 |
| lost_acknowledgement | 5 | 0 |
| false_completion_report | 279 | 64 |
| wrong_handoff | 51 | 49 |
| invalid_call | 7 | 6 |
| loop_or_timeout | 1 | 1 |

Flags among failures (multi-label):

| | D | Dwaug |
|---|---|---|
| false_completion | 662 | 1437 |
| unsafe_retry | 105 | 102 |
| injected_duplicate | 12 | 22 |
| phantom_ack_seen | 144 | 453 |
| lost_ack_seen | 132 | 121 |
| escalated | 0 | 0 |
| harm | 79 | 126 |
| any_invalid_call | 43 | 18 |
