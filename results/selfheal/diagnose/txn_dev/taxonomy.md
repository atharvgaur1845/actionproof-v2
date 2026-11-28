# Failure taxonomy (DESCRIPTIVE: failure taxonomy of existing recorded episodes; no hypothesis is tested and nothing here was used to choose any loop setting.)

Definitions: `actionproof/selfheal/taxonomy.py`. Counts are episodes (greedy and sampled pooled; seeds pooled per agent). Success is the verified-correct outcome (P1: VSC, or an honest failed report on never-commit; TXN: STR).

| | dev_D_none |
|---|---|
| episodes | 360 |
| verified completion (VSC/VTC) | 285/360 (0.792) |
| failures (not success) | 75 |
| unsafe_retry | 4 |
| duplicate_mutation | 9 |
| partial_commit | 1 |
| phantom_write | 8 |
| false_completion_report | 1 |
| wrong_handoff | 49 |
| invalid_call | 3 |

Flags among failures (multi-label):

| | dev_D_none |
|---|---|
| false_completion | 18 |
| unsafe_retry | 4 |
| injected_duplicate | 11 |
| phantom_ack_seen | 20 |
| lost_ack_seen | 17 |
| escalated | 0 |
| harm | 13 |
| any_invalid_call | 2 |
