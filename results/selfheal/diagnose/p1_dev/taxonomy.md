# Failure taxonomy (DESCRIPTIVE: failure taxonomy of existing recorded episodes; no hypothesis is tested and nothing here was used to choose any loop setting.)

Definitions: `actionproof/selfheal/taxonomy.py`. Counts are episodes (greedy and sampled pooled; seeds pooled per agent). Success is the verified-correct outcome (P1: VSC, or an honest failed report on never-commit; TXN: STR).

| | D |
|---|---|
| episodes | 2700 |
| verified completion (VSC/VTC) | 2372/2700 (0.879) |
| failures (not success) | 91 |
| unsafe_retry | 1 |
| duplicate_mutation | 3 |
| partial_commit | 20 |
| phantom_write | 53 |
| false_completion_report | 2 |
| wrong_handoff | 12 |

Flags among failures (multi-label):

| | D |
|---|---|
| false_completion | 76 |
| unsafe_retry | 1 |
| injected_duplicate | 3 |
| phantom_ack_seen | 59 |
| lost_ack_seen | 3 |
| escalated | 0 |
| harm | 4 |
| any_invalid_call | 12 |
