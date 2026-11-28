# Failure taxonomy (DESCRIPTIVE: failure taxonomy of existing recorded episodes; no hypothesis is tested and nothing here was used to choose any loop setting.)

Definitions: `actionproof/selfheal/taxonomy.py`. Counts are episodes (greedy and sampled pooled; seeds pooled per agent). Success is the verified-correct outcome (P1: VSC, or an honest failed report on never-commit; TXN: STR).

| | B4 | D | Dwaug |
|---|---|---|---|
| episodes | 440 | 3960 | 3960 |
| verified completion (VSC/VTC) | 380/440 (0.864) | 2864/3960 (0.723) | 2762/3960 (0.697) |
| failures (not success) | 0 | 1082 | 1198 |
| unsafe_retry | 0 | 68 | 189 |
| duplicate_mutation | 0 | 140 | 313 |
| wrong_or_collateral_write | 0 | 120 | 0 |
| partial_commit | 0 | 64 | 139 |
| phantom_write | 0 | 104 | 258 |
| lost_acknowledgement | 0 | 2 | 1 |
| false_completion_report | 0 | 63 | 14 |
| wrong_handoff | 0 | 460 | 279 |
| invalid_call | 0 | 29 | 5 |
| loop_or_timeout | 0 | 32 | 0 |

Flags among failures (multi-label):

| | B4 | D | Dwaug |
|---|---|---|---|
| false_completion | 0 | 459 | 873 |
| unsafe_retry | 0 | 68 | 189 |
| injected_duplicate | 0 | 189 | 341 |
| phantom_ack_seen | 0 | 223 | 327 |
| lost_ack_seen | 0 | 247 | 248 |
| escalated | 0 | 2 | 0 |
| harm | 0 | 328 | 502 |
| any_invalid_call | 0 | 87 | 59 |
