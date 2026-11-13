## Trained policies (Tiny Aya Global 3.35B), untouched confirmatory splits

`test_iid_c` ∪ `test_ood_fault_c`, n = 400 tasks, greedy. VSC = verified-success completion: the database shows the goal met **and** the report says so. One value per training seed.

| policy | seeds | VSC (%) per seed | Wilson 95% CI (seed 0) |
|---|---|---|---|
| warm start (SFT only) | 1 | 64.2 | 59–69 |
| tool_trust: believes acks + report | 2 | 63.5 / 64.5 | 59–68 |
| LLM judge (Aya 8B) | 1 | 63.5 | 59–68 |
| outcome: binary DB check | 2 | 82.0 / 83.5 | 78–85 |
| PCSTR | 1 | 84.2 | 80–87 |
| PCSTR-c (verification cost) | 1 | 83.8 | 80–87 |
| **PCSTR-d** (ours) | 2 | 84.2 / 84.2 | 80–87 |

Registered confirmatory test H1 (PCSTR-d − outcome): **+1.5 pp** [0.6, 2.6], p = 0.025. PCSTR-d − tool_trust: +20.25 pp.

## Untrained Cohere models, `test_iid` (n = 200; neutral prompt, and the “verify every write” prompt verify_a)

| model | VSC (%) | false-completion rate when failed (%) | verify calls / episode |
|---|---|---|---|
| aya-expanse-32b | 51.5 [45–58] | 98 | 1.45 |
| aya-expanse-32b + “verify” prompt (verify_a) | 33.0 [27–40] | 99 | 1.80 |
| tiny-aya-earth + “verify” prompt (verify_a) | 0.5 [0–3] | 29 | 1.77 |
| tiny-aya-global + “verify” prompt (verify_a) | 0.5 [0–3] | 34 | 3.21 |
| tiny-aya-water + “verify” prompt (verify_a) | 0.5 [0–3] | 56 | 1.96 |
| aya-expanse-8b | 0.0 [0–2] | 49 | 0.89 |
| aya-expanse-8b + “verify” prompt (verify_a) | 0.0 [0–2] | 47 | 0.17 |
| tiny-aya-earth | 0.0 [0–2] | 43 | 1.23 |
| tiny-aya-fire | 0.0 [0–2] | 62 | 1.32 |
| tiny-aya-fire + “verify” prompt (verify_a) | 0.0 [0–2] | 28 | 1.87 |
| tiny-aya-global | 0.0 [0–2] | 34 | 1.65 |
| tiny-aya-water | 0.0 [0–2] | 41 | 2.35 |

Sources: `results/confirmatory.json`, `results/leaderboard.json`.
