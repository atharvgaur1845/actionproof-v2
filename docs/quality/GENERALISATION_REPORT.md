# Generalisation report

2026-11-24. Ledger claims P1 (descriptive per-split numbers) and P2-SHIFT. Sampled VSC; intervals are 95% task bootstraps. Shift results are descriptive by registration.

## Inside the frozen test (P1, per split, seed-pooled)

| Split | What is new | A | C | D |
|---|---|---|---|---|
| p1_test_pp | unseen payment fractions | 0.319 | 0.860 | 0.931 |
| p1_test_wf4 | unseen workflow (order → receive → pay → post) | 0.174 | 0.397 | 0.486 |

Recovery transfers to new payment fractions. It does not transfer to a new step order: in the unseen workflow every arm skips the goods receipt and reports success (`PERFORMANCE_DIAGNOSIS.md` §3).

## Shift splits (P2; 150 tasks each; difference from A)

| Split | A | C − A | D − A | E − A |
|---|---|---|---|---|
| Hindi (s4_lang_hi) | 0.668 | +0.161 | +0.182 | +0.130 |
| Hinglish (s4_lang_hinglish) | 0.596 | +0.170 | +0.181 | +0.220 |
| Renamed tools (s4_api) | 0.157 | +0.159 | +0.045 | +0.096 |

- **Language shift:** the recovery gain holds in Hindi and Hinglish for every arm, with every interval above zero.
- **Renamed tools:** every arm stays low, though above A. D gains least there (lower bound 0.008), consistent with GRPO strengthening the trained tool-name procedure.
- **Not tested:** other base models, other domains (Stage 3's healthcare transfer was negative) and real systems.
