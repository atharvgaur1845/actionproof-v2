# Ablation report

Every arm in this programme differs from another in exactly one component. Each ablation below answers one question.
Numbers are filled only from `results/ad/d1_summary.json`, `d2_summary.json`, `d3_summary.json`,
`c1_summary.json` and `d1_mechanism.json`. **PENDING** marks a number that is not yet on file.

| # | Ablation | Arms compared (only this differs) | Question |
|---|---|---|---|
| A1 | Per-group std scaling | grpo vs nostd | Does group std, with group-shared hidden worlds, destroy the verification signal (P1)? |
| A2 | Rao-Blackwellised returns, no std | awrbn vs nostd | Does integrating out the hidden world help beyond removing the std (P5)? |
| A3 | Rao-Blackwellised returns, group std | awrb vs grpo | Does RB rescue GRPO, or introduce the mirror bias (P4)? |
| A4 | Scale type | nostd_ema vs nostd; awrbn_ema vs awrbn | Was no-std's slowness a step-size effect? |
| A5 | RB returns at matched scale | awrbn_ema vs nostd_ema | A2 without the step-size confound |
| A6 | Local credit for uninformative reads | nostd_ema_irc vs nostd_ema | Is the selectivity failure (F3) credit smear? |
| A7 | Verification cost | awrb, awrbn at c = 0.05 vs 0.30 | Does each estimator stop verifying when verification is not worth its cost (P4 vs P5)? |

## Results

PENDING: filled when the D1, D2 and D3 summaries are complete.
