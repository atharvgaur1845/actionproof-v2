# P5: does recovery SFT replicate on other Aya base models?

Registered 2026-11-26, about 01:20 IST, before any P5 model is trained or evaluated.

**Question.** P1's largest effect came from recovery demonstrations (Dw: SFT on `data/rft/p1_dwarm_rows.jsonl`;
exploratory 2×2: +0.344 over the warm start). Every trained result so far uses one base model, Tiny Aya Global. P5
asks whether the same demonstrations help other Cohere Labs bases.

**Arms.** The same recipe as `slurm/tasks/1153_p1_dwarm.sh`:
- `actionproof.train.rft`, 2 epochs, seed 0, LoRA r = 16, lr 1e-4, batch 2 × 8;
- the same 4,961 rows;
- only `--model` changes.

| m | Base | Base evaluation | Dw_m |
|---|---|---|---|
| fire | CohereLabs/tiny-aya-fire | new (`results/p5/base_fire`) | new |
| water | CohereLabs/tiny-aya-water | new | new |
| earth | CohereLabs/tiny-aya-earth | new | new |
| expanse8b | CohereLabs/aya-expanse-8b | B1's `aya8b` (same loop, already run) | new |

**Evaluation.** The frozen P1 test, five splits, 450 tasks. P1's loop is unchanged (`scripts/p5_eval.py`, a copy of
`scripts/b1_eval.py` writing to `results/p5/`): 1 greedy + 4 sampled episodes.

**Hypothesis (per model).** Dw_m > base_m in sampled VSC:
- paired task bootstrap, one-sided, Holm over the four models (`scripts/p5_stats.py`, committed now);
- one SFT seed per model, so this is a task-level test with the model fixed, and it is stated as such;
- descriptive: Dw_m against Tiny Aya Global's Dw, and per-split VSC.

**Priority.** P5 is tier 2 in the sprint plan: fire → water → earth → expanse8b. A model not run before the GPU gate
(Nov 27 16:00 IST) is reported as not run. The Holm family is then the models that ran.
