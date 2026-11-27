> **Superseded (2026-11-27, Stage 2 sprint).** This file is kept as history (last updated 2026-11-22) and is not current. The current source is `docs/submission/DEMO_RUN_OF_SHOW.md` (live Space plus replay Space). See `research_audit/STALE_SURFACES_AUDIT.md`.

# Three-minute demonstration: run of show

- **Screen:** the website at 1440 px (`demo/static/index.html`), built from commit stage4 HEAD. No live model inference: the simulator is real and precomputed at build time, and the model behaviour is real recorded evaluation episodes. Say so once, at 0:20.
- **Narrator:** Atharv.
- **Numbers:** every number is a row of `docs/final/FINAL_CLAIMS.md`.

| Time | Screen | Action | Narration (spoken) |
|---|---|---|---|
| 0:00–0:20 | Hero, "THE TOOL SAID SUCCESS. THE WORLD DISAGREED." | Let the transaction animation play: reply ok, then "payments stored: none", then MISMATCH, then the report "completed" | "An agent pays a supplier. The tool says ok. Nothing was stored, and the agent reports the job done. This is a real recorded episode from our evaluation." |
| 0:20–0:50 | 01 Three truths | Pick Fault = Partial commit, 25%. Toggle "trusts the reply" → "reads the record" | "This is our procurement simulator, running for real. The same ok comes back whether the write stored everything, part, or nothing. The hatched column is the database, which only our verifier sees. Trust the reply and you are wrong. Read the record and you can fix it. The two rewards score this run differently: the state reward gives 0, the acknowledgement reward gives 1." |
| 0:50–1:20 | 02 What the reward sees | Point at the seed strip, then the two recorded traces | "Same Cohere Tiny Aya model, same warm start, same GRPO. We change only what the reward reads. With the database reward, the agent reads the payment record after a phantom write in all five seeds. With the acknowledgement reward, never. One honest caveat: the starting model already did this sometimes. Training selected the habit; it did not invent it." |
| 1:20–1:50 | 03 Reading is not recovery | Model = "Trained on the state reward". Point left, then right | "But reading is not recovering. Same fault, same model. When payment comes first, it reads, refunds and pays again. After a goods receipt, it reads that only a quarter was paid, then posts and says completed. Across seeds: almost always repaired in one workflow, never in another." |
| 1:50–2:25 | 04 Training the recovery decision | Scroll the three cards, the decomposition bar, then the forest plot | "So we trained from the broken state: the same GRPO, with episodes that start at the faulted write. Plus 0.21 verified completion, five of five runs. About half of that is a fault type the earlier agent never saw. Against a fair control, it helps partial payments, at a small cost in unrefunded duplicates, and only on surfaces it practised. Rename the tools or add a step, and the gain is gone." |
| 2:25–3:00 | 05 Reproduction, then the rail back to the hero | Scroll the reproduction grid | "Every experiment was registered before it ran, every number is checked against its file, and the failures are on the page, including our own credit-assignment method that did not work. Reliable agents must be judged by the state changes they produce, not the success messages they generate." |

**Backup if the site cannot load:** open `demo/static/index.html` locally (a single file with no network calls except fonts).

**Before recording:**
- Hard-refresh the page.
- Check that the simulator's default (pay first, partial, 50%) shows the MISMATCH.
- Run `python scripts/s2_numcheck.py` and `pytest tests/product`.
