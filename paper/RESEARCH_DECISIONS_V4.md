# Research decisions for main_v4 (2026-11-28)

Source directive: `../ActionProof_RESEARCH_PIVOT_AND_PAPER_REBUILD.md`. Branch `research/v4-hidden-world`, created from
`final-sprint/winning-submission` at 6ebfe04 (remote `submission-candidate` = 17bb12f). Nothing was reset, deleted or
force-pushed; `main.tex`, `main_v2.tex`, `main_v3.tex` and `actionproof_main.tex` are untouched.

## What changed
- **Main story.** The theory-led paper (shared latent state × group standardisation × costed verification) is now the
  main paper (`paper/main_v4.tex`, EMNLP 2023 style, final mode, 6 + 6 format). The practical ActionProof study is
  Appendix `app4:practical` and remains a separate artifact (`paper/actionproof_main.pdf`).
- **Every number re-read from source** (`paper/EVIDENCE_LEDGER_V4.md`); 44 theory tests pass.
- **Citations** fetched from arXiv's BibTeX export or the ACL Anthology (`paper/refs_v4_fetched.bib`), including the
  Lexsi kits actually used (AgentTune, AuditKIT, CuratorKIT) and the Cohere models (Tiny Aya, Aya Expanse). AlignTune,
  SafeTune and CircuitKIT are not cited because the project did not use them.
- **Related work** compared on full text for Fontana et al. (ACL 2026), Bay & Yearick, Sakib and Calibrate-Then-Act;
  LIMBO, Evidence-Guided Recovery and Kaynar et al. from abstracts (stated in the paper).

## Audit findings that changed the text
1. **AD-C1's GRPO arm equals the warm start.** All five GRPO seeds and the warm start have identical greedy outcomes
   (reads 0.000, VSC 0.687, false success 0.313). The directive's "identical greedy outputs" is imprecise: conversations
   of seeds 0 and 3 agree on 175/300 tasks; the *outcomes* are identical.
2. **p = 0.016 is the floor** of the registered exact 5-vs-5 permutation test after Holm (1/252 × 4); it is reported
   as such.
3. **Sampled behaviour separates the explanations** (post hoc, descriptive): warm start 0.189 reads per episode under
   T = 1 sampling, GRPO 0.000–0.001 in every seed, std-free 1.051. GRPO did suppress reading, as the vote predicts.
4. Fontana et al.'s Prop. 2 is a *global* reward-scale invariance from AdamW, not a per-state one; the paper says so.
5. Calibrate-Then-Act reports GRPO collapsing to a static policy across cost regimes, but its group construction is
   not stated in the main text, so it is not claimed as an instance.

## New experiment
- **AD-C2** (`PREREGISTRATION_AD_C2.md`, registered 17:50 IST before any output; commit b089bf2): GRPO and std-free at
  c = 0.01 and GRPO at c = 0, seed 0, shared group law, sealed split `cw_c2` (sha 0f792406…). Closes AD-C1's T8 as a
  *limited* intervention.
- **Not run:** the full factorial (2 estimators × 3 costs × 2 group laws × seeds). The iid group law needs new rollout
  code that could not be built and checked before the cutoff; one H100 lease was available (a second H100 job is
  refused by QOS; H200 and A100 workers were submitted and stayed in the queue).
- **Status:** see the AD-C2 section of the paper, which states that the runs were not complete when this version was
  frozen; it will be filled from `results/ad/c2/c2_summary.json` exactly as registered.

## Not done, and why
- **No new estimator.** No candidate met the directive's acceptance gate in the time available; the paper proposes none.
- **No new schematic figure** of the shared-group mechanism; the two figures are generated from committed data
  (`scripts/paper/v4_figures.py`).
- **No human labels** exist for any judge evaluation.

## Clean room for v4 (2026-11-28, pushed commit 28e0b01)
- Fresh clone of `submission-candidate` on sharanga: `pytest tests/theory` 44 passed; `freeze_c2.py --check` OK;
  `s2_numcheck.py` 899 numbers, 0 failures; `scripts/paper/v4_figures.py` regenerates both figures byte-identical
  except the PDF creation date.
- Sparse clone of `paper/` (sharanga has no LaTeX): `main_v4` builds with 0 errors, 0 undefined references, 10 pages,
  no "pending" text.
- A full local clone was attempted first and filled the local disk (0 bytes free); it was deleted at once, and the
  clean room moved to the cluster.

## Remaining gaps
- AD-C2 outcome (running on sharanga); one seed per arm; one group law.
- The language-model evidence is one model family and one simulator.

## Response to the blind review (2026-11-28, 23:40 IST)
An independent blind judging pass (Opus; 21 entries under neutral codes) ranked this entry 4th at 6.95. Changes made:
- Both AD-C1 intervals are now reported: the registered seed-averaged task bootstrap [0.088, 0.153] and the two-level
  interval [0.075, 0.167] (`results/ad/c1_summary.json`), in the abstract, §AD-C1 and the deck.
- The practical study's 0.513 is labelled as trained arm A; the untrained Tiny Aya Global base (0.01) is given beside it.
- New appendix `app4:submission` (`paper/submission_v4.tex`): track and problem, one-line claim, same-model
  before → after, kits, data, metrics, declaration, credits plan, compute, use of AI, commands.
- New Track 3 table (`paper/tables_v4/track3.tex`, from `scripts/paper/v4_track3_table.py`; descriptive).
- One submitted paper: `main_v4.pdf`; `main_v3.pdf` moved to `paper/archive/`; README, WRITEUP, SLIDES, JUDGE_QA and
  the deck no longer call the theory a companion.
- `research_audit/` (internal notes, including competitor profiles) is no longer tracked on the submission branch;
  it stays in git history. Clean-room logs moved to `docs/reproducibility/cleanroom/`.
- Deck rendered to `docs/submission/ActionProof_deck.pdf`.
