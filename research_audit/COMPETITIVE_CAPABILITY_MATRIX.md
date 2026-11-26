# Competitive capability matrix

**Date.** 2026-11-27.

**Source.** `COMPETITOR_FORENSICS_TOP20.md`. Repository evidence only; no competitor was rerun.

**Scale.**
- **3** = materially stronger than the field median.
- **2** = solid.
- **1** = present but weak.
- **0** = absent.

These are ratings of evidence quality and engineering execution. **They do not compare task scores.** Every project
measures a different task, denominator and protocol, so their percentages are not comparable with ours or with each
other.

## The matrix

| # | Project | Novelty | Breadth | Stats | Baselines | Utility | Live demo | Presentation | Repro | Impact | Compliance |
|---|---|---|---|---|---|---|---|---|---|---|---|
| — | **ActionProof (ours)** | 2 | 3 | 3 | 3 | 2 | 3 | 1 | 2 | 1 | 3 |
| 01 | Lost in Injection | 2 | 3 | 2 | 3 | 1 | 0 | 3 | 2 | 2 | 2 |
| 02 | Escalation-Safe Aya | 2 | 3 | 2 | 3 | 2 | 1 | 2 | 3 | 2 | 3 |
| 03 | Desi Agent | 1 | 2 | 2 | 2 | 3 | 1 | 3 | 2 | 2 | 3 |
| 04 | Drift Monitor | 2 | 2 | 1 | 2 | 3 | 3 | 3 | 2 | 3 | 3 |
| 05 | Relapse | 2 | 3 | 3 | 2 | 1 | 1 | 2 | 2 | 3 | 3 |
| 06 | Hinglish Expense Agent | 1 | 2 | 2 | 2 | 2 | 2 | 3 | 3 | 1 | 3 |
| 07 | Lost in Refusal | 2 | 3 | 3 | 2 | 1 | 1 | 1 | 2 | 1 | 3 |
| 08 | EVIDENCELOCK | 2 | 1 | 2 | 2 | 2 | 1 | 2 | 2 | 1 | 3 |
| 09 | Safe from Fine-Tune to Phone | 1 | 2 | 2 | 2 | 1 | 0 | 1 | 2 | 1 | 2 |
| 10 | Lexsi AI Alignment | 2 | 1 | 1 | 1 | 1 | 2 | 2 | 1 | 1 | 2 |
| 11 | Against the Safety | 1 | 2 | 2 | 3 | 2 | 1 | 2 | 2 | 1 | 3 |
| 12 | Same Model, Different Language | 1 | 2 | 3 | 2 | 1 | 1 | 1 | 2 | 1 | 2 |
| 13 | RewardShield | 2 | 2 | 3 | 2 | 1 | 0 | 2 | 2 | 1 | 2 |
| 14 | VeriLoop | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| 15 | Reward Hacking Observatory | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 1 | 0 | 0 |
| 16 | Contextual Pseudonymizer | 1 | 1 | 0 | 1 | 1 | 0 | 1 | 1 | 1 | 2 |
| 17 | Tamil Clinical Triage | 1 | 1 | 0 | 1 | 1 | 0 | 1 | 1 | 1 | 2 |
| 18 | ICSA Grounded Assistant | 1 | 0 | 0 | 1 | 1 | 0 | 1 | 1 | 1 | 0 |
| 19 | FinSafe | 1 | 1 | 1 | 1 | 1 | 0 | 1 | 2 | 1 | 2 |
| 20 | Recovery Agent, Taught by Rules | 1 | 1 | 0 | 1 | 2 | 0 | 1 | 1 | 1 | 2 |

## Column definitions

- **Novelty.** Is the contribution a new benchmark, finding, intervention or system, or an application?
- **Breadth.** Models, seeds, datasets, task counts, adversarial and OOD splits, ablations.
- **Stats.** Registration, intervals, multiplicity correction, the seed/task distinction, independent labels.
- **Baselines.** Strength, fair budget, the same interface.
- **Utility.** Does it solve a meaningful task end to end?
- **Live demo.** Real model inference, interactivity, trace inspection, a fallback when the GPU is unavailable.
- **Presentation.** Hook, figures, a concise thesis, a finished video or slides **as they exist in the repository
  today**.
- **Repro.** Weights, data and traces public; hashes; a no-GPU check; pinned dependencies.
- **Impact.** Quantified, with grounded assumptions or real-world evidence.
- **Compliance.** Cohere core model, toolkits actually imported, Hub hosting, public deployment.

## Notes behind the ActionProof row

**Stats: 3.**
- 52 registration files; Holm corrections; template-clustered and paired task bootstraps.
- Seed and task uncertainty are reported separately in P4 and P5.
- `scripts/s2_numcheck.py` has 791 checks.

**Baselines: 3.**
- Scripted B0–B4 and ablations A1–A6 on the same tasks, budget and evaluator.
- Prompted Aya Expanse 32B, Qwen3-235B and gpt-oss-120b, labelled supplementary.

**Live demo: 3.** It is the only *awake* live-inference Space among the Track 3 entries at audit time. Its gaps:
- no guided judge path;
- no export of a run;
- no labelled replay fallback inside the Space itself;
- no failure gallery.

**Presentation: 1.**
- The slides exist only as Markdown source (`docs/submission/SLIDES.md`), with no rendered deck.
- There is no recorded video.
- The README's top block is good, but it is followed by about 150 lines of Stage 1–4 history.
- The write-up, paper and README open with three different theses: the reward information source, transaction
  recovery, and group-normalisation theory.
- **This is the largest gap against the field, and the round is judged only on the video or slides.**

**Repro: 2.**
- `verify_numbers.sh` recomputes every registered analysis, but it downloads about 2 GB.
- There is no 5-minute check and no artefact manifest with Hub revisions.
- The README's reproduce section has stale statements: "no CI is configured" while `.github/workflows/ci.yml`
  exists; a Nov 15 clone check.

**Impact: 1.** `docs/upgrade/ROI.md` is a projection on simulator rates. There is no real-world exposure evidence
(contrast Relapse's survey) and no impact graphic.

## What the matrix says

1. **Research quality is not our bottleneck.** Our statistics and baselines match the strongest research entries
   (Relapse, Lost in Refusal, RewardShield).
2. **Our weakest columns are Presentation and Impact,** exactly where judges spend their three minutes. The
   leaders in those columns (Drift Monitor, Desi Agent, Lost in Injection, Hinglish Expense) each have a finished
   video or a numbered judge path, a hero visual with the finding, and at most four headline findings.
3. **The live-demo column is our clearest structural lead.** It only counts if the demo is legible in 60 seconds
   and resilient when the GPU is unavailable.
4. **The open scientific flank is escalation.** Escalation-Safe Aya owns "stop and say so"; our H5 shows that our
   agent rarely escalates.
