# Competitive threat ranking (current evidence, 2026-11-27)

These rankings use the evidence found in each repository on 2026-11-27 (`COMPETITOR_FORENSICS_TOP20.md`,
`COMPETITIVE_CAPABILITY_MATRIX.md`), not the earlier snapshot scores. The 20 repositories are public candidates,
not a verified finalist list. No task score is compared across projects.

## 1. Strongest research package

1. **Relapse (05).** Six preregistrations with every hypothesis counted (11 held, 11 failed), a public correction
   of its own Stage 1 error, and a 150-run map. It also has real-world exposure evidence: 418 affected Hub repos.
2. **Lost in Refusal (07).** Three seeds, sections registered before each run, 120 + 300 human labels, a second
   machine, a second training stack and an outside model family.
3. **Lost in Injection (01).** 940,992 scored replies, 7 models in 3 families, equal-budget controls and three
   scoring rules, submitted to a workshop.
4. **Escalation-Safe Aya (02).** Five Aya bases, three seeds on the main recipe, a set sealed after the model froze,
   a real-server check and reward ablations.
5. **RewardShield (13).** An oracle independent in code, an external audit, and dated corrections.

*ActionProof's place.* With 52 registrations, P1/P4 at three seeds, P5 across five Cohere bases and a locked
counterfactual benchmark, ActionProof is in this tier. It is weaker than Relapse and Lost in Refusal on independent
labels: our reference verifier shares an author and a specification with the main one. It is stronger than all of
them on agent-level registered effect sizes.

## 2. Strongest Track 3 / agent system

1. **Escalation-Safe Aya (02).** Directly adjacent to our H5 gap.
2. **Desi Agent (03).**
3. **Hinglish Expense Agent (06).** Stage 1 only.
4. **EVIDENCELOCK (08).** Stage 1 only.
5. **Against the Safety (11).** Stage 1 only.
6. **RewardShield (13).**

## 3. Strongest demo and judge experience

1. **Drift Monitor (04).** A live tab with no key, replay animation, and idle/offline states.
2. **Hinglish Expense Agent (06).** A numbered "Judge's 60-second view", a CPU path that spares the GPU quota, and
   three galleries. The live Space was asleep at audit time.
3. **Desi Agent (03).** A tour, an outcome-filtered explorer over 486 conversations, a video, and a banner.
4. **Lost in Injection (01).** No demo, but the best visual storytelling: a GIF hero and a 4-panel findings figure.

## 4. Strongest reproducibility package

1. **Escalation-Safe Aya (02).** A 5-minute no-GPU check with a byte-identical rebuild, and per-command costs on HF
   Jobs.
2. **Hinglish Expense Agent (06).** 31 checks with no GPU and no token, plus a sha256 manifest re-hashed from the
   Hub.
3. **EVIDENCELOCK (08).** 53 checks of the headline numbers in one second.
4. **ActionProof.** `verify_numbers.sh` recomputes every registered analysis and 791 numbers, but needs a 2 GB
   download. There is no small smoke path and no Hub revision manifest.

## 5. Strongest overall public submission

1. **Escalation-Safe Aya (02).** Track 3, strong science, strong reproducibility. It owns our weakest flank.
2. **Lost in Injection (01).** Research breadth and the best story.
3. **Drift Monitor (04).** Product, live demo, video, cost.
4. **Desi Agent (03).** Polish and the judge explorer.
5. **Relapse (05).** Integrity and impact.
6. **Lost in Refusal (07).**
7. **Hinglish Expense Agent (06).** Strong but frozen since Nov 18.

The rest (08–20) are lower threats. Several are inactive since Stage 1 (06, 08, 09, 11, 16–20). Two have no Cohere
core model (15, 18). Two have no results (14, 15).

## Answers

**What do the top three (02, 01, 04) have that we do not?**
- A finished, rendered presentation: a video (01: 87 s Manim; 04: 2:48 with slides and an .srt) or a paper-quality
  figure set and an at-a-glance block (02).
- One thesis stated in one line at the top of every surface.
- 02: escalation safety as the central, measured result.
- 04: a live demo that degrades gracefully, and a cost per million operations.

**Which advantages can we match before the deadline?**
- A rendered six-slide deck and a video-ready run of show, with captions retimed.
- One thesis across the README, write-up, slides and paper status.
- A numbered judge path in the README and the Space.
- A small no-GPU smoke check (under 5 minutes) beside the full 2 GB route.
- A Hub revision manifest.
- An always-escalate baseline next to every escalation metric.
- An impact graphic.
- A selective-escalation experiment, if it passes its registered tests.

**Which advantages are structurally irrelevant and should not be copied?**
- Multilingual breadth (01, 07): our problem is execution state, not language. Hindi and Hinglish are already shown
  (S4).
- 70B-scale models.
- Closed-model API comparisons as headline rows: the official rules restrict non-Cohere models to supplementary use.
- Quantization studies.
- A workshop-paper target.

**Where does ActionProof have a defensible unique advantage?**
- **Counterfactual execution ambiguity.** The same request, tool, arguments and reply hide different database
  states. Success cannot be read from the transcript, and a deterministic database verifier scores it. No other
  entry has hidden execution state behind identical observations (04 and 09 have counterfactual *inputs*, not
  hidden *state*).
- **The only awake live Cohere-agent Space in Track 3** that reveals the hidden state after the run.
- **Registered multi-seed effects at agent level** across five Cohere bases, with every negative result reported.

**What is the single biggest reason a judge might rank a competitor above ActionProof?**
The judge cannot tell, in three minutes, what ActionProof's one contribution is. There are three theses across the
surfaces, a dense README, Markdown-only slides, and no video. The headline safety behaviour also fails: "rarely
escalates, 15 of 540", next to a competitor whose whole result is safe stopping.

**What is the smallest set of changes that can reverse that judgement?**
1. One thesis: *transaction-integrity recovery under ambiguous execution*. The benchmark (TXN) and the recovery agent
   are the spine. The theory paper is labelled a companion artefact.
2. A rendered six-slide deck and a video run of show that open on the counterfactual "ok".
3. A numbered 60-second judge path, in the README and in the live Space.
4. The escalation gap addressed by a registered selective-escalation experiment, or, if it fails, reported as the
   field's open problem with an always-escalate baseline.
5. A 5-minute no-GPU check and a revision manifest.
