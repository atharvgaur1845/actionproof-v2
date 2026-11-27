# Final competitive scorecard

> **These scores are our subjective assessments, not official scores.** They come from public repository evidence
> read on 2026-11-28 (`FINAL_20_COMPETITOR_INTELLIGENCE.md`) and from yesterday's profiles
> (`COMPETITOR_FORENSICS_TOP20.md`). No competitor code was run and no Space was driven in a browser. No task score
> is compared across projects, because every project uses its own benchmark. The judges' rubric is not public.

**ActionProof is scored as it stands at 2026-11-28 02:15 IST** (branch `final-sprint/winning-submission`, from
85657de). The Recovery Gate experiment (RG1) is in progress and has no results yet, so it does not enter any score.

## Rubric

Scale 1–5: 5 is the best seen in this set, 3 is solid, 1 is absent or weak.

| Key | Dimension | What earns a 5 |
|---|---|---|
| Q | Research question and novelty | A sharp question nobody else in the set poses, with honest prior-work positioning |
| V | Experimental validity and scale | Frozen or sealed tests, multiple seeds, large n, intervals, registered analyses |
| C | Causal or ablative evidence | Same-start arms or ablations isolating the mechanism, not a pile of metrics |
| A | Adversarial testing | Held-out attack or fault kinds, independent checks, self-red-teaming that found and fixed real holes |
| O | Operational value | Measured cost, latency or real-world exposure; a result a practitioner would act on |
| D | Live demonstration | A judge runs a real model in seconds, with honest labels and graceful failure states |
| R | Reproducibility | A fast no-GPU number check, public artefacts, pinned revisions, a clean-clone run |
| P | Presentation | A finished video, slides, figures and a paper a generalist understands in minutes |
| S | Story coherence | One thesis, the same on every surface, from problem to limits |

Confidence: **H** = based on result tables or limitations read in full; **M** = README plus partial files; **L** =
README only, or the evidence is indirect.

## Scores

| Project | Q | V | C | A | O | D | R | P | S | Sum /45 |
|---|---|---|---|---|---|---|---|---|---|---|
| **ActionProof (current state)** | 4 M | 4 H | 4 M | 4 M | 2 H | 5 M | 5 H | 2 H | 4 M | **34** |
| 02 Escalation-Safe Aya | 4 M | 4 H | 4 H | 4 H | 3 M | 2 H | 5 H | 3 M | 5 H | **34** |
| 04 Drift Monitor | 3 M | 3 H | 3 M | 2 M | 5 H | 5 H | 3 L | 5 H | 5 H | **34** |
| 01 Lost in Injection | 4 M | 5 H | 4 M | 4 M | 2 M | 1 H | 3 M | 5 H | 5 H | **33** |
| 05 Relapse | 4 M | 5 H | 4 M | 3 M | 4 H | 2 H | 4 M | 4 M | 3 M | **33** |
| 03 Desi Agent | 3 M | 3 H | 3 M | 3 M | 3 M | 3 H | 3 M | 5 H | 4 M | **30** |
| 07 Lost in Refusal | 3 M | 5 H | 4 M | 3 M | 2 M | 1 H | 4 M | 3 L | 3 M | **28** |
| 06 Hinglish Expense Agent (Stage 1 only) | 3 M | 3 H | 3 M | 4 M | 3 M | 3 H | 5 H | 4 M | 5 H | **33** |
| 12 Same Model, Different Language | 3 M | 4 H | 3 M | 3 M | 2 M | 1 M | 4 M | 3 M | 3 M | **26** |
| 13 RewardShield | 3 M | 3 H | 3 M | 4 H | 2 M | 1 H | 5 H | 2 M | 4 M | **27** |

The sums are only a summary. Six projects (ActionProof, 02, 04, 01, 05 and 06) lie within one point of each other,
which is inside the error of this method. 06 has no Stage 2 work. **No project, ours included, leads clearly.** A judge's weighting decides the order. For example, a judge
who weights presentation and product favours 04; one who weights a Track 3 safety result favours 02.

## Why ActionProof scores as it does

| Dim | Score | Evidence (this branch) | What would raise it |
|---|---|---|---|
| Q | 4 M | Hidden execution state behind identical replies: the same request, tool, arguments and reply hide different database states. No other entry poses this (04 and 09 counterfactual *inputs*, not hidden *state*). The paired design itself is not new (`NOVELTY_POSITIONING_FINAL.md`, addendum 2) | RG1: a learned-agent-plus-evidence-gate system that no competitor has |
| V | 4 H | 54 registrations and addenda; P1 / P4 at three seeds; 440 locked TXN tasks; P5 across five Cohere bases. Held back by our own finding that TXN v1 is weaker than designed: a read-free same-key double send reaches 308 / 440 VTC, and "unresolvable" keyed tasks settle by a same-key resend (TXN1 addendum 3) | A fresh sealed split for any new claim; the benchmark limits kept beside every TXN headline |
| C | 4 M | B4 ablations A1–A6; SEL1 against a matched-volume control. P1 arms are not nested, so C→D is not attributed to GRPO alone | Gate rule ablations on the same tasks and budget (planned for RG1) |
| A | 4 M | 25 scoring mutants killed; OOD alias, schema, workflow, combo, evidence and domain groups; three hostile reviews with a 35-row resolution table. The reference verifier shares an author and spec (a consistency check, not independent) | Hidden-state leakage tests for any runtime controller |
| O | 2 H | A hand-written script (B4) beats every trained agent. Reviewers rated impact 4/10. The impact projection is in both units, with assumptions stated. "Checks plus agent" was never measured | **The single largest gap.** A measured hybrid (agent plus gate) against B4, with cost per episode |
| D | 5 M | The only awake Track 3 Space with live Cohere inference, a scripted reference side by side, and the hidden database revealed after the run. Labelled fallbacks are API-tested. Not checked at 1280 px or on mobile; base-model options fall back without `HF_TOKEN` | Set `HF_TOKEN`; browser check; a gate option in the Space |
| R | 5 H | `smoke_check.py` takes about 15–30 s with no GPU, download or token. `verify_numbers.sh` recomputes every registered analysis. Two clean-room runs were byte-identical. A Hub revision manifest. The GitHub default branch still serves an older state until `submission-candidate` is fast-forwarded | The author fast-forwards `submission-candidate` |
| P | 2 H | A rendered six-slide deck (artifact, private until shared); a 2:45 video script and captions. **No recorded video.** The paper (`paper/main_v3.tex`) is a companion theory paper, not a paper of the main result. **No arXiv citations of the kits or of Tiny Aya** (see the intelligence file) | Record the video; export the deck; cite the kits; a short paper of the TXN / agent result in the EMNLP template |
| S | 4 M | One thesis across README, write-up, slides, deck, video script and Space since the Nov 27 revision. The companion paper is labelled as a separate claim | Keep RG1 inside the same thesis ("what the agent needs around it"), not a new story |

## Head-to-head notes against the four nearest

- **02 (Track 3, safe stopping).**
  - 02 leads on: one crisp safety finding (0.0% against 99.2%), and a narrative judges grasp at once.
  - We lead on: hidden-state ambiguity, a live model, and a deterministic state verifier. 02's escalation trigger is
    visible in the tool output.
  - 02 is equal or better on reproducibility: a no-GPU byte-identical check like ours, plus cost per command.
  - Our weak flank against 02: hand-off where it is the only safe action (SEL1: 164 / 432 unkeyed episodes,
    seed-fragile).
- **04 (product and presentation).**
  - 04 leads on: the video, slides, captions and paper; cost per million steps; a graceful live demo.
  - We lead on: executed tools and a database verifier (04's sessions are written and labelled by models); three
    seeds; registered tests.
  - 04 is not Track 3.
- **01 (breadth and story).**
  - 01 leads on: scale (940,992 replies, 7 models, 3 families); four numbered findings with figures; a video.
  - We lead on: a live demo (01 has none); agent-level effects well above the floor (01's agent benchmarks are near
    zero).
- **05 (integrity and impact).**
  - 05 leads on: real-world exposure evidence (418 Hub repos, 64,619 downloads); the hypothesis count (11 held,
    11 failed); finishing the organisers' review on Nov 27.
  - We lead on: a system, a live model, and a reusable benchmark.

## Where ActionProof is behind, in order of judge impact

1. **Presentation (P = 2).** No recorded video. The deck is not yet shared or exported. No short paper of the main
   result, while 02, 03, 04, 05 and 12 each ship one.
2. **Operational value (O = 2).** The script beats the agent, and no hybrid has been measured.
3. **Compliance with the organisers' citation review (new).** Three of the top five cite the kit and model arXiv
   papers; we cite none.
4. **Safe hand-off.** 02's whole result is safe stopping; ours is partial.
