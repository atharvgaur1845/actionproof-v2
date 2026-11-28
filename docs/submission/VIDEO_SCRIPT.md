# ActionProof — earlier video script (about 2:45; superseded)

> **Superseded on 2026-11-29.** The submitted video (3:18, https://github.com/atharvgaur1845/actionproof-v2/releases/tag/video-v4)
> follows `docs/submission/video/script.py`: a real screen-recorded live run on the L4 Space, then the research result.
> This file is kept for the record.

Narrator: the author. Screens: the live Space in judge mode, the deck (`docs/submission/ActionProof_deck.pdf`,
mirrored by `docs/submission/SLIDES.md`), and result files. The run of show, with fallbacks, is
`docs/submission/DEMO_RUN_OF_SHOW.md`. Every number spoken or on screen resolves through `claims/claims.yaml`.

**Labels.** Each segment keeps its label visible:
- `SCRIPTED`: a program, no model;
- `LIVE MODEL RUN`: a real model call during recording;
- `RECORDED MODEL RUN`: a stored real run, replayed;
- `RECORDED RESULT`: a number from a result file, with its path on screen.

The live Space's realities (lost acknowledgement, committed, phantom, partial) are not the TXN quadruples on slide 1.
Do not call them the same thing.

## 0:00–0:20 — The same "ok", different realities (deck slide 1)
- Screen: the four realities.
- Say: "The payment tool says ok. Did the payment happen? Same request, two replies, four realities."

## 0:20–1:15 — Judge mode: a script against the trained agent (live Space, Task 0004, Reality C)
- Press Run. Left column, `SCRIPTED: acknowledgement-trusting`: write, see ok, report done.
  Say: "A program that trusts the reply."
- Right column, `LIVE MODEL RUN · atharvgaur18/actionproof-agent-d`: it pays, reads the payment back, finds none, and
  pays again with the same idempotency key before it reports.
  Say: "Same task, same ok. The trained agent reads the record, finds no payment, and pays before it reports."
- The run takes about 70 s. Cut the wait in editing, and show an on-screen note "shortened: about 70 s live" while
  the banners stay visible.
- The reveal: one payment for the trained agent, none for the script. Click the JSON download.
  Say: "Every verdict is read from the database, not from the reply. Each run can be downloaded."
- If the live run fails, the Space shows the recorded run under `LIVE RUN FAILED … RECORDED MODEL RUN`. Keep that
  banner on screen and say "recorded".

## 1:15–1:45 — What training changes (`RECORDED RESULT`, deck slides 2 and 4)
- Say: "On 440 locked tasks, trusting the reply falsely completes 0.864 of the time. Recovery demonstrations then
  GRPO take the agent from 0.513 to 0.819 on 450 frozen tasks, three seeds. On the transaction test it falsely
  completes 0.121 and leaves far fewer duplicate payments than bounded retry. The retry test fails, and we say so."

## 1:45–2:15 — What fails, and what our own review found (`RECORDED RESULT`, deck slide 5)
- Say: "The hand-written reconciliation script still beats every learned agent. Our own hostile review then found
  the benchmark weaker than designed: a policy that never reads, but sends each payment twice with the same key,
  gets close on completion, though not on false completion. We registered one experiment before training: teach
  hand-off. It passed: on the locked test the agent hands off in 0.972 of unresolvable cases, against 0.002 for a
  control, at no cost elsewhere. But a same-key resend could have settled those cases. Where hand-off is the only
  safe action, it happens in about four of ten episodes, and one seed barely does it. We report all of it."

## 2:15–2:45 — What it is worth, and where it lives (`RECORDED RESULT`, projection; deck slide 6)
- Screen: the impact bars, then the links.
- Say: "Projected over a thousand faulted workflows, trusting the reply costs about ten times the trained agent's
  exposure, and the hand-written script cuts it further still. This is a simulator: the dollars are a projection,
  not savings. Every number has one file and one command. It is transaction-aware recovery under ambiguous
  execution, not a new RL optimizer and not a claim of safety."

## Recording checklist
- Press Run once before recording, to wake the GPU (the first load can take about a minute).
- Record the judge-mode segment first. If ZeroGPU refuses, record the labelled fallback or retry later.
- Never edit a label out. Never present one live run as a rate.
- Captions: `docs/submission/VIDEO_CAPTIONS.srt`, timed to the final cut.
