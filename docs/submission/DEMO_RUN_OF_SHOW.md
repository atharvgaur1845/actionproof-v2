# Demo run of show (final round)

**Surfaces.** This file replaces `docs/final/DEMO_RUN_OF_SHOW.md`, which described an earlier static-only demo. Three
surfaces exist. They are never conflated, and each one carries its label on screen:

| Surface | URL | What it is | Label on screen |
|---|---|---|---|
| Live Space | https://huggingface.co/spaces/atharvgaur18/actionproof-live | A real Cohere model per run (ZeroGPU), curated tasks, hidden reality revealed after the run | `LIVE MODEL RUN`, `SCRIPTED`, `RECORDED MODEL RUN` |
| Replay Space | https://huggingface.co/spaces/atharvgaur18/actionproof | The full registered study: every result and recorded episode; no model is called | "recorded episodes" |
| Deck | https://claude.ai/artifact/N5NxS6YHhbaGqGSUMqeYbW (export PDF or PPTX) | Six core slides and an appendix, mirrored by `docs/submission/SLIDES.md` | — |

**Before recording (author).**
1. Open the live Space and press Run once in judge mode. This wakes the GPU and loads the model, which can take about
   a minute the first time.
2. If ZeroGPU refuses (quota or queue), the Space falls back to the recorded run of the same model, task and reality,
   under a `LIVE RUN FAILED … RECORDED MODEL RUN` banner. Record that honestly with its label, or retry later.
   Never edit the label out.
3. **Set the base model's `HF_TOKEN` Space secret (author action, still open).** The base model is gated. On
   2026-11-27 the live Space had no token: a live base run failed with `GatedRepoError: 401` and fell back, labelled,
   to the recorded base run. Judge mode and the trained agent are unaffected (the trained agent is public).

## Timeline (about 2:45; slides in brackets)

| Time | Screen | Label | Action | Say |
|---|---|---|---|---|
| 0:00–0:20 | Deck slide 1 (hook) | — | Show the four realities | "The payment tool says ok. Did the payment happen? Same request, same reply, different realities." |
| 0:20–0:45 | Live Space, judge mode, Task 0004, Reality C | `SCRIPTED` (left) | Press Run; the left column streams | "A program that trusts the reply: write, see ok, report done." |
| 0:45–1:15 | Same run, right column | `LIVE MODEL RUN · atharvgaur18/actionproof-agent-d` | The trained agent reads the payment back; then the reveal | "Same task, same ok. The trained agent reads the record, finds no payment, and pays again before it reports. Below, the database: one payment for the trained agent, none for the script." |
| 1:15–1:30 | The reveal: verifier verdict and decision record; click the JSON download | — | Point at the verdict and the run id | "Every number comes from the database, never the transcript. Each run can be downloaded." |
| 1:30–1:50 | Deck slide 2, then 4 | `RECORDED RESULT` (deck) | Baselines; P1 bars; TXN H3 | "On 440 locked tasks, trusting the reply falsely completes 0.864. Recovery demonstrations then GRPO: 0.513 to 0.819 on 450 frozen tasks. Fewer duplicates than bounded retry; the retry test fails, and we say so." |
| 1:50–2:15 | Deck slide 5 | `RECORDED RESULT` | What fails; the SEL1 box | "It rarely handed off when the evidence could not settle it: 15 of 540. Registered before training, demonstrations of selective hand-off: 0.972 against 0.002 on the locked test, never on a transient outage, no cost on resolvable work. Fragile: on a held-out step one seed barely escalates." |
| 2:15–2:45 | Deck slide 6 | `RECORDED RESULT` (projection) | Impact bars; links | "Training removes most of the projected exposure; hand-written checks remove the rest, so ship both. Every number has one file and one command." |

## Fallback paths

- **No GPU at all.**
  - Agent "RECORDED: trained agent (no GPU)" replays the recorded run, with its banner.
  - Agent "SCRIPTED: transaction-aware oracle" shows the reference behaviour.
  - Say "recorded" and keep the banner visible.
- **The Space does not load.** Use the replay Space's TXN section and its episode explorer.
- **What not to say.**
  - Never present one live run as a rate.
  - Never call a recorded run "live".
  - Never call the scripted oracle "the model".

## Checklist (staging, then live; 2026-11-27)

Checked through the Space's API with `gradio_client` from the cluster, staging first (commit 61b1c77), then live
(cf90ad3; the previous live commit, 3bb0dc8, is the rollback). A human check in a browser is still needed for the
layout items.

- [x] Judge mode, Task 0004, Reality C: left `SCRIPTED: acknowledgement-trusting` (VSC 0, false success), right
      `LIVE MODEL RUN · atharvgaur18/actionproof-agent-d` (VSC 1). The JSON export lists both sides, with run id,
      model and decoding. End to end: 69 s on staging, 72 s live, GPU already warm.
- [x] Scripted agents need no GPU: oracle VSC 1 in 6 s live; acknowledgement-trusting VSC 0.
- [x] RECORDED options (task 1790, 48 runs, sha256 `bd88218a…`): `RECORDED MODEL RUN` banner with the recording
      time, hardware and decoding; no GPU; 9–15 s; trained VSC 1, base VSC 0 on Task 0004, Reality C.
- [x] A real live failure falls back with the `LIVE RUN FAILED … RECORDED MODEL RUN` banner (the live base run,
      missing `HF_TOKEN`, above). Also covered by unit test `test_live_failure_falls_back_to_labelled_recording`.
- [x] The session cap: unit test `test_session_cap_and_free_scripted_runs` (not exercised live, to save GPU quota).
- [ ] Cold start from a sleeping Space: not measured (the Space was awake). Press Run once before recording.
- [ ] Layout at 1280 px and on mobile (columns stack under 760 px, CSS `@media (max-width: 760px)`): needs a browser;
      the headless browser in this environment does not start.
