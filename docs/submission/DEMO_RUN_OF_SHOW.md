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
3. Keep the base model's `HF_TOKEN` Space secret set (it is gated); the trained agent is public.

## Timeline (about 2:45; slides in brackets)

| Time | Screen | Label | Action | Say |
|---|---|---|---|---|
| 0:00–0:20 | Deck slide 1 (hook) | — | Show the four realities | "The payment tool says ok. Did the payment happen? Same request, same reply, different realities." |
| 0:20–0:45 | Live Space, judge mode, Task 0004, Reality C | `SCRIPTED` (left) | Press Run; the left column streams | "A program that trusts the reply: write, see ok, report done." |
| 0:45–1:15 | Same run, right column | `LIVE MODEL RUN · atharvgaur18/actionproof-agent-d` | The trained agent reads the payment back; then the reveal | "Same task, same ok. The trained agent reads the record, finds no payment, and pays again before it reports. Below, the database: one payment for the trained agent, none for the script." |
| 1:15–1:30 | The reveal: verifier verdict and decision record; click the JSON download | — | Point at the verdict and the run id | "Every number comes from the database, never the transcript. Each run can be downloaded." |
| 1:30–1:50 | Deck slide 2, then 4 | `RECORDED RESULT` (deck) | Baselines; P1 bars; TXN H3 | "On 440 locked tasks, trusting the reply falsely completes 0.864. Recovery demonstrations then GRPO: 0.513 to 0.819 on 450 frozen tasks. Fewer duplicates than bounded retry; the retry test fails, and we say so." |
| 1:50–2:15 | Deck slide 5 | `RECORDED RESULT` | What fails; the SEL1 box | "It rarely hands off when the evidence cannot settle it." [Then SEL1's registered result, read from `results/sel1/sel1_summary.json`.] |
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

## Browser checklist (staging, then live; results in `docs/submission/REPRODUCIBILITY_FINAL.md`)

Each item is checked on staging first, then on the live Space:

- [ ] Cold start: the time to the first token in judge mode.
- [ ] Judge mode: left SCRIPTED, right LIVE; the reveal shows two decision records; the JSON downloads.
- [ ] Scripted agents run with no GPU.
- [ ] RECORDED options are labelled and need no GPU (after the recording task 1790 is deployed).
- [ ] The session cap message appears after 10 live runs.
- [ ] A forced failure falls back with the `LIVE RUN FAILED` banner (covered by unit test
      `test_live_failure_falls_back_to_labelled_recording`).
- [ ] At 1280 px and on mobile, the columns stack under 760 px.
