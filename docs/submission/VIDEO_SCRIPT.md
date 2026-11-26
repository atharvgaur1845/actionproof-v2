# ActionProof — final video script (about 2:45)

Narrator: the author. Screen: the live Space, the slides (`docs/submission/SLIDES.md`) and result files. Every
number on screen or spoken resolves through `claims/claims.yaml`. Each segment carries a small on-screen label:
**LIVE** (a real model call during recording) or **RECORDED RESULT** (from a file, path on screen).

## 0:00–0:20 — The same "ok", two realities (LIVE SIMULATOR, SCRIPTED AGENT: no model call)
- Screen: the live Space, task "pay post", Reality C (phantom), the scripted acknowledgement-trusting agent. The tool says ok.
- The live Space's realities are lost ack, committed, phantom and partial; they are not the TXN quadruples of slide 1.
- Say: "The payment tool says ok. Did the payment happen? Here, it did not."
- Reveal: the database shows no payment.

## 0:20–0:45 — Why trusting and retrying both fail (RECORDED RESULT, slide 2)
- Say: "An agent that trusts the reply falsely completes on every task where the ok hides a missing payment. One
  that retries without checking pays twice when the first payment did go through."
- Screen: the baseline table (B0 0.136, B3 0.491, B4 0.864 verified completion).

## 0:45–1:15 — ActionProof gathers evidence and acts (LIVE, live Space)
- Screen: same task, Reality C, "Trained agent (live)". It pays, reads the payment back, finds nothing, and pays
  again with the same idempotency key before reporting.
- Then Reality A (timeout, payment committed): it reads back, finds the payment, and does not pay again.
- Say: "Same model, same reply, different realities. It decides from the database, not from the reply."

## 1:15–1:45 — The registered results (RECORDED RESULT, slide 4)
- P1: recovery-start agent 0.513, recovery SFT 0.772, recovery SFT then GRPO 0.819; 450 frozen tasks, 3 seeds.
  The two recovery arms start from different checkpoints, so their gap is not GRPO's contribution.
- TXN H3: duplicate mutations 0.024 against 0.191 and 0.338 — passes. Unnecessary retries — fails, so the
  registered family is not supported. "We report it as registered."
- P4: randomised demonstrations lift the warm start, before any RL, on renamed tools from 0.137 to 0.741 and on an
  unseen workflow from 0.268 to 0.842. "The RL agent itself still fails there."
- P5: the same recipe raises four other Cohere bases (one SFT seed each).

## 1:45–2:15 — What did not work (RECORDED RESULT, slide 5)
- The agent rarely escalates when evidence cannot settle the outcome (15 of 540 cases).
- No-std GRPO did not help (0.813 vs 0.819). Healthcare transfer is negative. LLM judges are not verifier-grade.
- Two large open-weight models, prompted, complete none of the partial-payment tasks.

## 2:15–2:45 — Released (slide 6)
- Screen: the live Space URL, the model repos, the traces dataset, `bash scripts/verify_numbers.sh`.
- Say: "Every number has one file and one command. ActionProof is transaction-aware recovery under ambiguous
  execution — not a new RL optimizer, and not a claim of universal safety."

## Recording checklist
- Record the live segments first; if ZeroGPU quota fails, use the labelled recorded fallback.
- Keep the LIVE / RECORDED RESULT label visible in every segment.
- Captions: `docs/submission/VIDEO_CAPTIONS.srt`, timed to the final cut.
