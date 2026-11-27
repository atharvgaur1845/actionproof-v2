---
title: ActionProof Live
colorFrom: blue
colorTo: blue
sdk: gradio
sdk_version: 4.44.1
python_version: "3.10"
app_file: app.py
license: cc-by-nc-4.0
short_description: A Cohere Labs agent live in simulated procurement
---

# ActionProof live

Run a real Cohere Labs model as a tool-using agent in ProcureSim, ActionProof's simulated procurement environment,
and see what the systems of record actually hold after it says "done".

Deterministic replays of recorded runs: https://huggingface.co/spaces/atharvgaur18/actionproof

## What you do

1. Pick one of 6 curated tasks (each includes a supplier payment).
2. Pick a reality. The four realities all give the agent the same request; they differ only in what the first
   `pay_supplier` call does behind its reply. Which one you picked is not shown until the run ends:
   - Reality A: lost acknowledgement (payment committed, reply is a timeout)
   - Reality B: committed (payment committed, reply ok)
   - Reality C: phantom (reply ok, nothing committed)
   - Reality D: partial (reply ok, half the amount committed, ratio 0.5)
3. Pick an agent and press Run. The timeline streams each tool call and its raw reply; an "ok" acknowledgement is
   highlighted.
4. After the run: the hidden database rows for the purchase order, the verifier verdict per step, VSC and false
   success, and a transaction decision record (intent and postconditions, acknowledgements seen, evidence read,
   disposition, recovery actions, verification result).

## Live versus scripted

| Option | What runs | Needs GPU |
|---|---|---|
| Base model | `CohereLabs/tiny-aya-global`, live, greedy, max 16 turns, 256 new tokens per turn | yes (ZeroGPU) |
| Trained agent | `atharvgaur18/actionproof-agent-d` (merged), same settings | yes (ZeroGPU) |
| Compare | base and trained, one after the other, same task and reality, two columns | yes |
| SCRIPTED: acknowledgement-trusting | a fixed program: write, trust "ok", report completed | no |
| SCRIPTED: transaction-aware oracle | a fixed program: write, read the authoritative record, repair or retry, report honestly | no |
| **Judge mode** (default) | left: SCRIPTED acknowledgement-trusting; right: the trained agent, live; same task and reality | yes (one run) |
| RECORDED: trained agent / base model | a real run of that model recorded earlier through this Space's own episode loop (`record_runs.py`), replayed | no |

Every model run carries a banner `LIVE MODEL RUN · <model id> · <UTC timestamp> · greedy, max 16 turns`. Scripted runs
are labelled SCRIPTED. Recorded runs carry `RECORDED MODEL RUN · <model id> · recorded <UTC> on <hardware> · greedy · no
model is called now`.

**If a live run fails** (ZeroGPU quota, queue timeout, model download), the banner says `LIVE RUN FAILED (<reason>)` and
shows the recorded run of the same model, task and reality; a replay is never shown as a live run. The recordings were
made from `runs/p1_d_s0/merged`, whose `model.safetensors` sha256 equals that of the Hub revision `5418df5`
(checked in `slurm/tasks/1790_space_record.sh`).

**Export.** Every finished run can be downloaded as JSON: run id, agent option, model id and revision, decoding, task
id, reality, every call and raw reply, the report, the verifier's score and the decision record. The scripted policies are the repo's own reference policies; they never see the reality.

Live runs are not replays: they are one greedy run of the model on this task and reality, scored by the same
deterministic verifier used in the study. One run says nothing about a rate. Rates come from the full frozen
evaluation (see the replay site and the repository), not from this demo.

## Limits and safety

- Only simulated tools exist. Nothing touches a real system, and no data leaves the Space.
- Hidden state (the reality, the database rows, committed flags) is computed but not rendered until the run ends.
- One run at a time (queue); at most 10 live model runs per session (compare counts as 2). Scripted runs are free.
- The "transaction decision record" describes what one run did and what the verifier found. It is not a proof or a
  certificate.

## Code

- `app.py`: Gradio UI and the transformers generator (`@spaces.GPU(duration=120)`).
- `core.py`: realities, streaming, reveal, decision record, session cap; no model imports, unit-tested.
- `actionproof_min/`: minimal vendored copy of the environment, verifier, scripted policies, prompts and parser.
  Provenance in `VENDORED.md`.
- `data/tasks.json`: the 6 curated tasks (selection rule in `VENDORED.md`).
- `tests/test_app_logic.py`: `pytest space_live/tests` from the repository root.
