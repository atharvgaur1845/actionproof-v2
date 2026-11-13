---
title: ActionProof
emoji: 🧾
colorFrom: indigo
colorTo: green
sdk: gradio
python_version: "3.12"
app_file: app.py
pinned: false
license: cc-by-nc-4.0
short_description: Tool agents that check whether their writes landed
---

# ActionProof demo

Tools sometimes acknowledge writes that never committed. ActionProof is a SQLite customer-support simulator in which a
phantom write returns a success acknowledgement byte-identical to a real one. A deterministic verifier scores each
episode from the database, not from the transcript.

Tabs:

- **Replays (real eval episodes).** Recorded evaluation episodes, replayed deterministically against the simulator.
  Each write shows what the tool *claimed* next to what the database *actually* holds, and the episode is scored under
  every reward.
- **Same task, three templates.** One held-out store-credit task handled three ways: a void credit call, a refund
  substituted for the credit, a committed credit.
- **Leaderboard** and **Findings.** Headline numbers, each with its source file in the code repository.
- **Live.** Two agents run the same episode with a chosen injected fault. This tab needs a GPU and access to the
  gated base model (`HF_TOKEN` secret). Without them it says so; the Replays tab shows real episodes instead.

Base model: CohereLabs/tiny-aya-global (CC-BY-NC-4.0). Adapters and data: the ActionProof collections on the Hub.
Code: https://github.com/atharvgaur1845/actionproof-v2
