---
title: ActionProof
emoji: 🧾
colorFrom: indigo
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
license: cc-by-nc-4.0
---

# ActionProof demo

Agents say "done". Is it? Pick a task and an injected tool fault (phantom acknowledgement,
stale acknowledgement, partial commit, timeout-after-commit, prompt injection, or a policy that
forbids the refund). Two Tiny Aya agents then handle the same episode in identical environments.
Each write shows what the tool *claimed* next to what the database *actually* holds, and every
episode is scored under a transcript-trusting reward, a database outcome reward, and PCSTR.

Base model: CohereLabs/tiny-aya-global (CC-BY-NC). Adapters: see the ActionProof model collection.
Requires the `HF_TOKEN` secret (gated base model) and `AP_ADAPTERS` (JSON map of label → adapter repo).
