"""Self-healing loop (docs/selfheal/DESIGN.md): diagnose failed episodes from verifier state, generate corrective
continuations with a pluggable teacher, keep only those the simulator replay and the verifier accept, curate them
into SFT/DPO data, and decide deployment with a pre-declared gate on a fresh split.

Modules:
- `envs`: thin environment interface (P1 ProcureSim, TXN v1; TXN v2 and attack suites register here later).
- `taxonomy`: failure classes computed from the verifier record, never from text.
- `replay`: deterministic replay of a recorded conversation and turn-level prefix cuts.
- `teachers`: the teacher protocol, the B4 reconciliation teacher and an LLM teacher (OpenAI-compatible endpoint).
- `correct`: cut, continue, replay, verify; one correction record per accepted episode.
- `curate`: CuratorKIT deduplication, failure-class balance, hold-out by task id, SFT rows and DPO pairs.
- `ledger`: which task ids each loop round used, so no round evaluates on its own correction data.
- `gate`: candidate vs incumbent on a fresh split under pre-declared criteria; writes a promote/reject file.
"""
