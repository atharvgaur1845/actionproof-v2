# ActionProof: verifiable execution infrastructure for agentic workflows

ActionProof is an execution verification and recovery layer that sits between an AI agent and the systems it changes. The agent can act, but it cannot declare success: a transaction is finalized only when the gateway has checked every requested effect in the system that owns it.

It is not a distributed transaction manager. It does not make several services commit atomically. It checks after each write, controls what may happen next, recovers what can be recovered, and escalates the rest with the money at risk stated.

## Who it is for

A platform engineer who puts an LLM agent in front of purchase-to-pay systems (order management, a supplier portal, payments, inventory, accounting) and must be able to answer: did the money move exactly as approved, and can we prove it?

## How it works

1. **Contract.** For each task the operator binds a typed contract: the purchase order, the steps the agent may perform, the exact amount and quantity, the registered payee, money limits and how many write attempts are allowed. Anything not granted is refused (fail-closed). The contract has a content hash and cannot change once bound.
2. **Gateway.** Every tool call goes through the gateway. Only the gateway holds the service credentials; in the demo it runs as its own process and the agent talks to it over HTTP.
3. **Verification.** After every write the gateway reads the authoritative record of the service that owns it, through its own read path. A tool's reply is logged as a claim and never treated as proof. Evidence counts only if its version is at or after the version seen when the write was sent.
4. **Recovery.** A decision table chooses what to do:

| What the evidence shows | Action |
|---|---|
| Success reply, nothing stored | Retry with the same idempotency key (safe: absence was verified after dispatch) |
| Timeout, but it was stored | Do nothing; refuse the agent's own retry (prevents paying twice) |
| Timeout, nothing stored | Retry with the same key |
| Half the payment stored | Refund it (a new operation, itself checked), then pay the full amount |
| Paid twice | Refund the extra payment |
| Still not stored after the allowed attempts | Escalate to a human, with the money at risk |
| A step that has no safe repair | Escalate |

   Retry-safe and idempotent are different properties: a goods receipt has no key, so it is retried only after its absence was verified.
5. **Commit points.** A later step may not run until the earlier step it depends on is verified (no payment before the supplier order is confirmed; no journal entry before the payment is confirmed).
6. **Evidence ledger.** Every decision and every piece of evidence is appended to a SHA-256 hash chain. Heads can be anchored outside the store.
7. **Modes.** Observe (verify and record only), Enforce (also refuse, recover and escalate), Enforce and inform (also show the agent the verified result instead of the tool's reply).

## What it guarantees, and under which assumptions

- **No false finalization.** The gateway never reports success unless fresh evidence shows every requested effect. Tested on every scripted episode and on random fault plans.
- **At most one payment it caused**, per purchase order and transaction, if the payments provider scopes idempotency keys to (operation, entity), keeps them for the whole transaction, and no other call with the same key is in flight. When a key could have expired, the gateway escalates instead of retrying.
- **Tamper detection** of the evidence ledger: any change, deletion, insertion or reordering of stored events is detected relative to a trusted head; with external anchors, truncation after the last anchor is detected too.

## What it does not guarantee

- No exactly-once execution. When a write may still commit later with no bound, verification alone cannot be exactly-once (LIMBO, Proposition 1); the gateway escalates.
- No atomicity across services. Compensation (for example a refund) is a new operation that can fail; a failed compensation leaves the transaction unresolved, with its residual exposure, and is never shown as success.
- It cannot detect a system of record that lies about itself unless another service corroborates; then the verdict is ambiguous and the case is escalated.
- It is only as complete as its contract. On 161 independently written violations of the plain business rules, the full contract accepted 14 (8 of the 151 unambiguous ones), mostly around refunds and master data (`results/a13/h5_finite.json`). This describes one author's test set, not a rate for real violations.
- The in-process research harness used for training is not a security boundary; the separate gateway process is the demonstrated boundary.
- No real payments, real supplier APIs, signatures or multi-tenant operation.

## Try it

```
pip install -e ".[integrity]"
actionproof serve --port 8765          # dashboard at http://127.0.0.1:8765/
actionproof compare --task proc_dev-0001 --agent naive --fault pay_supplier:0=phantom
actionproof contract show --task proc_dev-0001
```

The scripted `naive` agent trusts every tool reply; `oracle` checks every write. A model agent can be driven through any OpenAI-compatible endpoint (`AP_LLM_URL`, see `actionproof/integrity/llm_agent.py`).
