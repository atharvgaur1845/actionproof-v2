# Stage 2 video script (draft; target 6 to 7 minutes)

Narrator: the user. Driver: Claude (commands in `docs/stage2/COMMANDS.md`, written at M6). Every number on screen resolves through `claims/claims.yaml`. Anything marked PENDING is filled only from a result file; if the file does not exist at the freeze (Tue Nov 24 12:00 IST), the segment says "not finished" instead.

Each segment shows a small label: **LIVE** (run during recording) or **RECORDED RESULT** (from a file, path on screen).

## 0:00 to 0:45: the problem (RECORDED RESULT)
- An AI agent tells you "done". Did the money actually move?
- Stage 1 (done before Stage 2, Nov 1 to 15), one sentence each:
  - Registered test: a reward that reads the database beat the best baseline by 1.5 points (0.8425 vs 0.8275, 400 tasks); a reward that trusts tool replies scored 0.640, a larger gap that was not the registered comparison. `results/confirmatory.json`
  - An LLM judge (Aya 32B, AuditKIT prompt) given the evidence still accepts 71% of the false completions it exposes. `results/judge/grounded_audit.md`
  - But what the agent learned was a fragile script. `results/a10/v2_env.md`

## 0:45 to 1:30: what ActionProof is now (LIVE dashboard, empty state, then architecture slide)
- Research question: does what the reward can see decide whether an agent checks its own work, and once the evidence is in front of it, does the agent act on it?
- Product: a gateway that verifies every write against the system that owns it. The agent can act but cannot declare success.

## 1:30 to 3:00: the hero run (LIVE)
- Procurement task, five services. Inject a phantom payment (the payment service says "ok", stores nothing).
- Compare modes table: Passthrough finalizes a false success; Observe catches it; Enforce retries safely and completes. Every row is a real run.
- Three lanes: agent said / gateway did / system of record shows; the contradicted reply struck through.

## 3:00 to 3:45: the hard cases (LIVE)
- Never-commit payment: two safe retries with the same key, then escalation; nothing is paid.
- Duplicate delivery: the extra payment refunded, the refund itself verified.
- Edit one ledger event: "Chain broken at event #N".

## 3:45 to 5:30: the research results (RECORDED RESULT)
- **What the reward sees decides what the agent checks.** Same warm start, same tasks, two rewards of the same form: one reads the database, one reads only the replies the agent got. On untouched tasks, after a payment that said "ok" but stored nothing, the database-reward agent read the records 1.00 of the time, the reply-reward agent 0.00, in every seed (153 of 153 tasks vs none). `results/s2/a12_summary.json`
- **Selection, not invention.** The warm start already read back when sampled; the database reward made it the default. The reply-reward agent ended exactly where it started: the same tool calls as the warm start on every untouched task. Under the reply reward, reading was punished in exactly the cases where acting on what you read costs an "ok" reply (lost ack -0.72, never commits -0.84, predicted before training). `results/a12b/h3_predictions.json`; the registered per-class test of this mechanism failed, and we show it as failed. `results/s2/h3_test.json`
- **Once the evidence is in front of it, the agent acts on it, up to a point.** We forced the read into the context of every model: all of them, even the one that never reads, re-pay when the record is empty and move on when it is there (correct-action accuracy 1.00). But when the record shows a payment for the wrong amount, only the database-reward agent refunds, in 0.35 to 0.48 of tasks depending on the seed; on duplicate payments both leave the extra one in place (0.58). `results/a14/a14_summary.json`, `results/s2/a12_secondary.json`
- **So put the evidence in front of the agent.** Showing the warm start the verified state through the gateway adds +0.25 verified completion. `results/s2/a11_summary.json`
- Contract coverage: the contract accepts none of the 194 single-fault and 479 two-fault violating configurations it was written for (an exhaustive enumeration, not a sample), and 14 of 161 independently written violations (8 of 151 unambiguous), mostly refunds and master data. `results/a13/h5_finite.json`
- Defences on identical fault traces: idempotency keys alone recover 0% of phantom writes; verify-before-retry 100% (deterministic; true by construction). `results/a11/h7_matrix.md`
- **Then teach the repair.** Registered P1, 450 fresh tasks, three seeds per arm. Recovery demonstrations followed by the same RL lift verified completion from 0.513 to 0.819 (+0.306), and partial payments from 0.319 to 0.931. A hint added at run time does not help (−0.030). `results/p1/p1_summary.json`
- **What did not work (P2).** A structured ledger at run time lowers completion to 0.253. A runtime guard removes false successes, but it blocks the taught refund-then-re-pay repair, so it lowers every agent. The hand-written checker still leads (0.924). `results/p2/p2_summary.json`
- **The limit.** On a workflow it has never seen, every agent skips the new step and says "done" (best agent 0.486).
- Track 3 metrics: task success, tool-call validity, latency, judge-human agreement. PENDING.

## 5:30 to 6:30: what it guarantees and what it does not (slide from `docs/stage2/PRODUCT.md`)

## 6:30 to 7:00: reproducibility and close
- Five pre-registrations this stage (A11, A12 with two addenda, A13, A14), each hashed before its runs; failed tests and deviations shown as such; independent verifier and golden traces; every number traced.
