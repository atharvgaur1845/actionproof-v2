# Stage 2 video script (draft; target 6 to 7 minutes)

Narrator: the user. Driver: Claude (commands in `docs/stage2/COMMANDS.md`, written at M6). Every number on screen resolves through `claims/claims.yaml`. Anything marked PENDING is filled only from a result file; if the file does not exist at the freeze (Tue Nov 24 12:00 IST), the segment says "not finished" instead.

Each segment shows a small label: **LIVE** (run during recording) or **RECORDED RESULT** (from a file, path on screen).

## 0:00 to 0:45: the problem (RECORDED RESULT)
- An AI agent tells you "done". Did the money actually move?
- Stage 1 (done before Stage 2, Nov 1 to 15), one sentence each:
  - Reward that reads the database: verified completion 0.8425 vs 0.640 for a reward that trusts tool replies (400 tasks, 2 seeds). `results/confirmatory.json`
  - LLM judges given the evidence still accept 72% of the false completions it exposes. `results/judge/grounded_audit.md`
  - But what the agent learned was a fragile script. `results/a10/v2_env.md`

## 0:45 to 1:30: what ActionProof is now (LIVE dashboard, empty state, then architecture slide)
- Research question: does what the reward can see decide whether an agent checks its own work, in a second environment?
- Product: a gateway that verifies every write against the system that owns it. The agent can act but cannot declare success.

## 1:30 to 3:00: the hero run (LIVE)
- Procurement task, five services. Inject a phantom payment (the payment service says "ok", stores nothing).
- Compare modes table: Passthrough finalizes a false success; Observe catches it; Enforce retries safely and completes. Every row is a real run.
- Three lanes: agent said / gateway did / system of record shows; the contradicted reply struck through.

## 3:00 to 3:45: the hard cases (LIVE)
- Never-commit payment: escalation with the money at risk.
- Duplicate delivery: the extra payment refunded, the refund itself verified.
- Edit one ledger event: "Chain broken at event N".

## 3:45 to 5:30: the research results (RECORDED RESULT)
- C1 reward observability, per seed: phantom read-back R_V vs R_A. PENDING `results/s2/a12_summary.json`.
- Mechanism predicted before training: under the acknowledgement reward, reading is penalised exactly where acting on what you read costs a success reply (lost ack -0.72, never commits -0.84, step 0). `results/a12b/h3_predictions.json`; observed: PENDING.
- C2 contract coverage: no error on 45,601 violations of the faults the contract was written for; 8.7% of independent adversarial violations slip through, mostly around refunds. `results/a13/h5_report.md`
- Defences on identical fault traces: idempotency keys alone recover 0% of phantom writes; verify-before-retry 100%. `results/a11/h7_matrix.md`
- C3 policy vs runtime vs information: PENDING `results/s2/a11_summary.json`.
- Track 3 metrics: task success, tool-call validity, latency, judge-human agreement. PENDING.

## 5:30 to 6:30: what it guarantees and what it does not (slide from `docs/stage2/PRODUCT.md`)

## 6:30 to 7:00: reproducibility and close
- Four pre-registrations this stage (A11, A12 + addendum, A13), each hashed before its runs; independent verifier and golden traces; every number traced.
