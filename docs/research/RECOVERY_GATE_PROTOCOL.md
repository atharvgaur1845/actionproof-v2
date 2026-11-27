# Recovery Gate: design specification (RG1)

Code: `actionproof/recovery_gate/gate.py` (the rules), `actionproof/recovery_gate/env.py` (how the gate is placed in
front of the TXN tools). Tests: `tests/recovery_gate/test_gate.py`. Registration: `PREREGISTRATION_RG1.md`.

## 1. What the gate is

The Recovery Gate is a small runtime controller between a tool-using agent and the tools of the TXN benchmark. The
agent proposes one call at a time. For each call, the gate does one of five things:
- **send** the call as proposed;
- **send it changed**: a read is sent with `consistent=true`;
- **read first**: before a repeated write it sends a read of the step's record, then decides on the write with the
  record in hand (send it, or block it with the reason);
- **block** the call: nothing is sent, and the agent gets a reason;
- **hand off**: it calls `escalate_to_human` with an evidence bundle, then reports `escalated`.

The gate is not a second agent. It does not plan a workflow, choose steps or fill in arguments. When the agent stops
without a report, the gate finishes the episode: completed if every requested step reads in effect, otherwise a hand-off.

## 2. What the gate may see

| The gate sees | The gate never sees |
|---|---|
| The task's intent, as the prompt states it: the PO id, the approved amount, the quantity, the requested steps | The fault plan, the regime, the reality of a quadruple, `escalation_correct`, any `meta.txn` label |
| Each call's name and arguments, as the agent sent them | The simulator's tables, idempotency store, pending late commits, replica lag state |
| Each call's reply, as the agent receives it | Trace metadata (`entry["txn"]`: staleness, primary version), attempt records, colleague writes |

This is enforced in three ways:
1. **No env reference.** The gate holds no reference to the environment. `GateMixin` gives it a callback that runs a
   call and returns the reply. A test walks the gate's object graph after an episode and finds no env object.
2. **No forbidden attributes.** An AST test fails if any gate function except `intent_of` accesses simulator or label
   attributes (`state`, `attempts`, `fault_plan`, `idem`, `trace`, `faults`, `regime`, ...). The gate module may
   import only `_observed` and the step–tool maps.
3. **Intent is in the prompt.** A test checks, for every task in txn_dev, txn_val, txn_test and rg1_test, that the
   prompt contains the PO id, `<qty> x` and the total amount.

A fourth test checks counterfactual blindness. The four tasks of a quadruple have identical prompts and different
hidden outcomes. Up to the first call whose reply differs, the gate's calls are identical in all four.

## 3. The rules

The gate keeps one view per step (order, pay, receive, post). A view holds:
- the write requests that reached a service, with their keys and replies;
- the latest read of the step's record since the last write or refund.

A request that was rejected before reaching a service (invalid arguments, unknown tool, not found) is not a write.

| Rule | Trigger | Action | Principle (prompt §4) |
|---|---|---|---|
| intent (always on) | A write names another PO, another amount, or a step the task does not ask for | Block | Intent binding |
| retry | A second write for a step, with no read of its record since the last write | Read first, then apply the rules below to the write | Acknowledgement non-authority; read-before-retry |
| retry | A write for a step whose current record shows it in effect | Block | Duplicate protection |
| retry | A retry after a reply other than ok that uses a different idempotency key | Block: resend with the same key, so a late commit is deduplicated | Duplicate protection (stable keys) |
| fresh | Any read | Send with `consistent=true` (the primary copy) | Evidence freshness |
| repair | A receipt whose quantity is not the missing quantity (approved quantity minus net received) | Block, stating the missing quantity | Partial-commit reasoning |
| repair | A payment while a live partial payment exists, or while two or more live payments exist | Block: refund first | Partial-commit reasoning; duplicate protection |
| repair | A refund that is not of a live partial or extra payment; or one that would reverse the only correct payment | Block (after a fresh read, if none is current) | Concurrency protection; no false compensation |
| repair | A replacement payment after a refund that reuses an acknowledged key | Block: the store would answer it and pay nothing | Duplicate protection |
| escalate | A write for a step that was written and whose record is unavailable | Block: do not resend; hand off | Safe abstention |
| escalate | A report other than `escalated` while a written step's record is unavailable; a `failed` report about an unread write whose record turns out to be unavailable | Hand off with the evidence bundle | Safe abstention; audit record |
| verify | A `completed` report | Read every requested step not confirmed since its last write. All in effect: send the report. Otherwise reject it, with the records, at most twice; then hand off. Unavailable: hand off | Post-action verification |
| escalate + verify | The agent stops without a report | Verify; complete if every step reads in effect, otherwise hand off | Post-action verification |

The thresholds are the only free choices: two rejected reports before a hand-off; reads always consistent. Neither
was tuned. The dev pilot (task 1900, txn_dev) changed two behaviours and added one rule; all three are listed in
`PREREGISTRATION_RG1.md` §6 with the evidence that motivated them.

## 4. Accounting

- Calls the gate makes are real agent-side calls. They reach the services, appear in the trace with `"gate": <tag>`,
  advance late commits, and count in the cost metrics (`calls`, `reads`).
- A blocked call reaches no service. It is absent from the trace, does not advance the fault schedule, and is
  recorded in the episode's `gate.log`.
- Each episode record carries `gate: {rules, log, rejections, gate_calls}`.

The agent's budget is unchanged: 16 model turns of at most 256 new tokens each, the same prompts, the same parser and
the same verifier as TXN1 (`scripts/txn_eval.py` → `scripts/p1_eval.py::run`). The gate's reads use no model turn.

## 5. What the gate cannot do

- **It cannot repair what the agent does not repair.** Wrapped around a scripted policy that ignores its replies
  (B0, B3), the gate removes false completions but lowers completion (dev pilot, §6 of the registration).
- **It encodes the benchmark's own invariants.** They are those of the scripted reference B4 and of the TXN
  specification. It does not discover them. A fault class outside this taxonomy would get no special handling.
- **It is no protection against a wrong intent.** It binds writes to the intent in the prompt; a wrong prompt
  produces a correctly enforced wrong transaction.
- **Its hand-off follows the benchmark's label.** In the unresolvable regime the specification defines escalation as
  correct. For keyed steps, a same-key resend would also be safe, because idempotency prevents a double charge (TXN1
  addendum 3). The gate hands off, as the specification asks.
- **It is evaluated in a simulator.** No claim is made about real payment systems.
