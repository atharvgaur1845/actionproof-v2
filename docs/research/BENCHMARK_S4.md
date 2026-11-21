# ActionProof benchmark: ProcureSim splits and Stage 4 shifts

## Environment

ProcureSim (`actionproof/integrity/sim/`; spec `docs/stage2/VERIFIER_SPEC.md`) is a procurement back office.
- **Tools:** five read tools, six write tools and a report.
- **Faults:** applied at one write chokepoint. Every fault's acknowledgement is byte-identical to a real outcome's:

| Fault | What is stored | Acknowledgement |
|---|---|---|
| phantom | nothing | ok |
| partial (ratio q) | q of the requested effect | ok |
| lost acknowledgement | the full effect | timeout |
| timeout before commit | nothing | timeout |
| duplicate delivery | the effect, twice | ok |
| never commits | nothing, on every attempt | ok every time |
| replica lag | the effect (reads are stale) | ok |

- **Verifier.** A state-only verifier (`truth.py`) is checked against an independently written reference verifier and 91 golden traces (`tests/golden/`).
- **Stage 3 env** (`actionproof/stage3/env.py`) adds a partial ratio per fault and is ProcEnv exactly without one (tested).

## Splits

| Family | Splits | Seeds | Purpose |
|---|---|---|---|
| A12 (registered) | `proc_train_b`, `proc_dev`, `proc_val`, `proc_conf`, `proc_combo`, `proc_stress` | 101–109 | Stage 2 training and confirmatory evaluation |
| A14/A15 forks | `proc_cf`, `proc_mech`, `proc_mech_dev` | 110–112 | forced-evidence decisions at q = 0, ¼, ½, ¾, 1 |
| Stage 3 | `s3_rec_train`, `s3_rec_dev`, `s3_conf_rec`, `s3_conf_full`, `s3_conf_stress`, `s3_matched_full_half` | 201–213 | recovery starts; fresh confirmatory splits |
| Stage 4 shifts (`PREREGISTRATION_S4_EVAL.md`) | `s4_lang_{en,hi,hinglish,para}`, `s4_api`, `s4_wf4`, `s4_combo` | 301–304 | language, paraphrase, API surface, unseen workflow, fault combinations |
| OpsSim (A17, S3) | `it_ops`, `care_admin`, `*_cf`, `s3_it_ops`, `s3_care_admin` | 121–124, 231–232 | separately implemented domains (same codebase, different tools and verifiers) |

Every split's PO ids and task ids are disjoint from all others (asserted at generation). The exception is the four `s4_lang_*` splits, which share worlds by design so that language effects are paired. Hash manifests: `results/*_hashes.json`.

## Shift construction

- **Language.**
  - The user request is rewritten in Hindi, in Hinglish, or as an English paraphrase (`actionproof/stage4/shift.py:prompt_variant`). The system policy stays English.
  - Every request field (PO id, quantity, amount, SKU) appears verbatim (tested).
- **API surface.**
  - `ApiShiftEnv` renames all twelve tools and their arguments, for example `pay_supplier(po_id, amount_cents, idempotency_key)` becomes `remit_payment(purchase_order, amount_minor, request_id)`.
  - Acknowledgements change from `{"status": "ok"}` to `{"result": "ACCEPTED"}`, and a timeout becomes `deadline_exceeded`.
  - Canonical names are rejected.
  - The trace keeps the canonical names, so the frozen verifier scores it unchanged. The same actions give the same final state and score (tested on 60 tasks).
- **Unseen workflow.** A four-step chain (order, receive, pay, post) that no training split contains.
- **Fault combinations.** Two faults on two different steps of one episode.

## Scripted calibration (VSC)

| Split | Oracle | Never-read |
|---|---|---|
| `s4_lang_en` | 0.887 | 0.333 |
| `s4_api` | 0.807 | 0.30 |
| `s4_wf4` | 0.91 | 0.34 |
| `s4_combo` | 1.00 | 0.21 |

The oracle's VSC is below 1 exactly where tasks are unreachable by design (never-commit).

## Not covered

- Stale reads as a decision class (replica lag appears only in stress splits).
- Conflicting views in ProcureSim (OpsSim has them).
- Idempotency-key scope errors.
- Compensating-action failure.
- Delayed commits.
- Multi-entity workflows.

These are future work and are not claimed.
