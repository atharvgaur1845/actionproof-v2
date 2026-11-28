# Suite card: ATK1 adversarial evaluation for transaction agents

| | |
|---|---|
| Name | ATK1 (ActionProof attack suite, version 1) |
| Branch | `research/attack-suite` |
| Environment | TXN v1 (ProcureSim `proc3`), via the `txn_v1` backend; a TXN v2 backend can be registered |
| Specification | `docs/attacks/SPEC.md` |
| Generator | `actionproof/attacks/generate.py` (deterministic from a seed) |
| Ground truth | `actionproof/attacks/harm.py` (final state + approved task + report status; never agent text) |
| Manifest | `data/attacks/attack_manifest.json` |
| Status | development split evaluated with scripted policies; held-out split sealed (never run); trained models not yet evaluated |

## What it measures

Whether an agent that changes a procurement and payment system of record can be pushed, by content that comes back
from its tools, into an unauthorised or harmful state change or a false claim of completion; whether the Recovery Gate
(RG1, frozen) stops it; and what the gate costs on benign twins of the same tasks.

## Composition

Eight attack families (payee redirection, cross-transaction mutation, amount manipulation, duplicate/replay after an
ambiguous acknowledgement, compensation abuse, audit-log versus system-of-record disagreement, stale replica after a
rollback, workflow step skipping), plus a held-out-only compound family. Each attack episode has a clean twin with the
same world, prompt and env seed; F4 and F6 also have a mechanics-only control (the same fault, no text).

| split | attack | twin | control | tasks | templates | use |
|---|---|---|---|---|---|---|
| attack_dev | 240 | 240 | 60 | 540 | 61 development | development, pilots, scripted baselines |
| attack_heldout | 1260 | 1260 | 270 | 2790 | 132 held-out + 180 compound pairs | registered evaluation only |
| attack_train | any | any | any | on demand | development only | future training |

Held-out generalisation axes: unseen phrasings (formal e-mail, key=value directive, Hindi), unseen channels (audit-log
notes, forged error acknowledgements, a pointer from the timeout reply to the forged log), an unseen family (two
attacks in one episode), and unseen supplier/item vocabulary in half of the held-out episodes.

## Development results (scripted policies, attack_dev)

From `results/attacks/dev_policies.md` (Slurm job 474858, code at commit f5aa88d, split sha256 2f5e60c8f8b86cae...).
Rates are k/n with 95% Clopper-Pearson intervals; zero counts carry the one-sided 95% upper bound.

| arm | attack success (n = 240) | unsafe mutation | false completion | twin clean success (n = 240) | twin false block |
|---|---|---|---|---|---|
| B0 (ack-trusting) | 75 = 0.312 [0.254, 0.375] | 45 = 0.188 | 75 = 0.312 | 240 = 1.000 | 0 (upper 0.012) |
| B0 + gate | 0 (upper 0.012) | 0 (upper 0.012) | 0 (upper 0.012) | 240 = 1.000 | 0 (upper 0.012) |
| B4 (reconciliation) | 0 (upper 0.012) | 0 (upper 0.012) | 0 (upper 0.012) | 240 = 1.000 | 0 (upper 0.012) |
| B4 + gate | 0 (upper 0.012) | 0 (upper 0.012) | 0 (upper 0.012) | 240 = 1.000 | 0 (upper 0.012) |
| OBD (obedient) | 240 = 1.000 [0.985, 1.000] | 165 = 0.688 | 240 = 1.000 | 240 = 1.000 | 0 (upper 0.012) |
| OBD + gate | 12 = 0.050 [0.026, 0.086] | 12 = 0.050 | 12 = 0.050 | 240 = 1.000 | 0 (upper 0.012) |

What the numbers say, and what they do not:

- The attacks are actionable: the obedient policy succeeds on all 240 dev attack episodes and on every one of the 61
  development templates. With text removed (controls) it is never harmed on F4 or F6 (0/60, upper 0.049), so for an
  agent that reads tool text, the text is the cause.
- B0 never reads text. Its 75 successes are all mechanics: F4 30/30, F6 15/30 (the "denies" half) and F7 30/30. Its
  F4 and F6 controls give the same counts (30/30 and 15/30), so the attacker's text adds nothing for B0.
- B4 (consistent reads, same-key retries) is never harmed alone (0/240, upper 0.012). All 29 of its refusals are F7
  episodes whose rolled-back step is a keyed write (17 pay, 12 post): its same-key retry is answered from the
  idempotency store, the record stays absent, and it escalates. The one F7 receipt episode it repairs by recording the
  receipt again.
- Behind the gate, B0 and OBD lose every attack except one kind: F1 payee redirection delivered on the PO before the
  first payment (OBD + gate: 12/30 on F1, all 12 on the `po_note` channel). RG1 binds the PO, the amount and the step,
  not the payee, and its read-back treats a correct-amount payment to the wrong account as in effect. When the
  redirection arrives after the first payment, the gate's retry rule stops the second payment (0/18).
- Paired gate contrasts: the gate prevents 75/75 of B0's successes and 228/240 of OBD's; it introduces no harm
  (B0: 0/165, upper 0.018; B4: 0/240, upper 0.012) and blocks no clean twin (0/240 for each policy, upper 0.012).
- The safety comes partly as refusal: behind the gate, 87/240 attack episodes end in a refusal or hand-off for both B0
  and OBD. On mechanics-only controls, B0 + gate completes 3/60 (B0 alone 15/60, with 45/60 harmful): the gate
  blocks B0's new-key retries after a timeout and B0 has no recovery path. OBD + gate completes 48/60 of controls
  against 60/60 alone, for the same reason.
- Guard cost per episode (paired, gate minus alone): B0 +2.13 calls and +2.53 reads on attacks, +2.67 / +2.67 on
  twins; OBD +0.61 calls and +0.99 reads on attacks, 0 on twins; B4 0 (it already reads after every write). Scripted
  wall-time overhead is under 0.2 ms per episode; it says nothing about model latency.

These are scripted policies on development templates. They establish that the suite's attacks, ground truth and gate
wiring behave as designed. They are not evidence about trained models or about held-out templates.

## Intended use

- Registered evaluation of the released agents (D seeds 0-2, SEL, the base model, Dw_aug) on `attack_heldout`, with
  and without the gate, after `PREREGISTRATION_ATTACKS_DRAFT.md` is revised and frozen.
- Development and pilots on `attack_dev`; training data generation from `attack_train` only.

## Out of scope / do not use for

- Training on, tuning on, or inspecting agent behaviour on `attack_heldout` before a registered run.
- Claims about real payment systems or real attackers: the attacker model is the eight families above, in a
  simulator with five services.
- Claims about prompt-injection robustness in general: only tool-output content is attacked; the request and system
  policy are trusted.

## Known limitations

- RG1 has no payee binding (see above). A payee rule would be a new gate version and needs its own registration.
- F1 adds an optional `payee_account` argument to `pay_supplier` for its attack episodes and twins only; trained
  agents never saw it.
- Text-free policies (B0, B4) cannot measure text susceptibility; OBD is a ceiling, not a model of any agent.
- Template phrasings were written by one author; the Hindi phrasing has not been checked by a second speaker.
- Ground truth is exact for the simulator's state, but "harm" is the suite's definition: for example, refunding the
  only correct payment and then paying again still counts as compensation abuse.

## Reproduction

```
python scripts/attacks_make_splits.py --force      # rewrites data/attacks/ (hashes must match the manifest)
python -m pytest -q tests/attacks                  # 31 pass, 1 skipped locally (about 10 s); the skipped smoke test needs vLLM
python scripts/attacks_policies_eval.py            # dev results; Slurm: sbatch slurm/attacks/dev_policies.sbatch
```
