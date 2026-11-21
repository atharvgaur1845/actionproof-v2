# Presentation storyboard (six slides)

- Every number on a slide is a row of `docs/final/FINAL_CLAIMS.md` (shown as `[C#]`).
- Every visual is a real recording or a chart built from a result file.

## Slide 1: THE TOOL SAID SUCCESS. THE WORLD DISAGREED.

- **Visual:** full-bleed, dark. One line of code, `pay_supplier(po_id="PO-7B3D9659", amount_cents=304850)`. Two lanes:
  - top: `reply: {"status": "ok", "payment_id": "PAY-F5D3F7"}` in chartreuse;
  - bottom: `payments stored for PO-7B3D9659: none` in vermilion.
  - A red MISMATCH bar joins them. The agent's report reads *"PO-7B3D9659: pay, post completed."*
- **Caption:** a real recorded evaluation episode (task `s3_conf_full-0001`, model before training).
- **Speaker line:** "The tool said the payment went through. Nothing was stored. The agent told the user it was done."

## Slide 2: THE THREE TRUTHS

- **Visual:** three columns for one write (from the site's simulator, real ProcureSim).
  - THE AGENT DID: `pay_supplier 498,000¢`.
  - THE TOOL REPLIED: `ok`.
  - THE DATABASE HOLDS: `249,000¢ of 498,000¢`, hatched as evaluator-only.
  - Underneath: "phantom · partial · lost reply · duplicate · never commits: **the same `ok`**" [C1].
- **Text:**
  - ActionProof-Bench (ProcureSim): 5 read tools, 6 write tools, one fault checkpoint.
  - A state-only verifier, checked against a second verifier and golden traces.
  - The policy never sees the database.
- **Speaker line:** "Tool-call success is not task success. The only ground truth is the state."

## Slide 3: WHAT THE REWARD SEES CHANGES WHAT THE AGENT CHECKS

- **Visual:** the seed strip, two rows of five cells.
  - State reward: 1.00 1.00 1.00 1.00 1.00 (chartreuse).
  - Acknowledgement reward: 0.00 0.00 0.00 0.00 0.00 (vermilion) [C2].
  - Below: VSC gain +0.27 to +0.32 per seed [C3].
- **Design line:** same Cohere Tiny Aya Global 3.35B, same warm start, same AgentTune GRPO; only the reward's information source changes. Five paired seeds; 300 untouched tasks.
- **Caveat box:** "Selection, not invention: the starting model already reads back 31% of the time when sampling. The acknowledgement-trained agent never moved from it."
- **Speaker line:** "Pay the agent for what the database says, and it learns to look."

## Slide 4: READING IS NOT RECOVERY

- **Visual:** two recorded traces side by side. The model is the state-reward agent, seed 0; the fault is the same in both (only 25% of the payment stored).
  - Left (pay first): read → refund → pay again → **VERIFIED DONE**.
  - Right (after a goods receipt): read shows 62,892¢ of 251,570¢ → post → report "completed" → **FALSE DONE** [C6].
  - Under them, the workflow table: 0.99–1.00 / 0.05–0.55 / 0.00 [C5].
- **Text:** shown a missing or complete record, every model acts correctly [C4]. A *partial* record needs a different, quantitative decision, and the learned repair is bound to where it was practised.
- **Speaker line:** "Looking is not knowing what to do."

## Slide 5: TRAINING FOR RECOVERY

- **Visual left:** +0.21 verified completion, 5 of 5 runs, every run beats every run (p = 0.004) [C7]. A decomposition bar: duplicate delivery 0.12, partial payments 0.09 [C8].
- **Visual right:** against a control trained on the same faults:
  - +0.09 on recovery starts and +0.22 on partial and duplicate tasks, per seed +0.01, +0.34, +0.32 [C9];
  - partial repair learned in 5 of 5 runs against 1 of 3 [C10].
- **Cost row** (vermilion):
  - +2.7 points duplicates left unrefunded, and duplicate stress −0.04 [C11];
  - after-receipt partial payments are still unrepaired (5 of 60) [C12];
  - renamed tools and a new workflow erase the gain [C14];
  - a sibling domain gets worse [C15].
- **Footer:** "Three control seeds; two more are trained and in evaluation. Recovery-start training is a known curriculum, not a new optimiser."
- **Speaker line:** "Practising from the broken state helps on the faults we practised, and only there."

## Slide 6: FROM RESEARCH TO A WORKING SYSTEM

- **Visual:** a screenshot of the website's truth timeline and model replay (real simulator; recorded model episodes). Pipeline strip: Tiny Aya Global → SFT warm start → AgentTune GRPO (state reward) → ProcureSim verifier → registered evaluation → claims ledger.
- **Artifacts:**
  - GitHub (branch `stage4`);
  - the website;
  - the registrations and hashes;
  - `claims/claims.yaml` with `s2_numcheck.py`.
- **Toolkit honesty:**
  - AgentTune trains every model;
  - AuditKIT judged Stage 1 transcripts (an AI judge accepted 72% of false completions even when shown the evidence);
  - CircuitKIT did not run in Stage 2.
- **Close:** **"Reliable agents must be judged by the state changes they actually produce, not the success messages they generate."**

## Supporting slides (only on request)

- S4 shift forest plot [C13, C14].
- FCA negative result [C16].
- The withdrawn addendum 3 and why.
- The statistics protocol.
- Study 1 (refunds).
