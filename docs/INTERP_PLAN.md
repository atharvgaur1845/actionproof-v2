# Mechanistic analysis of the reward-source effect: analysis plan (frozen before extraction)

Committed before any activation in this plan is extracted. Every choice below is fixed. Anything changed later is logged in SPEC §27 as a deviation.

## Audit of existing infrastructure (2026-11-10)

- `interp/extract.py`, `probe.py`, `text_baseline.py`, `h7_compare.py`: reused (task-hash split, logistic probe, TF-IDF baseline).
- **SAE (`sae_features.py`): not valid for our data.** FVU on agent-transcript activations is 3.9, against the go/no-go threshold of ≤ 0.3. It was trained on 256-token web text. It is not used.
- **The old pos-A probe failed its own negative control.** It reached test AUROC 0.60 overall, but 0.62 on F1 phantom acks, which are byte-identical to real acks, so no evidence signal can exist there. That probe therefore decoded something other than evidence (tool or position cues). It is not reused as evidence of an internal representation.
- H7 earlier: probe ≈ TF-IDF text baseline; all CIs include 0.

## Models (existing checkpoints only; no training)

All share one init: `rft_tiny_v2`, then GRPO with the reward named. Evaluated **per seed**:
- init;
- `ack_outcome` s0, s1;
- `outcome` s0, s1;
- `pcstr_d` s0;
- `tool_trust` s0.

## Prefix set (shared by all models)

- **Source:** the val T=1×8 episodes of all 6 GRPO models plus the init (`results/eval/a4/h11/*`), pooled with the greedy development-set episodes of the same models (`results/eval/r3/*__{test_iid,test_ood_fault}.jsonl`). The confirmatory `_c` sets are never used. (Amended before extraction: the sampled policies are near-deterministic, so val alone gave only 26 unique discrepancy prefixes.) Prefixes are deduplicated by text. Every model reads **the same** prefixes, with activations taken at the last token of the generation prompt, i.e. immediately before the next action.
- **A1 (post-ack):** prefix ends with the tool result of a *success-acknowledged* `issue_refund`, with no read since.
  - Label: did the refund land (from the attempt log)? On F1/F3 the ack text is identical to a landed one, so **chance is expected**. This is the built-in observation-equivalence control on the v2 domain.
- **A2 (post-read):** prefix ends with the result of a model-issued `query_ledger` / `lookup_order` / `get_refund_status` on the order, after a success-acked refund.
  - Label = **observable discrepancy**: the read shows no refund (ledger) or a non-refunded status (order / status endpoint).
- **Balance:** each probe set is subsampled to 50/50 labels, with a fixed seed of 0.
- **Splits:**
  - task-hash 60/20/20 (train / val / test);
  - plus a **scenario-held-out** evaluation: train on F3 (partial) prefixes and test on F1 (phantom), and the reverse. Benign negatives are split by task hash.

## Probes and controls

- Standardised L2 logistic regression with C = 1.0 (low capacity).
- The layer is chosen on val AUROC only; test is read once.
- Controls:
  - **shuffled-label** probe, which must be at chance;
  - **TF-IDF baseline** on the final tool-result message (word 1–2-grams, C = 1).
- CIs: cluster bootstrap over tasks, 2,000 resamples.

## Behaviour readout (same prefixes)

- P(next tool) is computed by teacher forcing over tool-name candidates after the model's preferred call header.
  - Both headers seen in the data, `[{"tool_name": "` and the pretty-printed one, are scored.
  - P(tool) = Σ_h P(h) · P(tool | h), renormalised over the candidate set.
- Groups:
  - **verify** = {query_ledger, lookup_order, get_refund_status};
  - **recover** = {issue_refund, reconcile_order};
  - **proceed** = {add_audit_note, close_ticket, submit_report}.
- **Readout is single-decision only.** No claim is made about full episodes.

## M4: knowing vs acting (A2, discrepancy = 1)

Per model:
- **decodability:** the model's own probe test AUROC;
- **action:** mean P(recover) on discrepancy prefixes minus mean P(recover) on consistent prefixes.

Outcome categories:
- *not decodable*: AUROC CI includes 0.5;
- *decodable but ignored*: AUROC > 0.5 and the Δ action CI includes 0;
- *decodable and acted on*: both hold.

## M3: intervention (only if hooks work; otherwise omitted)

- **Model:** `outcome` s0, A2 test prefixes.
- **Direction:** the model's own A2 probe weight vector at the chosen layer L, unit-normalised in residual space.
- **Ablation:** at layer L over the tokens of the final tool-result message, x ← x − α (x·d) d, with α ∈ {0, 0.5, 1, 2}. α = 0 is the sham.
- **Control:** 5 random unit directions, with α = 1 and α = 2.
- **Measures:** Δ P(recover) and Δ P(verify) on discrepancy prefixes.
- **Collateral check:** P(the episode's actual next tool) on 100 benign first-turn prefixes must change by less than 0.02 at α = 1.
- **Claim rule:** a causal role is claimed only if the probe direction's effect exceeds the maximum of the random-direction effects and is monotone in α.
- **Additive steering, reported separately:** add +β·σ·d in `ack_outcome` s0, β ∈ {1, 2, 4}, where σ is the std of the projection on the train set.

## M2: observation-equivalent pairs (A5 slice)

- **Episodes:** the init's (`rft_tiny_a5`) greedy a5_eval episodes on the `unbounded_late` / `never_commit` pairs.
- **Checks:**
  1. The full rendered input text is identical within each pair, at every turn.
  2. Activations are identical within each pair (max |Δ| reported).
  3. A probe for the hidden world (late vs never) is trained on task-hash splits. Its test AUROC must be ≈ 0.5.
- Any deviation triggers a leakage audit (ids, seeds, prompt construction, metadata) and is reported. It is never explained away.

## C/D on A5 (descriptive; the A5 policies are near the ceiling)

- **Model:** init `rft_tiny_a5`. **Positions:** after each `get_operation` / `list_operations` result.
- **Labels:**
  - **C** = lifecycle status of the agent's own op as observed (pending / committed / failed);
  - **D** = the oracle's appropriate next action (proceed / wait / report-uncertain / retry).
- Probe vs TF-IDF baseline, with leave-one-family-out evaluation. These labels are in the text, so a high probe score alone is uninformative; only probe − text is reported as evidence.
