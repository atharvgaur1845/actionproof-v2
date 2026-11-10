# ActionProof: Pre-registration Amendment 6 (selection of evidence-handling behaviours by reward observability)

**Status.** Registered before any Amendment 6 GPU run. The git commit that adds this file, `data/bench/a6/*.jsonl` and
`results/a6_hashes.json` is the timestamp. All hypotheses are exploratory unless listed in the primary family (§4).

**Preserved, unchanged:** `PREREGISTRATION.md` and its hashes (confirmatory H1, task 400); `PREREGISTRATION_A4.md`,
`results/a4_hashes.json` (pools, eval splits, A4 env, verifier, metrics, runner); `PREREGISTRATION_A5.md`. The A5 code
hash mismatch reported by `freeze_a5.py --check` is the disclosed D-A5-3 fix and predates this amendment.
New code is in new modules (`rewards/a6_rewards.py`, `bench/generator_a6.py`, `train/a6_demos.py`); the only edits
to existing files are additive dispatch entries in `rewards/trl_rewards.py` and `train/grpo.py` (unregistered).

## 0. Question and what was already seen

Q1 (replication on a new warm start): does the reward's goal-term information channel (acknowledgements vs
authoritative state) decide whether evidence acquisition and recovery are selected? Q2: does a report-aware state
reward change the policy's response to contradictory evidence, against an exactly matched report-blind reward?
Q3: is the selected behaviour evidence-conditioned (retry depth tracks persistence; transfers to an unseen depth and
tool) or a fixed script?

Seen before registration (cluster artifacts; all on the v2 lineage `rft_tiny_v2`): sampled post-ack refund
verification init 0.246, ack_outcome 0.102/0.100, tool_trust 0.153, outcome 0.713, pcstr_d 0.724 (val T=1x8); 0/150
honest failures on `e_unrecoverable` for every model and middleware; the trained policy retries the refund exactly
once on UR tasks and never retries the held-out credit tool despite reading `credits: []`; the warm start reports
failure on 0/48 sampled UR episodes of `a4_train_B` (95% upper bound 7.4%). This amendment is motivated by those
observations, which is why every contrast is between arms trained after registration.

## 1. Data (frozen here)

- **Training pool:** `data/bench/a4/a4_train_B.jsonl` (registered in A4, unchanged): 400 tasks = F5 55, OK 74,
  RP 222, UR 49. RP by (type, k_max): phantom 37/30/45 and timeout 43/36/31 for k_max 0/1/2. Fault tool: RP refund
  87, audit 71, close 64; UR refund 16, audit 14, close 19. (The generator draws F5 first at 15%, then OK/RP/UR at
  20/65/15% of the remainder.)
- **Demonstration pool `a6_demo`** (80): fresh seed stream, disjoint worlds, ids and prompts from both A4 pools and
  every eval split (tested). Faults on `issue_refund` only: 60 UR (persistent phantom), 20 RP (phantom, k_max = 2).
- **New eval splits:** `e6_twin_rp` (150): each `e_unrecoverable` world with the same prompt, ids and faulted tool,
  phantom for k_max = 2 then landing (recoverable twin). `e6_depth3` (150): phantom with k_max = 3 on refund/audit/
  close (one failure beyond any training task; the A4 oracle needs 5 attempts, 17 calls).
- **Reused eval splits (A4):** `e_unrecoverable`, `e_seen_refund`, `e_seen_other`, `e_heldout`, `e_heldout_ctrl`
  (18 turns); dev `test_iid`, `test_ood_fault` (10 turns).

## 2. Arms (one shared warm start, only the reward differs)

| Arm | Equation | Reads |
|---|---|---|
| M1 `pcstr_d` | G + 0.2P + 0.1V − 0.5I − 0.3ND − 0.1F_under − 0.2D | DB, receipts, report, feasibility |
| M2 `pcstr_d_noreport` | G + 0.2P + 0.1V − 0.5I − 0.2D | DB, receipts |
| M2b `pcstr_d_noND` | G + 0.2P + 0.1V − 0.5I − 0.1F_under − 0.2D | + report via F_under only |
| M3′ `outcome_G` | 1[G = 1] | DB, receipts |
| M3 `outcome` | 1[P = 1 ∧ I = 0] | DB; I from receipts |
| M4 `ack_outcome` | 1[∀(op,e) ∈ W: last ack success] · 1[I = 0] | acks (goal term); **state-derived invariant mask I** shared with M3 |
| M5 `tool_trust` | (1/|W|) Σ 1[ack success] · 1[rep = e(τ)] | acks, report, feasibility |

G = 1[P = 1 ∧ I = 0 ∧ D = 0 ∧ V = 1]; ND = 1[G = 0 ∧ rep ∉ {failed, escalated}]; F_under = 1[G = 1 ∧ rep ≠ e(τ)].
M4 is an acknowledgement-grounded **goal** term with a state mask, not a pure acknowledgement-channel reward; the
mask's binding frequency during M4 training is reported. M1 − M2 identifies the **bundle** of report terms; M1 − M2b
identifies ND; M2 − M3′ identifies density at a matched predicate; M3′ − M3 identifies the predicate (V, D); M3 − M4
identifies the goal-term channel at matched form.

**Fixed protocol:** warm start `runs/rft_tiny_a6` (§3); GRPO 200 steps × 4 prompts × 8 generations, lr 2e-5, β 0.02,
T = 1, ≤ 18 turns, ≤ 256 tokens/turn, LoRA r = 16, batched rollouts; seed 0 for every arm, seed 1 for M1–M4 if
compute allows; evaluation greedy, neutral prompt, saved adapter.

## 3. Warm start π₀′ and gate

`runs/rft_tiny_a6` = RFT on the v2 rows (rebuilt exactly as for A5: teacher `full_32b_verify_a`, expert
`expert_refund_f050`, seed 0) plus all correct-for-class `a6_demo` demonstrations (`train/a6_demos.py`; demonstrator
verifies only the refund, retries ≤ 4 times with one key, reports "failed" on a refund that never lands; never reads
audit/close), 2 epochs. This tests **reward-driven selection from an SFT-established repertoire**, not discovery.

**Gate G-A6 (before any arm), fixed now:** π₀′ at T = 1 × 4 on the first 100 tasks of `a4_train_B` (≤ 18 turns).
PASS iff HFR on UR (CF / UR episodes) ∈ [0.10, 0.70]. Reported, not gating: PWV on audit/close (expected < 0.05; if
not, H12/H13 are weakened and this is disclosed), PFR on RP, multi-retry rate. If the gate fails, no arm runs; the
result is reported and any change to the demonstration fraction is a new, logged deviation.

## 4. Hypotheses

**Primary family (Holm, α = 0.05, per seed 0; task-level paired bootstrap 10k + exact McNemar):**
- **P-A (channel):** M3 > M4 on (i) PWV_refund (post-ack refund verification, `e_seen_refund`) and (ii) RP VSC at
  k_max ≥ 1 on `e_seen_refund` ∪ `e_seen_other`. Claim requires both CIs to exclude 0.
- **P-B (report terms):** HFR(M1) > HFR(M2) on `e_unrecoverable`, with PFR(M1) − PFR(init) ≤ 5 pp on the RP tasks of
  `e_seen_*` and with the discrimination index HFR(e_unrecoverable) − PF(e6_twin_rp) higher for M1 than for M2.

**Secondary / exploratory:** M1 vs M2b (ND alone); M2 vs M3′ (density); every arm vs init (improvement, never
substituted for a between-arm contrast); attempts vs k_max (Spearman, per arm); `e6_depth3` RP VSC; held-out credit
retry after a read showing `credits: []` (`e_heldout`) and `e_heldout_ctrl` within 5 pp of `e_seen_*`; status
accuracy (confusion matrix of reported vs verifier status); M5 HFR < init.

**Kill criteria:** P-B CI includes 0 ⇒ no report-incentive claim. PFR rises ≥ HFR gain ⇒ "refusal", no honesty claim.
M1 ≈ M2b ⇒ the effect is F_under (underclaim), not ND. P-A fails ⇒ no channel replication on this warm start.
M2 ≈ M3′ ⇒ no density claim. A prompt-only baseline (`verify_a`, `verify_c` on π₀′) matching M1 on HFR and RP VSC
⇒ RL contribution weakened.

**Seeds.** No between-seed inference: with k runs per arm, exact permutation tests attain p ≥ 1/C(2k,k) one-sided
(k = 2: 1/6). Per-seed results with task-level CIs are within-run uncertainty only.

## 5. Gates before GPU

`tests/test_a6.py` passes (reward identities, report-blindness, honest-failure margin, disjointness, twin
construction, oracle solvability, pool B in the plain env, demonstrator correctness); `freeze_bench`, `freeze_a4`
and `freeze_a6 --check` print OK.

## 6. GPU plan (H200, estimates)

π₀′ RFT + gate ≈ 1.8 h; M1, M2, M3, M4 ≈ 3.5 h each + eval ≈ 0.5 h each; baselines ≈ 0.5 h; M2b, M3′, M5 after.
Experiment freeze 2026-11-13 18:00 IST; anything unfinished is reported as running.
