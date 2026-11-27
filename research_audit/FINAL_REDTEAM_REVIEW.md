# Final red-team review (prompt §11)

2026-11-27. Three hostile reviews were run as separate Claude sessions, started together at about 10:05 IST.

**Isolation.**
- No reviewer could see another's conclusions; each was told not to open any REVIEW or REDTEAM file.
- All three were read-only: no edits, no cluster, and no live-Space runs.
- Reviewer 2 ran probe scripts against the frozen simulator, in a scratch directory.

**The commit reviewed:** `stage2-superiority-sprint` at 378c7d8, after the SEL1 result and before any fix below.

**The verdict: all three returned FAIL.**
- The arithmetic was clean. Reviewer 1 re-read more than 20 numbers from the JSON files, and every one matched.
- The failures were in what the numbers were said to mean, and in the benchmark's construction.

**How the reviews were checked.** Every factual claim was re-checked before it was acted on:
- Reviewer 2's probes were read, then rerun.
- Reviewer 1's counts were recomputed.
- Exploratory task 1840 sized each finding over every recorded episode (`results/audit/txn_validity.json`).

The resolution table is at the end. The reviews are reproduced as they were returned; formatting is lightly
normalised.

---

## Reviewer 1 — skeptical RL / ML researcher

**Recommendation: FAIL.** "The arithmetic is clean, but the main headline counts duplicates that B3 never created.
The new SEL1 claim is decided on in-distribution tasks with one vacuous control, and on the locked test it barely
changes unsafe actions."

**Top blockers.**
1. **The headline "duplicates mutations" claim describes duplicates B3 mostly did not cause** (high confidence).
   - 50 of B3's 61 DMR episodes are `dup_delivery` or `concurrent`: the environment's duplicates, which B3 never
     reads back.
   - `late_commit` commits without updating the idempotency store.
   - The claim leads with a passing subtest of a registered family that is not supported.
2. **SEL1's deciding tests are in-distribution, and a selectivity control cannot fire** (high).
   - H1's 60 tasks follow the sel1_train construction.
   - `audit_down` is behaviourally identical to `std`, because B5 reads the audit log only after its primary reads
     fail.
   - `t2` (n = 2) is inside the demonstrated 3-read budget.
   - The only truly held-out test (H2) is outside the decision rule, and it fails for seed 2 on `u_receive`.
3. **SEL1 "fixes" a convention the prompt never states** (high on facts, medium on severity).
   - D reports `failed` in 473 of 540 unresolvable episodes.
   - Unsafe actions move only from 12 to 7 of 540.
   - H1 mostly replaces an honest failure report with an `escalate_to_human` call.
4. **A false B4 comparison appears on every surface** (high). SEL's 0.830 is below B4's 0.864; only CTL is above it.
5. **The SEL1 novelty claim misdescribes the method** (medium-high).
   - The trigger is an explicit `service_unavailable` error, not an outage hidden behind identical replies.
   - Competitor 02 and SURE-Act train stopping and escalation.
   - Some prior-work descriptions rest on search snippets.

**Major concerns.**
1. R7's GRPO increment has no equal-compute SFT control, and D's three seeds share one SFT warm start.
2. The teacher is the yardstick. B4 dominates D (FCR 0.121, DMR 0.052 against 0 and 0), and no regime shows a learned
   agent beating the script.
3. txn_test is reused, and the choice of held-out step coincides with the test file (intent: low confidence).
4. P5 is near the floor: one SFT seed, no plain-SFT or few-shot comparator.
5. Inference is task-level only: 3 seeds; "p 0.0003" is the bootstrap floor; H1 has 15 clusters; P1, P4 and P5 use
   unclustered bootstraps. The SEL1 data hashes are not in the repo, and the episodes are absent. Registrations rest
   on local git timestamps, and some stated registration times are later than their commits.

**Minor.**
- "789 numbers" is stale.
- "Retrying blindly pays twice" contradicts B3's idempotency keys.
- "Removes procedure replay" overstates P4 (one surface, one workflow, +21% rows).
- "Training removes most of the exposure" compares with B0; B4 is about 18× cheaper than D.
- The deck and write-up use two impact denominators.
- VTC credits completions on unresolvable tasks.
- H3's J reduces to esc_U.

**Claims to remove or reword.**
- The 0.024 against 0.191 headline.
- "SEL1 fixes that on the locked test".
- "15 of 540" without "and reports failure in 463".
- "Never escalates on a transient outage".
- `audit_down` as selectivity evidence.
- "Above B4's 0.864", on every surface.
- The "hidden behind identical replies" distinction.
- "Prior work is evaluation-only".
- "Registered and hashed before" while the SEL1 data hashes were absent.
- P4 "removes procedure replay".
- "Retrying blindly pays twice".

**Evidence missing.**
- The SEL1 training rows and their hashes in the repo, and the SEL1 episodes.
- An escalate-on-first-error demonstration arm and an instruction-only arm.
- Transient outages with n ≥ 3, a different error text, unresolvability signalled other than by an error, and an
  `audit_down` the policy observes.
- An equal-compute SFT control and multiple SFT seeds for D.
- DMR by origin.
- A registered D-against-B4 comparison.
- A hierarchical bootstrap.
- An external timestamp.
- A falsifier for "the learned agent adds value beyond B4".

---

## Reviewer 2 — agent / distributed systems researcher

**Recommendation: FAIL.** "The benchmark's key distinction between resolvable and unresolvable outcomes falls to the
simulator's own idempotency contract. The scorer gives verified success to over-refunds that leave the PO net
negative, and trivial read-free policies come close to the trained agent's VTC."

**Top blockers.**
1. **The "unresolvable" regime can be resolved, and it leaks the hidden truth through the idempotency store** (high).
   - A same-key resend never pays twice. It returns the same id if and only if the first write committed.
   - All 15 txn_test quadruples fault a keyed step, so all 60 unresolvable tasks can be resolved.
   - A policy that never reads and double-sends gets VTC 120 of 120 on the quadruples, 0 unsafe.
   - In sel1_test, 72 of 120 "escalation-correct" tasks are keyed. Only the receive tasks are not, and SEL is
     fragile exactly there.
2. **Over-refunds are scored as success** (high).
   - `refund_payment` never checks a prior refund, and the pay verdict ignores net money.
   - Probe: three refunds leave net −548,080 on a 548,080 PO, yet VTC is true.
   - It appears in real outputs: A_s0, A_s2.
3. **The H3 comparators cannot see the duplicates they are charged with** (high). Agent-caused DMR for B3 is 11 of
   320.
4. **Trivial policies come close** (medium-high). Fire-and-forget reaches 211 of 440; a read-free double send 308 of
   440 (0.700, against D's 0.724).

**Major concerns.**
1. The idempotency model is not Stripe's:
   - a parameter mismatch returns the original ok;
   - keys never expire, and there is no in-flight conflict;
   - `dup_delivery` writes two rows under one key while the audit log shows one request;
   - a phantom ok carries an id.
2. "Concurrency" is a scripted injection, not a race. Writes have no version precondition, and `consistent=True`
   bypasses the delayed regime.
3. Compensation and partial semantics are unrealistic. Refunds are instant, full and unkeyed. Paying the remainder of
   a partial payment scores DUPLICATE, while topping up a partial receipt is accepted.
4. Escalation metrics can be gamed:
   - always-escalate gets CER 60 of 60;
   - reading once and escalating on `service_unavailable` gets CER 60 of 60;
   - escalating any PO counts.
5. The verification claims are weaker than presented:
   - the mutants do not touch `integrity/sim/truth.py`;
   - the reference verifier is not independent;
   - the leak test is lexical;
   - B4 was tuned on dev while the benchmark was built, and it saturates the benchmark.

**Minor.**
- "SQLite" is false: the state is Python lists of dicts.
- The spec's H3 has 6 tests; the registration has 4.
- The audit log contradicts its own docstring under `log_conflict`.
- `txn_score` trusts a stored score.
- A late commit after a "failed" report is not scored as harm.
- The live demo uses the proc2 env, not TXN.
- The local Dw_aug and SEL1 episode files are truncated or absent.

**Claims to remove or reword.**
- "Escalation counts as success only where the outcome is truly unobservable".
- "Escalation is the safe action there".
- "Completes … without the evidence to know it".
- TXN_SPEC's "cannot tell … from any observation".
- "Reads the database, never the transcript".
- The 0.024 against 0.191 headline without the origin split.
- "Pay twice … are their own metrics".
- "25 mutants … all killed" (scope).
- "A SQLite procurement world".

**Evidence missing.**
- A same-key-resend baseline, and keyed-vs-unkeyed splits.
- A net-money postcondition, with re-scoring.
- Agent-caused DMR.
- Faults on compensations, mismatch errors, key expiry, and real interleavings.
- Grounding in real incident data.
- Tool-generated mutation of `integrity/sim/truth.py`.
- An independent verifier.
- The Dw_aug and SEL1 episodes.

---

## Reviewer 3 — hackathon judge / product evaluator

**Recommendation: FAIL as it stands.** "The science is strong and honest, but a judge can't reach the code, deck or
video. The submission also never shows why a trained agent beats its own hand-written reconciliation script, and its
headline slide repeats a wrong comparison (0.830 > 0.864)."

**Scores:** presentation 5/10, impact 4/10, technical depth 8/10.

**Top blockers.**
1. **A judge cannot get at the code.** The GitHub `origin/main` is 288 commits behind and states a different thesis,
   and `submission-candidate` lacks the sprint.
2. **The judged artefacts are not public.** The deck is private, with no PDF or PPTX in the repo, and there is no
   video.
3. **There is no product case against the script.**
   - B4 is about 18× cheaper than D.
   - Judge mode compares only with B0, while the oracle script is in the dropdown.
4. **SEL1 is invisible in the demo, and the demo shows only successes.** The recordings are 24 of 24 for the trained
   agent.

**Major concerns.**
1. The wrong B4 comparison on four surfaces.
2. Two inconsistent impact models. "2% incidence" is listed as an assumption for a per-faulted figure, and per 1,000
   real workflows the moderate figures are about $29.9k and $2.9k.
3. Jargon is used before it is defined: VSC, STR, B5, DMR, URR, J, Dw_aug, SR1.
4. Theses compete. WRITEUP §3 leads with A12 and omits SEL1.
5. Live-path friction:
   - "60 seconds" against a measured 69–72 s;
   - `HF_TOKEN` missing, so the "(live)" base options always fall back;
   - a false "models are public" comment;
   - the dataset viewer errors;
   - the README judge path differs from the default.

**Minor.**
- Registration counts disagree: 53, 52 and 33.
- The video's judge segment (55 s) is shorter than the run, and the timings differ from the run of show.
- The demo and benchmark are conflated ("440 locked tasks" in the demo intro).
- The hook says "same reply" over two replies.
- Deck overflow risk: the matters.html URLs, and the fails SEL box.
- The author-action list is stale.
- "0.989" with no metric named.
- The D card says "one base model".

**Claims to remove or reword.**
- "Above B4's 0.864".
- "Training removes most of the exposure. Checks remove the rest" and "ship both" (never measured).
- "2% fault incidence" for a per-faulted figure.
- Lead with H2 rather than H1.
- "Never hands off on a transient outage".
- "60 seconds" and "(live)".
- "Every number is scored by a deterministic verifier".
- "The full registered study: every result and recorded episode".

**Evidence missing.**
- D plus B4 measured against B4 alone.
- SEL in the live demo.
- A real-system check.
- An independent verifier.
- The video, and a deck export.
- A clean-room run.
- The cost of human review.
- A browser check of the layout and the cold start.

---

## Resolution table

Status: **fixed** (changed and verified), **disclosed** (cannot be fixed before the freeze; stated prominently where
the claim is made), **open** (author action, or work beyond this round). Severity follows the reviewers' own
ranking: B = blocker, M = major, m = minor.

| # | Concern | Reviewer | Sev. | Evidence (re-checked) | Fix | Status |
|---|---|---|---|---|---|---|
| 1 | Keyed "unresolvable" tasks settle by a same-key resend; escalation is not the only safe action | R2 (B1), R1 | B | Probes rerun; task 1840: all 60 txn_test unresolvable tasks keyed; read-free BD 0 unsafe of 60 | `PREREGISTRATION_TXN1_ADDENDUM3.md` §1; H5, SEL1 and shortcut wording changed on every surface; keyed/unkeyed split reported | disclosed (benchmark locked; v2 fix listed) |
| 2 | SEL1 H1 mostly replaces an honest "failed"; unsafe 12 → 7 of 540 | R1 (B3) | B | 1840: CTL "other" 354 of 540, D 482 | SEL1 addendum 1 §1; README, write-up, slides, deck, cards and site restate SEL1 as "hand-off after repeated outage errors" | fixed (wording) |
| 3 | `audit_down` cannot fire; `t2` within the 3-read budget; H1 in-distribution | R1 (B2) | B | `actionproof/sel1/policy.py` read; SEL behaviour on `audit_down` = `std` | SEL1 addendum 1 §2; selectivity evidence withdrawn from every surface | fixed (wording); controls listed as missing |
| 4 | Where hand-off is the only safe action, the effect is partial and seed-fragile | R1, R2 | B | 1840: unkeyed 164 of 432 escalated (55/106/3), unsafe 215 against 293 | Leads the SEL1 statement everywhere | disclosed |
| 5 | False "both SEL1 arms pass B4" (SEL 0.830 < 0.864) | R1 (B4), R3 (M1) | B | `results/impact/impact_txn.json` | Corrected in README, write-up, SLIDES, deck, story decision; logged in the sprint log | fixed |
| 6 | H3 duplicate headline counts injected duplicates B3 cannot see | R1 (B1), R2 (B3) | B | 1840: B3 50 of 50 injected, 11 of 270 other; D 54 of 450, 15 of 2430 | Origin split added next to every 0.024/0.191; the registered test stands | fixed (wording); registered result unchanged |
| 7 | Over-refunds credited by the verdict code | R2 (B2) | B | Probe rerun (net −548,080, VTC true); 1840: 3 of 74,000 episodes, none D's | TXN1 addendum 3 §4; the shortcuts table now says it | disclosed (no conclusion changes; v2 fix listed) |
| 8 | Read-free policies close to D on VTC (BD 308 of 440) | R2 (B4) | B | 1840 policies | Reported as a reference row in README; TXN read with FCR and DMR | disclosed |
| 9 | A judge cannot reach the code (stale `origin`; sprint unmerged) | R3 (B1) | B | `git branch -vv` | — | **open: author must approve the push and merge** |
| 10 | Deck private; no video | R3 (B2) | B | — | Deck finished; video script and captions ready | **open: author shares/exports the deck and records the video** |
| 11 | No case that a learned agent beats the script; judge mode shows only B0 | R1 (M2), R3 (B3) | B | B4 dominates on TXN | Stated as the first "what fails" item; the Space intro now says the oracle script also succeeds; the impact slide says "the script cuts more" | disclosed (no such regime exists in the evidence) |
| 12 | SEL1 invisible in the live demo; recordings show only successes | R3 (B4) | B | `space_live/core.py` | Not changed (the demo env has no unresolvable regime); the replay site gained SEL1 and benchmark-limit text | disclosed |
| 13 | SEL1 novelty claim misdescribes the trigger; prior work that trains | R1 (B5) | B | `actionproof/sel1/env.py` | Novelty addendum 2: distinction withdrawn, claim narrowed | fixed |
| 14 | SEL1 data hashes and rows missing from the repo | R1 (M5) | M | Cluster `results/sel1_hashes.json`: data hashes recorded 05:42, first training 05:44 | Hash file and stats committed; rows released in the traces dataset (task 1850); `freeze_sel1 --check` separates absent from mismatched | fixed |
| 15 | Impact: two denominators; "2% incidence" misplaced; "ship both" never measured | R1, R3 (M2) | M | `impact_txn.json` `unit` | Both units on the slide and in write-up §24; "ship both" removed; "the script cuts more" | fixed |
| 16 | Jargon before definition | R3 (M3) | M | — | README "Terms" block before the tables; write-up §3 rewritten | fixed (README); deck appendix keeps the acronyms |
| 17 | Competing theses; write-up §3 led with A12 | R3 (M4) | M | — | Write-up §3 rebuilt on the spine; A12 moved to "Also" | fixed |
| 18 | Live-path friction: "60 s", "(live)" base without a token, a false comment, the dataset viewer | R3 (M5) | M | Measured 69–72 s; base `GatedRepoError` | Honest timing; base label says it needs a token; comment fixed; dataset viewer disabled with an explanation | fixed; **open: author sets the `HF_TOKEN` secret** |
| 19 | GRPO increment not attributable; D seeds share one warm start | R1 (M1) | M | P1 registration | Already labelled exploratory (R7) and "not nested"; stated in limitations | disclosed |
| 20 | Task-level inference only; bootstrap floor; unclustered P1/P4/P5 | R1 (M5) | M | — | Stated in write-up §25 and §20 | disclosed |
| 21 | P5 near floor, one seed | R1 (M4) | M | `p5_summary.json` | Already stated ("one SFT seed per model; a task-level test") | disclosed |
| 22 | Idempotency, concurrency and compensation semantics simplified | R2 (M1–M3) | M | Code read | TXN1 addendum 3 "also recorded"; README limits section | disclosed (v2) |
| 23 | Mutants miss ProcureSim's verdict code; the verifier is not independent | R2 (M5) | M | `scripts/verifier_mutation.py` | Wording changed on README, slides and deck | disclosed |
| 24 | "SQLite" false | R2 (m) | m | `actionproof/integrity/sim/env.py` docstring | Corrected on every current surface and card | fixed |
| 25 | "Never the transcript" | R2, R3 | m | FCR reads the report | "From the final state; the report is checked against it" | fixed |
| 26 | "Retrying blindly pays twice" | R1 (m) | m | B3 uses keys | Slide 2 and deck rewritten | fixed |
| 27 | P4 "removes procedure replay" | R1 (m) | m | One surface, one workflow | Abstract and §3 say "raise … on the tested surface and workflow" | fixed (abstract, summary and the README P4 heading: "reduce … on the tested surface and workflow") |
| 28 | Registration counts 53, 52, 33 | R3 (m) | m | 54 files now | README and write-up §6 say 54 | fixed |
| 29 | "789 numbers" stale | R1 (m) | m | numcheck 866 | Count dropped from the reproduction table | fixed |
| 30 | Video and run-of-show timings differ; 55 s segment against a 72 s run | R3 (m) | m | — | One timeline; a labelled cut | fixed |
| 31 | "440 locked tasks" in the demo intro; the demo is not TXN | R2, R3 | m | `space_live/core.py` uses P1 tasks | Intro says "ProcureSim tasks from the P1 test; not the benchmark" | fixed |
| 32 | Deck overflow (URLs) | R3 (m) | m | Estimate | `hf.co` short URLs at 19 px with wrapping | fixed (estimate; no headless browser here) |
| 33 | Registration times later than commits; local timestamps only | R1 (M5) | M | git log | Not changed; a push would add an external timestamp | **open (push)** |
| 34 | Tool-call accuracy not a named metric | compliance audit | m | `research_audit/COMPLIANCE_AUDIT.md` | Disclosed there | disclosed |
| 35 | No real-system check; no human-review cost | R2, R3 | M | — | Limitations | disclosed |

**After the fixes.**
- No unresolved critical blocker remains that this session can fix.
- Rows 9 and 10 are the author's: push and merge; share the deck; record the video.
- Rows 1, 4, 7, 8, 11 and 12 are disclosed limits of the evidence, not fixable without new experiments. Each is
  stated where its claim is made, and in the README's "What does not work".
- The submission does not claim to beat any competitor.
