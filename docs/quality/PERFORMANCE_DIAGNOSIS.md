# Performance diagnosis

2026-11-24, branch `quality-upgrade`. Descriptive. The registered verdicts are in `results/p1/p1_summary.json` (ledger claim P1); nothing here changes them.

**Sources.** Every share below is from `docs/quality/diagnosis_tables.md`, which `scripts/diagnosis_report.py` generates from `results/quality/diagnosis_p1_{dev,test}.json` (`scripts/diagnose_failures.py`, run on Slurm). Shares pool all seeds and samples, so they are episode shares, not the registered seed-averaged task statistic.

**Failure stages.** Each episode gets the first stage it fails at:
1. no_read: the agent never reads after the faulted write;
2. read_no_problem_shown: the read did not expose the fault;
3. read_not_acted: the read showed the problem and the agent did not act on it;
4. acted_not_resolved: the agent acted, but the state is still wrong;
5. resolved_other_step_failed: the fault was repaired, but another step failed.

Never-commit tasks, which are unreachable by design, are classified only by whether the agent reported honestly.

## 1. Where the recovery-start agent (A) fails on the frozen test (sampled)

A loses episodes at every stage, not at one:
- no read: 0.138;
- read, not acted: 0.117;
- acted, not resolved: 0.089.

Its reports are false successes in 0.385 of episodes. Across stages 1–3, A's training (recovery starts with a handed read) taught it to read in many episodes. It did not teach it what to do after a short read.

## 2. What each intervention changes

| Change from A (sampled, test) | B: reconciliation line | C: recovery SFT | D: recovery SFT, then GRPO |
|---|---|---|---|
| no read | unchanged (0.136) | almost gone (0.010) | low (0.045) |
| read, not acted | lower (0.067) | 0.024 | 0.019 |
| acted, not resolved | **doubles** (0.168) | 0.054 | **0.007** |
| false-success reports | higher (0.420) | 0.129 | 0.110 |

Readings:
- **B moves failures from stage 3 to stage 4.** The line makes the agent act, but on short payments it tops up with the original idempotency key. The service deduplicates that call, so nothing changes. This is the development mechanism (`docs/research/P1_DEV_DIAGNOSIS.md`), now on the test. B's registered result is −0.030.
- **C fixes reading and acting, but not always the repair.** Its remaining loss is mostly acted_not_resolved.
- **D's GRPO stage trades a little reading for correct repair.** D skips the read more often than C (0.045 against 0.010). When it acts, though, the repair almost always resolves (0.007). Per registration, D beats A by +0.306 and C beats A by +0.259; the D–C difference was not registered and is not tested here.

## 3. Where the best arm still fails

- **Unseen workflow** (`p1_test_wf4`, order → receive → pay → post). D reaches 0.486 sampled VSC (A 0.174).
  - The largest single loss is a skipped step. Every test task of this workflow injects a fault, and the scripted oracle solves all 60 (`results/p2/ref/O__p1_test_wf4.summary.json`). Yet in 158 of D's sampled episodes the fault never fires, and none of them succeeds. They report "completed" in 149 (`results/quality/diagnosis_p1_test.json`, cell `D|sampled|order_receive_pay|clean`).
  - In every one of those episodes the faulted step is the goods receipt, and the agent never calls `record_goods_receipt`. It runs order → pay → post, the three-step order it was trained on, and reports success.
  - The same holds for every arm: no arm completes any such episode (A 0 of 296, B 0 of 305, C 0 of 150). This is the Stage 4 procedure-replay finding, and P1 does not remove it. The diagnosis labels these episodes "clean" because no injected fault fired. That label means the fault never fired; it does not mean the task was fault-free. Counts differ between arms because whether the fault fires depends on the agent's own actions.
  - Two further losses in this workflow: a lost acknowledgement on the goods receipt (read, then not acted on, in most episodes), and short payments where the payment is repaired but another step fails.
- **Never-commit tasks** cap VSC by design. The VSC ceiling of the test union is 0.940 (27 never-commit tasks).

## 4. Greedy and sampled decoding

Greedy decoding gives every arm higher VSC and fewer false successes than sampled decoding (tables, "greedy" sections). The registered statistic is sampled. Greedy numbers are never reported as sampled.

## 5. What this implies for P2

- **The structured ledger (B2, E)** presents payment state as fields instead of a sentence. Whether that avoids B's same-key top-up is P2's H3.
- **The action guard (rule R3, no refunded key; rule R1, read before retry)** blocks the exact stage-4 failure B shows. P2's H4 and S1–S3 test whether blocking helps.
- **The unseen workflow** is the gap no P1 arm closes. P2's shift splits measure it again.

## 6. P2: why the ledger and the guard lower completion (written after the P2 results)

Sources: `results/quality/p2_guard_diagnosis.json` (`scripts/p2_guard_diagnosis.py`) and `results/quality/diagnosis_p2_test.json`, both computed on Slurm. Registered verdicts: claims P2-H2 and P2.

**The guard does what it specifies; the specification is the problem.**
- *Rule R1 (read before retry) blocks the registered repair.* The registered repair for a short payment is refund, then re-pay with a new key. A refund counts as a write to payments, so R1 demands a read in between. Replaying the scripted oracle through the guard, R1 would block it on 90 of 90 partial-payment tasks. Every would-block across all test splits is R1.
- *Blocked agents rarely recover.* Sampled VSC after any block: Ag 0.12, Dg 0.145, Eg 0.37. (Blocked and unblocked episodes are not comparable groups: whether a block happens depends on what the agent does.)
- *The guard converts false successes into non-completions.* E reports a false success in 702 of 5,400 sampled episodes; Eg in one. But Eg has 522 episodes without a report (turn limit) and 465 under-claims, where the work was done and the agent did not report it as done. VSC counts both as failures.
- *Failure stage.* With the guard, "acted, not resolved" grows for C and D (`diagnosis_p2_test.json`): blocked mid-repair, the agents improvise a different repair, which is often wrong.

**The ledger at inference raises false success.** B2 (A with the ledger block) reports false success in 3,092 of 5,400 episodes, against 2,081 for A. A was never trained on the block. The text is out of distribution for it, and it does not use the "below the PO total" field to repair.

**What would fix it, not tested here.** A guard that allows a refund-then-re-pay pair under a new key, without a read in between, would not block the oracle. This is a post-hoc design observation. Testing it needs a new registration, and it is not claimed.

