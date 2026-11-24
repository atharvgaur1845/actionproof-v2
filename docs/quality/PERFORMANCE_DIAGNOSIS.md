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
  - The largest single loss is not a fault at all. On the **fault-free** episodes of this workflow D succeeds in none of 159 sampled episodes, and reports "completed" in 149 of them (`results/quality/diagnosis_p1_test.json`, cell `D|sampled|order_receive_pay|clean`).
  - The same holds for every arm: no arm completes any such episode (A 0 of 296, B 0 of 305, C 0 of 150). The agents do not finish the four-step order they were never trained on, and say they did. This is the Stage 4 procedure-replay finding, and P1 does not remove it. Episode counts differ between arms because the stage "clean" is assigned only when no injected fault was triggered, and that depends on the agent's own actions.
  - Two further losses in this workflow: a lost acknowledgement on the goods receipt (read, then not acted on, in most episodes), and short payments where the payment is repaired but another step fails.
- **Never-commit tasks** cap VSC by design. The VSC ceiling of the test union is 0.940 (27 never-commit tasks).

## 4. Greedy and sampled decoding

Greedy decoding gives every arm higher VSC and fewer false successes than sampled decoding (tables, "greedy" sections). The registered statistic is sampled. Greedy numbers are never reported as sampled.

## 5. What this implies for P2

- **The structured ledger (B2, E)** presents payment state as fields instead of a sentence. Whether that avoids B's same-key top-up is P2's H3.
- **The action guard (rule R3, no refunded key; rule R1, read before retry)** blocks the exact stage-4 failure B shows. P2's H4 and S1–S3 test whether blocking helps.
- **The unseen workflow** is the gap no P1 arm closes. P2's shift splits measure it again.

P2 sections are added when `results/p2/p2_summary.json` exists.
