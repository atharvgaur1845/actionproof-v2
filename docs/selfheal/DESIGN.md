# Self-healing loop: design

Branch `research/self-healing`. Post-submission research. Code: `actionproof/selfheal/`, runner
`scripts/selfheal_run.py`, GPU eval wrapper `scripts/selfheal_eval.py`, tests `tests/selfheal/`, CPU jobs
`slurm/selfheal/*.sbatch`, prepared (not submitted) GPU tasks `slurm/tasks/21xx_sh_*.sh` run by
`slurm/selfheal/worker.sh` (ROOT `/scratch/pabitra/selfheal`, never the frozen project).

## Motivation

On TXN v1 (`txn_test`), agent D reaches verified completion 2864/3960 episodes (0.723, seeds pooled), while the
scripted reconciliation policy B4 reaches 380/440 (0.864) and is never wrong by STR
(`results/selfheal/diagnose/txn_test/taxonomy.md`). The loop tries to close part of that gap by training on
corrections that the simulator and the deterministic verifier have accepted, never on text judgments.

## The loop

1. **Run** the current agent on a loop pool (`sh_pool_r{1,2}_*`) with P1's episode loop (`scripts/selfheal_eval.py`,
   which calls `scripts/p1_eval.py::run` or `scripts/txn_eval.py::run_txn`).
2. **Diagnose** (`taxonomy.py`): one primary class per failed episode, from final-state verdicts, the attempts table
   and the trace. Precedence (most harmful state first): unsafe_retry, duplicate_mutation,
   wrong_or_collateral_write, partial_commit, phantom_write, then (no report) invalid_call / loop_or_timeout,
   (failed or escalated report) wrong_handoff or lost_acknowledgement, (completed report) false_completion_report.
   Multi-label flags are kept too. Each class also fixes `first_bad_idx`, the trace index where the failure shows.
3. **Correct** (`correct.py`, `teachers.py`, `replay.py`): replay the recorded conversation through the same parser
   and env (the trace must match call for call, else the episode is skipped); cut before the turn holding
   `first_bad_idx`; a teacher continues; the continuation is replayed from scratch and scored by the verifier; it is
   kept only if the env spec's verified-correct outcome holds (P1: VSC, or an honest failed report on never-commit;
   TXN: STR). If the primary cut does not verify, earlier turn boundaries are tried (`cut_kind = backoff`).
   - B4 teacher: TXN's B4 started mid-episode. It reads before writing, reuses the last idempotency key for an absent
     effect unless the visible history shows that key's effect was used up, refunds partial or extra payments, and
     escalates (TXN) or reports failed (P1) when retries run out. It sees only agent-visible calls and observations;
     a test runs it on an env handle that raises on any other attribute.
   - LLM teacher: any OpenAI-compatible endpoint (Aya Expanse 32B via local vLLM, task 2110). Not run yet.
4. **Curate** (`curate.py`): hold out by task id (hash), CuratorKIT exact + MinHash (0.95) dedup as in
   `actionproof/p1/build.py`, caps per task (2) and per class (150); writes `sft_{train,holdout}.jsonl`
   (`actionproof/train/rft.py` format, loss on teacher turns only) and `dpo_{train,holdout}.jsonl` (primary cut:
   teacher's first turn vs the agent's original turn, `actionproof/train/dpo.py` format). `mix` adds an equal number of
   the incumbent's own warm-start rows as a replay buffer.
5. **Train** (prepared only): `rft` from the same-seed incumbent, 1 epoch, lr 5e-5, LoRA r16 (P1 arm C settings);
   optional DPO. `configs/selfheal/round_v1.json`.
6. **Gate** (`gate.py`, `configs/selfheal/gate_v1.json`): candidate vs incumbent on fresh splits, paired by task,
   cluster bootstrap. Promote iff verified completion is non-inferior (lower 95% bound > -0.02), false completion
   and unsafe final state do not rise (upper 95% bound < +0.01), and OOD retention holds (lower bound > -0.05 per
   split). Writes a decision file with input and criteria hashes; a descriptive share of the D-to-B4 gap closed.

## Data isolation

- Fresh splits (`actionproof/selfheal/splits.py`, `data/bench/proc/selfheal_manifest.json`), disjoint in task id,
  PO id and prompt from every existing split: loop pools r1, r2 (120 P1 full, 120 P1 recovery, 184 TXN each); gate
  primary `sh_gate_p1full` 120, `sh_gate_p1rec` 120, `sh_gate_txn` 280; gate retention `sh_gate_txn_ood` 120,
  `sh_gate_p1wf4` 60, `sh_gate_p1combo` 60.
- Ledger (`ledger.py`, `results/selfheal/ledger.json`): gate splits are registered; `correct` skips their tasks and
  the ledger refuses them as loop data; the gate refuses any evaluated task that was corrected or trained on.
- `correct` refuses any split whose name contains "test"; `selfheal_eval.py` runs only `sh_*` splits and runs a
  gate split once per tag.
- Env interface (`envs.py`): one `EnvSpec` per simulator (P1, TXN v1). TXN v2 or the attack suite registers another.

## What ran (CPU, 2026-11-29)

**Diagnose, DESCRIPTIVE** (`results/selfheal/diagnose/*/taxonomy.{md,json}`, per-episode rows `episodes__*.jsonl.gz`),
from `/scratch/pabitra/actionproof/results` read only. TXN test, episodes (1 greedy + 2 sampled, 3 seeds):

| class | D (1082 failures / 3960) | Dwaug (1198 / 3960) |
|---|---|---|
| wrong_handoff | 460 | 279 |
| duplicate_mutation | 140 | 313 |
| wrong_or_collateral_write | 120 | 0 |
| phantom_write | 104 | 258 |
| unsafe_retry | 68 | 189 |
| partial_commit | 64 | 139 |
| false_completion_report | 63 | 14 |
| loop_or_timeout | 32 | 0 |
| invalid_call | 29 | 5 |
| lost_acknowledgement | 2 | 1 |

Notes: 432 of D's 460 wrong_handoff episodes are "failed" reports on the unresolvable counterfactual tasks, where the
label wants "escalated" (D never calls escalate_to_human); whether that is a real error is the construct question
of TXN1 addendum 3. 110 of D's 120 wrong/collateral writes are on `ood_workflow` (a step that was not requested,
mostly posting a journal entry in the no-post workflow). P1 test and dev tables are in the same directory.

**Corrections, B4 teacher, dev splits only** (`results/selfheal/r0_dev/`): every dev failure was corrected and
verified, which mostly reflects that B4 is near-perfect on these envs, not that corrections are hard to find.
- P1 dev (D_s0-2 on p1_dev_full + p1_dev_rec, 2700 episodes): 91 failures, 91 verified (78 at the primary cut,
  13 by backoff, 6 of those at cut 0, which is B4 from scratch, not a repair of the agent); outcomes 73 completed,
  18 honest failed on never-commit.
- TXN dev (D_s0, RG1 no-gate run on txn_dev, 360 episodes): 75 failures, 75 verified, all at the primary cut;
  53 of them are escalations on the unresolvable regime, 22 verified completions.
- Pilot curation (CuratorKIT available): train 45 corrections, 269 SFT rows, 43 DPO pairs (43 tasks); hold-out
  17 corrections, 100 rows, 16 pairs. Dedup is aggressive because 3 seeds x 5 episodes share tasks.
- A first pass (before the teacher stopped re-reading keyed steps) is kept on the cluster as `r0_dev_pre_readfix`.

## Known limitations

- B4 escalates on unreadable keyed steps; TXN1 addendum 3 shows a same-key resend would settle them. Training on
  these escalations teaches the labelled disposition, not the better one.
- Backoff corrections at early cuts are mostly the teacher's own episodes; curation keeps `cut_kind` so a run can
  drop them.
- Teacher targets are terse JSON calls; the LLM teacher would add variety but is untested on real failures.
- If round 2 is gated on the same gate splits as round 1, that is a second look; the draft pre-registration
  addresses it.
