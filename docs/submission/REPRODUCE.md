# Reproduce ActionProof

Start here. This page says which command reproduces which claim, what it needs, how long it takes and what a pass
looks like. The evidence that these commands work from a fresh clone (two clean-room runs) is in
[`REPRODUCIBILITY_FINAL.md`](REPRODUCIBILITY_FINAL.md). That evidence is linked here, not repeated.

**Two kinds of reproduction.**
- **CPU only (section 2).** Checks every stored result against its released episodes. It needs no GPU and no token.
- **GPU (sections 3 and 4).** Produces new episodes or new models. It needs a GPU, the pinned stack and, for the
  base model, a Hugging Face token.

---

## 1. Get the code

```bash
git clone https://github.com/atharvgaur1845/actionproof-v2.git
cd actionproof-v2
git checkout <submission commit>        # the SHA given in the submission form
git status --short                      # must print nothing
```

**Which commit.**
- The clean-room runs verified commit `319a383` of `stage2-superiority-sprint`. Later commits on that branch change
  only documents and the artefact manifest.
- On 2026-11-28 the remote `stage2-superiority-sprint` was at `85657de`.
- The remote `submission-candidate` was still at `7ae863f`, an older state. It holds the final work only after the
  author merges the final branch into it.

**Software.** Python 3.12. Package versions used in both clean rooms: numpy 2.5.5, scipy 1.18.1, PyYAML 6.0.4,
huggingface_hub 2.5.0, pytest 9.2.0, hypothesis 6.170.17, pydantic 2.14.4
(`research_audit/cleanroom/pip_freeze_and_head.txt`).

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]" huggingface_hub
```

## 2. CPU-only verification (no GPU, no token)

| Step | Command | Needs | Time (clean room) | What it reproduces |
|---|---|---|---|---|
| 2a. Smoke check | `python scripts/smoke_check.py` | CPU; no download | 27 s in both runs (about 15–30 s) | The 152 TXN and SEL1 tests (including 25 scoring mutants); all six frozen splits, regenerated from their seeds; all 11 scripted baselines on `txn_test`; the claims ledger. No model result |
| 2b. Every registered analysis | `bash scripts/verify_numbers.sh` | CPU; about 2 GB anonymous download; `git` | 367 s and 375 s | P1, P2, P3, the exploratory 2×2, P4, P5, TXN1 (with its invariants recheck and descriptive rerun), SEL1, and the impact projection, recomputed from the released episodes and compared with the committed results |
| 2c. Claims ledger alone | `python scripts/s2_numcheck.py` | CPU | seconds | Every number in the 14 scanned documents resolves through `claims/claims.yaml` to its result file |
| 2d. Post-hoc validity analysis (exploratory) | `python scripts/audit/txn_validity.py` | CPU; run after 2b (it reads the downloaded episodes) | — | `results/audit/txn_validity.json`: read-free reference policies, same-key resolution of "unresolvable" tasks, net-money re-scoring, duplicates by origin |
| 2e. Registration hashes | `python scripts/freeze_<id>.py --check` for `p1`, `p2`, `p3`, `p4`, `p5`, `txn1`, `sel1`, `j1` | CPU | seconds | The frozen files of each registration (registration text, code, task files, task scripts) still match the hashes recorded before the outcome. `freeze_sel1.py --check` also needs the SEL1 training rows from the dataset (2f) |
| 2f. SEL1 training data against the pre-training hashes | Download `data/rft/sel1_sel_rows.jsonl`, `sel1_ctl_rows.jsonl`, `sel1_rows.stats.json` and `p4_aug_rows.jsonl` from `atharvgaur18/actionproof-traces` into `data/rft/`, then `python scripts/freeze_sel1.py --check` | CPU; small download | — | The released SEL1 training rows are byte-identical to what was hashed at 05:42 IST, before the first SEL1 training run at 05:44 (`results/sel1_hashes.json`) |

The clean rooms ran `freeze_sel1.py --check` (2f) but not the other registration checks in 2e. Those run when each
registration is frozen, and again in each task script before training.

### What a pass looks like

**2a, smoke check.** The last lines:

```
[PASS] claims ledger (s2_numcheck): 866 numbers checked against files, 14 documents scanned, 0 failures
ALL PASSED in 27 s (smoke test only; model results: bash scripts/verify_numbers.sh)
```

Above them, one `[PASS]` line per check. Examples:
- `[PASS] TXN + SEL1 test suites: 152 passed`;
- `[PASS] txn_test regenerates from seed 26112603: 01437c0b341828d0`;
- `[PASS] scripted B4 on txn_test: VTC 380/440, FCR 0/440`.

**2b, `verify_numbers.sh`.** One `ok` line per analysis, then:

```
== 3. recomputed summaries versus the committed ones (git diff --exit-code results/)
identical to the committed results
== 4. claims ledger (scripts/s2_numcheck.py)
866 numbers checked against files, 14 documents scanned, 0 failures
PASS: the recomputed analyses match the committed results and every claim resolves.
```

Every bootstrap uses a fixed seed, so a rerun is deterministic. The TXN statistics sum over a Python set. The script
therefore compares its two TXN JSON outputs within 1e-9 relative and prints `TXN JSON outputs: equal up to float
noise`. If a different numpy build gives a difference above that, run it again with `ALLOW_FLOAT_NOISE=1`.

**2f, SEL1 training rows.** `OK: every present SEL1 file matches the registration`, with exit code 0 and nothing
reported absent.

**What the CPU route does not prove.** It verifies the stored results, not that a model produces those episodes.
That needs section 3.

## 3. GPU: run a released model

**Needs.**
- One GPU with 24 GB or more.
- vLLM and the pinned stack: Python 3.12, then `bash slurm/setup_env.sh`. `requirements.lock` pins torch
  2.11.0+cu130, transformers 5.22.1, trl 1.7.1, vLLM 0.23.0 and peft 0.21.2. The toolkits are pinned at AgentTune
  `d4db4c12d523`, AuditKIT `71316da34578` and CuratorKIT `dd9e835301fc`.
- Base-model runs need `HF_TOKEN`: the base is gated, so accept the Tiny Aya terms on the Hub first. The released
  agent `atharvgaur18/actionproof-agent-d` is public.

```bash
# The trained agent D on the TXN dev split: 1 greedy + 2 sampled episodes per task, about 10 minutes
python scripts/txn_eval.py --model atharvgaur18/actionproof-agent-d --tag mine_D --split txn_dev --samples 2

# The locked test: use a new tag. The script refuses to overwrite an existing tag-and-split output.
python scripts/txn_eval.py --model atharvgaur18/actionproof-agent-d --tag mine_D --split txn_test --samples 2
```

For reference, our D evaluation of `txn_test` (1,320 episodes) took 563 s on one cluster GPU
(`results/txn/D_s0__txn_test.summary.json`, field `seconds`).

**What to expect.** Episodes can differ across hardware and library versions, sampled ones especially. Compare
rates, not bytes.

Other evaluation entry points share the same episode loop (prompts, tool schemas, parser, 16 turns, 256 new tokens):
`scripts/p1_eval.py` (P1 splits) and `scripts/sel1_eval.py` (`txn_test`, `sel1_test`).

## 4. GPU: retrain

Each registered run is one task script under `slurm/tasks/`. The scripts check the registration hashes before
training (`freeze_<id>.py --check`).

| Run | Scripts | Hardware | Time |
|---|---|---|---|
| D (P1): recovery-SFT warm start Dw, then GRPO | `slurm/tasks/1153_p1_dwarm.sh`, then `1154_p1_d_s0.sh` (seeds 1 and 2: `1155`, `1156`) | 1 × H100/H200 | about 4–5 h |
| SEL1 arms | `slurm/tasks/1800_sel1_build.sh`, then `1801`–`1806` (parents: `1807`; analysis: `1812`) | 1 × H100/H200 | a task budget of 150 min per arm and seed, training plus both evaluations |

**Training data.**
- **SEL1.** Its training rows are public (2f). They were checked against their pre-training hashes in the final
  clean room (task 1861).
- **D's warm start.** `data/rft/p1_dwarm_rows.jsonl` is in neither git nor the dataset; only its hash is committed
  (`data/rft/p1_dwarm_rows.jsonl.sha256`). Task 1153 builds it as `proc_demo_rows.jsonl` +
  `p1_recovery_rows.jsonl`. The second file is in git. The first is generated by
  `python -m actionproof.train.proc_demos --out data/rft/proc_demo_rows.jsonl`.
- **Check before training:** `sha256sum -c data/rft/p1_dwarm_rows.jsonl.sha256`. We have not run this regeneration
  in a clean room.

Training was not rerun in this round. Only the analyses were reproduced from a clean clone.

## 5. Recovery Gate (RG1)

RG1 and RG2: `python scripts/rg1_stats.py`, `scripts/rg1x_stats.py`, `scripts/rg1_addendum2.py` and `scripts/rg2_stats.py` recompute the registered analyses from the released episodes (`results/rg1`, `results/rg2`); `bash scripts/verify_numbers.sh` runs the first, second and last.

## 6. Released artefacts

All are public to an anonymous request (`release/ARTIFACT_MANIFEST.json`, checked without credentials).

| Artefact | Kind | Revision | Licence |
|---|---|---|---|
| `atharvgaur18/actionproof-agent-d`: recommended agent D (merged, P1 arm D seed 0) | model | `fbf4bed` | CC-BY-NC-4.0 |
| `atharvgaur18/actionproof-agent-d-adapters`: D adapters (seeds 0–2) and the Dw adapter | model | `85a7a27` | CC-BY-NC-4.0 |
| `atharvgaur18/actionproof-warmstart-dw`: recovery-SFT warm start Dw | model | `4ce231d` | CC-BY-NC-4.0 |
| `atharvgaur18/actionproof-warmstart-dwaug`: P4 warm starts Dw_aug (seeds 0–2) | model | `21d9af5` | CC-BY-NC-4.0 |
| `atharvgaur18/actionproof-agent-a`: reference agent A (recovery-start GRPO) | model | `e09037f` | CC-BY-NC-4.0 |
| `atharvgaur18/actionproof-agent-dn`: Dn, no-std GRPO (P3, not supported) | model | `4748fd9` | CC-BY-NC-4.0 |
| `atharvgaur18/actionproof-sel1-adapters`: SEL1 arms SEL and CTL (seeds 0–2) | model | `ea96b04` | CC-BY-NC-4.0 |
| `atharvgaur18/actionproof-traces`: every episode behind a reported number, the frozen splits, judge cases, SEL1 training rows | dataset | `3d28cbf` | CC-BY-NC-4.0 |
| `atharvgaur18/actionproof-live`: live Space (real model inference) | Space | `bac5889` | CC-BY-NC-4.0 |
| `atharvgaur18/actionproof`: replay Space (recorded episodes, no model calls) | Space | `db7fc43` | CC-BY-NC-4.0 |
| `CohereLabs/tiny-aya-global`: base model (third party, gated) | model | `4694bd7` | CC-BY-NC-4.0 |

- **Collection.**
  https://huggingface.co/collections/atharvgaur18/actionproof-final-round-stage-2-6b090c4faec34bf64ccdb0f0 (10
  items).
- **Hashes.** The sha256 of every frozen split and headline result file is in the manifest.
- **Card matches.** The cards on the Hub match `release/model_cards/*.md` and `release/dataset_card.md` byte for
  byte (`REPRODUCIBILITY_FINAL.md`).
- **The D weights.** `model.safetensors` of `actionproof-agent-d` has sha256 `0e9a257c…` at `fbf4bed`, the same as at
  `5418df5`, the revision the live Space pins.

**Frozen splits** (sha256, first 16 characters; full values in `data/bench/proc/*_manifest.json`).

| Split | Tasks | sha256 |
|---|---|---|
| `txn_dev` | 120 | `f07c2a637fc7ad11` |
| `txn_val` | 120 | `3378d614af9c39e9` |
| `txn_test` (locked) | 440 | `01437c0b341828d0` |
| `sel1_train` | 336 | `696cfb26a92dc4c7` |
| `sel1_ctl_pool` | 280 | `6e6d40d4dad35974` |
| `sel1_test` (sealed) | 240 | `644f7530a250f5c8` |

## 7. Licences

- **Code:** Apache-2.0 (`LICENSE`).
- **Models and adapters:** CC-BY-NC-4.0 with Cohere attribution, plus the Cohere Labs acceptable-use policy. They
  derive from `CohereLabs/tiny-aya-global`. They are for research use only, not production payments.
- **Episode traces and judge cases:** CC-BY-NC-4.0. The agent turns are generations of the models above.
  Prompted-model episodes keep the terms of the model named in their folder.
- **Task splits:** generated by our code from synthetic data. They are Apache-2.0 in the code repository and
  CC-BY-NC-4.0 in the dataset copy.
- **Personal data:** none. All names, ids and account numbers are synthetic.
- **Third-party judge models (J1):** Gemma 4 31B, Qwen3-235B-A22B and gpt-oss-120b (Apache-2.0); DeepSeek V3.2
  (MIT) (`docs/submission/WRITEUP.md` §11).

## 8. Known limits of reproduction

- **Not rerun in a clean room.** Inference and training. Only the analyses from released episodes were.
- **Not released.**
  - The P5 SFT models (results and episodes only).
  - The P2 and AD-C1 adapters (cluster only).
  - D's warm-start training rows (only their hash, section 4).
- **The live Space has no `HF_TOKEN` secret** (author action, open on 2026-11-27). Its base-model option fails with
  the gated-model error and falls back, labelled, to the recorded base run. The trained agent and judge mode are
  unaffected.
- **Anonymous Hub downloads can be reset or rate-limited.** One clean-room download failed with
  `Connection reset by peer` and passed on rerun. Rerun, or set a read token.
- **Float noise** in the TXN statistics (2b).
- **Everything is a simulator result.** Reproducing it reproduces ProcureSim's numbers, not behaviour on a real
  payment system.

## 9. Common failures

| Symptom | Fix |
|---|---|
| `results/ has uncommitted changes` | Use a fresh clone. `verify_numbers.sh` compares against git |
| `GatedRepoError: 401` on the base model | Accept the Tiny Aya terms on the Hub and set `HF_TOKEN` |
| `IncompleteSnapshotError` or `Connection reset by peer` during the download | Rerun the command; the cache resumes |
| A float difference below 1e-9 in the TXN statistics JSON | `ALLOW_FLOAT_NOISE=1 bash scripts/verify_numbers.sh` |
| `txn_eval.py` refuses to run on `txn_test` | The tag already has an output for that split. Choose a new `--tag` |

Command reference for earlier stages: `docs/stage2/COMMANDS.md`. Other analyses (ROI model, R1 scoring, J1) are
listed in `docs/submission/WRITEUP.md` §29.
