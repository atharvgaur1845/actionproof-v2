---
license: cc-by-nc-4.0
base_model: CohereLabs/tiny-aya-global
base_model_relation: finetune
library_name: transformers
pipeline_tag: text-generation
language:
- en
tags:
- actionproof
- tool-use
- agents
- grpo
- lora-merged
- procuresim
- verification
- research-only
datasets:
- atharvgaur18/actionproof-traces
---

# ActionProof agent D (merged, seed 0)

**Recommended ActionProof agent.** A 3.35B tool-using procurement agent: `CohereLabs/tiny-aya-global` fine-tuned with recovery SFT and
then verifier-reward GRPO. This repository is a **full merged model** (seed 0 of three), the one used by the live Space
(`atharvgaur18/actionproof-live`). The adapters of all three seeds are in
[`atharvgaur18/actionproof-agent-d-adapters`](https://huggingface.co/atharvgaur18/actionproof-agent-d-adapters).

It is trained to read back its own writes, notice when a tool acknowledgement ("ok") disagrees with the records,
and repair the discrepancy (for a partial payment: refund, then re-pay with a new idempotency key, then verify)
before reporting. Code, pre-registrations and the claims ledger: https://github.com/atharvgaur1845/actionproof-v2.

## Lineage (exact parent chain)

1. `CohereLabs/tiny-aya-global` (base, 3.35B, CC-BY-NC-4.0).
2. **Dw merge:** LoRA r16 recovery-SFT warm start, trained on the original warm-start demonstrations plus verified
   oracle recovery demonstrations (`runs/p1_dwarm/final`, 2 epochs, lr 1e-4, seed 0), merged into the base
   (`runs/p1_dwarm/merged`). Adapter: [`atharvgaur18/actionproof-warmstart-dw`](https://huggingface.co/atharvgaur18/actionproof-warmstart-dw).
3. **D merge:** LoRA r16 trained by GRPO on top of the merged Dw model (`runs/p1_d_s0/final`; reward R_V, state-based;
   200 steps, 4 prompts x 8 rollouts, lr 2e-5, beta 0.02, T = 1; pools `s3_trainb_half` + `s3_rec_train_half`),
   merged into it with `actionproof/train/merge.py` (bf16). This is the checkpoint `scripts/p1_eval.py` evaluated
   as `D_s0` (`--model runs/p1_dwarm/merged --adapter runs/p1_d_s0/final`).

## Tool-call format and system prompt

The model is trained and evaluated in AgentTune's Cohere fallback format. There is no chat-template tool block.
- **System prompt** = ProcureSim policy text (`actionproof.integrity.sim.prompts.PROC_NEUTRAL`, also in
  `data/bench/proc/system_prompt.txt` of the code repository) + a JSON tool-schema block built by
  `actionproof.rollout.episode.system_prompt(env, PROC_NEUTRAL)`. That function uses AgentTune's prompt and schema
  builder when AgentTune is installed, and a dependency-free copy (`actionproof/rollout/lite_format.py`) otherwise.
- **The model answers with one JSON list per turn and nothing else:**
  `[{"tool_name": "pay_supplier", "parameters": {"po_id": "PO-1A2B3C", "amount_cents": 1000, "idempotency_key": "pay-PO-1A2B3C"}}]`
- **Tool results come back as a user turn:**
  `Tool result for <name>: <json>` followed by `Continue the task. Call submit_report when you are finished.`
- **The episode ends** when the model calls `submit_report` with status `completed`, `failed` or `escalated`, or
  when a turn has no parseable call (16 turns, 256 new tokens per turn in the evaluation).
- The ProcureSim tools are `get_purchase_order`, `find_supplier_order`, `place_supplier_order`, `record_goods_receipt`,
  `list_goods_receipts`, `pay_supplier`, `list_payments`, `refund_payment`, `post_journal_entry`,
  `get_journal_entries`, `escalate_to_human`, `submit_report` (`actionproof/integrity/sim/env.py`).

## Usage (one turn)

```python
# pip install torch transformers; git clone https://github.com/atharvgaur1845/actionproof-v2 && cd actionproof-v2 && pip install -e .
import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from actionproof.bench.generator_proc import DATA, load_tasks
from actionproof.stage3.env import make_env_s3
from actionproof.integrity.sim.prompts import PROC_NEUTRAL
from actionproof.rollout.episode import system_prompt, _agenttune

repo = "atharvgaur18/actionproof-agent-d"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.bfloat16, device_map="auto")

task = load_tasks(DATA / "p1_test_full.jsonl")[0]          # a frozen-test task; any ProcureSim task works
env = make_env_s3(task, "card-demo")                        # the simulator and its verifier (CPU, deterministic)
conv = [{"role": "system", "content": system_prompt(env, PROC_NEUTRAL)},
        {"role": "user", "content": task.prompt}]
text = tok.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
ids = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
out = model.generate(**ids, max_new_tokens=256, do_sample=False)
reply = tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
print(reply)                                                # e.g. [{"tool_name": "record_goods_receipt", ...}]
_, _, parse = _agenttune()
for c in parse(reply) or []:
    print(env.call(c["function"]["name"], json.loads(c["function"]["arguments"] or "{}")))
```

For a full episode loop use `scripts/p1_eval.py` (below) or `actionproof.rollout.episode.run_episode`.

## Results (frozen P1 test, 450 tasks, 3 seeds, sampled VSC)

VSC = verified-success completion: the goal holds in the database after the episode and nothing forbidden happened.
Sampled = 4 episodes per task at T = 1, averaged per task, then over the 3 seeds of the arm. Numbers are the
3-seed arm D, of which this repository is seed 0.

**By split (arm means)**

| Model | full | pp | rec | wf4 | combo | union (450) |
|---|---|---|---|---|---|---|
| W: RFT warm start | 0.583 | 0.017 | 0.421 | 0.183 | 0.296 | 0.335 |
| A: recovery-start GRPO | 0.706 | 0.319 | 0.624 | 0.174 | 0.535 | 0.513 |
| Dw: recovery SFT only | 0.706 | 0.614 | 0.881 | 0.325 | 0.671 | 0.679 |
| **D** (this model's arm) | 0.810 | 0.931 | 0.909 | 0.486 | 0.824 | 0.819 |

**Contrasts with 95% paired task-bootstrap intervals** (450 tasks, B = 10,000)

| Contrast | Status | Delta VSC [95% interval] |
|---|---|---|
| D - A | registered P1-H3, **supported** (Holm p 0.0003, seed deltas all positive: +0.293, +0.292, +0.334) | +0.306 [+0.276, +0.337] |
| C - A | registered P1-H2, supported (C and D are not nested) | +0.259 [+0.231, +0.287] |
| B - A | registered P1-H1, **not supported** (reconciliation line at inference) | -0.030 [-0.041, -0.019] |
| D - Dw | exploratory: GRPO increment on recovery SFT | +0.140 [+0.121, +0.160] |
| Dw - W | exploratory: recovery SFT without RL | +0.344 [+0.311, +0.378] |
| Dn - D | registered P3-H1, **null** (no-std GRPO) | -0.007 [-0.013, -0.001] |

Union means: A 0.513, D 0.819 (`results/p1/p1_summary.json`, `test.D`). Per-seed union sampled VSC of D:
s0 0.823, s1 0.806, s2 0.829; this repository is s0, for which no separate interval is stored.

**Behaviour of arm D by split (sampled, seed mean)**

| Split | VSC | false success | partial payment repaired | duplicate unrefunded |
|---|---|---|---|---|
| full | 0.810 | 0.028 | 0.981 | 0.002 |
| pp | 0.931 | 0.058 | 0.931 | 0.003 |
| rec | 0.909 | 0.020 | 0.994 | 0.009 |
| wf4 | 0.486 | 0.474 | 0.417 | 0.058 |
| combo | 0.824 | 0.167 | 0.779 | 0.010 |

Reading it: the partial-payment cell that Stage 3 left unsolved is repaired on most episodes here, but the unseen
four-step workflow (`wf4`) and two-fault episodes (`combo`) remain hard, and false success is not zero on them.

## Limitations

- **Simulator, not production payments.** Every result is on ProcureSim, an in-memory procurement simulator with injected
  write faults (phantom, partial, lost-acknowledgement, duplicate, never-commit). No real payment system, real money
  or real supplier was involved, and nothing here shows the agent is safe to deploy.
- **The RL agents (A, D, Dn) fail on renamed tools and on the four-step workflow.** Recovery-start training with randomised
  tool names (registered SR1, `results/s4/sr1_summary.json`) did not help on the renamed API (+0.000 [-0.040, +0.042]) and cost VSC on
  the canonical tools (-0.248 [-0.296, -0.201]).
  On the unseen four-step workflow (`wf4`, order, receive, pay, post) the models run the trained three-step sequence
  and skip the new step (the `p1_test_wf4` split of `results/p1/p1_summary.json`).
  Randomising the **demonstrations** instead (registered P4, `results/p4/p4_summary.json`; warm starts only, no GRPO
  on top) raises the warm start on the tested renamed-tool surface and four-step workflow: +0.604 [+0.540, +0.666] and +0.574 [+0.536, +0.610], P1 test +0.048 [+0.026, +0.072]
  (Dw_aug vs Dw, 3 seeds each; [`atharvgaur18/actionproof-warmstart-dwaug`](https://huggingface.co/atharvgaur18/actionproof-warmstart-dwaug)).
- **Arms C and D are not nested.** C is the recovery-start agent A plus recovery SFT. D starts from its own
  recovery-SFT warm start (Dw) and then runs GRPO. The C to D gap is therefore not GRPO's contribution; the
  exploratory 2x2 (`results/audit/dwarm_2x2.json`) is the right reading of what GRPO adds on top of SFT.
- **P3 is not supported.** Removing the group standard deviation from the GRPO advantage (no-std GRPO, agent Dn)
  leaves it slightly below D (registered P3-H1, `results/p3/p3_summary.json`).
- **Intervals are over tasks only.** Paired task bootstrap over 450 tasks (B = 10,000) holds the three trained
  models fixed; with three seeds, seed-level power is low. Per-split cells carry no stored interval.
- One base model (3.35B), one simulated domain, English-language training; Hindi and Hinglish were evaluated as shifts
  only (`docs/submission/WRITEUP.md` section 25).
- Non-commercial licence inherited from the base model.

## Intended use

Research only: reproducing the ActionProof study, probing how training signals shape verification and recovery
behaviour in tool agents, and benchmarking against ProcureSim. Not for production financial operations, not for
automated payment decisions, and not for commercial use (CC-BY-NC-4.0, plus the Cohere Labs acceptable-use policy).

## Citation

```bibtex
@misc{actionproof2026,
  title        = {ActionProof: what reinforcement learning teaches tool agents when "ok" does not mean it happened},
  author       = {atharvgaur18},
  year         = {2026},
  howpublished = {\url{https://github.com/atharvgaur1845/actionproof-v2}},
  note         = {Model atharvgaur18/actionproof-agent-d; code, claims ledger and pre-registrations in the repository}
}
```

## How to evaluate

Frozen P1 test: 450 tasks over five splits, 4 sampled episodes (T = 1) plus one greedy episode per task, 16 turns,
256 new tokens, scored by the deterministic verifier (verified-success completion, VSC). Needs vLLM and a GPU.

```bash
git clone https://github.com/atharvgaur1845/actionproof-v2 && cd actionproof-v2          # data/bench/proc/p1_test_*.jsonl are in the repository
python scripts/p1_eval.py --model atharvgaur18/actionproof-agent-d --tag <tag> \
    --split p1_test_full --phase test --samples 4      # repeat for p1_test_pp, p1_test_rec, p1_test_wf4, p1_test_combo
```

`runs/...` is the cluster layout of the original runs. With the public repositories, pass a downloaded directory
(`huggingface-cli download <repo> --local-dir <dir>`); for an adapter repository `--model` is the merged parent
(see Lineage; the tool merges the adapter itself, as `actionproof/train/merge.py`) and `--adapter` is `<dir>/<sub-folder>`.
Evaluate seeds 0, 1, 2 for the three-seed arm (`--tag <ARM>_s0` and so on).

`scripts/p1_eval.py` refuses to overwrite a frozen-test output, so write to a fresh checkout or use `--phase dev` on
`p1_dev_*` splits. The recorded episodes behind every number on this card are in the dataset
[`atharvgaur18/actionproof-traces`](https://huggingface.co/datasets/atharvgaur18/actionproof-traces); `scripts/verify_numbers.sh`
recomputes the summaries from them on a CPU.

This merged checkpoint equals the cluster evaluation of `--model runs/p1_dwarm/merged --adapter runs/p1_d_s0/final` (tag `D_s0`).

*Sources: results/p1/p1_summary.json, results/p1/test/D_s*__*.summary.json, results/audit/dwarm_2x2.json, results/p3/p3_summary.json. Numbers on this card were generated from those files by `release/build_cards.py`; none is typed by hand.*
