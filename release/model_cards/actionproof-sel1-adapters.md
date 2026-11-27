---
license: cc-by-nc-4.0
base_model: CohereLabs/tiny-aya-global
base_model_relation: adapter
library_name: peft
pipeline_tag: text-generation
language:
- en
tags:
- actionproof
- tool-use
- agents
- sft
- lora
- procuresim
- escalation
- research-only
datasets:
- atharvgaur18/actionproof-traces
---

# ActionProof SEL1: escalation adapters (registered experiment, SFT only)

Six LoRA (r16) adapters from the registered SEL1 experiment (`PREREGISTRATION_SEL1.md`, registered before any
training): `sel/s0..s2` (arm SEL) and `ctl/s0..s2` (arm CTL; `runs/sel1_{SEL,CTL}_s{k}/final`). Each continues
from the merged Dw_aug seed-k warm start
([`atharvgaur18/actionproof-warmstart-dwaug`](https://huggingface.co/atharvgaur18/actionproof-warmstart-dwaug) `s{k}/`), with a
new LoRA, 1 epoch, lr 1e-4, batch 2 × 8, max length 6144 (`actionproof.train.rft`).
- **SEL** sees demonstrations from the scripted selective rule B5: escalate to a person when the faulted write's
  outcome cannot be read back for the whole episode; re-read up to three times through a transient outage. Plus
  2,000 replay rows from Dw_aug's data.
- **CTL** sees the same number of resolvable demonstrations plus the same replay.

**No RL was run on these adapters**, and no model selection was done.

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

The SEL1 environments add `escalate_to_human` outcomes and transient read outages: `actionproof/sel1/env.py`.

## Usage (one turn)

The adapters are LoRA (r16) on a *parent*, which is not always the public base model (see Lineage). Merge the
parent chain first, then attach the adapter. Parent chain and merge helper: `actionproof/train/merge.py`.

```python
# pip install torch transformers peft; git clone https://github.com/atharvgaur1845/actionproof-v2 && cd actionproof-v2 && pip install -e .
import json, torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from actionproof.bench.generator_proc import DATA, load_tasks
from actionproof.stage3.env import make_env_s3
from actionproof.integrity.sim.prompts import PROC_NEUTRAL
from actionproof.rollout.episode import system_prompt, _agenttune

repo = "atharvgaur18/actionproof-sel1-adapters"
local = snapshot_download(repo)
# parent = the base model with the Dw_aug seed-k adapter merged in (k = the seed of the sub-folder)
k = 0
parent = AutoModelForCausalLM.from_pretrained("CohereLabs/tiny-aya-global", torch_dtype=torch.bfloat16, device_map="auto")
parent = PeftModel.from_pretrained(parent, f"{snapshot_download('atharvgaur18/actionproof-warmstart-dwaug')}/s{k}").merge_and_unload()
tok = AutoTokenizer.from_pretrained("CohereLabs/tiny-aya-global")
model = PeftModel.from_pretrained(parent, f"{local}/sel/s0")   # sub-folder of this repository

task = load_tasks(DATA / "p1_test_full.jsonl")[0]
env = make_env_s3(task, "card-demo")
conv = [{"role": "system", "content": system_prompt(env, PROC_NEUTRAL)},
        {"role": "user", "content": task.prompt}]
text = tok.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
ids = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
out = model.generate(**ids, max_new_tokens=256, do_sample=False)
reply = tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
print(reply)
_, _, parse = _agenttune()
for c in parse(reply) or []:
    print(env.call(c["function"]["name"], json.loads(c["function"]["arguments"] or "{}")))
```

## Results (registered SEL1; arm = mean of 3 seed models, 3 episodes per task, cluster bootstrap B = 10,000)

| Test | SEL | CTL | SEL − CTL [95% CI] | Holm p | Seed differences | Result |
|---|---|---|---|---|---|---|
| H1 escalation, txn_test unresolvable (60 tasks) | 0.972 | 0.002 | +0.970 [+0.952, +0.987] | 0.0003 | +0.967, +0.961, +0.983 | supported |
| H2 escalation, held-out step and workflows (80; reported on its own) | 0.624 | 0.006 | +0.618 [+0.486, +0.754] | 0.0003 | +0.629, +0.821, +0.404 | passes |
| H3 selectivity J (200) | 0.740 | 0.007 | +0.732 [+0.625, +0.836] | 0.0003 | +0.747, +0.856, +0.594 | supported |
| H4 VTC, txn_test resolvable (380; non-inferiority) | 0.960 | 0.958 | +0.002 [-0.006, +0.010] | bound -0.03 | +0.003, +0.004, -0.002 | supported |

**SEL1 is supported** (the rule: H1, H3 and H4). Scripted
selectivity J: B5 1, escalate-on-first-error B4 0.5, always-escalate 0, never-escalate 0.

**Descriptive** (episodes, 3 seeds pooled):
- Transient outage of held-out length: SEL escalates in 0 of 360 and finishes
  safely in 350 of 360 (CTL 264 of 360).
- Audit-log-only outage: SEL escalates in 0 of 360.
- **Held-out step `receive`, the weak spot:** SEL escalates in 146 of 360
  and is unsafe in 188 of 360 (CTL 275 of 360).
  Seed 2 escalates in 2 of 120 and is unsafe in 112, more than its control (83).
- txn_test VTC over all 440 tasks: SEL 3285 of 3960, CTL
  3451 of 3960. CTL completes 173 of 540
  unresolvable episodes without the evidence to know it; SEL hands them off.

## Limitations

- **Simulator, not production payments.** ProcureSim is an in-memory procurement simulator; nothing here shows the agent is
  safe to deploy.
- **Supervised, from a scripted rule.** The behaviour is taught by demonstrations of B5; it is not an RL result, and
  the scripted rule itself still scores higher on selectivity.
- **Seed-fragile transfer to a held-out step** (above).
- **Narrower than designed (post hoc, `PREREGISTRATION_SEL1_ADDENDUM1.md`).**
  - Every locked-test H1 task faults a keyed step, which a same-key resend settles. SEL's escalations there mostly
    replace an honest "failed" report; unsafe actions fall from 12 to 7 of 540.
  - The audit-log-only control cannot fire, and the 2-read outage is inside the demonstrated 3-read budget.
  - Where hand-off is the only safe action (unkeyed receive steps), SEL hands off in 164 of 432 episodes (by seed: 55,
    106, 3 of 144).
  - The trigger is an explicit `service_unavailable` read error.
  - The benchmark's own limits are in `PREREGISTRATION_TXN1_ADDENDUM3.md`.
- The cost of a human review is not measured; escalation is scored as safe only where the outcome is truly
  unobservable (`actionproof/txn/truth.py`).
- Non-commercial licence inherited from the base model. Research use only.

## How to evaluate

```bash
git clone https://github.com/atharvgaur1845/actionproof-v2 && cd actionproof-v2
python scripts/sel1_eval.py --model <merged Dw_aug s0 dir> --adapter <dir>/sel/s0 --tag SEL_s0 --split sel1_test
python scripts/sel1_eval.py --model <merged Dw_aug s0 dir> --adapter <dir>/sel/s0 --tag SEL_s0 --split txn_test
python scripts/sel1_stats.py      # the registered analysis, from results/sel1/
```

*Sources: results/sel1/sel1_summary.json. Numbers on this card were generated from those files by `release/build_cards.py`; none is typed by hand.*
