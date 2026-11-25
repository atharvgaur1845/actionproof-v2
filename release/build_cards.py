"""Regenerate release/model_cards/*.md from the result files, so no number in a card is typed by hand.

    python release/build_cards.py            # writes release/model_cards/<repo>.md

Sources (all committed in git):
- results/p1/p1_summary.json          registered P1 contrasts (B, C, D versus A), per-split descriptive metrics
- results/p1/test/*.summary.json      per-model, per-split sampled VSC (n episodes per split gives the union weights)
- results/audit/dwarm_2x2.json        W / A / Dw / D / Dn arm means per split and the exploratory 2x2 contrasts
- results/p3/p3_summary.json          registered P3 contrast (Dn - D)
Needs only the standard library.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "release/model_cards"
USER = "atharvgaur18"
CODE = "https://github.com/atharvgaur1845/actionproof-v2"
BASE = "CohereLabs/tiny-aya-global"
SPLITS = ("p1_test_full", "p1_test_pp", "p1_test_rec", "p1_test_wf4", "p1_test_combo")
SPLIT_NOTE = {"p1_test_full": "full episodes (120 tasks)", "p1_test_pp": "partial payment, unseen ratios 0.4 and 0.6 (90)",
              "p1_test_rec": "recovery starts (120)", "p1_test_wf4": "unseen four-step workflow (60)",
              "p1_test_combo": "two faults in one episode (60)"}


def J(p: str):
    return json.loads((ROOT / p).read_text())


P1 = J("results/p1/p1_summary.json")
P3 = J("results/p3/p3_summary.json")
W2 = J("results/audit/dwarm_2x2.json")


def f3(x: float) -> str:
    return f"{x:.3f}"


def sg(x: float) -> str:
    return f"{x:+.3f}"


def ci(c: dict) -> str:
    return f"{sg(c['delta'])} [{sg(c['ci95'][0])}, {sg(c['ci95'][1])}]"


def model_union(tag: str, directory: str = "results/p1/test") -> float | None:
    """Sampled VSC of one trained model on the 450-task union, weighted by episodes per split."""
    num = den = 0
    for sp in SPLITS:
        p = ROOT / directory / f"{tag}__{sp}.summary.json"
        if not p.exists():
            return None
        s = json.loads(p.read_text())["sampled"]
        num += s["vsc"] * s["n"]
        den += s["n"]
    return num / den


def per_split_table(arms: list[tuple[str, str]]) -> str:
    """arms: (label, key in dwarm_2x2 arm_means). Rows are the arms, columns the five splits and the union."""
    head = "| Model | " + " | ".join(sp.replace("p1_test_", "") for sp in SPLITS) + " | union (450) |\n"
    head += "|---|" + "---|" * (len(SPLITS) + 1) + "\n"
    rows = ""
    for label, k in arms:
        cells = [f3(W2["per_split"][sp]["arm_means"][k]) for sp in SPLITS]
        rows += f"| {label} | " + " | ".join(cells) + f" | {f3(W2['arm_means'][k])} |\n"
    return head + rows


def seed_line(arm: str) -> str:
    vals = [model_union(f"{arm}_s{s}") for s in range(3)]
    return ", ".join(f"s{s} {f3(v)}" for s, v in enumerate(vals))


def behaviour_table(arm: str) -> str:
    d = P1["descriptive"][arm]["test"]
    out = "| Split | VSC | false success | partial payment repaired | duplicate unrefunded |\n|---|---|---|---|---|\n"
    for sp in SPLITS:
        s = d[sp]["sampled"]
        out += (f"| {sp.replace('p1_test_', '')} | {f3(s['vsc'])} | {f3(s['false_success'])} | "
                f"{f3(s['partial_pay_repaired'])} | {f3(s['duplicate_unrefunded'])} |\n")
    return out


def dw_behaviour_table() -> str:
    out = "| Split | VSC | false success | partial payment repaired | duplicate unrefunded |\n|---|---|---|---|---|\n"
    for sp in SPLITS:
        s = J(f"results/p1/test/Dw__{sp}.summary.json")["sampled"]
        out += (f"| {sp.replace('p1_test_', '')} | {f3(s['vsc'])} | {f3(s['false_success'])} | "
                f"{f3(s['partial_pay_repaired'])} | {f3(s['duplicate_unrefunded'])} |\n")
    return out


FRONT = """---
license: cc-by-nc-4.0
base_model: {base}
base_model_relation: {rel}
library_name: {lib}
pipeline_tag: text-generation
language:
- en
tags:
{tags}
datasets:
- {user}/actionproof-traces
---
"""

HOW_TO_PROMPT = """## Tool-call format and system prompt

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
"""

USAGE = '''## Usage (one turn)

```python
# pip install torch transformers; git clone {code} && cd actionproof-v2 && pip install -e .
import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from actionproof.bench.generator_proc import DATA, load_tasks
from actionproof.stage3.env import make_env_s3
from actionproof.integrity.sim.prompts import PROC_NEUTRAL
from actionproof.rollout.episode import system_prompt, _agenttune

repo = "{repo}"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.bfloat16, device_map="auto")

task = load_tasks(DATA / "p1_test_full.jsonl")[0]          # a frozen-test task; any ProcureSim task works
env = make_env_s3(task, "card-demo")                        # the simulator and its verifier (CPU, deterministic)
conv = [{{"role": "system", "content": system_prompt(env, PROC_NEUTRAL)}},
        {{"role": "user", "content": task.prompt}}]
text = tok.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
ids = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
out = model.generate(**ids, max_new_tokens=256, do_sample=False)
reply = tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
print(reply)                                                # e.g. [{{"tool_name": "record_goods_receipt", ...}}]
_, _, parse = _agenttune()
for c in parse(reply) or []:
    print(env.call(c["function"]["name"], json.loads(c["function"]["arguments"] or "{{}}")))
```

For a full episode loop use `scripts/p1_eval.py` (below) or `actionproof.rollout.episode.run_episode`.
'''

ADAPTER_USAGE = '''## Usage (one turn)

The adapters are LoRA (r16) on a *parent*, which is not always the public base model (see Lineage). Merge the
parent chain first, then attach the adapter. Parent chain and merge helper: `actionproof/train/merge.py`.

```python
# pip install torch transformers peft; git clone {code} && cd actionproof-v2 && pip install -e .
import json, torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from actionproof.bench.generator_proc import DATA, load_tasks
from actionproof.stage3.env import make_env_s3
from actionproof.integrity.sim.prompts import PROC_NEUTRAL
from actionproof.rollout.episode import system_prompt, _agenttune

repo = "{repo}"
local = snapshot_download(repo)
{chain}
tok = AutoTokenizer.from_pretrained("{base}")
model = PeftModel.from_pretrained(parent, f"{{local}}/{sub}")   # sub-folder of this repository

task = load_tasks(DATA / "p1_test_full.jsonl")[0]
env = make_env_s3(task, "card-demo")
conv = [{{"role": "system", "content": system_prompt(env, PROC_NEUTRAL)}},
        {{"role": "user", "content": task.prompt}}]
text = tok.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
ids = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
out = model.generate(**ids, max_new_tokens=256, do_sample=False)
reply = tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
print(reply)
_, _, parse = _agenttune()
for c in parse(reply) or []:
    print(env.call(c["function"]["name"], json.loads(c["function"]["arguments"] or "{{}}")))
```
'''

CHAIN_BASE = '''parent = AutoModelForCausalLM.from_pretrained("{base}", torch_dtype=torch.bfloat16, device_map="auto")'''
CHAIN_DW = '''# parent = the base model with the Dw adapter merged in (the "warm start")
parent = AutoModelForCausalLM.from_pretrained("{base}", torch_dtype=torch.bfloat16, device_map="auto")
parent = PeftModel.from_pretrained(parent, snapshot_download("{user}/actionproof-warmstart-dw")).merge_and_unload()'''
CHAIN_W = '''# parent = the base model with the W adapter merged in (the RFT warm start, shipped in this repository)
parent = AutoModelForCausalLM.from_pretrained("{base}", torch_dtype=torch.bfloat16, device_map="auto")
parent = PeftModel.from_pretrained(parent, f"{{local}}/w").merge_and_unload()'''

EVAL = """## How to evaluate

Frozen P1 test: 450 tasks over five splits, 4 sampled episodes (T = 1) plus one greedy episode per task, 16 turns,
256 new tokens, scored by the deterministic verifier (verified-success completion, VSC). Needs vLLM and a GPU.

```bash
git clone {code} && cd actionproof-v2          # data/bench/proc/p1_test_*.jsonl are in the repository
python scripts/p1_eval.py --model {model_arg}{adapter_arg} --tag <tag> \\
    --split p1_test_full --phase test --samples 4      # repeat for p1_test_pp, p1_test_rec, p1_test_wf4, p1_test_combo
```

`runs/...` is the cluster layout of the original runs. With the public repositories, pass a downloaded directory
(`huggingface-cli download <repo> --local-dir <dir>`); for an adapter repository `--model` is the merged parent
(see Lineage; the tool merges the adapter itself, as `actionproof/train/merge.py`) and `--adapter` is `<dir>/<sub-folder>`.
Evaluate seeds 0, 1, 2 for the three-seed arm (`--tag <ARM>_s0` and so on).

`scripts/p1_eval.py` refuses to overwrite a frozen-test output, so write to a fresh checkout or use `--phase dev` on
`p1_dev_*` splits. The recorded episodes behind every number on this card are in the dataset
[`{user}/actionproof-traces`](https://huggingface.co/datasets/{user}/actionproof-traces); `scripts/verify_numbers.sh`
recomputes the summaries from them on a CPU.
"""

HONESTY = """\n## Limitations

- **Simulator, not production payments.** Every result is on ProcureSim, a SQLite procurement simulator with injected
  write faults (phantom, partial, lost-acknowledgement, duplicate, never-commit). No real payment system, real money
  or real supplier was involved, and nothing here shows the agent is safe to deploy.
- **Generalisation fails on renamed tools and on the four-step workflow.** Recovery-start training with randomised
  tool names (registered SR1, `results/s4/sr1_summary.json`) did not help on the renamed API ({sr1_h1}) and cost VSC on
  the canonical tools ({sr1_h2}).
  On the unseen four-step workflow (`wf4`, order, receive, pay, post) the models run the trained three-step sequence
  and skip the new step; the table above shows the wf4 column far below the others.
- **Arms C and D are not nested.** C is the recovery-start agent A plus recovery SFT. D starts from its own
  recovery-SFT warm start (Dw) and then runs GRPO. The C to D gap is therefore not GRPO's contribution; the
  exploratory 2x2 (`results/audit/dwarm_2x2.json`) is the right reading of what GRPO adds on top of SFT.
- **P3 is a null result.** Removing the group standard deviation from the GRPO advantage (no-std GRPO, agent Dn) did
  not improve on D (registered P3-H1, `results/p3/p3_summary.json`).
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
@misc{{actionproof2026,
  title        = {{ActionProof: what reinforcement learning teaches tool agents when "ok" does not mean it happened}},
  author       = {{{author}}},
  year         = {{2026}},
  howpublished = {{\\url{{{code}}}}},
  note         = {{Model {repo}; code, claims ledger and pre-registrations in the repository}}
}}
```
"""

SOURCES = """
*Sources: {src}. Numbers on this card were generated from those files by `release/build_cards.py`; none is typed by hand.*
"""


def regen_note(srcs: str) -> str:
    return SOURCES.format(src=srcs)



SR1 = J("results/s4/sr1_summary.json")
SR1_H1 = ci({"delta": SR1["SR-H1"]["seed_avg"]["est"], "ci95": SR1["SR-H1"]["seed_avg"]["ci95"]})
SR1_H2 = ci({"delta": SR1["SR-H2"]["seed_avg"]["est"], "ci95": SR1["SR-H2"]["seed_avg"]["ci95"]})


def honesty(repo: str) -> str:
    return HONESTY.format(code=CODE, repo=repo, author="atharvgaur18", sr1_h1=SR1_H1, sr1_h2=SR1_H2)


def card_d() -> str:
    repo = f"{USER}/actionproof-agent-d"
    d, c = P1["test"]["D"], P1["test"]["C"]
    c2 = W2["contrasts"]
    return FRONT.format(base=BASE, rel="finetune", lib="transformers", user=USER,
                        tags="\n".join(f"- {t}" for t in ["actionproof", "tool-use", "agents", "grpo", "lora-merged", "procuresim",
                                                          "verification", "research-only"])) + f"""
# ActionProof agent D (merged, seed 0)

**Recommended ActionProof agent.** A 3.35B tool-using procurement agent: `{BASE}` fine-tuned with recovery SFT and
then verifier-reward GRPO. This repository is a **full merged model** (seed 0 of three), the one used by the live Space
(`{USER}/actionproof`). The adapters of all three seeds are in
[`{USER}/actionproof-agent-d-adapters`](https://huggingface.co/{USER}/actionproof-agent-d-adapters).

It is trained to read back its own writes, notice when a tool acknowledgement ("ok") disagrees with the records,
and repair the discrepancy (for a partial payment: refund, then re-pay with a new idempotency key, then verify)
before reporting. Code, pre-registrations and the claims ledger: {CODE}.

## Lineage (exact parent chain)

1. `{BASE}` (base, 3.35B, CC-BY-NC-4.0).
2. **Dw merge:** LoRA r16 recovery-SFT warm start, trained on the original warm-start demonstrations plus verified
   oracle recovery demonstrations (`runs/p1_dwarm/final`, 2 epochs, lr 1e-4, seed 0), merged into the base
   (`runs/p1_dwarm/merged`). Adapter: [`{USER}/actionproof-warmstart-dw`](https://huggingface.co/{USER}/actionproof-warmstart-dw).
3. **D merge:** LoRA r16 trained by GRPO on top of the merged Dw model (`runs/p1_d_s0/final`; reward R_V, state-based;
   200 steps, 4 prompts x 8 rollouts, lr 2e-5, beta 0.02, T = 1; pools `s3_trainb_half` + `s3_rec_train_half`),
   merged into it with `actionproof/train/merge.py` (bf16). This is the checkpoint `scripts/p1_eval.py` evaluated
   as `D_s0` (`--model runs/p1_dwarm/merged --adapter runs/p1_d_s0/final`).

{HOW_TO_PROMPT}
{USAGE.format(code=CODE, repo=repo)}
## Results (frozen P1 test, 450 tasks, 3 seeds, sampled VSC)

VSC = verified-success completion: the goal holds in the database after the episode and nothing forbidden happened.
Sampled = 4 episodes per task at T = 1, averaged per task, then over the 3 seeds of the arm. Numbers are the
3-seed arm D, of which this repository is seed 0.

**By split (arm means)**

{per_split_table([("W: RFT warm start", "W"), ("A: recovery-start GRPO", "A"), ("Dw: recovery SFT only", "Dw"), ("**D** (this model's arm)", "D")])}
**Contrasts with 95% paired task-bootstrap intervals** (450 tasks, B = 10,000)

| Contrast | Status | Delta VSC [95% interval] |
|---|---|---|
| D - A | registered P1-H3, **supported** (Holm p {d['holm_p']:.4f}, seed deltas all positive: {', '.join(sg(x) for x in d['seed_deltas'])}) | {ci(d)} |
| C - A | registered P1-H2, supported (C and D are not nested) | {ci(c)} |
| B - A | registered P1-H1, **not supported** (reconciliation line at inference) | {ci(P1['test']['B'])} |
| D - Dw | exploratory: GRPO increment on recovery SFT | {ci(c2['D-Dw (GRPO on recovery SFT)'])} |
| Dw - W | exploratory: recovery SFT without RL | {ci(c2['Dw-W (SFT, no RL)'])} |
| Dn - D | registered P3-H1, **null** (no-std GRPO) | {ci(P3['H1'])} |

Union means: A {f3(d['A_mean'])}, D {f3(d['X_mean'])} (`results/p1/p1_summary.json`, `test.D`). Per-seed union sampled VSC of D:
{seed_line('D')}; this repository is s0, for which no separate interval is stored.

**Behaviour of arm D by split (sampled, seed mean)**

{behaviour_table('D')}
Reading it: the partial-payment cell that Stage 3 left unsolved is repaired on most episodes here, but the unseen
four-step workflow (`wf4`) and two-fault episodes (`combo`) remain hard, and false success is not zero on them.
{honesty(repo)}
{EVAL.format(code=CODE, user=USER, model_arg=repo, adapter_arg="")}
This merged checkpoint equals the cluster evaluation of `--model runs/p1_dwarm/merged --adapter runs/p1_d_s0/final` (tag `D_s0`).
{regen_note("results/p1/p1_summary.json, results/p1/test/D_s*__*.summary.json, results/audit/dwarm_2x2.json, results/p3/p3_summary.json")}"""


def card_d_adapters() -> str:
    repo = f"{USER}/actionproof-agent-d-adapters"
    c2 = W2["contrasts"]
    d = P1["test"]["D"]
    return FRONT.format(base=BASE, rel="adapter", lib="peft", user=USER,
                        tags="\n".join(f"- {t}" for t in ["actionproof", "tool-use", "agents", "grpo", "lora", "procuresim", "research-only"])) + f"""
# ActionProof agent D: LoRA adapters (seeds 0, 1, 2) and the Dw warm-start adapter

Four LoRA (r16) adapters. The merged seed-0 model is [`{USER}/actionproof-agent-d`](https://huggingface.co/{USER}/actionproof-agent-d)
(the recommended single download); this repository is for the three-seed reproduction and for anyone who wants to
merge themselves.

| Folder | Adapter | Parent it must be loaded on |
|---|---|---|
| `dw/` | recovery-SFT warm start Dw (`runs/p1_dwarm/final`) | `{BASE}` |
| `s0/`, `s1/`, `s2/` | D, seed k: GRPO on the merged Dw model (`runs/p1_d_s{{k}}/final`) | **merged Dw** = base + `dw/` |

`LINEAGE.json` in each folder records the original `base_model_name_or_path` of the cluster run, the real parent and
the merge command. In the uploaded `adapter_config.json` the field `base_model_name_or_path` is rewritten to
`{BASE}` (the cluster path would not resolve); **the `s*/` adapters are not valid on the raw base**, only on merged Dw.

## Lineage and merge instructions

```
{BASE}  --(LoRA dw/, merge)-->  merged Dw  --(LoRA sK/, merge)-->  D seed K
```

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
base = "{BASE}"
m = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16)
m = PeftModel.from_pretrained(m, "{repo}", subfolder="dw").merge_and_unload()      # merged Dw
m = PeftModel.from_pretrained(m, "{repo}", subfolder="s0").merge_and_unload()      # D seed 0
m.save_pretrained("d_s0_merged"); AutoTokenizer.from_pretrained(base).save_pretrained("d_s0_merged")
```

The repository's own helper does the same (`actionproof/train/merge.py`: bf16, `merge_and_unload`, tokenizer copied
from the base). Evaluation used merged checkpoints because vLLM 0.23 with LoRA crashes on Cohere's tied embeddings.

{HOW_TO_PROMPT}
{ADAPTER_USAGE.format(code=CODE, repo=repo, base=BASE, sub="s0", chain=CHAIN_DW.format(base=BASE, user=USER))}
## Results (frozen P1 test, 450 tasks, 3 seeds, sampled VSC)

Arm D is the three seeds in `s0`, `s1`, `s2`.

{per_split_table([("W: RFT warm start", "W"), ("A: recovery-start GRPO", "A"), ("Dw: recovery SFT only (`dw/`)", "Dw"), ("**D** (`s0`, `s1`, `s2`)", "D")])}
| Contrast (paired task bootstrap, B = 10,000) | Status | Delta VSC [95% interval] |
|---|---|---|
| D - A | registered P1-H3, **supported** (Holm p {d['holm_p']:.4f}); seed deltas {', '.join(sg(x) for x in d['seed_deltas'])} | {ci(d)} |
| D - Dw | exploratory: GRPO increment on recovery SFT | {ci(c2['D-Dw (GRPO on recovery SFT)'])} |
| Dw - W | exploratory: recovery SFT without RL | {ci(c2['Dw-W (SFT, no RL)'])} |
| Dn - D | registered P3-H1, **null** | {ci(P3['H1'])} |

Per-seed union sampled VSC of D: {seed_line('D')}. Dw (single checkpoint, no seeds): {f3(W2['arm_means']['Dw'])}.

{behaviour_table('D')}
{honesty(repo)}
{EVAL.format(code=CODE, user=USER, model_arg="runs/p1_dwarm/merged", adapter_arg=" --adapter runs/p1_d_s0/final")}
{regen_note("results/p1/p1_summary.json, results/p1/test/D_s*__*.summary.json, results/audit/dwarm_2x2.json, results/p3/p3_summary.json")}"""


def card_dw() -> str:
    repo = f"{USER}/actionproof-warmstart-dw"
    c2 = W2["contrasts"]
    return FRONT.format(base=BASE, rel="adapter", lib="peft", user=USER,
                        tags="\n".join(f"- {t}" for t in ["actionproof", "tool-use", "agents", "sft", "lora", "procuresim", "warm-start", "research-only"])) + f"""
# ActionProof warm start Dw (recovery SFT, no RL)

The LoRA (r16) adapter `Dw` on `{BASE}`: supervised fine-tuning on the original warm-start demonstrations plus verified
oracle recovery demonstrations, with the exact settings of the earlier RFT warm start (2 epochs, lr 1e-4, seed 0;
`runs/p1_dwarm/final`). It is the **starting point of agent D** (and of the no-std variant Dn) and the missing cell of
the exploratory 2x2 that separates what recovery SFT adds from what GRPO adds. **No RL was run on it.**

Data: 554 verified oracle episodes after deduplication and a balance cap (`actionproof/p1/build.py`; CuratorKIT exact and
MinHash deduplication, then a cap per workflow, fault class and action pattern). The teacher is a scripted oracle that decides only from read
results and the request.

## Lineage

`{BASE}` + this adapter = merged Dw (`runs/p1_dwarm/merged`), the parent of all D and Dn adapters
([`{USER}/actionproof-agent-d-adapters`](https://huggingface.co/{USER}/actionproof-agent-d-adapters),
[`{USER}/actionproof-agent-dn`](https://huggingface.co/{USER}/actionproof-agent-dn)). This adapter loads directly on the
public base model.

{HOW_TO_PROMPT}
{ADAPTER_USAGE.format(code=CODE, repo=repo, base=BASE, sub=".", chain=CHAIN_BASE.format(base=BASE)).replace('f"{local}/."', 'local')}
## Results (frozen P1 test, 450 tasks, sampled VSC)

Single checkpoint (the SFT has one seed), so only task-level uncertainty applies.

{per_split_table([("W: RFT warm start", "W"), ("**Dw** (this adapter)", "Dw"), ("D: Dw + GRPO", "D")])}
| Contrast (exploratory, paired task bootstrap, B = 10,000) | Delta VSC [95% interval] |
|---|---|
| Dw - W: recovery SFT, no RL | {ci(c2['Dw-W (SFT, no RL)'])} |
| D - Dw: GRPO on recovery SFT | {ci(c2['D-Dw (GRPO on recovery SFT)'])} |

Dw union sampled VSC {f3(W2['arm_means']['Dw'])}. Behaviour by split (sampled):

{dw_behaviour_table()}
The 2x2 contrasts were written down before Dw's output existed (`research_audit/NOTE_DWARM_EVAL.md`) but are labelled
exploratory; nothing on this card is a registered hypothesis test.
{honesty(repo)}
{EVAL.format(code=CODE, user=USER, model_arg="runs/p1_dwarm/merged", adapter_arg="")}
{regen_note("results/p1/test/Dw__*.summary.json, results/audit/dwarm_2x2.json")}"""


def card_a() -> str:
    repo = f"{USER}/actionproof-agent-a"
    c2 = W2["contrasts"]
    return FRONT.format(base=BASE, rel="adapter", lib="peft", user=USER,
                        tags="\n".join(f"- {t}" for t in ["actionproof", "tool-use", "agents", "grpo", "lora", "procuresim", "research-only"])) + f"""
# ActionProof agent A (recovery-start GRPO, "RSC"), seeds 0, 1, 2

The reference agent of the P1 study: three LoRA (r16) adapters (`s0`, `s1`, `s2` = `runs/s3_rv_mix_s{{0,1,2}}/final`)
trained by GRPO with the state reward R_V on recovery-start and full-episode pools (`s3_trainb_half` +
`s3_rec_train_half`; 200 steps, 4 prompts x 8 rollouts, lr 2e-5, beta 0.02). It is the **baseline** that D, C and B are
compared against; it is not the recommended agent
([`{USER}/actionproof-agent-d`](https://huggingface.co/{USER}/actionproof-agent-d) is).

## Lineage

`{BASE}` + `w/` (the RFT warm start W, SFT on teacher trajectories; `runs/rft_proc/final`) = merged W
(`runs/rft_proc/merged`) + `s0`, `s1` or `s2` = A. **The A adapters load on merged W, not on the raw base.**
`w/` is included so the chain can be rebuilt. `LINEAGE.json` in each folder records the original parent path.

{HOW_TO_PROMPT}
{ADAPTER_USAGE.format(code=CODE, repo=repo, base=BASE, sub="s0", chain=CHAIN_W.format(base=BASE))}
## Results (frozen P1 test, 450 tasks, 3 seeds, sampled VSC)

{per_split_table([("W: RFT warm start (`w/`)", "W"), ("**A** (`s0`, `s1`, `s2`)", "A"), ("D: recovery SFT then GRPO", "D")])}
| Contrast (paired task bootstrap, B = 10,000) | Status | Delta VSC [95% interval] |
|---|---|---|
| A - W | exploratory: GRPO on the plain warm start | {ci(c2['A-W (GRPO on warm start)'])} |
| B - A | registered P1-H1, not supported (reconciliation line at inference on A) | {ci(P1['test']['B'])} |
| C - A | registered P1-H2, supported (A plus recovery SFT) | {ci(P1['test']['C'])} |
| D - A | registered P1-H3, supported | {ci(P1['test']['D'])} |

Per-seed union sampled VSC of A: {seed_line('A')}.

**Behaviour of arm A by split (sampled, seed mean)**

{behaviour_table('A')}
Partial payments are repaired on a minority of episodes and the unseen four-step workflow is mostly false success;
that is what P1 set out to improve.
{honesty(repo)}
{EVAL.format(code=CODE, user=USER, model_arg="runs/rft_proc/merged", adapter_arg=" --adapter runs/s3_rv_mix_s0/final")}
{regen_note("results/p1/p1_summary.json, results/p1/test/A_s*__*.summary.json, results/audit/dwarm_2x2.json")}"""


def card_dn() -> str:
    repo = f"{USER}/actionproof-agent-dn"
    h = P3["H1"]
    return FRONT.format(base=BASE, rel="adapter", lib="peft", user=USER,
                        tags="\n".join(f"- {t}" for t in ["actionproof", "tool-use", "agents", "grpo", "dr-grpo", "negative-result", "lora", "research-only"])) + f"""
# ActionProof agent Dn (no-std GRPO): the registered P3 null result

**Published for reproducibility, not for use.** Dn is agent D with exactly one change: the group standard deviation is
removed from the GRPO advantage (TRL `scale_rewards="none"`, Dr. GRPO's advantage without its length normalisation;
`actionproof/p3/train_nostd.py`). Same warm start (merged Dw), pools, reward R_V, 200 steps, LoRA r16, lr 2e-5, beta 0.02,
seeds 0-2. Three adapters: `s0`, `s1`, `s2` = `runs/p3_dn_s{{0,1,2}}/final`.

**P3 is a null result.** Registered hypothesis P3-H1 (`PREREGISTRATION_P3.md`): Dn beats D. It does not
(Dn - D {ci(h)}, one-sided p {h['p_one_sided']:.3f}, seed deltas {', '.join(sg(x) for x in h['seed_deltas'])}, **not supported**).
The interval excludes zero on the negative side, so there is no gain and a very small loss. The authors' reading that this fits the theory, because reads are free in ProcureSim,
was written after the result and is not a tested prediction.

## Lineage

`{BASE}` + Dw adapter ([`{USER}/actionproof-warmstart-dw`](https://huggingface.co/{USER}/actionproof-warmstart-dw), merged) + `s0`, `s1` or `s2`
(this repository). The adapters load on merged Dw, not on the raw base.

{HOW_TO_PROMPT}
{ADAPTER_USAGE.format(code=CODE, repo=repo, base=BASE, sub="s0", chain=CHAIN_DW.format(base=BASE, user=USER))}
## Results (frozen P1 test, 450 tasks, 3 seeds, sampled VSC)

| Split | D | Dn |
|---|---|---|
""" + "".join(f"| {sp.replace('p1_test_', '')} | {f3(v['D'])} | {f3(v['Dn'])} |\n" for sp, v in P3["descriptive_per_split_sampled_vsc"].items()) + f"""| union (450) | {f3(h['D_mean'])} | {f3(h['Dn_mean'])} |

| Contrast (paired task bootstrap, B = 10,000) | Status | Delta VSC [95% interval] |
|---|---|---|
| Dn - D | registered P3-H1, **not supported (null)** | {ci(h)} |
| Dn - Dw | exploratory: no-std GRPO on recovery SFT | {ci(W2['contrasts']['Dn-Dw (no-std GRPO on recovery SFT)'])} |
| D - Dw | exploratory: standard GRPO on recovery SFT | {ci(W2['contrasts']['D-Dw (GRPO on recovery SFT)'])} |

Per-seed union sampled VSC of Dn: {seed_line('Dn')}.
{honesty(repo)}
{EVAL.format(code=CODE, user=USER, model_arg="runs/p1_dwarm/merged", adapter_arg=" --adapter runs/p3_dn_s0/final")}
{regen_note("results/p3/p3_summary.json, results/audit/dwarm_2x2.json, results/p1/test/Dn_s*__*.summary.json")}"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in (("actionproof-agent-d", card_d), ("actionproof-agent-d-adapters", card_d_adapters),
                     ("actionproof-warmstart-dw", card_dw), ("actionproof-agent-a", card_a),
                     ("actionproof-agent-dn", card_dn)):
        (OUT / f"{name}.md").write_text(fn())
        print("wrote", OUT / f"{name}.md")


if __name__ == "__main__":
    main()
