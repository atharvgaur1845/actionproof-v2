"""Publish the ActionProof Hub deliverables (run on the cluster copy, where the adapters are).

  models   one public repo per adapter (3 warm starts, the v2 / A6 / A7 GRPO arms), each with a model card
  datasets actionproof-bench (every frozen split) and actionproof-traces (evaluation episodes and reports)
  space    the Gradio demo (staged folder: demo/ + the actionproof package)
  collections  "ActionProof: adapters" and "ActionProof: benchmark, traces, demo"

    python scripts/publish_hf.py models [--only RUN ...] [--dry-run]
    python scripts/publish_hf.py datasets --bench DIR --traces DIR
    python scripts/publish_hf.py space --space-dir DIR
    python scripts/publish_hf.py collections
Idempotent: repos are created with exist_ok and re-uploads replace files.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

USER = "atharvgaur18"
BASE = "CohereLabs/tiny-aya-global"
CODE = "https://github.com/atharvgaur1845/actionproof-v2"
PFX = f"{USER}/actionproof-"

WARM = {  # run -> (description, prereg, eval summary for test_iid)
    "rft_tiny_v2": ("v2 warm start: rejection-sampling fine-tuning on Aya Expanse 32B trajectories plus expert "
                    "refund demonstrations (half of them verify the refund).", "PREREGISTRATION_A4.md",
                    "results/eval/r3/rft_tiny_v2__neutral__test_iid.summary.json"),
    "rft_tiny_a6": ("A6 warm start: the v2 SFT rows plus 80 oracle demonstrations (60 honest failures on "
                    "unrecoverable refunds, 20 multi-retry recoveries).", "PREREGISTRATION_A6.md",
                    "results/eval/a6/init_a6__neutral__test_iid.summary.json"),
    "rft_tiny_a7": ("A7 warm start: the A6 SFT rows plus 20 store-credit workflow demonstrations.",
                    "PREREGISTRATION_A7.md", "results/eval/a7/a7_init__neutral__test_iid.summary.json"),
}
ARMS = {  # reward arm -> (goal-term channel, equation)
    "pcstr": ("state", "G + 0.2P + 0.1V − 0.5I − 0.3F_over − 0.1F_under − 0.2D"),
    "pcstr_d": ("state", "G + 0.2P + 0.1V − 0.5I − 0.3ND − 0.1F_under − 0.2D"),
    "pcstr_c": ("state", "PCSTR-d − 0.02 · (number of verification calls)"),
    "pcstr_d_noreport": ("state", "G + 0.2P + 0.1V − 0.5I − 0.2D (PCSTR-d without report terms)"),
    "pcstr_d_noND": ("state", "PCSTR-d without the nondisclosure term ND"),
    "pcstr_d_noFunder": ("state", "PCSTR-d without the underclaim term F_under"),
    "outcome": ("state", "1[P = 1 ∧ I = 0] (database outcome, report-blind)"),
    "ack_outcome": ("acknowledgements", "1[last ack of every required write is success] · 1[I = 0]"),
    "tool_trust": ("acknowledgements", "share of required writes acked success · 1[report = expected]"),
    "llm_judge": ("transcript", "Aya Expanse 8B judge, P(COMPLETE | transcript)"),
}
GRPO = ([(f"grpo_tiny_v2_{a}_s{s}", "rft_tiny_v2", a, s, f"results/eval/r3/grpo_tiny_v2_{a}_s{s}__neutral__test_iid.summary.json")
         for a, ss in (("pcstr_d", (0, 1)), ("outcome", (0, 1)), ("ack_outcome", (0, 1)), ("tool_trust", (0, 1)),
                       ("pcstr", (0,)), ("pcstr_c", (0,)), ("llm_judge", (0,))) for s in ss]
        + [(f"grpo_tiny_a6_{a}_s{s}", "rft_tiny_a6", a, s, f"results/eval/a6/a6_{a}_s{s}__neutral__test_iid.summary.json")
           for a, ss in (("pcstr_d", (0, 1)), ("pcstr_d_noreport", (0, 1)), ("pcstr_d_noND", (0,)),
                         ("pcstr_d_noFunder", (0,)), ("outcome", (0,)), ("ack_outcome", (0,))) for s in ss]
        + [(f"grpo_tiny_a7_{a}_s{s}", "rft_tiny_a7", a, s, f"results/eval/a6/a7_{a}_s{s}__neutral__test_iid.summary.json")
           for a, s in (("pcstr_d_noreport", 0), ("ack_outcome", 0))]
        # Amendment 8: X1 (target classes removed) and its placebo (60 random tasks removed); ack_outcome, seed 0
        + [(f"grpo_tiny_{t}_ack_outcome_s0", "rft_tiny_a7", "ack_outcome", 0,
            f"results/eval/a6/{t}_ack_outcome_s0__neutral__test_iid.summary.json") for t in ("a8x1", "a8x1p")])


def repo_id(run: str) -> str:
    return PFX + run.replace("_", "-").lower()


def metrics(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return "Evaluation summary not available in this repository snapshot."
    d = json.loads(p.read_text())
    pwv = (d.get("a4") or {}).get("pwv_issue_refund") or {}
    ver = f"; post-write refund verification {pwv['rate']:.2f} (n = {pwv.get('n')})" if pwv.get("rate") is not None else ""
    return (f"Greedy evaluation on `test_iid` (n = {d.get('n')}): verified-success completion (VSC) "
            f"**{d['vsc']:.3f}**{ver}. Source: `{path}` in the code repository.")


HEADER = """---
license: cc-by-nc-4.0
base_model: {base}
library_name: peft
tags: [lora, grpo, tool-use, agents, reinforcement-learning, actionproof]
---
"""


def card(run: str) -> str:
    if run in WARM:
        desc, prereg, summ = WARM[run]
        chain = f'model = PeftModel.from_pretrained(base, "{repo_id(run)}")'
        body = f"""# ActionProof warm start `{run}`

LoRA adapter (r = 16) on [{BASE}](https://huggingface.co/{BASE}). {desc}

GRPO arms in the ActionProof collection were trained as a fresh LoRA on this adapter **merged** into the base.
Design and registration: `{prereg}` in the [code repository]({CODE}).

{metrics(summ)}
"""
    else:
        _, init, arm, seed, summ = next(g for g in GRPO if g[0] == run)
        ch, eq = ARMS[arm]
        chain = (f'base = PeftModel.from_pretrained(base, "{repo_id(init)}").merge_and_unload()  # warm start\n'
                 f'model = PeftModel.from_pretrained(base, "{repo_id(run)}")')
        body = f"""# ActionProof GRPO arm `{arm}`, seed {seed}

LoRA adapter (r = 16) trained with GRPO on top of the warm start [`{repo_id(init)}`](https://huggingface.co/{repo_id(init)})
merged into [{BASE}](https://huggingface.co/{BASE}).

- **Reward:** `{arm}`, goal term read from the **{ch}**: {eq}.
- **Training:** 200 steps × 4 prompts × 8 rollouts, lr 2e-5, β 0.02, T = 1, seed {seed}; every arm in a family shares
  warm start, tasks, optimiser and budget (see `run_config.json`, `log_history.json`).

{metrics(summ)}
"""
    return HEADER.format(base=BASE) + body + f"""
## Load

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained("{BASE}", torch_dtype="bfloat16")
{chain}
```

The environment, verifier, prompts and evaluation code are in the [code repository]({CODE}). Episodes are scored from
the database by a deterministic verifier; acknowledgements can be false by design.

License: CC-BY-NC-4.0, derived from Cohere's Tiny Aya Global (Cohere Labs). Research use only.
"""


def publish_models(api: HfApi, only: list[str], dry: bool) -> None:
    runs = list(WARM) + [g[0] for g in GRPO]
    for run in [r for r in runs if not only or r in only]:
        src = Path("runs") / run
        if not (src / "final" / "adapter_model.safetensors").exists():
            print("skip (no adapter)", run)
            continue
        rid = repo_id(run)
        print(("DRY " if dry else "") + "model", rid)
        if dry:
            continue
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            for f in ("adapter_config.json", "adapter_model.safetensors", "chat_template.jinja",
                      "tokenizer_config.json", "tokenizer.json"):
                if (src / "final" / f).exists():
                    shutil.copy(src / "final" / f, t / f)
            for f in ("run_config.json", "log_history.json", "lexsi_provenance.json", "grpo_training_config.yaml"):
                if (src / f).exists():
                    shutil.copy(src / f, t / f)
            (t / "README.md").write_text(card(run))
            api.create_repo(rid, repo_type="model", private=False, exist_ok=True)
            api.upload_folder(repo_id=rid, folder_path=t, commit_message="ActionProof adapter and model card")


BENCH_CARD = f"""---
license: cc-by-4.0
tags: [agents, tool-use, benchmark, reinforcement-learning, actionproof]
---
# ActionProof-Bench

Frozen task splits of ActionProof, a SQLite customer-support simulator in which tool acknowledgements can be false by
design (phantom, partial, stale and timeout-after-commit writes). Each task holds the prompt, the initial database
state, the fault schedule and the goal predicates the deterministic verifier checks.

- `*.jsonl` at the top level: the registered v1 splits (train / val / test_iid / test_ood_fault / variants / banking).
- `a4/` … `a8/`: the splits and pools of Amendments 4–8 (training pools A/B, held-out credit workflow,
  unrecoverable tasks, recoverable twins, depth-3, forced prefixes, the Amendment 8 task list).

The generator, environment, verifier and the hashes that freeze every split (`results/*hashes*.json`) are in the
[code repository]({CODE}). Tasks are synthetic; no real customer data.
"""
TRACES_CARD = f"""---
license: cc-by-nc-4.0
tags: [agents, tool-use, evaluation, reinforcement-learning, actionproof]
---
# ActionProof traces and reports

Evaluation episodes (one JSON line per episode: tool calls, acknowledgements, final report, verifier verdict) and the
generated reports behind every number in the ActionProof write-up. Paths mirror `results/` in the
[code repository]({CODE}); each report names the script that produced it.

Model outputs come from Tiny Aya Global adapters and Aya Expanse teachers (CC-BY-NC-4.0, Cohere Labs). Interpretability
activation dumps are not included.
"""


def publish_datasets(api: HfApi, bench: str, traces: str) -> None:
    for name, folder, cardtxt, ignore in (("bench", bench, BENCH_CARD, None),
                                          ("traces", traces, TRACES_CARD, ["interp/**", "**/*.npz", "**/*.pt"])):
        rid = PFX + name
        Path(folder, "README.md").write_text(cardtxt)
        api.create_repo(rid, repo_type="dataset", private=False, exist_ok=True)
        api.upload_folder(repo_id=rid, repo_type="dataset", folder_path=folder, ignore_patterns=ignore,
                          commit_message="ActionProof data")
        print("dataset", rid)


def publish_space(api: HfApi, space_dir: str) -> None:
    rid = f"{USER}/actionproof"
    api.create_repo(rid, repo_type="space", space_sdk="gradio", private=False, exist_ok=True)
    api.upload_folder(repo_id=rid, repo_type="space", folder_path=space_dir, ignore_patterns=["**/__pycache__/**"],
                      commit_message="ActionProof demo")
    print("space", rid)


def publish_collections(api: HfApi) -> None:
    def coll(title: str, desc: str):
        hit = [c for c in api.list_collections(owner=USER) if c.title == title]
        return hit[0] if hit else api.create_collection(title=title, description=desc, private=False)
    m = coll("ActionProof: adapters", "Tiny Aya Global LoRA adapters: three warm starts and the GRPO reward arms "
             "(state- vs acknowledgement-grounded goal terms).")
    have = {i.item_id for i in api.get_collection(m.slug).items}
    for run in list(WARM) + [g[0] for g in GRPO]:
        rid = repo_id(run)
        if rid not in have and api.repo_exists(rid):
            api.add_collection_item(m.slug, item_id=rid, item_type="model", exists_ok=True)
    d = coll("ActionProof: benchmark, traces, demo", "Frozen task splits, evaluation episodes and reports, and the demo Space.")
    for rid, typ in ((PFX + "bench", "dataset"), (PFX + "traces", "dataset"), (f"{USER}/actionproof", "space")):
        if api.repo_exists(rid, repo_type=typ):
            api.add_collection_item(d.slug, item_id=rid, item_type=typ, exists_ok=True)
    print("collections", m.slug, d.slug)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("models", "datasets", "space", "collections"))
    ap.add_argument("--only", nargs="*", default=[])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--bench")
    ap.add_argument("--traces")
    ap.add_argument("--space-dir")
    a = ap.parse_args()
    api = HfApi()
    if a.what == "models":
        publish_models(api, a.only, a.dry_run)
    elif a.what == "datasets":
        publish_datasets(api, a.bench, a.traces)
    elif a.what == "space":
        publish_space(api, a.space_dir)
    else:
        publish_collections(api)


if __name__ == "__main__":
    main()
