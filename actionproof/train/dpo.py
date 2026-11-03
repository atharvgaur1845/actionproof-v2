"""Offline DPO through AgentTune (spec §13.2, rung R1).

Starts from the shared RFT adapter (merged) so DPO and GRPO arms share an initialisation.

    python -m actionproof.train.dpo --pairs data/dpo/pairs.jsonl --init-adapter runs/rft_tiny/final \
        --out runs/dpo_tiny
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--model", default="CohereLabs/tiny-aya-global")
    ap.add_argument("--init-adapter", default=None, help="RFT LoRA, merged into the base before DPO")
    ap.add_argument("--out", required=True)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=6144)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from agenttune.core.backend_factory import create_agentic_trainer

    rows = [json.loads(l) for l in open(a.pairs)]
    ds = Dataset.from_list([{k: r[k] for k in ("prompt", "chosen", "rejected")} for r in rows])
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_config.json").write_text(json.dumps({**vars(a), "pairs": len(rows)}, indent=2))

    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16)
    if a.init_adapter:  # merge RFT so the reference model == the RFT policy
        model = PeftModel.from_pretrained(model, a.init_adapter).merge_and_unload()

    trainer = create_agentic_trainer(
        "dpo", backend="trl", model=model, processing_class=tok, train_dataset=ds,
        output_dir=str(out), beta=a.beta, learning_rate=a.lr, num_train_epochs=a.epochs,
        per_device_train_batch_size=a.batch, gradient_accumulation_steps=a.grad_accum,
        max_length=a.max_length, bf16=torch.cuda.is_available(), logging_steps=5,
        save_strategy="epoch", seed=a.seed, report_to=[], gradient_checkpointing=True,
        peft_config=LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                               target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                               "gate_proj", "up_proj", "down_proj"]),
    )
    trainer.train()
    inner = getattr(trainer, "trainer", trainer)
    inner.save_model(str(out / "final"))
    (out / "log_history.json").write_text(json.dumps(inner.state.log_history, indent=2))


if __name__ == "__main__":
    main()
