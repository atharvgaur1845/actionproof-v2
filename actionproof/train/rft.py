"""RFT: LoRA SFT on verifier-filtered teacher turns (spec §13.1, rung R1).

The output adapter is the SHARED initialisation for every GRPO arm (spec §13), so all RL
arms differ only in their reward.

    python -m actionproof.train.rft --data data/rft/curated_tiny.jsonl \
        --model CohereLabs/tiny-aya-global --out runs/rft_tiny_curated
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="CohereLabs/tiny-aya-global")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=2)  # 262k vocab: keep micro-batches small
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=6144)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    rows = [json.loads(l) for l in open(a.data)]
    ds = Dataset.from_list([{"prompt": r["prompt"], "completion": r["completion"]} for r in rows])
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_config.json").write_text(json.dumps({**vars(a), "rows": len(rows)}, indent=2))

    import math
    total_steps = math.ceil(len(rows) / (a.batch * a.grad_accum)) * a.epochs
    cfg = SFTConfig(
        output_dir=str(out), num_train_epochs=a.epochs, learning_rate=a.lr,
        per_device_train_batch_size=a.batch, gradient_accumulation_steps=a.grad_accum,
        lr_scheduler_type="cosine", warmup_steps=max(1, int(0.05 * total_steps)), logging_steps=5, save_strategy="epoch",
        bf16=torch.cuda.is_available(), max_length=a.max_length, completion_only_loss=True,
        gradient_checkpointing=True, seed=a.seed, report_to=[],
        model_init_kwargs={"torch_dtype": torch.bfloat16},
    )
    trainer = SFTTrainer(
        model=a.model, args=cfg, train_dataset=ds, processing_class=tok,
        peft_config=LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.05, task_type="CAUSAL_LM",
                               target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                               "gate_proj", "up_proj", "down_proj"]),
    )
    trainer.train()
    trainer.save_model(str(out / "final"))
    (out / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=2))


if __name__ == "__main__":
    main()
