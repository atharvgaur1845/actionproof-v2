"""Merge a LoRA adapter into its base model and save a standalone checkpoint.

vLLM 0.23 + LoRA crashes on Cohere's tied embeddings ('VocabParallelEmbeddingWithLoRA' has no
'quant_method'), so adapters are evaluated as merged checkpoints. Cached at <adapter>/../merged.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def merged_path(adapter: str) -> Path:
    return Path(adapter).resolve().parent / "merged"


def merge(adapter: str, base: str | None = None) -> str:
    out = merged_path(adapter)
    if (out / "config.json").exists():
        return str(out)
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    base = base or json.loads((Path(adapter) / "adapter_config.json").read_text())["base_model_name_or_path"]
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(base).save_pretrained(out)
    (out / "merged_from.json").write_text(json.dumps({"base": base, "adapter": str(Path(adapter).resolve())}))
    return str(out)


if __name__ == "__main__":
    print(merge(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None))
