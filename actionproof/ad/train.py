"""Costed-verification GRPO entry point. Identical to actionproof/stage3/train.py (same trainer, warm start, LoRA,
4 prompts x 8 rollouts, lr 2e-5, beta 0.02, T 1, 16 turns, 256 tokens) except:
  rollout   actionproof/ad/rollout.py (CW environment; records the world posterior)
  reward    actionproof/ad/rewards.py, chosen by --estimator, with read cost --cost
  scaling   TRL's scale_rewards, chosen by --estimator (group for grpo/awrb/cvar, none for nostd, batch for batchstd)

    python -m actionproof.ad.train --estimator awrb --pools data/bench/proc/cw_train.jsonl --seed 0 \
        --init-adapter runs/rft_proc/final --out runs/ad_awrb_s0 --use-vllm
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    from actionproof.ad.rewards import COST, ESTIMATORS
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimator", choices=sorted(ESTIMATORS), required=True)
    ap.add_argument("--cost", type=float, default=COST)
    ap.add_argument("--pools", required=True)
    ap.add_argument("--model", default="CohereLabs/tiny-aya-global")
    ap.add_argument("--init-adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use-vllm", action="store_true")
    ap.add_argument("--vllm-mem", type=float, default=0.30)
    a = ap.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer

    from agenttune.backends.trl.agentic.grpo.agentic_grpo import TrlAgenticGrpo
    from actionproof.ad.rewards import METRICS, make_reward
    from actionproof.ad.rollout import make_rollout_func
    from actionproof.bench.generator_proc import load_tasks
    from actionproof.integrity.sim.prompts import PROC_NEUTRAL
    from actionproof.train.merge import merge

    field, scale = ESTIMATORS[a.estimator]
    tasks = [t for p in a.pools.split(",") for t in load_tasks(p)]
    ds = Dataset.from_dict({"prompt": [[{"role": "system", "content": PROC_NEUTRAL}, {"role": "user", "content": t.prompt}]
                                       for t in tasks], "task_id": [t.task_id for t in tasks]})
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_config.json").write_text(json.dumps({**vars(a), "reward_field": field, "scale_rewards": scale,
                                                     "n_tasks_used": len(tasks)}, indent=2))
    grpo = TrlAgenticGrpo(
        model=merge(a.init_adapter, a.model), processing_class=tok,
        reward_funcs=[make_reward(field, a.cost), *METRICS], reward_weights=[1.0] + [0.0] * len(METRICS),
        train_dataset=ds, rollout_func=make_rollout_func(tasks, PROC_NEUTRAL, max_turns=16),
        output_dir=str(out),
        peft_config=LoraConfig(r=16, lora_alpha=32, task_type="CAUSAL_LM",
                               target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
        num_generations=8, per_device_train_batch_size=1, gradient_accumulation_steps=32, generation_batch_size=32,
        gradient_checkpointing=True, max_steps=a.steps, learning_rate=2e-5, beta=0.02, temperature=1.0,
        max_completion_length=256, max_grad_norm=0.5, logging_steps=1, save_steps=25, save_total_limit=None,
        bf16=torch.cuda.is_available(), seed=a.seed, report_to=[], use_vllm=a.use_vllm, scale_rewards=scale,
        resume_from_checkpoint=any(out.glob("checkpoint-*")),
        **({"vllm_mode": "colocate", "vllm_gpu_memory_utilization": a.vllm_mem, "vllm_enable_sleep_mode": False,
            "vllm_max_model_length": 8192} if a.use_vllm else {}),
    )
    grpo.train()
    grpo.trainer.save_model(str(out / "final"))
    (out / "log_history.json").write_text(json.dumps(getattr(grpo.trainer.state, "log_history", []), indent=1))


if __name__ == "__main__":
    main()
