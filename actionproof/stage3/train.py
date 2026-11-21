"""Stage 3 GRPO entry point. Same trainer, warm start, LoRA and settings as A12 (`actionproof/train/grpo_proc.py`,
frozen): 4 prompts x 8 rollouts per step, lr 2e-5, KL 0.02, T = 1, LoRA r16, 16 turns. Varies only:
  --reward  rv | safe | lag            (actionproof/stage3/rewards.py)
  --pools   comma-separated task files, concatenated (e.g. proc_train_b half + a recovery-start pool half)
Rollouts use actionproof/stage3/rollout.py (replays `start_prefix` when a task has one; otherwise identical).

    python -m actionproof.stage3.train --reward safe --pools data/bench/proc/s3_train_mix.jsonl --seed 0 \
        --init-adapter runs/rft_proc/final --out runs/s3_safe_mix_s0 --use-vllm
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reward", choices=("rv", "safe", "lag"), required=True)
    ap.add_argument("--pools", required=True)
    ap.add_argument("--model", default="CohereLabs/tiny-aya-global")
    ap.add_argument("--init-adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--prompts-per-step", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--beta", type=float, default=0.02)
    ap.add_argument("--max-turns", type=int, default=16)
    ap.add_argument("--max-completion-length", type=int, default=256)
    ap.add_argument("--save-steps", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use-vllm", action="store_true")
    ap.add_argument("--vllm-mem", type=float, default=0.30)
    a = ap.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer

    from agenttune.backends.trl.agentic.grpo.agentic_grpo import TrlAgenticGrpo
    from actionproof.bench.generator_proc import load_tasks
    from actionproof.integrity.sim.prompts import PROC_NEUTRAL
    from actionproof.stage3.rewards import METRICS, REWARDS, LagrangianReward
    from actionproof.stage3.rollout import make_rollout_func
    from actionproof.train.merge import merge

    tasks = [t for p in a.pools.split(",") for t in load_tasks(p)]
    ds = Dataset.from_dict({"prompt": [[{"role": "system", "content": PROC_NEUTRAL},
                                        {"role": "user", "content": t.prompt}] for t in tasks],
                            "task_id": [t.task_id for t in tasks]})
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_config.json").write_text(json.dumps({**vars(a), "n_tasks_used": len(tasks)}, indent=2))
    reward = LagrangianReward() if a.reward == "lag" else REWARDS[a.reward]
    grpo = TrlAgenticGrpo(
        model=merge(a.init_adapter, a.model), processing_class=tok,
        reward_funcs=[reward, *METRICS], reward_weights=[1.0] + [0.0] * len(METRICS),
        train_dataset=ds, rollout_func=make_rollout_func(tasks, PROC_NEUTRAL, max_turns=a.max_turns),
        output_dir=str(out),
        peft_config=LoraConfig(r=16, lora_alpha=32, task_type="CAUSAL_LM",
                               target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
        num_generations=a.num_generations, per_device_train_batch_size=1,
        gradient_accumulation_steps=a.num_generations * a.prompts_per_step,
        generation_batch_size=a.num_generations * a.prompts_per_step, gradient_checkpointing=True,
        max_steps=a.steps, learning_rate=a.lr, beta=a.beta, temperature=1.0,
        max_completion_length=a.max_completion_length, max_grad_norm=0.5, logging_steps=1, save_steps=a.save_steps,
        save_total_limit=None, bf16=torch.cuda.is_available(), seed=a.seed, report_to=[], use_vllm=a.use_vllm,
        resume_from_checkpoint=any(out.glob("checkpoint-*")),
        **({"vllm_mode": "colocate", "vllm_gpu_memory_utilization": a.vllm_mem, "vllm_enable_sleep_mode": False,
            "vllm_max_model_length": 8192} if a.use_vllm else {}),
    )
    grpo.train()
    grpo.trainer.save_model(str(out / "final"))
    (out / "log_history.json").write_text(json.dumps(getattr(grpo.trainer.state, "log_history", []), indent=1))
    if isinstance(reward, LagrangianReward):
        (out / "lagrangian.json").write_text(json.dumps(reward.history))


if __name__ == "__main__":
    main()
