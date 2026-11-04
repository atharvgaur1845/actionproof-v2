"""GRPO with a swappable reward arm on ActionProof tasks (spec §13.3).

All arms share: init adapter (--init-adapter, the RFT checkpoint), data, steps, group size,
LR, lengths and decoding. Only --arm changes. `vsc` is always logged as a 0-weight metric
so every arm reports verified success, not just its own reward.

    python -m actionproof.train.grpo --arm pcstr --model CohereLabs/tiny-aya-global \
        --out runs/tiny_pcstr_s0 --steps 200 --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ARMS = ("pcstr", "pcstr_d", "pcstr_c", "tool_trust", "outcome")


def build_rewards(arm: str, tasks):
    from actionproof.rewards import trl_rewards as R
    goals = {t.task_id: t.goal for t in tasks}
    main = {"pcstr": R.pcstr_reward, "pcstr_d": R.pcstr_d_reward, "pcstr_c": R.make_pcstr_c_reward(0.02),
            "outcome": R.outcome_reward, "tool_trust": R.make_tool_trust_reward(goals)}[arm]
    return [main, R.vsc_metric], [1.0, 0.0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--model", default="CohereLabs/tiny-aya-global")
    ap.add_argument("--init-adapter", default=None, help="shared RFT LoRA to start every arm from")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n-tasks", type=int, default=None)
    ap.add_argument("--task-file", default=None, help="jsonl of task ids (learnable-band curriculum)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--prompts-per-step", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--beta", type=float, default=0.02)
    ap.add_argument("--max-steps-per-episode", type=int, default=10)
    ap.add_argument("--max-completion-length", type=int, default=256)
    ap.add_argument("--save-steps", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use-vllm", action="store_true")
    ap.add_argument("--rollout", choices=("sequential", "batched"), default="sequential",
                    help="batched = lockstep multi-turn rollouts on colocated vLLM (requires --use-vllm)")
    ap.add_argument("--vllm-mem", type=float, default=0.30)
    a = ap.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer

    from agenttune.backends.trl.agentic.grpo.agentic_grpo import TrlAgenticGrpo
    from actionproof.bench.generator import make_split
    from actionproof.bench.prompts import NEUTRAL
    from actionproof.rollout.agenttune_rollout import make_rollout_func
    from actionproof.rollout.batched_vllm import make_batched_rollout_func

    tasks = make_split(a.split, a.n_tasks)
    if a.task_file:
        keep = {json.loads(l)["task_id"] for l in open(a.task_file)}
        tasks = [t for t in tasks if t.task_id in keep]
    ds = Dataset.from_dict({"prompt": [[{"role": "system", "content": NEUTRAL},
                                        {"role": "user", "content": t.prompt}] for t in tasks],
                            "task_id": [t.task_id for t in tasks]})
    reward_funcs, weights = build_rewards(a.arm, tasks)
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_config.json").write_text(json.dumps({**vars(a), "n_tasks_used": len(tasks)}, indent=2))
    resume = any(out.glob("checkpoint-*"))

    from actionproof.train.merge import merge
    # Shared init: every arm starts from the SAME merged RFT weights and trains a fresh LoRA.
    base = merge(a.init_adapter, a.model) if a.init_adapter else a.model
    grpo = TrlAgenticGrpo(
        model=base,
        processing_class=tok,
        reward_funcs=reward_funcs,
        reward_weights=weights,
        train_dataset=ds,
        rollout_func=(make_batched_rollout_func(tasks, NEUTRAL, max_turns=a.max_steps_per_episode)
                      if a.rollout == "batched" else make_rollout_func(tasks, max_steps=a.max_steps_per_episode)),
        output_dir=str(out),
        peft_config=LoraConfig(
            r=16, lora_alpha=32, task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
        num_generations=a.num_generations,
        # Tiny Aya's 262k vocab makes full-trajectory logits huge: train one sequence per
        # micro-batch, but still generate a whole step's groups at once.
        per_device_train_batch_size=1,
        gradient_accumulation_steps=a.num_generations * a.prompts_per_step,  # explicit (AgentTune default 16)
        generation_batch_size=a.num_generations * a.prompts_per_step,
        gradient_checkpointing=True,
        max_steps=a.steps,
        learning_rate=a.lr,
        beta=a.beta,
        temperature=1.0,
        max_completion_length=a.max_completion_length,
        max_grad_norm=0.5,
        logging_steps=1,
        save_steps=a.save_steps,
        save_total_limit=None,  # keep every checkpoint: #7 probes them
        bf16=torch.cuda.is_available(),
        seed=a.seed,
        report_to=[],
        use_vllm=a.use_vllm,
        **({"vllm_mode": "colocate", "vllm_gpu_memory_utilization": a.vllm_mem,
            "vllm_enable_sleep_mode": False, "vllm_max_model_length": 8192} if a.use_vllm else {}),
    )
    if resume:  # allocation ended mid-run: continue from the latest checkpoint
        grpo.trainer.train(resume_from_checkpoint=True)
    else:
        grpo.train()
    (out / "log_history.json").write_text(json.dumps(grpo.trainer.state.log_history, indent=2))
    grpo.trainer.save_model(str(out / "final"))


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
