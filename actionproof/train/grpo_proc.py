"""Amendment 12 GRPO on ProcureSim: R_V (verdict reward) vs R_A (acknowledgement reward), same warm start.

Training settings are those of A6/A7 (`train/grpo.py`, hash-frozen and not edited): 200 steps x 4 prompts x 8
rollouts, lr 2e-5, beta 0.02, T = 1, LoRA r16 on a merged warm start, batched lockstep vLLM rollouts. The only
differences are the environment and the reward. `rollout/batched_vllm.py` imports `make_env` and
`episode_record` as module globals, so they are rebound here to the ProcureSim versions (the pattern
`eval/run_eval_a4.py` uses); no frozen file changes.

    python -m actionproof.train.grpo_proc --arm proc_rv --init-adapter runs/rft_proc --seed 0 \
        --task-jsonl data/bench/proc/proc_train.jsonl --out runs/a12_proc_rv_s0 --use-vllm
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("proc_rv", "proc_ra"), required=True)
    ap.add_argument("--model", default="CohereLabs/tiny-aya-global")
    ap.add_argument("--init-adapter", required=True, help="the shared A12 warm start (one for every arm and seed)")
    ap.add_argument("--task-jsonl", default="data/bench/proc/proc_train.jsonl")
    ap.add_argument("--n-tasks", type=int, default=None)
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
    import actionproof.rollout.batched_vllm as BV
    from actionproof.bench.generator_proc import load_tasks
    from actionproof.integrity.sim.prompts import PROC_NEUTRAL
    from actionproof.integrity.sim.runtime import episode_record_proc, make_env_proc
    from actionproof.rewards.proc_rewards import ARMS, proc_phantom_readback_metric, proc_vsc_metric
    from actionproof.train.merge import merge

    BV.make_env = make_env_proc            # ProcureSim instead of the e-commerce factory
    BV.episode_record = episode_record_proc

    tasks = load_tasks(a.task_jsonl)[: a.n_tasks]
    ds = Dataset.from_dict({"prompt": [[{"role": "system", "content": PROC_NEUTRAL},
                                        {"role": "user", "content": t.prompt}] for t in tasks],
                            "task_id": [t.task_id for t in tasks]})
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_config.json").write_text(json.dumps({**vars(a), "n_tasks_used": len(tasks)}, indent=2))
    resume = any(out.glob("checkpoint-*"))
    base = merge(a.init_adapter, a.model)
    grpo = TrlAgenticGrpo(
        model=base,
        processing_class=tok,
        reward_funcs=[ARMS[a.arm], proc_vsc_metric, proc_phantom_readback_metric],
        reward_weights=[1.0, 0.0, 0.0],
        train_dataset=ds,
        rollout_func=BV.make_batched_rollout_func(tasks, PROC_NEUTRAL, max_turns=a.max_turns),
        output_dir=str(out),
        peft_config=LoraConfig(
            r=16, lora_alpha=32, task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
        num_generations=a.num_generations,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=a.num_generations * a.prompts_per_step,
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
        save_total_limit=None,
        bf16=torch.cuda.is_available(),
        seed=a.seed,
        report_to=[],
        use_vllm=a.use_vllm,
        resume_from_checkpoint=resume,
        **({"vllm_mode": "colocate", "vllm_gpu_memory_utilization": a.vllm_mem,
            "vllm_enable_sleep_mode": False, "vllm_max_model_length": 8192} if a.use_vllm else {}),
    )
    grpo.train()
    grpo.trainer.save_model(str(out / "final"))
    hist = getattr(grpo.trainer.state, "log_history", [])
    (out / "log_history.json").write_text(json.dumps(hist, indent=1))


if __name__ == "__main__":
    main()
