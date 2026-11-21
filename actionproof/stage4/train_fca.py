"""FCA training entry point (docs/research/algorithm/FCA_GRPO.md). Identical to actionproof/stage3/train.py (reward R_V,
same trainer, settings, warm start and pools) except: the FCA rollout function, and per-token advantages computed by
actionproof/stage4/advantage.py, installed by wrapping the trainer class's `_generate_and_score_completions`
(TRL's loss accepts (B, T) advantages; nothing else in TRL is changed).

    python -m actionproof.stage4.train_fca --pools data/bench/proc/s3_trainb_half.jsonl,data/bench/proc/s3_rec_train_half.jsonl \
        --seed 0 --init-adapter runs/rft_proc/final --out runs/s4_fca_s0 --use-vllm
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def install(trainer_cls, log_path: Path) -> None:
    """Wrap `_generate_and_score_completions` on the trainer class (AgentTune builds the GRPOTrainer inside `train()`;
    this process trains exactly one trainer)."""
    import torch
    from actionproof.stage4.advantage import fca_advantages
    orig = trainer_cls._generate_and_score_completions

    def wrapped(self, inputs):
        out = orig(self, inputs)
        meta, rewards = self._fca_meta, self._fca_rewards
        B, T = out["completion_ids"].shape
        assert len(meta) == len(rewards) == B, (len(meta), len(rewards), B)
        base = out["advantages"].float().tolist()
        adv, diag = fca_advantages(rewards, meta, T, base)
        out["advantages"] = torch.tensor(adv, dtype=out["advantages"].dtype, device=out["advantages"].device)
        with log_path.open("a") as f:
            f.write(json.dumps({"step": int(self.state.global_step), "groups": diag}) + "\n")
        return out

    trainer_cls._generate_and_score_completions = wrapped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", required=True)
    ap.add_argument("--model", default="CohereLabs/tiny-aya-global")
    ap.add_argument("--init-adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--n-on", type=int, default=4)
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
    from actionproof.stage3.rewards import METRICS, rv_reward
    from actionproof.stage4.rollout_fca import make_rollout_func
    from actionproof.train.merge import merge

    tasks = [t for p in a.pools.split(",") for t in load_tasks(p)]
    ds = Dataset.from_dict({"prompt": [[{"role": "system", "content": PROC_NEUTRAL}, {"role": "user", "content": t.prompt}]
                                       for t in tasks], "task_id": [t.task_id for t in tasks]})
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_config.json").write_text(json.dumps({**vars(a), "algo": "fca", "n_tasks_used": len(tasks)}, indent=2))
    grpo = TrlAgenticGrpo(
        model=merge(a.init_adapter, a.model), processing_class=tok,
        reward_funcs=[rv_reward, *METRICS], reward_weights=[1.0] + [0.0] * len(METRICS),
        train_dataset=ds, rollout_func=make_rollout_func(tasks, PROC_NEUTRAL, max_turns=16, n_on=a.n_on),
        output_dir=str(out),
        peft_config=LoraConfig(r=16, lora_alpha=32, task_type="CAUSAL_LM",
                               target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
        num_generations=8, per_device_train_batch_size=1, gradient_accumulation_steps=32, generation_batch_size=32,
        gradient_checkpointing=True, max_steps=a.steps, learning_rate=2e-5, beta=0.02, temperature=1.0,
        max_completion_length=256, max_grad_norm=0.5, logging_steps=1, save_steps=25, save_total_limit=None,
        bf16=torch.cuda.is_available(), seed=a.seed, report_to=[], use_vllm=a.use_vllm,
        resume_from_checkpoint=any(out.glob("checkpoint-*")),
        **({"vllm_mode": "colocate", "vllm_gpu_memory_utilization": a.vllm_mem, "vllm_enable_sleep_mode": False,
            "vllm_max_model_length": 8192} if a.use_vllm else {}),
    )
    from trl import GRPOTrainer
    install(GRPOTrainer, out / "fca_groups.jsonl")
    grpo.train()
    grpo.trainer.save_model(str(out / "final"))
    (out / "log_history.json").write_text(json.dumps(getattr(grpo.trainer.state, "log_history", []), indent=1))


if __name__ == "__main__":
    main()
