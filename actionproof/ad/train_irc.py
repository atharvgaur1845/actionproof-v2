"""Candidate D3 entry point: uninformative-read local credit (IRC) on top of the AD-D2 EMA scale.

The return is decomposed as R_c = [R_V - c * n_informative_reads] + sum over uninformative reads of (-c).
- The bracket is the episode-level return. TRL centres it within the group (scale_rewards='none'), then it is divided
  by the EMA divisor (actionproof/ad/scale.py), as in D2.
- Each uninformative read's -c is credited only to the tokens of the turn that made it: -c / divisor is added there.

A read is uninformative iff replay shows that its observation could not eliminate any world of the prior
(actionproof/ad/readinfo.py). No read is normalised against any write: the C5 failure mode is excluded by
construction. Informative reads are untouched.

  --estimator nostd_ema_irc   bracket uses R_V
  --estimator awrbn_ema_irc   bracket uses E[R_V | traj]

    python -m actionproof.ad.train_irc --estimator nostd_ema_irc --pools data/bench/proc/cw_train.jsonl --seed 0 \
        --init-adapter runs/rft_proc/final --out runs/ad_nostd_ema_irc_s0 --use-vllm
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

IRC_ESTIMATORS = {"nostd_ema_irc": "r_v", "awrbn_ema_irc": "r_bar"}


def local_advantages(base: list[float], spans: list[list[tuple[int, int]]], T: int, div: float,
                     cost: float) -> list[list[float]]:
    """base[i]: the episode advantage (already divided); spans[i]: uninformative-read turn spans of episode i."""
    out = []
    for b, sp in zip(base, spans):
        row = [b] * T
        for s, e in sp:
            for t in range(s, min(e, T)):
                row[t] = b - cost / div
        out.append(row)
    return out


def make_inf_reward(field: str, cost: float):
    from actionproof.ad.rewards import _eps, base_value

    def f(prompts, completions, **kw):
        return [base_value(e, field) - cost * e["n_inf_reads"] for e in _eps(completions, kw)]
    f.__name__ = f"{field}_inf_reward"
    return f


def install(trainer_cls, log_path: Path, cost: float, G: int = 8) -> None:
    import torch
    from actionproof.ad.scale import EmaScale, group_stds
    ema = None
    if log_path.exists():
        rows = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]
        ema = rows[-1]["ema_after"] if rows else None
    sc = EmaScale(ema=ema)
    orig = trainer_cls._generate_and_score_completions

    def wrapped(self, inputs):
        out = orig(self, inputs)
        a = out["advantages"].float()
        flat = a.tolist() if a.dim() == 1 else a[:, 0].tolist()
        div, m = sc.divisor_and_update(group_stds(flat, G))
        B, T = out["completion_ids"].shape
        meta = self._ad_meta
        assert len(meta) == B, (len(meta), B)
        adv = local_advantages([x / div for x in flat], [mm["uninf_spans"] for mm in meta], T, div, cost)
        out["advantages"] = torch.tensor(adv, dtype=out["advantages"].dtype, device=out["advantages"].device)
        with log_path.open("a") as f:
            f.write(json.dumps({"step": int(self.state.global_step), "divisor": div, "batch_mean_group_std": m,
                                "ema_after": sc.ema, "uninf_turns": sum(len(mm["uninf_spans"]) for mm in meta)}) + "\n")
        return out

    trainer_cls._generate_and_score_completions = wrapped


def main() -> None:
    from actionproof.ad.rewards import COST
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimator", choices=sorted(IRC_ESTIMATORS), required=True)
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
    from actionproof.ad.rewards import METRICS
    from actionproof.ad.rollout_turns import make_rollout_func
    from actionproof.bench.generator_proc import load_tasks
    from actionproof.integrity.sim.prompts import PROC_NEUTRAL
    from actionproof.train.merge import merge

    field = IRC_ESTIMATORS[a.estimator]
    tasks = [t for p in a.pools.split(",") for t in load_tasks(p)]
    ds = Dataset.from_dict({"prompt": [[{"role": "system", "content": PROC_NEUTRAL}, {"role": "user", "content": t.prompt}]
                                       for t in tasks], "task_id": [t.task_id for t in tasks]})
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_config.json").write_text(json.dumps({**vars(a), "reward_field": field, "scale_rewards": "none+ema+irc",
                                                     "n_tasks_used": len(tasks)}, indent=2))
    grpo = TrlAgenticGrpo(
        model=merge(a.init_adapter, a.model), processing_class=tok,
        reward_funcs=[make_inf_reward(field, a.cost), *METRICS], reward_weights=[1.0] + [0.0] * len(METRICS),
        train_dataset=ds, rollout_func=make_rollout_func(tasks, PROC_NEUTRAL, max_turns=16),
        output_dir=str(out),
        peft_config=LoraConfig(r=16, lora_alpha=32, task_type="CAUSAL_LM",
                               target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
        num_generations=8, per_device_train_batch_size=1, gradient_accumulation_steps=32, generation_batch_size=32,
        gradient_checkpointing=True, max_steps=a.steps, learning_rate=2e-5, beta=0.02, temperature=1.0,
        max_completion_length=256, max_grad_norm=0.5, logging_steps=1, save_steps=25, save_total_limit=None,
        bf16=torch.cuda.is_available(), seed=a.seed, report_to=[], use_vllm=a.use_vllm, scale_rewards="none",
        resume_from_checkpoint=any(out.glob("checkpoint-*")),
        **({"vllm_mode": "colocate", "vllm_gpu_memory_utilization": a.vllm_mem, "vllm_enable_sleep_mode": False,
            "vllm_max_model_length": 8192} if a.use_vllm else {}),
    )
    from trl import GRPOTrainer
    install(GRPOTrainer, out / "irc_scale.jsonl", a.cost)
    grpo.train()
    grpo.trainer.save_model(str(out / "final"))
    (out / "log_history.json").write_text(json.dumps(getattr(grpo.trainer.state, "log_history", []), indent=1))


if __name__ == "__main__":
    main()
