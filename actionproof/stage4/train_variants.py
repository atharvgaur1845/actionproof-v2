"""Stage 3 trainer with one registered change at a time (PREREGISTRATION_S3_ADDENDUM3.md, PREREGISTRATION_SR1.md).

The Stage 3 entry point (`actionproof/stage3/train.py`, unchanged) runs with the Stage 3 rollout
(`actionproof/stage3/rollout.py`, unchanged) after one of these patches of that module:

  --hide-prefix     state-only recovery starts: a task's scripted prefix is replayed into the environment (the world
                    is in the faulted state) but NOT into the conversation; the policy sees only the system prompt and
                    the user request, exactly as in a full episode
  --surface-random  tool-surface randomisation (actionproof/stage4/surface.py): one surface per (task, step), half
                    canonical

Everything else (reward R_V, pools, warm start, settings, seeds, scoring) is the Stage 3 trainer's.
    python -m actionproof.stage4.train_variants --hide-prefix --reward rv --pools ... --seed 0 \
        --init-adapter runs/rft_proc/final --out runs/s3_stateonly_s0 --use-vllm
"""
from __future__ import annotations

import sys


def hide_prefix(R) -> None:
    def start(env, task, base_prompt):
        for tool, args in task.meta.get("start_prefix") or []:
            env.call(tool, dict(args))
        return [{"role": "system", "content": R.system_prompt(env, base_prompt)}, {"role": "user", "content": task.prompt}]
    R.start_conversation = start


def surface_random(R) -> None:
    from actionproof.stage4 import surface
    wrap = surface.install(R)
    orig = R.make_rollout_func

    def make_rollout_func(*a, **kw):
        return wrap(orig(*a, **kw))
    R.make_rollout_func = make_rollout_func


def main() -> None:
    import actionproof.stage3.rollout as R
    flags = {"--hide-prefix": hide_prefix, "--surface-random": surface_random}
    chosen = [f for f in flags if f in sys.argv]
    assert len(chosen) == 1, "exactly one variant flag"
    sys.argv = [a for a in sys.argv if a not in flags]
    flags[chosen[0]](R)
    from actionproof.stage3 import train
    train.main()


if __name__ == "__main__":
    main()
