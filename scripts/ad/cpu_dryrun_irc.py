"""CPU dry run of the D3 rollout (turn spans, read flags, rewards, local advantages) with the scripted fake LLM of
scripts/ad/cpu_dryrun.py. Not an experiment.  HF_HUB_OFFLINE=1 python scripts/ad/cpu_dryrun_irc.py"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cpu_dryrun as D  # noqa: E402

from actionproof.ad.rollout_turns import make_rollout_func  # noqa: E402
from actionproof.ad.train_irc import local_advantages, make_inf_reward  # noqa: E402
from actionproof.integrity.sim.prompts import PROC_NEUTRAL  # noqa: E402


def main():
    rf = make_rollout_func(D.tasks, PROC_NEUTRAL)
    trainer = SimpleNamespace(processing_class=D.tok, vllm_generation=SimpleNamespace(llm=D.FakeLLM([True, False])),
                              state=SimpleNamespace(global_step=0))
    prompts = [[{"role": "system", "content": PROC_NEUTRAL}, {"role": "user", "content": t.prompt}] for t in D.tasks for _ in range(2)]
    out = rf(prompts, trainer=trainer)
    eps = out["ap_episodes"]
    for e, m in zip(eps, trainer._ad_meta):
        print(e["task_id"], e["fault_class"], "reads", e["n_verify"], "inf", e["n_inf_reads"], "uninf", e["n_uninf_reads"],
              "uninf turns", e["n_uninf_turns"], m["uninf_spans"])
        assert e["n_inf_reads"] + e["n_uninf_reads"] == e["n_verify"]
    r = make_inf_reward("r_v", 0.05)(None, [0] * len(eps), ap_episodes=eps)
    T = max(len(x) for x in out["completion_ids"])
    adv = local_advantages([0.0] * len(eps), [m["uninf_spans"] for m in trainer._ad_meta], T, 0.1, 0.05)
    assert any(min(row) < 0 for row in adv), "journal reads should be uninformative in read episodes"
    print("rewards", [round(x, 3) for x in r])
    print("DRYRUN_IRC_OK")


if __name__ == "__main__":
    main()
