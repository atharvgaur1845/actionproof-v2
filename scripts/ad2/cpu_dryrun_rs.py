"""CPU dry run of condition B (resampled worlds) through the unchanged first-stage rollout function, with the
first-stage fake LLM (scripts/ad/cpu_dryrun.py). Plumbing check only, not an experiment.

    HF_HUB_OFFLINE=1 python scripts/ad2/cpu_dryrun_rs.py
"""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from actionproof.ad2 import rollout_rs  # noqa: E402

rollout_rs.install()
spec = importlib.util.spec_from_file_location("dry", ROOT / "scripts/ad/cpu_dryrun.py")
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)
from actionproof.ad.rollout import make_rollout_func  # noqa: E402
from actionproof.integrity.sim.prompts import PROC_NEUTRAL  # noqa: E402

rf = make_rollout_func(D.tasks, PROC_NEUTRAL)
tr = SimpleNamespace(processing_class=D.tok, vllm_generation=SimpleNamespace(llm=D.FakeLLM([True, False])),
                     state=SimpleNamespace(global_step=0))
prompts = [[{"role": "system", "content": PROC_NEUTRAL}, {"role": "user", "content": t.prompt}]
           for t in D.tasks for _ in range(8)]
eps = rf(prompts, trainer=tr)["ap_episodes"]
by_task = {}
for e in eps:
    assert e["ad"]["r_bar"] is not None and e["ad"]["consistent_actual"], e["ad"]
    by_task.setdefault(e["task_id"], set()).add(e["fault_class"])
for k, v in by_task.items():
    print(k, sorted(v))
assert any(len(v) > 1 for v in by_task.values()), "worlds should vary within a task's group"
print("DRYRUN_RS_OK")
