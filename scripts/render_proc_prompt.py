"""Write the exact ProcureSim system prompt (neutral, with AgentTune's tool block) to data/bench/proc/system_prompt.txt."""
from pathlib import Path

from actionproof.bench.generator_proc import make_split
from actionproof.integrity.sim.prompts import PROC_NEUTRAL
from actionproof.integrity.sim.runtime import make_env_proc
from actionproof.rollout.episode import system_prompt

out = Path(__file__).resolve().parents[1] / "data/bench/proc/system_prompt.txt"
out.write_text(system_prompt(make_env_proc(make_split("proc_dev", 1)[0], "p"), PROC_NEUTRAL))
print(out, len(out.read_text()))
