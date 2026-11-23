"""Per-read information flags from observation-equivalent replay (diagnostic for FAILURE_ANALYSIS F3; candidate D3).

For a recorded trajectory, replay its calls in every world of the prior and record, per world, the index of the first
call whose observation differs from the record (None if none does). A read call at trace index k is *informative* iff
some world with positive prior mass first diverges exactly at k: only then can the read's observation have eliminated
a world that was still consistent. Otherwise its observation was fully determined by the prior and the history.
Under the prior, its information value is 0 (for example the journal read under cw1, or a re-read of a record already
read with no write in between).
"""
from __future__ import annotations

from typing import Any

from actionproof.ad.worlds import make_env_cw, world_prior
from actionproof.integrity.sim.env import READ_TOOLS


def first_divergence(task: Any, trace: list[dict[str, Any]], faults: list[dict[str, Any]], seed: int) -> int | None:
    env = make_env_cw(task, "readinfo", seed=seed, faults=faults)
    for i, st in enumerate(trace):
        if env.call(st["tool"], dict(st["args"]), by=st.get("by", "agent")) != st["obs"]:
            return i
    return None


def read_flags(task: Any, trace: list[dict[str, Any]], seed: int) -> list[tuple[int, str, bool]]:
    """[(trace index, tool, informative)] for every agent read call in the trace."""
    div = [first_divergence(task, trace, fs, seed) for fs, p in world_prior(task.meta["requested"]) if p > 0]
    hit = {d for d in div if d is not None}
    return [(i, st["tool"], i in hit) for i, st in enumerate(trace)
            if st["tool"] in READ_TOOLS and st.get("by", "agent") == "agent"]
