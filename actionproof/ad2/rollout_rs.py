"""Group-generation condition B (resampled hidden world) for the LLM CW testbed.

First-stage training (condition A) gives all G rollouts of a group the task's single pool world `meta.faults`.
Condition B draws an independent world for every rollout from the same registered cw1 prior
(actionproof/ad/worlds.py:sample_world), keyed by the rollout's unique id. Prompt, initial state, tools and
observation model are unchanged, so the agent observes the hidden world under neither condition; only the joint
distribution of worlds within a group changes (shared vs i.i.d.).

Note (audit finding, 2026-11-23): the first-stage rollout's numeric `seed` is
uuid.UUID(hex12.ljust(32, "0")).int % 2**31, which is 0 for every rollout (the low 80 bits are the zero padding).
Every rollout therefore shares env seed and id seed 0. That is harmless for the first stage, because replay uses the
same seed, but the seed cannot key a per-rollout world. Worlds are keyed by rollout id here, and found again at
marginalisation through the trace object.

`install()` rebinds three names inside actionproof.ad.rollout (environment factory, marginaliser, episode record) so
that the unchanged first-stage rollout function runs condition B. No first-stage file is edited.
"""
from __future__ import annotations

import copy
import random
from typing import Any

import actionproof.ad.rollout as R
from actionproof.ad.worlds import make_env_cw, marginalise, sample_world, world_class
from actionproof.integrity.sim.runtime import episode_record_proc

_WORLDS: dict[int, list[dict[str, Any]]] = {}  # id(env.trace) -> world


def world_for(task: Any, key: Any) -> list[dict[str, Any]]:
    return sample_world(random.Random(f"ad2-rs:{key}"), task.meta["requested"])


def with_world(task: Any, faults: list[dict[str, Any]]) -> Any:
    t = copy.copy(task)
    t.meta = {**task.meta, "faults": copy.deepcopy(faults), "fault_class": world_class(faults)}
    return t


def env_rs(task: Any, rollout_id: str, seed: int | None = None, faults: list[dict[str, Any]] | None = None):
    if faults is None:
        faults = world_for(task, rollout_id)
    env = make_env_cw(task, rollout_id, seed=seed, faults=faults)
    env._ad2_faults = faults
    _WORLDS[id(env.trace)] = faults
    return env


def record_rs(task: Any, env: Any) -> dict[str, Any]:
    fs = getattr(env, "_ad2_faults", None)
    rec = episode_record_proc(task if fs is None else with_world(task, fs), env)
    if fs is not None:
        rec["world"] = fs
    return rec


def marginalise_rs(task: Any, trace: list[dict[str, Any]], seed: int, alpha: float = 0.25) -> dict[str, Any]:
    fs = _WORLDS.pop(id(trace), None)
    if fs is None:
        raise RuntimeError("condition B: no recorded world for this trace")
    return marginalise(with_world(task, fs), trace, seed, alpha)


def install() -> None:
    R.make_env_cw = env_rs
    R.marginalise = marginalise_rs
    R.episode_record_proc = record_rs
