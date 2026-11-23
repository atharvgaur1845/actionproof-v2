"""Hidden fault worlds for the costed-verification testbed (CW) and observation-equivalent world marginalisation.

World prior (registered in docs/research/algorithm_discovery/EXPERIMENT_PROTOCOL.md §2; never changed after
registration). For a task with requested steps R:
  P(clean) = 0.6.
  Otherwise one fault on attempt 0 of one requested write, site uniform over R minus "post" (accounting stays
  unfaulted, as in every ActionProof training split), kind from BASE_MIX restricted to the kinds an agent tool can
  repair at that site, renormalised; a partial fault's ratio is uniform over RATIOS.

`CWEnv` is the Stage 3 environment with one change: generated ids are a hash of (rollout seed, call index, draw
number within the call) instead of a running RNG. A dup_delivery's hidden second row then cannot shift the ids shown
by later calls, so two worlds that differ only in an unobserved fault give byte-identical observations to the same
action sequence. Ids stay pseudo-random strings; the agent cannot tell the schemes apart.

`marginalise` replays the agent's exact calls in every world of the prior and keeps the worlds whose every
observation equals the one recorded. Under the prior these are exactly the worlds with the same likelihood for the
observed trajectory (the policy's action probabilities depend on observations only), so the posterior-weighted mean
of R_V over them is E[R_V | trajectory]: the Rao-Blackwellised return.
"""
from __future__ import annotations

import random
from typing import Any

from actionproof.integrity.sim.env import STEP_TOOL
from actionproof.integrity.sim.truth import score
from actionproof.stage3.env import S3ProcEnv

P_CLEAN = 0.6
BASE_MIX = (("phantom", 0.35), ("partial", 0.25), ("dup_delivery", 0.15), ("lost_ack", 0.125),
            ("precommit_timeout", 0.125))
RATIOS = (0.25, 0.5, 0.75)
SITE_KINDS = {"order": ("phantom", "lost_ack", "precommit_timeout"),
              "receive": ("phantom", "partial", "lost_ack", "precommit_timeout"),
              "pay": ("phantom", "partial", "dup_delivery", "lost_ack", "precommit_timeout")}


def sites(requested: list[str]) -> list[str]:
    return [s for s in requested if s != "post"]


def world_prior(requested: list[str]) -> list[tuple[list[dict[str, Any]], float]]:
    """Every world of the prior as (fault list, probability); probabilities sum to 1."""
    out: list[tuple[list[dict[str, Any]], float]] = [([], P_CLEAN)]
    ss = sites(requested)
    for s in ss:
        kinds = SITE_KINDS[s]
        z = sum(p for k, p in BASE_MIX if k in kinds)
        for k, p in BASE_MIX:
            if k not in kinds:
                continue
            pk = (1 - P_CLEAN) / len(ss) * p / z
            if k == "partial":
                for r in RATIOS:
                    out.append(([{"tool": STEP_TOOL[s], "k": 0, "kind": k, "ratio": r}], pk / len(RATIOS)))
            else:
                out.append(([{"tool": STEP_TOOL[s], "k": 0, "kind": k}], pk))
    return out


def sample_world(rng: random.Random, requested: list[str]) -> list[dict[str, Any]]:
    x, acc = rng.random(), 0.0
    worlds = world_prior(requested)
    for faults, p in worlds:
        acc += p
        if x < acc:
            return faults
    return worlds[-1][0]


def world_key(faults: list[dict[str, Any]]) -> tuple:
    return tuple(sorted((f["tool"], int(f["k"]), f["kind"], f.get("ratio")) for f in faults))


def world_class(faults: list[dict[str, Any]]) -> str:
    if not faults:
        return "cw:clean"
    f = faults[0]
    step = next(s for s, t in STEP_TOOL.items() if t == f["tool"])
    return f"cw:{f['kind']}:{step}"


class CWEnv(S3ProcEnv):
    def __init__(self, *args: Any, id_seed: int = 0, **kw: Any) -> None:
        super().__init__(*args, **kw)
        self.id_seed = id_seed
        self._id_n = 0

    def call(self, tool: str, args: dict[str, Any] | None = None, by: str = "agent") -> dict[str, Any]:
        self._id_n = 0
        return super().call(tool, args, by)

    def _next_id(self, prefix: str) -> str:
        n = self._id_n
        self._id_n += 1
        h = random.Random(f"{self.id_seed}:{self._ctx[0]}:{n}").randrange(16**6)
        return f"{prefix}-{h:06X}"


def make_env_cw(task: Any, rollout_id: str, seed: int | None = None,
                faults: list[dict[str, Any]] | None = None) -> CWEnv:
    m = task.meta
    fs = (m.get("faults") or []) if faults is None else faults
    ratios = {(f["tool"], int(f["k"])): float(f["ratio"]) for f in fs if f.get("ratio") is not None}
    s = m["env_seed"] if seed is None else seed
    return CWEnv(task.initial_state, [{k: f[k] for k in ("tool", "k", "kind")} for f in fs], rollout_id=rollout_id,
                 seed=s, never_commit=m.get("never_commit"), ratios=ratios, id_seed=s)


def _record(task_json: dict[str, Any], env: CWEnv) -> dict[str, Any]:
    return {"task": task_json, "trace": env.trace, "attempts": env.attempts, "report": env.final_report,
            "final_state": env.state}


def replay(task: Any, trace: list[dict[str, Any]], faults: list[dict[str, Any]], seed: int,
           task_json: dict[str, Any] | None = None) -> int | None:
    """R_V of the recorded calls replayed in world `faults`, or None if any observation differs from the record."""
    env = make_env_cw(task, "replay", seed=seed, faults=faults)
    for st in trace:
        obs = env.call(st["tool"], dict(st["args"]), by=st.get("by", "agent"))
        if obs != st["obs"]:
            return None
    return int(score(_record(task_json or task.to_json(), env))["r_v"])


def marginalise(task: Any, trace: list[dict[str, Any]], seed: int, alpha: float = 0.25) -> dict[str, Any]:
    """Posterior over the prior's worlds given the recorded trajectory, and the returns it implies.

    r_bar: E[R_V | trajectory]; cvar: mean R_V over the worst alpha posterior mass; post_fault: posterior mass of
    faulted worlds; in_prior: the actual world is one of the prior's worlds (else r_bar falls back to the actual R_V).
    """
    tj = task.to_json()
    actual = world_key(task.meta.get("faults") or [])
    cons: list[tuple[float, int, bool]] = []
    in_prior, actual_rv = False, None
    for faults, p in world_prior(task.meta["requested"]):
        rv = replay(task, trace, faults, seed, tj)
        is_actual = world_key(faults) == actual and not task.meta.get("never_commit")
        if is_actual:
            in_prior = True
            actual_rv = rv
        if rv is not None:
            cons.append((p, rv, bool(faults)))
    den = sum(p for p, _, _ in cons)
    if not in_prior or actual_rv is None or den <= 0:
        return {"in_prior": in_prior, "consistent_actual": actual_rv is not None, "r_bar": None, "cvar": None,
                "post_fault": None, "n_worlds": len(cons)}
    r_bar = sum(p * rv for p, rv, _ in cons) / den
    tail, acc, cv = alpha * den, 0.0, 0.0
    for p, rv, _ in sorted(cons, key=lambda x: x[1]):
        take = min(p, tail - acc)
        if take <= 0:
            break
        cv += take * rv
        acc += take
    return {"in_prior": True, "consistent_actual": True, "r_bar": r_bar, "cvar": cv / tail,
            "post_fault": sum(p for p, _, f in cons if f) / den, "n_worlds": len(cons)}
