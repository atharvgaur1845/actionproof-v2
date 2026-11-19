"""Amendment 11 evaluation entry: run a policy through the e-commerce gateway in one mode over a split.

Scripted agents (CPU, from rewards/sanity_agents.py):
    python scripts/eval_gateway.py --agent task_oracle --mode enforce --split test_iid_c2 --out results/a11/x.jsonl
vLLM policy (GPU; mirrors eval/run_eval.py: its `run` is reused UNCHANGED, only the module-level env factory and
episode-record functions are swapped, exactly as eval/run_eval_a4.py does):
    python scripts/eval_gateway.py --model runs/rft_tiny_v2/merged --adapter runs/X/final --mode enforce_inform \
        --split v2_stress_c2 --max-turns 12 --out results/a11/X__enforce_inform__v2_stress_c2.jsonl
Observe mode offline, from passthrough episodes (replays each recorded call sequence in a fresh environment through
an observe gateway, asserts every observation is byte-identical to the recorded one, and adds the verdicts):
    python scripts/eval_gateway.py --offline-observe results/a11/X__passthrough__test_iid_c2.jsonl \
        --split test_iid_c2 --out results/a11/X__observe__test_iid_c2.jsonl

Writes per-episode JSONL to --out and a summary (system level and agent level kept separate, plus a per-cell
breakdown over the frozen taxonomy labels) to <out>.summary.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from actionproof.bench.generator import Task  # noqa: E402
from actionproof.integrity.adapters.ecom import (MODES, EcomGateway, gateway_episode_record,  # noqa: E402
                                                 records_equal, summarize_gateway)
from actionproof.rollout.agenttune_rollout import make_env as base_make_env  # noqa: E402

C2 = ROOT / "data/bench/c2"
LABELS = C2 / "fault_labels.jsonl"
SCRIPTED = ("task_oracle", "lean_expert", "oracle", "over_verifier", "lazy_liar", "blind_retrier",
            "policy_violator", "over_escalator", "no_op")


def load_tasks(split: str | None, tasks: str | None, n: int | None = None) -> list[Task]:
    path = Path(tasks) if tasks else C2 / f"{split}.jsonl"
    out = [Task.from_json(json.loads(line)) for line in path.open()]
    return out[:n] if n is not None else out


def file_sha(p: str | Path | None) -> str | None:
    if not p or not Path(p).exists():
        return None
    h = hashlib.sha256()
    for f in sorted(Path(p).rglob("*")) if Path(p).is_dir() else [Path(p)]:
        if f.is_file() and f.suffix in (".safetensors", ".jsonl", ".json"):
            h.update(f.name.encode())
            h.update(f.read_bytes())
    return h.hexdigest()


def scripted_agent(name: str) -> Callable[[Any, Task], None]:
    from actionproof.rewards import sanity_agents as S
    if name == "task_oracle":
        return lambda env, t: S.task_oracle(env, t.workflow, t.meta["ids"]["order"], t.meta["ids"]["ticket"])
    if name == "lean_expert":
        return lambda env, t: S.lean_expert(env, t.meta["ids"]["order"], t.meta["ids"]["ticket"])
    fn = S.AGENTS[name]
    return lambda env, t: fn(env, S.RefundTask(t.meta["ids"]["order"], t.meta["ids"]["ticket"],
                                               t.meta["ids"]["amount"], t.meta["ids"]["method"]))


def run_scripted(agent: str, tasks: list[Task], mode: str, seed: int = 0, limits: bool = True) -> list[dict]:
    play = scripted_agent(agent)
    recs = []
    for t in tasks:
        gw = EcomGateway(base_make_env(t, f"ev-{t.task_id}-0", seed=seed), t, mode, limits=limits)
        err = None
        try:
            play(gw, t)
        except (KeyError, TypeError, IndexError) as e:  # a scripted agent indexing a refusal or a finished episode
            err = f"{type(e).__name__}: {e}"
        r = gateway_episode_record(t, gw)
        r.update(agent=agent, agent_error=err, env_seed=seed, sample=0, workflow=t.workflow, split=t.split,
                 benign=t.meta.get("benign"))
        recs.append(r)
    return recs


def offline_observe(path: str, tasks: list[Task], seed: int = 0) -> list[dict]:
    """Observe-mode records from passthrough records: same call sequence, fresh env, observe gateway."""
    by_id = {t.task_id: t for t in tasks}
    out = []
    for line in open(path):
        src = json.loads(line)
        if src.get("gateway", {}).get("mode", "passthrough") != "passthrough":
            raise SystemExit(f"{path}: offline observe needs passthrough episodes, got {src['gateway']['mode']}")
        t = by_id[src["task_id"]]
        gw = EcomGateway(base_make_env(t, src["rollout_id"], seed=src.get("env_seed", seed)), t, "observe")
        for i, step in enumerate(src["trace"]):
            obs = gw.call(step["tool"], step["args"])
            if not records_equal(obs, step["obs"]):
                raise SystemExit(f"{src['task_id']} step {i}: replayed observation differs from the recorded one")
        r = gateway_episode_record(t, gw)
        if r["vsc"] != src["vsc"] or not records_equal(r["verdict"], src["verdict"]):
            raise SystemExit(f"{src['task_id']}: replayed verdict differs from the passthrough record")
        keep = ("sample", "turns", "unparsed", "invalid", "ended_by", "workflow", "split", "benign", "model",
                "adapter", "prompt", "agent", "agent_error")
        r.update({k: src[k] for k in keep if k in src}, env_seed=src.get("env_seed", seed), offline_from=path)
        out.append(r)
    return out


def run_vllm(a: argparse.Namespace, tasks: list[Task]) -> list[dict]:
    import actionproof.eval.run_eval as RE
    mode, limits = a.mode, not a.no_limits
    RE.make_env = lambda t, rid, seed=0: EcomGateway(base_make_env(t, rid, seed=seed), t, mode, limits=limits)
    RE.episode_record = gateway_episode_record  # swap factory and record only; decoding and turn loop unchanged
    from vllm import LLM
    from actionproof.train.merge import merge
    weights = merge(a.adapter, a.model) if a.adapter else a.model
    llm = LLM(weights, tensor_parallel_size=a.tp, gpu_memory_utilization=a.gpu_mem,
              max_model_len=a.max_model_len, dtype="bfloat16", seed=a.seed)
    recs = RE.run(llm, tasks, RE.PROMPTS[a.prompt], a.samples, a.temperature, a.max_turns, a.max_tokens, None, a.seed)
    for r in recs:
        r.update(env_seed=a.seed, model=a.model, adapter=a.adapter, prompt=a.prompt)
    return recs


def by_cell(records: list[dict], labels_path: Path) -> dict[str, Any]:
    if not labels_path.exists():
        return {}
    labels = {l["task_id"]: l for l in map(json.loads, labels_path.open())}
    cells: dict[str, list[dict]] = {}
    for r in records:
        lab = labels.get(r["task_id"])
        cells.setdefault(lab["cell"] if lab else "unlabelled", []).append(r)
    return {c: {"n": len(rs), "system_vsc": sum(r["system_level"]["vsc"] for r in rs) / len(rs),
                "agent_vsc": sum(r["agent_level"]["vsc"] for r in rs) / len(rs)} for c, rs in sorted(cells.items())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=MODES, default="passthrough")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--split", help="c2 split name (data/bench/c2/<split>.jsonl)")
    src.add_argument("--tasks", help="any task jsonl")
    ap.add_argument("--n-tasks", type=int, default=None)
    pol = ap.add_mutually_exclusive_group(required=True)
    pol.add_argument("--agent", choices=SCRIPTED, help="scripted agent (CPU)")
    pol.add_argument("--model", help="vLLM policy weights (GPU)")
    pol.add_argument("--offline-observe", metavar="PASSTHROUGH_JSONL")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--prompt", default="neutral")
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-turns", type=int, default=12)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-limits", action="store_true", help="enforce modes: disable the refund policy limits")
    ap.add_argument("--labels", default=str(LABELS))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    tasks = load_tasks(a.split, a.tasks, a.n_tasks)
    t0 = time.time()
    if a.offline_observe:
        a.mode = "observe"
        recs = offline_observe(a.offline_observe, tasks, a.seed)
        policy = {"offline_from": a.offline_observe}
    elif a.agent:
        recs = run_scripted(a.agent, tasks, a.mode, a.seed, limits=not a.no_limits)
        policy = {"agent": a.agent}
    else:
        recs = run_vllm(a, tasks)
        policy = {"model": a.model, "adapter": a.adapter, "adapter_sha256": file_sha(a.adapter), "prompt": a.prompt}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in recs:
            f.write(json.dumps({**r, "mode": a.mode}, sort_keys=True, default=str) + "\n")
    tasks_file = a.tasks or str(C2 / f"{a.split}.jsonl")
    s = {**summarize_gateway(recs), "by_cell": by_cell(recs, Path(a.labels)), "mode": a.mode, "policy": policy,
         "split": a.split, "tasks": tasks_file, "tasks_sha256": file_sha(tasks_file), "limits": not a.no_limits,
         "max_turns": a.max_turns, "temperature": a.temperature, "seed": a.seed, "seconds": round(time.time() - t0)}
    out.with_suffix(".summary.json").write_text(json.dumps(s, indent=2, default=str))
    print(json.dumps({k: s[k] for k in ("n", "mode", "system", "agent")}, default=str))


if __name__ == "__main__":
    main()
