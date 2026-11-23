"""P2 evaluation (PREREGISTRATION_P2.md §5): P1's episode loop (`scripts/p1_eval.py`, imported unchanged) with two
optional, visible-only additions:
- `--ledger`: every tool-result message gets the observable ledger (`actionproof/p2/ledger.py`);
- `--guard`: every write and every "completed" report passes the deterministic guard first
  (`actionproof/p2/guard.py`). A blocked call is not executed, and the policy sees the guard's reply.

`--api` evaluates in the renamed-tool environment of S4 (`actionproof/stage4/shift.py::make_env_api`).

Decoding, turns, tokens, context guard, seeds and metrics are P1's.

**Provenance.** Each output gets `<out>.provenance.json`, with:
- the git commit, hashes of this script, of P1's evaluator, of the ledger, guard and verifier modules, of the split
  file and of the adapter weights;
- vLLM sampling settings;
- expected and actual episode counts.

The `test` phase refuses to overwrite.

    python scripts/p2_eval.py --model runs/p1_dwarm/merged --adapter runs/p1_d_s0/final --tag Dg_s0 \
        --split p1_test_full --phase test --guard
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("p1_eval", ROOT / "scripts/p1_eval.py")
P1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P1)
E1 = P1.E1


def sha(p: Path) -> str | None:
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def start(env, task, ledger: bool) -> list[dict]:
    from actionproof.p2.ledger import with_ledger
    conv = P1.start(env, task, recon=False)
    if ledger:  # the scripted prefix's tool results carry the ledger too, as every later result does
        for i in range(3, len(conv), 2):
            conv[i] = {"role": "user", "content": with_ledger(conv[i]["content"], conv[:i])}
    return conv


def run(llm, tasks, samples: int, ledger: bool, guard: bool, api: bool, max_turns: int = 16, max_tokens: int = 256):
    from vllm import SamplingParams
    from actionproof.bench.prompts import TOOL_RESULT_FORMAT
    from actionproof.p2 import guard as G
    from actionproof.p2.ledger import with_ledger
    from actionproof.rollout.episode import _agenttune
    from actionproof.stage3.env import make_env_s3
    make_env = make_env_s3
    if api:
        from actionproof.stage4.shift import make_env_api
        make_env = make_env_api
    _, _, parse = _agenttune()
    eps = []
    for t in tasks:
        for j in [-1] + list(range(samples)):
            env = make_env(t, f"p2-{t.task_id}")
            eps.append({"task": t, "sample": j, "env": env, "conv": start(env, t, ledger), "turns": 0, "invalid": 0,
                        "ended_by": "max_turns", "done": False, "n0": len(env.trace), "cfg": G.config(t),
                        "blocks": []})
    tok = llm.get_tokenizer()
    ctx = llm.llm_engine.model_config.max_model_len
    for turn in range(max_turns):
        for e in eps:
            if not e["done"] and len(tok.encode(tok.apply_chat_template(e["conv"], add_generation_prompt=True, tokenize=False),
                                                add_special_tokens=False)) + max_tokens > ctx:
                e["ended_by"], e["done"] = "context", True
        live = [e for e in eps if not e["done"]]
        if not live:
            break
        params = [SamplingParams(temperature=0, max_tokens=max_tokens) if e["sample"] < 0 else
                  SamplingParams(temperature=1.0, max_tokens=max_tokens, seed=E1.seed_of(e["task"].task_id, e["sample"], turn))
                  for e in live]
        outs = llm.chat([e["conv"] for e in live], params, use_tqdm=False)
        for e, o in zip(live, outs):
            text = o.outputs[0].text
            e["turns"] += 1
            e["conv"].append({"role": "assistant", "content": text})
            calls = parse(text) or []
            if not calls:
                e["ended_by"], e["done"] = "no_call", True
                continue
            res = []
            for c in calls:
                name = c["function"]["name"]
                try:
                    args = json.loads(c["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = None
                if not isinstance(args, dict):
                    obs = {"status": "error", "error": "invalid_arguments"}
                else:
                    blocked = G.check(name, args, e["conv"], e["cfg"]) if guard and not api else None
                    if blocked:
                        e["blocks"].append({"turn": turn, "tool": name, "rule": blocked["error"]})
                        obs = blocked
                    else:
                        obs = e["env"].call(name, args)
                e["invalid"] += obs.get("error") in ("unknown_tool", "invalid_arguments")
                res.append(TOOL_RESULT_FORMAT.format(name=name, content=json.dumps(obs)))
            msg = "\n\n".join(res)
            if ledger:
                msg = with_ledger(msg, e["conv"])
            e["conv"].append({"role": "user", "content": msg})
            if e["env"].done:
                e["ended_by"], e["done"] = "report", True
    out = []
    for e in eps:
        r = E1.S3E.episode_record_proc(e["task"], e["env"])
        status = (r["report"] or {}).get("status")
        agent = e["env"].trace[e["n0"]:]
        r.update(sample=e["sample"], turns=e["turns"], invalid=e["invalid"], ended_by=e["ended_by"],
                 conversation=e["conv"], workflow=e["task"].workflow, start_type=e["task"].meta.get("start_type"),
                 violation=E1.S3E.R.violation(r), honest_fail=status in ("failed", "escalated") and not r["score"]["goal_met"],
                 policy_calls=len(agent) + len(e["blocks"]), guard_blocks=e["blocks"], ledger=ledger, guard=guard,
                 **P1.episode_metrics({**r, "task": {"meta": e["task"].meta}}, e["n0"]))
        out.append(r)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--phase", choices=("smoke", "dev", "test", "shift", "ref"), required=True)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--ledger", action="store_true")
    ap.add_argument("--guard", action="store_true")
    ap.add_argument("--n-tasks", type=int, default=None)
    a = ap.parse_args()
    test_like = a.split.startswith("p1_test_") or a.split.startswith("s4_")
    if a.phase != "smoke" and (a.phase in ("test", "shift", "ref")) != test_like:
        raise SystemExit("test/shift/ref phases run only p1_test_* / s4_* splits, and those splits run only there")
    api = a.split == "s4_api"
    if api and a.guard:
        raise SystemExit("the guard reads canonical tool names; it is not defined on the renamed-tool API")
    out = ROOT / f"results/p2/{a.phase}/{a.tag}__{a.split}.jsonl"
    if a.phase in ("test", "shift", "ref") and out.exists():
        raise SystemExit(f"{out} exists; each model and split is run once")
    from actionproof.bench.generator_proc import DATA, load_tasks
    from actionproof.train.merge import merge
    from vllm import LLM
    split_file = DATA / f"{a.split}.jsonl"
    tasks = load_tasks(split_file)[: a.n_tasks]
    model = merge(a.adapter, a.model) if a.adapter else a.model
    llm = LLM(model=model, max_model_len=8192, gpu_memory_utilization=0.85, seed=0, enable_prefix_caching=True)
    t0 = time.time()
    rows = run(llm, tasks, a.samples, a.ledger, a.guard, api)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))
    s = {"tag": a.tag, "split": a.split, "phase": a.phase, "ledger": a.ledger, "guard": a.guard, "model": a.model,
         "adapter": a.adapter, "samples": a.samples, "seconds": round(time.time() - t0, 1), **P1.summarise(rows),
         "guard_blocks_per_episode": sum(len(r["guard_blocks"]) for r in rows) / len(rows)}
    out.with_suffix(".summary.json").write_text(json.dumps(s, indent=2))
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip() or None
    prov = {"tag": a.tag, "split": a.split, "phase": a.phase, "git_commit": git,
            "split_sha256": sha(split_file), "task_ids": [t.task_id for t in tasks],
            "expected_episodes": len(tasks) * (1 + a.samples), "actual_episodes": len(rows),
            "decoding": {"greedy": {"temperature": 0, "max_tokens": 256},
                         "sampled": {"temperature": 1.0, "top_p": 1.0, "max_tokens": 256, "n": a.samples,
                                     "seed": "E1.seed_of(task_id, sample, turn)"},
                         "max_turns": 16, "max_model_len": 8192},
            "model": a.model, "adapter": a.adapter,
            "adapter_sha256": sha(Path(a.adapter) / "adapter_model.safetensors") if a.adapter else None,
            "code_sha256": {p: sha(ROOT / p) for p in ("scripts/p2_eval.py", "scripts/p1_eval.py", "scripts/e1_eval.py",
                                                       "actionproof/p2/ledger.py", "actionproof/p2/guard.py",
                                                       "actionproof/integrity/sim/truth.py", "actionproof/integrity/sim/env.py")},
            "ledger": a.ledger, "guard": a.guard, "api": api}
    out.with_suffix(".provenance.json").write_text(json.dumps(prov, indent=2))
    print(json.dumps({"tag": a.tag, "split": a.split, "greedy_vsc": s["greedy"]["vsc"], "sampled_vsc": s["sampled"]["vsc"]}))


if __name__ == "__main__":
    main()
