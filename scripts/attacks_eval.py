"""Model evaluation on the attack suite (docs/attacks/SPEC.md §8). PREPARED, NOT YET RUN.

The TXN v1 episode loop of scripts/txn_eval.py (p1_eval's `run`: same prompts, tool schemas, parser, 16 turns, 256 new
tokens), with the attack layer in front of the tools and, with --rules, the frozen RG1 Recovery Gate in front of that.
As in scripts/rg1_eval.py only two things are patched for the duration of the run:
- the env factory (`actionproof.stage3.env.make_env_s3`) builds `make_env_attack(task, ..., rules)`;
- the episode-record builder lets the gate finalise an episode that ended without a report and attaches the gate log
  and the attack log (backend.record).
Every row gets `txn`, `invariants`, `decision_record` (as txn_eval) and `harm` (actionproof/attacks/harm.py).

    python scripts/attacks_eval.py --model runs/p1_dwarm/merged --adapter runs/p1_d_s0/final --tag D_s0_none \
        --split attack_dev --rules none --samples 2

attack_heldout is refused unless a FROZEN registration exists (PREREGISTRATION_ATTACKS.md, not the draft) that quotes
the held-out sha256 of data/attacks/attack_manifest.json, and each tag runs once (an existing output is never
overwritten).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location("txn_eval", ROOT / "scripts/txn_eval.py")
TE = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(TE)

SPLITS = ("attack_dev", "attack_heldout")
REGISTRATION = ROOT / "PREREGISTRATION_ATTACKS.md"
CODE = ("actionproof/attacks/env.py", "actionproof/attacks/harm.py", "actionproof/attacks/backend.py",
        "actionproof/attacks/generate.py", "actionproof/attacks/templates.py", "actionproof/attacks/spec.py",
        "actionproof/recovery_gate/gate.py", "actionproof/recovery_gate/env.py", "scripts/attacks_eval.py")


def code_hashes() -> dict[str, str]:
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in CODE}


def check_registration() -> None:
    man = json.loads((ROOT / "data/attacks/attack_manifest.json").read_text())
    want = man["splits"]["attack_heldout"]["sha256"]
    if not REGISTRATION.exists():
        raise SystemExit(f"{REGISTRATION.name} does not exist: the held-out split runs only after a frozen registration")
    text = REGISTRATION.read_text()
    if "DRAFT" in text.splitlines()[0] or want not in text:
        raise SystemExit("the registration is a draft or does not quote the held-out sha256")


def run_attack(llm, tasks, samples: int, rules, backend: str = "txn_v1") -> list[dict]:
    import actionproof.stage3.env as S3
    from actionproof.attacks import harm
    from actionproof.attacks.backend import get_backend, make_env_attack
    S3E = TE.P1.E1.S3E
    b = get_backend(backend)
    orig_env, orig_rec = S3.make_env_s3, S3E.episode_record_proc
    S3.make_env_s3 = lambda task, rid, seed=None: make_env_attack(task, rid, seed=seed, rules=rules, backend=backend)
    S3E.episode_record_proc = b.record
    try:
        rows = TE.P1.run(llm, tasks, samples, recon=False)
    finally:
        S3.make_env_s3, S3E.episode_record_proc = orig_env, orig_rec
    out = []
    for r in rows:
        r = TE.annotate(r)
        r["harm"] = harm.flags(r)
        out.append(r)
    return out


def main() -> None:
    from actionproof.attacks.generate import load_split
    from actionproof.attacks.metrics import breakdown, summarise
    from actionproof.recovery_gate.gate import ABLATIONS, rules_of
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--split", choices=SPLITS, required=True)
    ap.add_argument("--rules", choices=sorted(ABLATIONS) + ["none"], required=True)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--n-tasks", type=int, default=None)
    ap.add_argument("--backend", default="txn_v1")
    a = ap.parse_args()
    out = ROOT / f"results/attacks/models/{a.tag}__{a.split}.jsonl"
    if out.exists():
        raise SystemExit(f"{out} exists; each tag and split is run once")
    if a.split == "attack_heldout":
        if a.n_tasks is not None:
            raise SystemExit("the held-out split is run in full")
        check_registration()
    rules = None if a.rules == "none" else rules_of(a.rules)
    tasks = load_split(a.split)[: a.n_tasks]
    t0 = time.time()
    from vllm import LLM
    from actionproof.train.merge import merge
    model = merge(a.adapter, a.model) if a.adapter else a.model
    llm = LLM(model=model, max_model_len=8192, gpu_memory_utilization=a.gpu_mem, seed=0, enable_prefix_caching=True)
    rows = run_attack(llm, tasks, a.samples, rules, a.backend)
    for r in rows:
        r["tag"] = a.tag
        r["task_id"] = r["task"]["task_id"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, default=str, ensure_ascii=False) + "\n" for r in rows))
    flags = [{**r["harm"], "task_id": r["task_id"], "sample": r["sample"], "wall_ms": 0.0} for r in rows]
    greedy = [f for f in flags if f["sample"] < 0]
    s = {"tag": a.tag, "split": a.split, "rules": a.rules, "model": a.model, "adapter": a.adapter, "samples": a.samples,
         "n_tasks": len(tasks), "seconds": round(time.time() - t0, 1), "code_sha256": code_hashes(),
         "jsonl_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
         "greedy": summarise(greedy), "all": summarise(flags), "greedy_by_family": breakdown(greedy, "family"),
         "txn": TE.txn_summary(rows)}
    out.with_suffix(".summary.json").write_text(json.dumps(s, indent=2, ensure_ascii=False))
    g = s["greedy"]
    print(json.dumps({"tag": a.tag, "split": a.split, "rules": a.rules,
                      "ASR": g.get("attack", {}).get("attack_success"), "clean": g.get("twin", {}).get("clean_success")}))


if __name__ == "__main__":
    main()
