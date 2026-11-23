"""Run a LAB1 family (PREREGISTRATION_LAB1.md).

    python scripts/lab/lab1_sweep.py L1|L2N|L3G [--workers 14]
    python scripts/lab/lab1_sweep.py L1 --dry 6     # plumbing/timing check on seed 99 (not a registered seed); writes nothing

Writes results/lab/tabular/<FAM>.jsonl (one line per run; resumable, runs on file are skipped).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from actionproof.ad2.tabular import envs as E  # noqa: E402

OUT = ROOT / "results/lab/tabular"
SETTINGS = [(m, c, L) for m in ("single", "multi") for c in (0.02, 0.05, 0.1) for L in (0.0, 1.0)]
SEEDS = range(5)
BUDGETS = {"E1": 6400, "E4": 25600}
STEP_BASE = ["grpo", "nostd", "rloo", "critic_v", "critic_q", "ppo", "ppo_norm", "igc_v1", "awrb", "awrbn"]
ONLY_E1 = {"grpo", "awrb", "ppo_norm"}
CDVO = [("fork_act", None), ("fork_act_nocrn", None), ("cdvo_fixed", 0.25), ("cdvo_adapt", 0.25),
        ("cdvo_adapt", 0.10), ("cdvo_adapt_critic", 0.25), ("cdvo_ia", 0.25)]


def family(fam: str) -> list[dict]:
    runs = []
    if fam == "L1":
        for (mode, c, L) in SETTINGS:
            for seed in SEEDS:
                for bname, eps in BUDGETS.items():
                    for est in STEP_BASE:
                        if bname != "E1" and est in ONLY_E1:
                            continue
                        runs.append({"id": f"L1|{mode}|{c}|{L}|{est}|{bname}|{seed}", "env": ["rich", mode, c, L],
                                     "est": est, "seed": seed, "steps": eps // 32, "n_mc": 1000})
                    for est, b in CDVO:
                        tag = est if b is None else f"{est}@{b}"
                        runs.append({"id": f"L1|{mode}|{c}|{L}|{tag}|{bname}|{seed}", "env": ["rich", mode, c, L],
                                     "est": est, "seed": seed, "episode_budget": eps, "branch_frac": b or 0.25,
                                     "fork_k": 1, "n_mc": 1000})
    elif fam == "L2N":
        for p in (0.1, 0.2, 0.4, 0.6):
            for c in (0.01, 0.05, 0.2):
                for s in (0.6, 1.0):
                    for est in ("grpo", "nostd", "indep_world_std", "indep_task_std"):
                        for seed in range(3):
                            runs.append({"id": f"L2N|{p}|{c}|{s}|{est}|{seed}", "env": ["core", p, c, s, 0.0],
                                         "est": est, "seed": seed, "steps": 200, "n_mc": 300, "n_ind": 16})
    elif fam == "L3G":
        for est, b in (("nostd", None), ("rloo", None), ("igc_v1", None), ("ppo", None), ("critic_q", None),
                       ("cdvo_adapt", 0.25), ("cdvo_ia", 0.25), ("fork_act", None)):
            for seed in SEEDS:
                tag = est if b is None else f"{est}@{b}"
                r = {"id": f"L3G|{tag}|{seed}", "env": ["gen"], "est": est, "seed": seed, "n_mc": 200, "mode": "feat",
                     "fork_k": 1, "branch_frac": b or 0.25}
                if est.startswith(("cdvo", "fork")):
                    r["episode_budget"] = 4 * 12800
                else:
                    r["steps"] = 1600
                runs.append(r)
    else:
        raise SystemExit(f"unknown family {fam}")
    return runs


def build(env: list):
    if env[0] == "core":
        t = [E.core(*env[1:])]
        return t, t
    if env[0] == "rich":
        t = E.rich(env[1], env[2], env[3])
        return t, t
    if env[0] == "gen":
        tr = E.gen_tasks("train")
        return tr, tr[:6] + E.gen_tasks("test_iid") + E.gen_tasks("test_shift") + E.gen_tasks("test_long")
    raise ValueError(env)


def run_one(r: dict) -> dict:
    from actionproof.lab.cdvo import train
    tasks, ev = build(r["env"])
    spec = {k: v for k, v in r.items() if k not in ("id", "env")}
    spec.update(tasks=tasks, eval_tasks=ev)
    t = time.time()
    out = train(spec)
    out.pop("theta", None)
    out.pop("curve", None) if r["env"][0] == "core" else None
    return {"id": r["id"], "env": r["env"], **out, "seconds": time.time() - t}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("family")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--dry", type=int, default=0)
    ap.add_argument("--smoke-dir", default=None, help="write a seed-99 smoke subset (seeds 0-1 relabelled) here")
    a = ap.parse_args()
    fam = family(a.family)
    if a.dry:
        step = max(1, len(fam) // a.dry)
        for r in fam[::step][: a.dry]:
            r = {**r, "seed": 99, "id": r["id"] + "|dry"}
            res = run_one(r)
            print(r["id"], f"{res['seconds']:.1f}s eps={res['episodes_used']:.0f} forks={res['n_forks']}", flush=True)
        return
    out_dir = OUT
    if a.smoke_dir:  # analysis smoke test: two settings, seeds 98/99 relabelled 0/1, written outside results/
        out_dir = Path(a.smoke_dir)
        keep = [r for r in fam if r["seed"] in (0, 1) and ("|multi|0.05|1.0|" in r["id"] or "|single|0.1|0.0|" in r["id"]
                                                          or r["id"].startswith(("L2N|0.4|", "L3G")))]
        fam = [{**r, "seed": 98 + r["seed"]} for r in keep]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{a.family}.jsonl"
    done = set()
    if path.exists():
        done = {json.loads(l)["id"] for l in path.read_text().splitlines() if l.strip()}
    todo = [r for r in fam if r["id"] not in done]
    # longest first (fork runs), for load balance
    todo.sort(key=lambda r: (0 if "episode_budget" in r else 1, -r.get("steps", 0)))
    print(f"{a.family}: {len(todo)} runs to do ({len(done)} on file)", flush=True)
    t0 = time.time()
    with Pool(a.workers) as pool, path.open("a") as f:
        for i, res in enumerate(pool.imap_unordered(run_one, todo)):
            f.write(json.dumps(res, default=float) + "\n")
            f.flush()
            if i % 100 == 0:
                print(f"{i}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    print(f"{a.family} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
