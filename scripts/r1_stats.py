"""R1 capability retention: registered analysis (PREREGISTRATION_R1.md §6). Frozen before any output.

Reads `results/r1/score/<tag>.jsonl`; writes `results/r1/r1_summary.json`.

Estimand for an arm with seeds s = 1..S and an endpoint E:
    Δ = E(mean over seeds of the per-item scores) − E(reference per-item scores).
E is a stratified mean: per-category accuracy, then the unweighted mean over the endpoint's categories
(BFCL's own non-live aggregation; for single-category endpoints this is plain accuracy).
Uncertainty: a paired bootstrap over items, stratified by category, B = 10,000, the same resampled
items for both sides; the trained models are held fixed (task-level uncertainty only). Seed-level Δs are
reported next to it.

    .venv-r1/bin/python scripts/r1_stats.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCORE = ROOT / "results/r1/score"
B, SEED, MARGIN = 10_000, 20261123, 0.03

ENDPOINTS = {
    "ifeval_prompt_strict": ("ifeval",),
    "bfcl_nonlive_ast": ("simple_python", "multiple", "parallel", "parallel_multiple"),
    "bfcl_irrelevance": ("irrelevance",),
    "bfcl_live_ast": ("live_simple", "live_multiple", "live_parallel", "live_parallel_multiple"),
    "bfcl_live_irrelevance": ("live_irrelevance",),
}
ARMS = {
    "a12_rv": [f"a12_rv_s{i}" for i in range(5)],
    "a12_ra": [f"a12_ra_s{i}" for i in range(5)],
    "rsc": [f"rsc_s{i}" for i in range(5)],
    "control": [f"control_s{i}" for i in range(5)],
    "ackonly": [f"ackonly_s{i}" for i in range(5)],
    "sr1": [f"sr1_s{i}" for i in range(3)],
    "s1_pcstr_d": ["s1_pcstr_d_s0", "s1_pcstr_d_s1"],
    "s1_tool_trust": ["s1_tool_trust_s0", "s1_tool_trust_s1"],
}
PRIMARY = [(arm, ep) for arm in ("a12_rv", "rsc") for ep in ("ifeval_prompt_strict", "bfcl_nonlive_ast")]


def load(tag: str) -> dict:
    p = SCORE / f"{tag}.jsonl"
    if not p.exists():
        return {}
    return {(r["category"], r["id"]): float(r["correct"]) for r in map(json.loads, p.read_text().splitlines())}


def strata(rows: dict, cats: tuple) -> dict[str, list]:
    out: dict[str, list] = {c: [] for c in cats}
    for (c, i) in sorted(rows):
        if c in out:
            out[c].append((c, i))
    return out


def endpoint(vec_by_cat: dict[str, np.ndarray]) -> float:
    return float(np.mean([v.mean() for v in vec_by_cat.values() if len(v)]))


def contrast(arm_tags: list[str], ref_tag: str | list[str], cats: tuple, rng: np.random.Generator) -> dict | None:
    """Δ = arm (seed-averaged) − reference; a list reference is seed-averaged the same way."""
    refs = [load(t) for t in ([ref_tag] if isinstance(ref_tag, str) else ref_tag)]
    arms = [load(t) for t in arm_tags]
    if any(not x for x in refs + arms):
        return None
    common = set.intersection(*(set(x) for x in refs))
    ref = {k: float(np.mean([x[k] for x in refs])) for k in common}
    keys = strata(ref, cats)
    for c in keys:  # items scored for every model (over-length items are absent for all alike)
        keys[c] = [k for k in keys[c] if all(k in a for a in arms)]
    r = {c: np.array([ref[k] for k in ks]) for c, ks in keys.items()}
    m = {c: np.array([np.mean([a[k] for a in arms]) for k in ks]) for c, ks in keys.items()}
    per_seed = [endpoint({c: np.array([a[k] for k in ks]) for c, ks in keys.items()}) - endpoint(r) for a in arms]
    est = endpoint(m) - endpoint(r)
    boots = np.empty(B)
    idx = {c: rng.integers(0, len(ks), size=(B, len(ks))) for c, ks in keys.items() if len(ks)}
    for b in range(B):
        boots[b] = np.mean([m[c][idx[c][b]].mean() - r[c][idx[c][b]].mean() for c in idx])
    return {"arm": arm_tags, "ref": ref_tag, "ref_value": endpoint(r), "arm_value": endpoint(m),
            "delta": est, "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
            "p_noninferior": float((1 + np.sum(boots <= -MARGIN)) / (B + 1)),
            "p_less_than_zero": float((1 + np.sum(boots >= 0)) / (B + 1)),
            "per_seed_delta": per_seed, "n_items": {c: len(ks) for c, ks in keys.items()}}


def holm(ps: list[float]) -> list[float]:
    order = np.argsort(ps)
    adj, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def main() -> None:
    rng = np.random.default_rng(SEED)
    out: dict = {"margin": MARGIN, "B": B, "contrasts": {}}
    # H1: the SFT warm start against the base model
    for ep, cats in ENDPOINTS.items():
        out["contrasts"][f"warm_vs_base:{ep}"] = contrast(["warm_start"], "base", cats, rng)
        out["contrasts"][f"s1warm_vs_base:{ep}"] = contrast(["s1_warm_start"], "base", cats, rng)
    # H2-H4 and descriptive rows: every arm against its own warm start
    for arm, tags in ARMS.items():
        ref = "s1_warm_start" if arm.startswith("s1_") else "warm_start"
        for ep, cats in ENDPOINTS.items():
            out["contrasts"][f"{arm}_vs_warm:{ep}"] = contrast(tags, ref, cats, rng)
    # H4 (exploratory): tool-name randomisation against RSC on unseen function schemas
    for ep in ("bfcl_live_ast", "bfcl_nonlive_ast"):
        out["contrasts"][f"sr1_vs_rsc:{ep}"] = contrast(ARMS["sr1"], ARMS["rsc"], ENDPOINTS[ep], rng)
    c = out["contrasts"]
    # H2 (primary): non-inferiority at margin 0.03, Holm over four tests
    prim = [c.get(f"{a}_vs_warm:{e}") for a, e in PRIMARY]
    if all(prim):
        adj = holm([x["p_noninferior"] for x in prim])
        out["H2"] = {f"{a}:{e}": {"delta": x["delta"], "ci95": x["ci95"], "holm_p": p, "noninferior": p < 0.05}
                     for (a, e), x, p in zip(PRIMARY, prim, adj)}
    # H3: irrelevance accuracy falls, Holm over two tests
    h3 = [c.get(f"{a}_vs_warm:bfcl_irrelevance") for a in ("a12_rv", "rsc")]
    if all(h3):
        adj = holm([x["p_less_than_zero"] for x in h3])
        out["H3"] = {a: {"delta": x["delta"], "ci95": x["ci95"], "holm_p": p, "supported": p < 0.05}
                     for a, x, p in zip(("a12_rv", "rsc"), h3, adj)}
    # H1: the warm start loses capability (both endpoints; two-sided CI excludes 0 on the negative side)
    h1 = {e: c.get(f"warm_vs_base:{e}") for e in ("ifeval_prompt_strict", "bfcl_nonlive_ast")}
    if all(h1.values()):
        out["H1"] = {e: {"delta": x["delta"], "ci95": x["ci95"], "supported": x["ci95"][1] < 0} for e, x in h1.items()}
    (ROOT / "results/r1/r1_summary.json").write_text(json.dumps(out, indent=2))
    for k in ("H1", "H2", "H3"):
        print(k, json.dumps(out.get(k), indent=1) if out.get(k) else "pending")


if __name__ == "__main__":
    main()
