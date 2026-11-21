"""FCA per-token advantages (docs/research/algorithm/FCA_GRPO.md §2). Pure function; unit-tested on CPU.

meta[i] = {"group": str, "kind": "on" | "forced" | "plain", "cls": str | None, "span": (start, end) | None}
rewards[i] = episode reward; T = padded completion length; base[i] = TRL's sequence advantage (used for "plain").
"""
from __future__ import annotations

from collections import defaultdict
from statistics import mean, pstdev
from typing import Any

EPS = 1e-4


def fca_advantages(rewards: list[float], meta: list[dict[str, Any]], T: int, base: list[float]) -> tuple[list[list[float]], list[dict]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, m in enumerate(meta):
        groups[m["group"]].append(i)
    adv = [[base[i]] * T for i in range(len(rewards))]
    diag = []
    for g, idx in groups.items():
        if all(meta[i]["kind"] == "plain" for i in idx):
            continue
        on = [i for i in idx if meta[i]["kind"] == "on"]
        if not on:
            raise ValueError(f"group {g} has no on-policy rollout")
        v = mean(rewards[i] for i in on)
        sigma = pstdev([rewards[i] for i in idx]) + EPS
        q: dict[str, list[float]] = defaultdict(list)
        for i in idx:
            if meta[i]["cls"] is not None:
                q[meta[i]["cls"]].append(rewards[i])
        qhat = {c: mean(v_) for c, v_ in q.items()}
        for i in on:
            adv[i] = [(rewards[i] - v) / sigma] * T
        for i in idx:
            if meta[i]["kind"] != "forced":
                continue
            row = [0.0] * T
            s, e = meta[i]["span"]
            a = (qhat[meta[i]["cls"]] - v) / sigma
            for t in range(s, min(e, T)):
                row[t] = a
            adv[i] = row
        diag.append({"group": g, "V": v, "sigma": sigma, "Q": qhat, "n_on": len(on),
                     "on_zero_var": len({rewards[i] for i in on}) == 1})
    return adv, diag
