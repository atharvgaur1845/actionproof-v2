"""Statistics for ActionProof reporting (spec §7, §15).

Wilson intervals for rates; paired bootstrap over tasks for model-vs-model deltas;
hierarchical bootstrap (seed -> task) for multi-seed arms.
"""
from __future__ import annotations

import math
import random
from typing import Sequence


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Point estimate and Wilson score interval for k successes in n trials."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, centre - half), min(1.0, centre + half)


def paired_bootstrap(a: dict[str, float], b: dict[str, float], iters: int = 10_000,
                     seed: int = 0) -> tuple[float, float, float]:
    """Mean of (a - b) over shared task ids with a 95% percentile interval.

    `a` and `b` map task_id -> per-task score (e.g. VSC averaged over that task's samples).
    """
    ids = sorted(set(a) & set(b))
    if not ids:
        raise ValueError("no shared tasks")
    d = [a[i] - b[i] for i in ids]
    rng = random.Random(seed)
    n = len(d)
    means = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    return sum(d) / n, means[int(0.025 * iters)], means[int(0.975 * iters) - 1]


def hierarchical_bootstrap(a_by_seed: Sequence[dict[str, float]], b: dict[str, float],
                           iters: int = 10_000, seed: int = 0) -> tuple[float, float, float]:
    """Delta of a multi-seed arm vs a reference: resample seeds, then tasks."""
    rng = random.Random(seed)
    ids = sorted(set(b).intersection(*[set(s) for s in a_by_seed]))
    n, k = len(ids), len(a_by_seed)
    point = sum(sum(s[i] for s in a_by_seed) / k - b[i] for i in ids) / n
    means = []
    for _ in range(iters):
        seeds = [a_by_seed[rng.randrange(k)] for _ in range(k)]
        tot = 0.0
        for _ in range(n):
            i = ids[rng.randrange(n)]
            tot += sum(s[i] for s in seeds) / k - b[i]
        means.append(tot / n)
    means.sort()
    return point, means[int(0.025 * iters)], means[int(0.975 * iters) - 1]
