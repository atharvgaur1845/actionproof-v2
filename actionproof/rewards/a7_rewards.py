"""Amendment 7 reward arm (PREREGISTRATION_A7 §2). New module: a6_rewards.py stays hash-registered and unchanged.

Completes the 2 x 2 over PCSTR-d's two report terms on the A6 warm start:

    pcstr_d           = G + 0.2P + 0.1V - 0.5I - 0.3ND - 0.1F_under - 0.2D     (M1: both)
    pcstr_d_noND      = G + 0.2P + 0.1V - 0.5I          - 0.1F_under - 0.2D     (M2b: F_under only)
    pcstr_d_noFunder  = G + 0.2P + 0.1V - 0.5I - 0.3ND                - 0.2D     (M2c: ND only, this module)
    pcstr_d_noreport  = G + 0.2P + 0.1V - 0.5I                        - 0.2D     (M2: neither)
"""
from __future__ import annotations

from actionproof.env.verifier import Verdict
from actionproof.rewards.a6_rewards import pcstr_d_noreport_score
from actionproof.rewards.pcstr import WEIGHTS, nondisclosure


def pcstr_d_noFunder_score(v: Verdict) -> float:
    return pcstr_d_noreport_score(v) - WEIGHTS["F_over"] * nondisclosure(v)
