"""Amendment 6 reward arms (PREREGISTRATION_A6 §2). New module: `pcstr.py` and `baselines.py` are hash-registered and
unchanged. Each arm is written out term by term so the contrasts are exact:

    pcstr_d          = G + 0.2P + 0.1V - 0.5I - 0.3ND - 0.1F_under - 0.2D     (rewards/pcstr.py, disclosure=True)
    pcstr_d_noreport = G + 0.2P + 0.1V - 0.5I                      - 0.2D     (M2: both report terms removed)
    pcstr_d_noND     = G + 0.2P + 0.1V - 0.5I          - 0.1F_under - 0.2D     (M2b: only ND removed)
    outcome_G        = 1[G = 1]                                                (M3': binary on the SAME predicate G)

G = 1[P = 1 and I = 0 and D = 0 and V = 1] (env/verifier.py). None of these reads the transcript or the report except
pcstr_d (ND, F_under) and pcstr_d_noND (F_under).
"""
from __future__ import annotations

from actionproof.env.verifier import Verdict
from actionproof.rewards.pcstr import WEIGHTS


def pcstr_d_noreport_score(v: Verdict) -> float:
    w = WEIGHTS
    return w["G"] * v.G + w["P"] * v.P + w["V"] * v.V - w["I"] * v.I - w["D"] * v.D


def pcstr_d_noND_score(v: Verdict) -> float:
    return pcstr_d_noreport_score(v) - WEIGHTS["F_under"] * v.F_under


def outcome_G_score(v: Verdict) -> float:
    return float(v.G == 1)
