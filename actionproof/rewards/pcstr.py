"""PCSTR: Proof-Carrying State-Transition Reward (spec §11).

Scores a trajectory purely from the verifier's `Verdict`, which is computed from the final DB
snapshot, the environment's receipt log and the hidden goal spec. The transcript cannot move
any term except through the final report's status (F_over / F_under).

The weights are pre-registered initial values; they may only be tuned on val, and every
setting must keep the ordering property tested in tests/test_reward_sanity.py:
    min reward of any VSC=1 trajectory  >  max reward of any G=0 trajectory.
"""
from __future__ import annotations

from actionproof.env.verifier import Verdict

WEIGHTS = {"G": 1.0, "P": 0.2, "V": 0.1, "I": 0.5, "F_over": 0.3, "F_under": 0.1, "D": 0.2}


def pcstr_score(v: Verdict, n_verification_calls: int = 0, verification_cost: float = 0.0) -> float:
    """Scalar PCSTR reward for one trajectory.

    `verification_cost` > 0 gives the `pcstr+c` arm (calibrated trust, upgrade 1): each
    verification call costs that much. The primary `pcstr` arm uses 0.
    """
    w = WEIGHTS  # read from the dict so ablations (A1 no V, A3 no I, A4 no F) can zero a term
    r = (w["G"] * v.G + w["P"] * v.P + w["V"] * v.V
         - w["I"] * v.I - w["F_over"] * v.F_over - w["F_under"] * v.F_under - w["D"] * v.D)
    return r - verification_cost * n_verification_calls
