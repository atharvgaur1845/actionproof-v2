from __future__ import annotations

import pytest

from actionproof.env.faults import Fault, FaultSchedule
from actionproof.env.tools_ecom import EcomEnv
from actionproof.bench.fixtures import (  # noqa: F401  re-exported for tests
    CUST, ORDER, SKU, TICKET, approval_goal, refund_goal, refund_state)

def make_env(schedule: dict | None = None, state: dict | None = None, **kw) -> EcomEnv:
    sched = FaultSchedule(by_call={k: Fault(v) for k, v in (schedule or {}).items()})
    return EcomEnv(state or refund_state(), sched, rollout_id="ro-test", seed=7, **kw)


@pytest.fixture
def env_factory():
    return make_env
