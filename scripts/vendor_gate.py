"""Write space_live/actionproof_min/gate.py from actionproof/recovery_gate/gate.py: the same file with the two import
paths changed (tests/recovery_gate/test_space_vendor.py checks it)."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC, DST = ROOT / "actionproof/recovery_gate/gate.py", ROOT / "space_live/actionproof_min/gate.py"
SUBS = (("from actionproof.integrity.sim.agents import _observed", "from actionproof_min.agents import _observed"),
        ("from actionproof.integrity.sim.env import STEP_READ, STEP_TOOL, TOOL_STEP",
         "from actionproof_min.env import STEP_READ, STEP_TOOL, TOOL_STEP"))


def vendored() -> str:
    s = SRC.read_text()
    for a, b in SUBS:
        assert s.count(a) == 1, a
        s = s.replace(a, b)
    return s


if __name__ == "__main__":
    DST.write_text(vendored())
    print("wrote", DST.relative_to(ROOT))
