"""The live Space's gate is the repo's gate with two import paths changed."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_space_gate_is_the_repo_gate():
    spec = importlib.util.spec_from_file_location("vendor_gate", ROOT / "scripts/vendor_gate.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert (ROOT / "space_live/actionproof_min/gate.py").read_text() == m.vendored()
