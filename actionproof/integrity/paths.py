"""Where the ProcureSim splits and result files live. An editable install finds them next to the package; an ordinary
install finds them through AP_ROOT or the current directory (run from a repository checkout)."""
from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    if os.environ.get("AP_ROOT"):
        return Path(os.environ["AP_ROOT"])
    for c in (Path(__file__).resolve().parents[2], Path.cwd()):
        if (c / "data/bench/proc").is_dir():
            return c
    raise FileNotFoundError("ProcureSim splits not found (data/bench/proc): run from the repository root or set AP_ROOT")
