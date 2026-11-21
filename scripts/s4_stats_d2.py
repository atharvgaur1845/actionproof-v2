"""Deviation D-S4E-2 wrapper around the frozen `scripts/s4_stats.py` (not edited).

The frozen script pairs tasks across splits by `task_id`. The four `s4_lang_*` splits share their worlds by design,
but their ids carry the split name (`s4_lang_en-0007` vs `s4_lang_hi-0007`). So the registered language-cost
secondary found no common tasks and the script stopped before writing its output. This wrapper renames language-split
ids to `lang-<index>` as they are loaded (tested: the same index is the same world, faults and state), then runs the
frozen `main()` unchanged. Within-split contrasts (G1-G4 and the start-state comparison) pair each split with itself,
so their keys and values are identical to the frozen script's.
    python scripts/s4_stats_d2.py      # same arguments and outputs as s4_stats.py
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("s4_stats", ROOT / "scripts/s4_stats.py")
S4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S4)
_load = S4.load


def load(tag: str, split: str, d: str):
    out = _load(tag, split, d)
    if out is None or not split.startswith("s4_lang_"):
        return out
    return {"lang-" + t.rsplit("-", 1)[1]: v for t, v in out.items()}


S4.load = load

if __name__ == "__main__":
    S4.main()
