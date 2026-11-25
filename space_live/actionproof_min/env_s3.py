"""ProcureSim with a partial ratio per fault (`{"kind": "partial", "ratio": 0.25}`). Without a ratio it is ProcEnv
exactly (partial = half), so every Stage 2 task behaves as before (tested)."""
from __future__ import annotations

from typing import Any

from actionproof_min.env import ProcEnv

SCALED = {"pay_supplier": "amount_cents", "record_goods_receipt": "qty", "place_supplier_order": "qty",
          "post_journal_entry": "amount_cents"}


class S3ProcEnv(ProcEnv):
    def __init__(self, *args: Any, ratios: dict[tuple[str, int], float] | None = None, **kw: Any) -> None:
        super().__init__(*args, **kw)
        self.ratios = ratios or {}

    def _write(self, tool, po_id, key, prefix, rows, id_field):  # type: ignore[override]
        r = self.ratios.get((tool, self._ctx[2]))
        if r is None or tool not in SCALED:
            return super()._write(tool, po_id, key, prefix, rows, id_field)
        col = SCALED[tool]

        def scaled(i: str, part: bool):
            if not part:
                return rows(i, False)
            out = []
            for t, row in rows(i, False):
                v = max(1, int(row[col] * r))
                out.append((t, {**row, col: min(v, row[col] - 1)}))
            return out
        return super()._write(tool, po_id, key, prefix, scaled, id_field)


def make_env_s3(task: Any, rollout_id: str, seed: int | None = None) -> S3ProcEnv:
    m = task.meta
    faults = m.get("faults") or []
    ratios = {(f["tool"], int(f["k"])): float(f["ratio"]) for f in faults if f.get("ratio") is not None}
    return S3ProcEnv(task.initial_state, [{k: f[k] for k in ("tool", "k", "kind")} for f in faults],
                     rollout_id=rollout_id, seed=m["env_seed"] if seed is None else seed,
                     never_commit=m.get("never_commit"), ratios=ratios)
