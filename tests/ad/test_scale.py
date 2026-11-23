"""CPU tests for the AD-D2 world-independent advantage scale."""
from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest

from actionproof.ad.scale import FLOOR, EmaScale, group_stds, install


def test_group_stds():
    assert group_stds([1, -1, 1, -1, 0, 0, 0, 0], 4) == [1.0, 0.0]


def test_divisor_uses_only_past_steps_after_step0():
    sc = EmaScale(decay=0.5, floor=0.0)
    d0, m0 = sc.divisor_and_update([0.4])
    assert d0 == pytest.approx(0.4) and sc.ema == pytest.approx(0.4)
    d1, _ = sc.divisor_and_update([0.0])          # this step's own batch must not change its divisor
    assert d1 == pytest.approx(0.4) and sc.ema == pytest.approx(0.2)
    d2, _ = sc.divisor_and_update([10.0])
    assert d2 == pytest.approx(0.2)


def test_floor():
    sc = EmaScale(floor=FLOOR)
    d, _ = sc.divisor_and_update([0.0, 0.0])
    assert d == FLOOR


def test_sign_preserved_and_common_divisor():
    """Every advantage in a step is divided by the same positive scalar: signs and within-step ratios unchanged."""
    torch = pytest.importorskip("torch")

    class Fake:
        def _generate_and_score_completions(self, inputs):
            return {"advantages": torch.tensor(inputs, dtype=torch.float32)}

    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        log = Path(d) / "ema.jsonl"
        install(Fake, log, G=4)
        f = Fake()
        f.state = SimpleNamespace(global_step=0)
        a = [0.5, -0.5, 0.25, -0.25, 0.0, 0.0, 0.0, 0.0]
        out = f._generate_and_score_completions(a)["advantages"].tolist()
        ratios = {round(o / x, 6) for o, x in zip(out, a) if x}
        assert len(ratios) == 1 and next(iter(ratios)) > 0
        f.state.global_step = 1
        out2 = f._generate_and_score_completions([1.0, -1.0, 0, 0, 0, 0, 0, 0])["advantages"].tolist()
        rows = [json.loads(l) for l in log.read_text().splitlines()]
        assert len(rows) == 2 and math.isclose(rows[1]["divisor"], max(FLOOR, rows[0]["ema_after"]))
        assert out2[0] > 0 > out2[1]
