"""P5 addendum 1 (D-P5-1): an unhashable idempotency key must not crash P1's behaviour metrics."""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def p5():
    spec = importlib.util.spec_from_file_location("p5_eval", ROOT / "scripts/p5_eval.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _trace(key):
    return [{"tool": "pay_supplier", "args": {"po_id": "PO-1", "idempotency_key": key}},
            {"tool": "refund_payment", "args": {"po_id": "PO-1"}},
            {"tool": "pay_supplier", "args": {"po_id": "PO-1", "idempotency_key": key}}]


def test_dict_key_does_not_crash_and_counts_as_reused(p5):
    r = {"trace": _trace({"k": "a"}), "task": {"meta": {}}}
    m = p5.P1.episode_metrics(r, 0)
    assert m["repay_after_refund"] == 1 and m["repay_new_key"] == 0
    assert isinstance(r["trace"][0]["args"]["idempotency_key"], dict)  # stored trace untouched


def test_hashable_keys_unchanged(p5):
    for key in ("k1", 7, None):
        r = {"trace": _trace(key), "task": {"meta": {}}}
        assert p5.P1.episode_metrics(r, 0) == p5._episode_metrics(r, 0)
