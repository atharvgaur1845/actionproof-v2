import math

from actionproof.eval.stats import hierarchical_bootstrap, paired_bootstrap, wilson


def test_wilson_known_values():
    p, lo, hi = wilson(0, 60)
    assert p == 0 and lo == 0 and 0.05 < hi < 0.07  # rule of three-ish: 0/60 -> upper ~6%
    p, lo, hi = wilson(30, 60)
    assert abs(p - 0.5) < 1e-9 and lo < 0.5 < hi
    assert math.isnan(wilson(0, 0)[0])


def test_paired_bootstrap_detects_clear_gain_and_null():
    a = {f"t{i}": 1.0 for i in range(200)}
    b = {f"t{i}": float(i % 2) for i in range(200)}
    d, lo, hi = paired_bootstrap(a, b, iters=2000)
    assert abs(d - 0.5) < 1e-9 and lo > 0.4
    d, lo, hi = paired_bootstrap(b, b, iters=2000)
    assert d == 0 and lo <= 0 <= hi


def test_hierarchical_bootstrap_matches_point():
    b = {f"t{i}": 0.0 for i in range(50)}
    seeds = [{f"t{i}": 1.0 for i in range(50)}, {f"t{i}": 0.5 for i in range(50)}]
    d, lo, hi = hierarchical_bootstrap(seeds, b, iters=500)
    assert abs(d - 0.75) < 1e-9 and 0.5 <= lo <= d <= hi <= 1.0


def test_post_ack_check_needs_same_entity_after_success():
    from actionproof.eval.by_fault import post_ack_check
    w = {"tool": "issue_refund", "args": {"order_id": "O-1"}, "obs": {"status": "success"}}
    assert post_ack_check([w, {"tool": "query_ledger", "args": {"order_id": "O-1"}, "obs": {}}])
    assert not post_ack_check([w, {"tool": "query_ledger", "args": {"order_id": "O-2"}, "obs": {}}])
    assert not post_ack_check([{"tool": "query_ledger", "args": {"order_id": "O-1"}, "obs": {}}, w])
    assert not post_ack_check([{**w, "obs": {"status": "error"}}, {"tool": "query_ledger", "args": {"order_id": "O-1"}, "obs": {}}])


def test_calibration_profile_counts_same_entity_checks_per_service():
    from actionproof.eval.calibration import acked_writes
    tr = [{"tool": "issue_refund", "args": {"order_id": "O-1"}, "obs": {"status": "success"}},
          {"tool": "close_ticket", "args": {"ticket_id": "T-1"}, "obs": {"status": "success"}},
          {"tool": "query_ledger", "args": {"order_id": "O-1"}, "obs": {}}]
    assert acked_writes(tr) == [("payments", True), ("ticketing", False)]
