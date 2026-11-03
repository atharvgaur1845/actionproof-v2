from actionproof.train.curriculum import band


def recs(task, wins, n=8):
    return [{"task_id": task, "vsc": int(i < wins), "faults": ["F1_phantom"], "workflow": "refund"} for i in range(n)]


def test_band_keeps_only_learnable_tasks():
    rs = recs("always", 8) + recs("never", 0) + recs("mid", 4) + recs("edge", 1)
    kept, stats = band(rs, 0.10, 0.90)
    assert {k["task_id"] for k in kept} == {"mid", "edge"} and stats["tasks"] == 4
