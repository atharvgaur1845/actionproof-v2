from actionproof.eval.confirmatory import boot_p, fcr, holm


def test_holm_step_down():
    # 0.01 <= 0.05/3, 0.02 <= 0.05/2, 0.06 > 0.05/1
    assert holm({"a": 0.01, "b": 0.02, "c": 0.06}) == {"a": True, "b": True, "c": False}
    # once one fails, all larger p fail
    assert holm({"a": 0.03, "b": 0.001, "c": 0.04}) == {"b": True, "a": False, "c": False}


def test_boot_p_sides():
    assert boot_p([0.1] * 100) == 0.0
    assert boot_p([-0.1] * 50 + [0.1] * 50) == 1.0


def test_fcr_conditions_on_own_failures():
    eps = {"t1": {"verdict": {"G": 0, "reported": "completed"}},
           "t2": {"verdict": {"G": 0, "reported": "failed"}},
           "t3": {"verdict": {"G": 1, "reported": "completed"}}}
    assert fcr(eps, ["t1", "t2", "t3"]) == 0.5
