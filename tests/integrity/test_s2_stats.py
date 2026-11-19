"""s2_stats on scripted stand-ins (oracle as 'R_V', naive as 'R_A') in a temporary tree; never touches results/."""
import importlib.util
import json
from pathlib import Path

from actionproof.bench.generator_proc import make_split
from actionproof.integrity.sim.agents import naive, oracle
from actionproof.integrity.sim.runtime import episode_record_proc, make_env_proc

ROOT = Path(__file__).resolve().parents[2]


def test_s2_stats_end_to_end(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("s2", ROOT / "scripts/s2_stats.py")
    s2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s2)
    monkeypatch.setattr(s2, "ROOT", tmp_path)
    monkeypatch.setattr(s2, "B", 200)
    d = tmp_path / "results/eval/a12/conf"
    d.mkdir(parents=True)
    tasks = make_split("proc_dev", 60)
    for seed in (0, 1, 2):
        for arm, pol in (("proc_rv", oracle), ("proc_ra", naive)):
            with open(d / f"a12_{arm}_s{seed}__neutral__proc_conf.jsonl", "w") as f:
                for t in tasks:
                    env = make_env_proc(t, "x")
                    pol(env, t)
                    r = episode_record_proc(t, env)
                    f.write(json.dumps({"task_id": t.task_id, "score": r["score"]}) + "\n")
    monkeypatch.setattr("sys.argv", ["s2", "--out", "out.json"])
    s2.main()
    res = json.loads((tmp_path / "out.json").read_text())
    pr = res["phantom_readback"]
    assert res["seeds"] == [0, 1, 2] and pr["same_sign_all_seeds"]
    assert pr["per_seed"]["0"]["diff"] > 0.5 and pr["sign_flip_p_one_sided"] == 1 / 8
    assert res["vsc"]["per_seed"]["0"]["mcnemar"]["rv_only"] > 0
