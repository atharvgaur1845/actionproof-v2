"""Write release/ARTIFACT_MANIFEST.json: every released artefact with its immutable revision, every frozen split and
headline result file with its sha256, and the script that produces each result. Hub entries are read from the
public API WITHOUT credentials, so "public" here means an anonymous request saw it.

    python scripts/build_manifest.py
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HUB = [("model", "atharvgaur18/actionproof-agent-d", "Recommended agent D (merged, P1 arm D seed 0)"),
       ("model", "atharvgaur18/actionproof-agent-d-adapters", "D adapters (seeds 0-2) and the Dw adapter"),
       ("model", "atharvgaur18/actionproof-warmstart-dw", "Recovery-SFT warm start Dw"),
       ("model", "atharvgaur18/actionproof-warmstart-dwaug", "P4 warm starts Dw_aug (seeds 0-2)"),
       ("model", "atharvgaur18/actionproof-agent-a", "Reference agent A (recovery-start GRPO)"),
       ("model", "atharvgaur18/actionproof-agent-dn", "Dn, no-std GRPO (P3, not supported)"),
       ("model", "atharvgaur18/actionproof-sel1-adapters", "SEL1 arms SEL and CTL (seeds 0-2), registered SEL1"),
       ("dataset", "atharvgaur18/actionproof-traces", "Episode traces, frozen splits, judge cases"),
       ("space", "atharvgaur18/actionproof-live", "Live Space (real model inference)"),
       ("space", "atharvgaur18/actionproof", "Replay Space (recorded episodes, no model calls)"),
       ("model", "CohereLabs/tiny-aya-global", "Base model (third party, gated)")]
RESULTS = {  # file -> (registration or status, script)
    "results/p1/p1_summary.json": ("PREREGISTRATION_P1.md", "scripts/p1_stats.py"),
    "results/p3/p3_summary.json": ("PREREGISTRATION_P3.md", "scripts/p3_stats.py"),
    "results/p4/p4_summary.json": ("PREREGISTRATION_P4.md", "scripts/p4_stats.py"),
    "results/p5/p5_summary.json": ("PREREGISTRATION_P5.md", "scripts/p5_stats.py"),
    "results/txn/stats__txn_test.json": ("PREREGISTRATION_TXN1.md", "scripts/txn_stats.py"),
    "results/txn/compact_descriptive__txn_test.json": ("descriptive", "scripts/txn_compact.py"),
    "results/audit/r7_grpo_increment.json": ("exploratory", "scripts/audit/r7_grpo_increment.py"),
    "results/audit/dwarm_2x2.json": ("exploratory", "scripts/audit/dwarm_2x2.py"),
    "results/impact/impact_txn.json": ("descriptive projection", "scripts/impact_txn.py"),
    "space_live/data/recorded.json": ("recorded real model runs for the live Space (task 1790)", "space_live/record_runs.py"),
    "results/sel1/sel1_summary.json": ("PREREGISTRATION_SEL1.md", "scripts/sel1_stats.py"),
    "results/p4_hashes.json": ("P4 registration hashes", "scripts/freeze_p4.py"),
    "results/sel1_hashes.json": ("SEL1 registration hashes", "scripts/freeze_sel1.py"),
}
SPLIT_MANIFESTS = ["data/bench/proc/txn_manifest.json", "data/bench/proc/sel1_manifest.json"]
ENV = {"python": "3.12", "torch": "2.11.0+cu130", "transformers": "5.22.1", "trl": "1.7.1", "vllm": "0.23.0",
       "peft": "0.21.2", "AgentTune": "d4db4c12d523", "AuditKIT": "71316da34578", "CuratorKIT": "dd9e835301fc",
       "gpus": "NVIDIA H200 / H100 / A100 80GB (sharanga, BITS Pilani)", "lock_file": "requirements.lock"}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def hub(kind: str, repo: str) -> dict:
    url = f"https://huggingface.co/api/{kind}s/{repo}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "manifest-check"}), timeout=30) as r:
            d = json.loads(r.read().decode())
    except Exception as e:  # not visible anonymously, or a network error
        return {"status": f"not verified anonymously ({type(e).__name__})"}
    card = d.get("cardData") or {}
    out = {"revision": d.get("sha"), "last_modified": d.get("lastModified"),
           "status": "private" if d.get("private") else ("public, gated" if d.get("gated") else "public"),
           "license": card.get("license"), "base_model": card.get("base_model")}
    if kind == "space":
        out["sdk"] = d.get("sdk")
        out["hardware"] = ((d.get("runtime") or {}).get("hardware") or {}).get("current")
    return out


def main() -> None:
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    m = {"generated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
         "code": {"repository": "https://github.com/atharvgaur1845/actionproof-v2", "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
                  "commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))},
         "licence": {"code": "Apache-2.0", "models": "CC-BY-NC-4.0 (derived from Cohere Aya; research use only)"},
         "environment": ENV, "hub": {}, "splits": {}, "results": {}, "not_released": [
             "P5 SFT models on Tiny Aya Fire, Water, Earth and Aya Expanse 8B (results and episodes only)",
             "P2 and AD-C1 adapters (cluster only)"]}
    for kind, repo, what in HUB:
        m["hub"][repo] = {"kind": kind, "what": what, **hub(kind, repo)}
    for f in SPLIT_MANIFESTS:
        d = json.loads((ROOT / f).read_text())
        for name, v in d["files"].items():
            p = ROOT / "data/bench/proc" / name
            m["splits"][name] = {"n_tasks": v["n"], "sha256": v["sha256"], "file_matches": p.exists() and sha(p) == v["sha256"],
                                 "manifest": f}
    for f, (status, script) in RESULTS.items():
        p = ROOT / f
        m["results"][f] = ({"sha256": sha(p), "status": status, "script": script} if p.exists()
                           else {"status": f"{status}: not yet produced", "script": script})
    out = ROOT / "release/ARTIFACT_MANIFEST.json"
    out.write_text(json.dumps(m, indent=1, sort_keys=False) + "\n")
    bad = [r for r, v in m["hub"].items() if not v["status"].startswith("public")]
    print(f"wrote {out}: {len(m['hub'])} hub entries ({len(bad)} not public anonymously: {bad}), "
          f"{len(m['splits'])} splits (all match: {all(v['file_matches'] for v in m['splits'].values())}), "
          f"{sum('sha256' in v for v in m['results'].values())}/{len(m['results'])} result files")


if __name__ == "__main__":
    main()
