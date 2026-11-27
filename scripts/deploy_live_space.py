"""Deploy space_live/ to a Hugging Face Space (network only; run where an HF write token is configured).

    python scripts/deploy_live_space.py --repo atharvgaur18/actionproof-live-staging --private   # staging first
    python scripts/deploy_live_space.py --repo atharvgaur18/actionproof-live                     # then the live Space

Uploads the Space files (app, core, vendored package, data, tests, README, requirements); never caches or build
products. With --private, creates the repo as a private ZeroGPU Space if it does not exist. Prints the new commit.
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--message", default="Deploy space_live from the ActionProof repository")
    a = ap.parse_args()
    from huggingface_hub import HfApi
    api = HfApi()
    if a.private:
        api.create_repo(a.repo, repo_type="space", space_sdk="gradio", private=True, exist_ok=True,
                        space_hardware="zero-a10g")
    info = api.upload_folder(repo_id=a.repo, repo_type="space", folder_path=str(ROOT / "space_live"),
                             commit_message=a.message,
                             ignore_patterns=["**/__pycache__/**", "*.pyc", ".pytest_cache/**", "record_runs.py"])
    print("deployed", a.repo, getattr(info, "oid", info))


if __name__ == "__main__":
    main()
