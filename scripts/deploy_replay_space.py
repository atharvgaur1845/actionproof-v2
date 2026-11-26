"""Redeploy the static replay Space (atharvgaur18/actionproof) with the built demo/static/index.html.

Replaces the page at every path named index.html that the Space already serves (root and/or static/), nothing else.
Retries network errors. Prints the commit and the sha256 of what was uploaded.

    python scripts/deploy_replay_space.py
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

ROOT = Path(__file__).resolve().parents[1]
RID = "atharvgaur18/actionproof"


def retry(fn, n=6):
    for i in range(n):
        try:
            return fn()
        except Exception as e:  # network resets from the cluster
            if i == n - 1:
                raise
            print("retry", i + 1, type(e).__name__)
            time.sleep(20 * (i + 1))


def main() -> None:
    page = ROOT / "demo/static/index.html"
    api = HfApi()
    files = retry(lambda: api.list_repo_files(RID, repo_type="space"))
    targets = [f for f in files if f in ("index.html", "static/index.html")]
    if not targets:
        raise SystemExit(f"no index.html in {RID}; refusing to guess the layout")
    ops = [CommitOperationAdd(path_in_repo=t, path_or_fileobj=str(page)) for t in targets]
    c = retry(lambda: api.create_commit(RID, repo_type="space", operations=ops,
                                        commit_message="Site: TXN1, P4 and B1-API sections (built from result files)"))
    print("deployed", targets, "sha256", hashlib.sha256(page.read_bytes()).hexdigest(), c.oid)


if __name__ == "__main__":
    main()
