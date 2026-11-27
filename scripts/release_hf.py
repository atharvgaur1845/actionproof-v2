"""Public Hugging Face release of ActionProof (run on the cluster, where the adapters and episode files live).

    python scripts/release_hf.py                       # DRY RUN (default): plan, sizes, sha256 manifest; uploads nothing
    python scripts/release_hf.py --check-remote        # dry run that also asks the Hub which files are already there
    python scripts/release_hf.py --build-only          # only build the merged D seed-0 model, then exit (needs torch + peft)
    python scripts/release_hf.py --execute             # create the public repos and upload what is missing

Repos, all under `atharvgaur18` and public:

| Repo | Type | Content |
|---|---|---|
| actionproof-agent-d | model | merged full model: base + merge(runs/p1_dwarm/final) + merge(runs/p1_d_s0/final) |
| actionproof-agent-d-adapters | model | dw/ (runs/p1_dwarm/final), s0..s2/ (runs/p1_d_s{k}/final) |
| actionproof-warmstart-dw | model | the Dw adapter at the repo root |
| actionproof-agent-a | model | w/ (runs/rft_proc/final, the parent of A), s0..s2/ (runs/s3_rv_mix_s{k}/final) |
| actionproof-agent-dn | model | s0..s2/ (runs/p3_dn_s{k}/final) |
| actionproof-warmstart-dwaug | model | s0..s2/ (runs/p4_dwaug_s{k}/final), P4 |
| actionproof-sel1-adapters | model | sel/s0..s2, ctl/s0..s2 (runs/sel1_{SEL,CTL}_s{k}/final), SEL1 |
| actionproof-traces | dataset | results/{p1,p2,p3}/**, data/bench/proc/p1_*.jsonl (+ s4_*), evaluation/judge_next/data/*.jsonl, and results/{p4,b1,txn} if present |

Each repo's README.md is release/model_cards/<repo>.md (dataset: release/dataset_card.md); regenerate the model cards
from the result files with `python release/build_cards.py` first.

**Merged D seed 0.** `actionproof/train/merge.py` merges an adapter into the model given as base and caches the result
at `<adapter dir>/../merged`. The chain is `merge(runs/p1_dwarm/final)` (reuses runs/p1_dwarm/merged) then
`merge(runs/p1_d_s0/final, base=runs/p1_dwarm/merged)` (reuses runs/p1_d_s0/merged, which is the checkpoint the P1
evaluation used). An existing merged directory is reused; its `merged_from.json` must name the Dw merge as base, else
the script stops (override: `--trust-merged`). Use `--merged-dir DIR` to publish a different directory.

**Adapters.** Only `adapter_model.safetensors` and `adapter_config.json` are uploaded, plus a `LINEAGE.json` per
adapter folder. In `adapter_config.json` the cluster path in `base_model_name_or_path` is replaced by the public base
model id (the original value is kept in LINEAGE.json); adapters whose real parent is a merged model say so in LINEAGE.json
and on the card.

**Idempotent.** Every file is hashed (sha256 and git blob sha1). Files whose hash matches the remote copy (LFS sha256, or
git blob id for small files) are skipped, so a re-run uploads only what changed or is new. Models go up with one
`create_commit` per repo; the dataset goes up with `upload_large_folder` from a staging folder of symlinks to the
missing files (resumable). The manifest (repo, path, size, sha256) is printed and written to `release/hf_manifest.json`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
USER = "atharvgaur18"
BASE = "CohereLabs/tiny-aya-global"
CODE = "https://github.com/atharvgaur1845/actionproof-v2"
SEEDS = (0, 1, 2)
ADAPTER_FILES = ("adapter_model.safetensors", "adapter_config.json")
MERGED_SKIP_SUFFIX = (".bin", ".pt", ".log", ".md")
MERGED_SKIP_NAMES = ("merged_from.json", "README.md")

# repo short name -> (repo name, adapters: [(folder in repo or ".", runs/<dir>, real parent description)])
DW_PARENT = "merged Dw = base + adapter of actionproof-warmstart-dw (runs/p1_dwarm/merged)"
W_PARENT = "merged W = base + the w/ adapter (runs/rft_proc/merged)"
ADAPTER_REPOS: dict[str, tuple[str, list[tuple[str, str, str]]]] = {
    "d-adapters": ("actionproof-agent-d-adapters", [("dw", "p1_dwarm", "base")] + [(f"s{s}", f"p1_d_s{s}", DW_PARENT) for s in SEEDS]),
    "dw": ("actionproof-warmstart-dw", [(".", "p1_dwarm", "base")]),
    "a": ("actionproof-agent-a", [("w", "rft_proc", "base")] + [(f"s{s}", f"s3_rv_mix_s{s}", W_PARENT) for s in SEEDS]),
    "dn": ("actionproof-agent-dn", [(f"s{s}", f"p3_dn_s{s}", DW_PARENT) for s in SEEDS]),
    "dwaug": ("actionproof-warmstart-dwaug", [(f"s{s}", f"p4_dwaug_s{s}", "base") for s in SEEDS]),
    "sel1": ("actionproof-sel1-adapters",
             [(f"{arm.lower()}/s{s}", f"sel1_{arm}_s{s}",
               f"merged Dw_aug s{s} = base + s{s}/ adapter of actionproof-warmstart-dwaug (runs/p4_dwaug_s{s}/merged)")
              for arm in ("SEL", "CTL") for s in SEEDS]),
}
MERGED_REPO = "actionproof-agent-d"
DATASET_REPO = "actionproof-traces"
ALL = ("merged-d", *ADAPTER_REPOS, "dataset")

DATA_GLOBS = ("results/p1/**/*", "results/p2/**/*", "results/p3/**/*", "data/bench/proc/p1_*.jsonl",
              "data/bench/proc/s4_*.jsonl", "data/bench/proc/sel1_*.jsonl", "data/rft/sel1_*_rows.jsonl",
              "data/rft/sel1_rows.stats.json", "data/rft/p4_aug_rows.jsonl", "data/rft/p1_dwarm_rows.jsonl",
              "data/rft/proc_demo_rows.jsonl", "data/rft/proc_demo_rows.stats.json", "data/bench/proc/rg1_*",
              "evaluation/judge_next/data/*.jsonl")
EXTRA_DIRS = ("results/p4", "results/p5", "results/b1", "results/txn", "results/sel1", "results/rg1")  # included when they exist
SKIP_PARTS = {"smoke", "__pycache__", ".cache"}


class Item:
    """One file to publish: where it comes from (a path or generated bytes), where it goes, and its hashes."""

    def __init__(self, repo_path: str, src: Path | None = None, data: bytes | None = None):
        self.repo_path, self.src, self.data = repo_path, src, data
        h, g = hashlib.sha256(), hashlib.sha1()
        if data is not None:
            self.size = len(data)
            h.update(data)
            g.update(b"blob %d\0" % self.size + data)
        else:
            self.size = src.stat().st_size
            g.update(b"blob %d\0" % self.size)
            with src.open("rb") as f:
                for chunk in iter(lambda: f.read(1 << 22), b""):
                    h.update(chunk)
                    g.update(chunk)
        self.sha256, self.blob_sha1 = h.hexdigest(), g.hexdigest()


def sha_matches(item: Item, remote: dict) -> bool:
    r = remote.get(item.repo_path)
    if r is None:
        return False
    lfs, blob = r
    return (lfs == item.sha256) if lfs else (blob == item.blob_sha1)


def remote_files(api, repo_id: str, repo_type: str) -> dict:
    """path -> (lfs sha256 or None, git blob id). Empty when the repo does not exist yet."""
    from huggingface_hub.errors import RepositoryNotFoundError
    try:
        out = {}
        for e in api.list_repo_tree(repo_id, repo_type=repo_type, recursive=True):
            if hasattr(e, "blob_id"):
                lfs = getattr(e, "lfs", None)
                out[e.path] = ((lfs["sha256"] if isinstance(lfs, dict) else getattr(lfs, "sha256", None)) if lfs else None, e.blob_id)
        return out
    except RepositoryNotFoundError:
        return {}


# ---------------------------------------------------------------------------------------------------- plans

def card(name: str) -> Item:
    p = ROOT / "release" / ("dataset_card.md" if name == DATASET_REPO else f"model_cards/{name}.md")
    if not p.exists():
        raise SystemExit(f"missing card {p}; run python release/build_cards.py")
    return Item("README.md", src=p)


def lineage_bytes(folder: str, run: str, parent: str, orig_base: str, repo: str) -> bytes:
    d = {"repo": f"{USER}/{repo}", "folder": folder, "source_run": f"runs/{run}/final", "root_base_model": BASE,
         "real_parent": parent, "original_base_model_name_or_path": orig_base,
         "note": "adapter_config.json in this folder has base_model_name_or_path rewritten to the public root base; "
                 "when real_parent is not 'base', merge the parent chain first (see the repository README).",
         "merge_helper": "actionproof/train/merge.py", "code": CODE}
    return (json.dumps(d, indent=2) + "\n").encode()


def adapter_items(repo: str, runs: Path, problems: list) -> list[Item]:
    items = [card(repo)]
    for folder, run, parent in ADAPTER_REPOS[next(k for k, v in ADAPTER_REPOS.items() if v[0] == repo)][1]:
        d = runs / run / "final"
        pre = "" if folder == "." else f"{folder}/"
        missing = [f for f in ADAPTER_FILES if not (d / f).exists()]
        if missing:
            problems.append(f"{repo}: {d} lacks {missing}")
            continue
        cfg = json.loads((d / "adapter_config.json").read_text())
        orig = cfg.get("base_model_name_or_path")
        cfg["base_model_name_or_path"] = BASE
        items.append(Item(pre + "adapter_config.json", data=(json.dumps(cfg, indent=2) + "\n").encode()))
        items.append(Item(pre + "adapter_model.safetensors", src=d / "adapter_model.safetensors"))
        items.append(Item(pre + "LINEAGE.json", data=lineage_bytes(folder, run, parent, str(orig), repo)))
    return items


def build_merged(runs: Path, dry: bool, trust: bool, problems: list) -> Path | None:
    """Return the merged D seed-0 directory, building the chain base -> Dw -> D s0 with the repo's merge.py if absent."""
    dw_final, d_final = runs / "p1_dwarm/final", runs / "p1_d_s0/final"
    dw_out, d_out = runs / "p1_dwarm/merged", runs / "p1_d_s0/merged"
    for p in (dw_final, d_final):
        if not (p / "adapter_model.safetensors").exists():
            problems.append(f"merged-d: adapter {p} not found")
            return None
    if (d_out / "config.json").exists():
        mf = d_out / "merged_from.json"
        base_ok = False
        if mf.exists():
            b = json.loads(mf.read_text()).get("base", "")
            base_ok = Path(b).resolve() == dw_out.resolve() or Path(b).name == "merged" and Path(b).parent.name == "p1_dwarm"
        if not base_ok and not trust:
            problems.append(f"merged-d: {d_out} exists but merged_from.json does not name {dw_out} as its base; "
                            "inspect it or pass --trust-merged")
            return None
        return d_out
    if dry:
        print(f"[plan] merged D s0 not built yet: would run merge({dw_final}) -> {dw_out} then merge({d_final}, base={dw_out})")
        return None
    from actionproof.train.merge import merge
    dw = merge(str(dw_final))
    return Path(merge(str(d_final), dw))


def merged_items(d: Path) -> list[Item]:
    items = [card(MERGED_REPO)]
    for p in sorted(d.iterdir()):
        if p.is_file() and p.name not in MERGED_SKIP_NAMES and not p.name.endswith(MERGED_SKIP_SUFFIX):
            items.append(Item(p.name, src=p))
    d_info = {"repo": f"{USER}/{MERGED_REPO}", "root_base_model": BASE,
              "chain": ["CohereLabs/tiny-aya-global", f"+ merge adapter {USER}/actionproof-warmstart-dw (runs/p1_dwarm/final)",
                        "+ merge adapter runs/p1_d_s0/final (GRPO seed 0; also in " + f"{USER}/actionproof-agent-d-adapters s0/)"],
            "merge_helper": "actionproof/train/merge.py (bf16, merge_and_unload)", "code": CODE}
    items.append(Item("LINEAGE.json", data=(json.dumps(d_info, indent=2) + "\n").encode()))
    return items


def dataset_items() -> list[Item]:
    seen, paths = set(), []
    pats = list(DATA_GLOBS) + [f"{d}/**/*" for d in EXTRA_DIRS]
    for pat in pats:
        for p in sorted(ROOT.glob(pat)):
            rel = p.relative_to(ROOT)
            if p.is_file() and not (set(rel.parts) & SKIP_PARTS) and rel.suffix != ".pyc" and str(rel) not in seen:
                seen.add(str(rel))
                paths.append(rel)
    for d in EXTRA_DIRS:
        if not (ROOT / d).exists():
            print(f"[note] {d} does not exist yet (to be added); a later run will upload it")
    return [card(DATASET_REPO)] + [Item(str(r), src=ROOT / r) for r in paths]


# ------------------------------------------------------------------------------------------------- uploading

def sync(api, repo: str, rtype: str, items: list[Item], execute: bool, check_remote: bool) -> dict:
    repo_id = f"{USER}/{repo}"
    remote = remote_files(api, repo_id, rtype) if (execute or check_remote) else {}
    todo = [i for i in items if not sha_matches(i, remote)]
    size = sum(i.size for i in todo)
    state = "remote checked" if (execute or check_remote) else "remote not checked (dry run): all files counted as new"
    print(f"[{repo}] {len(items)} files, {len(todo)} to upload ({size / 1e6:.1f} MB); {state}")
    if execute and todo:
        from huggingface_hub import CommitOperationAdd
        api.create_repo(repo_id, repo_type=rtype, private=False, exist_ok=True)
        if rtype == "model":
            ops = [CommitOperationAdd(i.repo_path, i.data if i.data is not None else str(i.src)) for i in todo]
            api.create_commit(repo_id, ops, repo_type="model", commit_message="ActionProof release: upload files")
        else:
            with tempfile.TemporaryDirectory(prefix="release_stage_") as stage:
                for i in todo:
                    dst = Path(stage) / i.repo_path
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if i.data is not None:
                        dst.write_bytes(i.data)
                    else:
                        os.symlink(i.src.resolve(), dst)
                api.upload_large_folder(repo_id=repo_id, repo_type="dataset", folder_path=stage)
    elif execute:
        api.create_repo(repo_id, repo_type=rtype, private=False, exist_ok=True)
    if execute:
        after = remote_files(api, repo_id, rtype)
        bad = [i.repo_path for i in items if not sha_matches(i, after)]
        print(f"[{repo}] verify: {len(items) - len(bad)}/{len(items)} files match the remote" + (f"; MISMATCH {bad[:5]}" if bad else ""))
    return {"repo": repo_id, "type": rtype, "uploaded": [i.repo_path for i in todo] if execute else [],
            "files": [{"path": i.repo_path, "size": i.size, "sha256": i.sha256} for i in items]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--execute", action="store_true", help="create repos and upload (default: dry run)")
    ap.add_argument("--check-remote", action="store_true", help="in a dry run, also compare against the Hub")
    ap.add_argument("--only", nargs="+", choices=ALL, default=list(ALL))
    ap.add_argument("--runs", default=str(ROOT / "runs"), help="directory with the run folders (default <repo>/runs)")
    ap.add_argument("--merged-dir", default=None, help="publish this merged D s0 directory instead of runs/p1_d_s0/merged")
    ap.add_argument("--trust-merged", action="store_true", help="accept an existing merged dir without a merged_from.json check")
    ap.add_argument("--build-only", action="store_true", help="build the merged D s0 model and exit")
    ap.add_argument("--manifest", default=str(ROOT / "release/hf_manifest.json"))
    a = ap.parse_args()
    runs, problems, plans = Path(a.runs), [], []
    dry = not a.execute

    if a.build_only:
        d = build_merged(runs, False, a.trust_merged, problems)
        print("merged:", d, "problems:", problems)
        return 1 if problems or d is None else 0

    api = None
    if a.execute or a.check_remote:
        from huggingface_hub import HfApi
        api = HfApi()
        who = api.whoami()["name"]
        print(f"authenticated as {who}")
        if a.execute and who != USER:
            raise SystemExit(f"logged in as {who}, expected {USER}; refusing to create repos under another account")
    print("MODE:", "EXECUTE" if a.execute else "DRY RUN (nothing is uploaded; pass --execute)")

    for name in a.only:
        if name == "merged-d":
            d = Path(a.merged_dir) if a.merged_dir else build_merged(runs, dry, a.trust_merged, problems)
            if d is not None and (d / "config.json").exists():
                plans.append((MERGED_REPO, "model", merged_items(d)))
        elif name == "dataset":
            plans.append((DATASET_REPO, "dataset", dataset_items()))
        else:
            repo = ADAPTER_REPOS[name][0]
            plans.append((repo, "model", adapter_items(repo, runs, problems)))
    for p in problems:
        print("PROBLEM:", p)
    if problems and a.execute:
        raise SystemExit("not uploading: fix the problems above (nothing was changed on the Hub)")

    manifest = []
    for repo, rtype, items in plans:
        manifest.append(sync(api, repo, rtype, items, a.execute, a.check_remote))
    Path(a.manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(a.manifest).write_text(json.dumps(manifest, indent=1))
    print("\nMANIFEST (repo / path / size / sha256), also written to", a.manifest)
    for m in manifest:
        for f in m["files"]:
            print(f"{m['repo']}\t{f['path']}\t{f['size']}\t{f['sha256']}")
    if dry:
        print("\nDry run only. Re-run with --execute to upload.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
