"""Materialize repo snapshots into S3 for the AgentCore agent (which has no internet).

For every unique (repo, base_commit) in the given task files: blobless clone, `git archive` at
the commit (which honours each repo's export-ignore attributes), filter to small text/code files in
memory, upload to `s3://$SWEGREP_CORPUS_BUCKET/repos/{slug}__{commit}.tar.gz`. Run after cdk/deploy.sh.

    uv run python scripts/prepare_corpus.py [TASKS.jsonl ...] [--workers 16]

Default task files: data/train.jsonl, data/val.jsonl and both benchmarks under data/benchmarks/.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import tarfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from rollout.repo_store import S3_PREFIX, snapshot_key

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASKS = [
    ROOT / "data/train.jsonl",
    ROOT / "data/val.jsonl",
    ROOT / "data/benchmarks/swe-explore.jsonl",
    ROOT / "data/benchmarks/contextbench-lite.jsonl",
]

# File filters — keep the snapshot small and text-only so grep/read tools work.
MAX_FILE_BYTES = 1_000_000  # skip files larger than 1 MB (data blobs, minified)
MAX_TOTAL_BYTES = 400_000_000  # safety cap on a single snapshot (~400 MB)
# Extensions worth searching for code localization. Extend as needed.
KEEP_SUFFIXES = {
    ".py",
    ".pyx",
    ".pyi",
    ".txt",
    ".rst",
    ".md",
    ".cfg",
    ".ini",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
    ".c",
    ".h",
    ".cc",
    ".cpp",
    ".hpp",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".java",
    ".go",
    ".rs",
    ".rb",
    ".sh",
    ".html",
    ".css",
    ".xml",
    ".sql",
    ".tex",
    ".in",
    ".template",
    # Multilingual benchmark gold files: a missing suffix silently caps recall.
    ".pony",
    ".svelte",
    ".vue",
    ".mjs",
    ".cjs",
    ".mts",
    ".cts",
    ".scss",
    ".less",
    ".pcss",
    ".styl",
    ".tpl",
    ".hbs",
    ".erb",
    ".asciidoc",
    ".adoc",
    ".mod",
    ".sum",
    ".cue",
    ".pot",
    ".snap",
    ".kt",
    ".kts",
    ".scala",
    ".swift",
    ".cs",
    ".php",
    ".m",
    ".mm",
    ".proto",
    ".gradle",
    ".properties",
    ".bzl",
    ".cmake",
    ".zig",
    ".ex",
    ".exs",
    ".clj",
    ".hs",
    ".lua",
    ".pl",
    ".pm",
    ".r",
    ".jl",
    # Rarer extensions of ContextBench gold files, found by checking that every gold file is in its snapshot.
    ".plist",
    ".preview",
    ".vcproj",
    ".vcxproj",
    ".txtar",
    ".j2",
    ".feature",
    ".options",
    ".ps1",
    ".test",
    ".mdx",
    ".jq",
    ".bats",
    ".lock",
}
# Extensionless files worth keeping (Makefile etc. are ContextBench gold targets).
KEEP_NAMES = {
    "Makefile",
    "makefile",
    "GNUmakefile",
    "Dockerfile",
    "Containerfile",
    "Jenkinsfile",
    "Rakefile",
    "Gemfile",
    "Procfile",
    "Vagrantfile",
    "Brewfile",
    "BUILD",
    "WORKSPACE",
    "CHANGELOG",
    "README",
    "LICENSE",
    "NOTICE",
    "CODEOWNERS",
    "managed-schema",
}

# Directories never worth shipping. "javadoc" (generated HTML) blew the size cap on
# jackson-databind; do NOT extend to "api-docs" and keep .html in KEEP_SUFFIXES: both hold gold.
SKIP_DIRS = {
    ".git",
    "node_modules",
    ".tox",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    "javadoc",
}


def _unique_pairs(paths: list[Path]) -> list[tuple[str, str]]:
    pairs: dict[tuple[str, str], None] = {}
    for path in paths:
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                pairs[(row["repo"], row["base_commit"])] = None
    return list(pairs)


# One lock per repo so concurrent workers never clone/fetch the same repo twice;
# `git archive` reads are safe to run concurrently against a settled clone.
_repo_locks: dict[str, threading.Lock] = {}
_repo_locks_guard = threading.Lock()


def _repo_lock(repo: str) -> threading.Lock:
    with _repo_locks_guard:
        return _repo_locks.setdefault(repo, threading.Lock())


def _clone(repo: str, cache_dir: Path) -> Path:
    """Blobless clone of a repo (cached so multiple commits reuse it).

    Keep --filter=blob:none: full clones of the training manifest filled a 500 GB disk.
    """
    dest = cache_dir / repo.replace("/", "__")
    with _repo_lock(repo):
        if not (dest / ".git").exists():
            url = f"https://github.com/{repo}.git"
            subprocess.run(
                ["git", "clone", "--quiet", "--filter=blob:none", url, str(dest)],
                check=True,
            )
    return dest


def _git_archive(clone_dir: Path, repo: str, commit: str) -> bytes:
    """`git archive` the tree at `commit`, fetching it first if missing."""
    cmd = ["git", "-C", str(clone_dir), "archive", "--format=tar", commit]
    try:
        return subprocess.run(cmd, check=True, capture_output=True).stdout
    except subprocess.CalledProcessError:
        # Fetch the exact sha first: base_commits often sit on PR refs `fetch --all` misses.
        with _repo_lock(repo):
            by_sha = subprocess.run(
                ["git", "-C", str(clone_dir), "fetch", "--quiet", "origin", commit],
                capture_output=True,
                check=False,
            )
            if by_sha.returncode != 0:
                subprocess.run(["git", "-C", str(clone_dir), "fetch", "--quiet", "--all"], check=True)
        return subprocess.run(cmd, check=True, capture_output=True).stdout


def _filtered_snapshot(clone_dir: Path, repo: str, commit: str) -> tuple[io.BytesIO, int, int]:
    """Repack the `git archive` tar as tar.gz, keeping only small text/code files.

    Streams tar -> tar.gz entirely in memory (no extract-to-disk round trip).
    Returns (buffer, num_files_kept, total_bytes).
    """
    raw = _git_archive(clone_dir, repo, commit)
    buf = io.BytesIO()
    total = kept = skipped_oversize = 0
    with (
        tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as src,
        tarfile.open(fileobj=buf, mode="w:gz", compresslevel=6) as dst,
    ):
        for member in src:
            if not member.isreg():
                continue
            rel = Path(member.name)
            if any(part in SKIP_DIRS for part in rel.parts):
                continue
            if rel.suffix.lower() not in KEEP_SUFFIXES and rel.name not in KEEP_NAMES:
                continue
            if member.size == 0 or member.size > MAX_FILE_BYTES:
                continue
            # Skip-and-continue, NOT break: break silently dropped the whole archive tail.
            if total + member.size > MAX_TOTAL_BYTES:
                skipped_oversize += 1
                continue
            fobj = src.extractfile(member)
            if fobj is None:
                continue
            dst.addfile(member, fobj)
            total += member.size
            kept += 1
    if skipped_oversize:
        print(
            f"  WARNING {repo}@{commit[:8]}: snapshot hit MAX_TOTAL_BYTES "
            f"({MAX_TOTAL_BYTES:,}) -- dropped {skipped_oversize} file(s); "
            f"kept {kept} ({total:,} B). Gold files may be missing.",
            flush=True,
        )
    buf.seek(0)
    return buf, kept, total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tasks", nargs="*", type=Path, default=DEFAULT_TASKS, help="task JSONL files")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N unique (repo, base_commit) pairs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-upload snapshots even if the S3 key already exists.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Concurrent snapshot builders (default 8). The work is git "
        "subprocesses + zlib + S3 I/O, so threads parallelize fully.",
    )
    parser.add_argument(
        "--repos",
        type=str,
        default="",
        help="Comma-separated repo slugs to restrict to (re-snapshot a subset).",
    )
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")  # SWEGREP_CORPUS_BUCKET, written by cdk/deploy.sh
    bucket = os.environ.get("SWEGREP_CORPUS_BUCKET")
    if not bucket:
        raise SystemExit(
            "SWEGREP_CORPUS_BUCKET not set. Run cdk/deploy.sh first (it writes it " "into .env), or export it manually."
        )
    # Shared by all workers, each upload using several connections; the default pool holds 10.
    s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION"), config=Config(max_pool_connections=64))

    missing = [str(p) for p in args.tasks if not p.exists()]
    if missing:
        raise SystemExit(f"Missing task files {missing}: run data/build_training_data.py and data/build_benchmarks.py")
    pairs = _unique_pairs(args.tasks)
    if args.repos:
        want = {r.strip() for r in args.repos.split(",") if r.strip()}
        pairs = [(r, c) for r, c in pairs if r in want]
    if args.limit is not None:
        pairs = pairs[: args.limit]
    print(f"{len(pairs)} unique (repo, base_commit) snapshots -> s3://{bucket}/{S3_PREFIX}/")

    # Not tempfile.gettempdir(): /tmp is often a tmpfs. SWEGREP_CLONE_DIR overrides.
    clone_cache = Path(os.environ.get("SWEGREP_CLONE_DIR", str(Path.home() / ".cache" / "swegrep_clones")))
    clone_cache.mkdir(parents=True, exist_ok=True)

    def build_one(i: int, repo: str, commit: str) -> str:
        """Runs on the pool. Returns 'uploaded' | 'skipped' (raises on failure)."""
        key = snapshot_key(repo, commit)
        if not args.overwrite:
            try:
                s3.head_object(Bucket=bucket, Key=key)
                return "skipped"
            except ClientError as exc:
                if exc.response["Error"]["Code"] not in {"404", "NoSuchKey", "NotFound"}:
                    raise
        clone_dir = _clone(repo, clone_cache)
        buf, kept, total = _filtered_snapshot(clone_dir, repo, commit)
        s3.upload_fileobj(buf, bucket, key)
        print(
            f"[{i}/{len(pairs)}] {repo}@{commit[:8]}  files={kept} bytes={total}  -> {key}",
            flush=True,
        )
        return "uploaded"

    uploaded = skipped = failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(build_one, i, repo, commit): (i, repo, commit) for i, (repo, commit) in enumerate(pairs, 1)
        }
        for fut in as_completed(futures):
            i, repo, commit = futures[fut]
            try:
                result = fut.result()
                uploaded += result == "uploaded"
                skipped += result == "skipped"
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"[{i}/{len(pairs)}] FAILED {repo}@{commit[:8]}: {exc}", flush=True)

    print(f"Done. uploaded={uploaded} skipped={skipped} failed={failed}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
