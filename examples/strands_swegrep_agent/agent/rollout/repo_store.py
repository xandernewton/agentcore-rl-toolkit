"""Repo-snapshot access for the agent: ``ensure_checkout(repo, base_commit) -> Path``.

The runtime has no internet, so ``scripts/prepare_corpus.py`` uploads
each ``{repo}@{base_commit}`` as ``s3://$SWEGREP_CORPUS_BUCKET/repos/{slug}__{commit}.tar.gz``;
this module downloads + extracts it into a per-key-locked local cache. boto3 + stdlib only.
"""

from __future__ import annotations

import os
import tarfile
import threading
from pathlib import Path

import boto3
from botocore.config import Config

# Where snapshots are extracted inside the container. A warm container reuses
# this across sessions, so repeated instances on the same repo pay the S3
# download once. Overridable for local testing.
CACHE_ROOT = Path(os.environ.get("SWEGREP_CACHE_DIR", "/tmp/swegrep_repos"))

# S3 prefix the snapshots live under.
S3_PREFIX = os.environ.get("SWEGREP_CORPUS_PREFIX", "repos")

# One lock per snapshot key so two rollout threads can't extract the same repo
# concurrently, while different repos still extract in parallel.
_key_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()

# Module-level S3 client (boto3 clients are thread-safe). Region comes from the
# execution-role env; unset is fine for the global S3 endpoint. Many threads share it (benchmark
# harness, scorers, corpus upload), each download using several connections: botocore's default
# pool of 10 drops the extras with "Connection pool is full" warnings.
_s3 = None
_s3_guard = threading.Lock()


def checkout_dir(repo: str, base_commit: str) -> Path:
    """Where a snapshot is extracted. ``owner/name`` -> ``owner__name`` so it is a
    single path segment (mirrors SWE-bench's instance_id slugging)."""
    return CACHE_ROOT / f"{repo.replace('/', '__')}__{base_commit}"


def snapshot_key(repo: str, base_commit: str) -> str:
    """S3 object key for a repo snapshot."""
    return f"{S3_PREFIX}/{checkout_dir(repo, base_commit).name}.tar.gz"


def _bucket() -> str:
    bucket = os.environ.get("SWEGREP_CORPUS_BUCKET")
    if not bucket:
        raise RuntimeError(
            "SWEGREP_CORPUS_BUCKET is not set; the runtime needs the corpus "
            "bucket name to fetch repo snapshots (set by the CDK stack)."
        )
    return bucket


def _client():
    global _s3
    with _s3_guard:
        if _s3 is None:
            _s3 = boto3.client(
                "s3",
                region_name=os.environ.get("AWS_REGION"),
                config=Config(max_pool_connections=64),
            )
        return _s3


def _lock_for(key: str) -> threading.Lock:
    with _locks_guard:
        return _key_locks.setdefault(key, threading.Lock())


def ensure_checkout(repo: str, base_commit: str) -> Path:
    """Return a local path holding the extracted ``repo@base_commit`` tree.

    Downloads + extracts the S3 snapshot on first use; later calls (any thread,
    any session in the same warm container) return the cached path. Raises if the
    snapshot is missing — that means ``prepare_corpus.py`` never uploaded it.
    """
    key = snapshot_key(repo, base_commit)
    dest = checkout_dir(repo, base_commit)
    done_marker = dest / ".swegrep_extracted"

    if done_marker.exists():
        return dest

    with _lock_for(key):
        # Re-check under the lock: another thread may have finished while we waited.
        if done_marker.exists():
            return dest

        CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        tar_path = dest.with_name(f"{dest.name}.tar.gz")
        bucket = _bucket()
        try:
            _client().download_file(bucket, key, str(tar_path))
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download repo snapshot s3://{bucket}/{key}: {exc}. "
                "Did scripts/prepare_corpus.py run for this task file?"
            ) from exc

        dest.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_path, "r:gz") as tar:
            _safe_extractall(tar, dest)
        tar_path.unlink(missing_ok=True)
        done_marker.touch()
        return dest


def _safe_extractall(tar: tarfile.TarFile, dest: Path) -> None:
    """Extract, refusing members that would escape ``dest`` (path traversal)."""
    dest = dest.resolve()
    for member in tar.getmembers():
        target = (dest / member.name).resolve()
        # Require a path-separator boundary so a sibling dir sharing the dest
        # prefix (e.g. .../repo__c vs .../repo__c2) can't pass the check.
        if target != dest and not str(target).startswith(str(dest) + "/"):
            raise RuntimeError(f"Unsafe path in snapshot: {member.name}")
    # filter="data" also validates symlink targets, which the name check can't see
    tar.extractall(dest, filter="data")
