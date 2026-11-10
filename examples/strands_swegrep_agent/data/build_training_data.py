"""Rebuild the RL training data (data/train.jsonl, data/val.jsonl) from public sources.

    python data/build_training_data.py [--check] [--window W]

Inputs:
  data/manifests/training_tasks.jsonl  ordered task ids, split and source dataset (our task selection)
  data/manifests/excluded_files.json   six patch files export-ignore'd by their repo (absent from snapshots)
  SWE-rebench / SWE-rebench-V2 on Hugging Face, pinned revisions (problem statement, reference patch)
  GitHub, at each task's base commit (file lengths, to clip windows at end of file)

Labels ("patch window 20"):
  core_lines  old-side span of every hunk of the reference patch, context lines included, keeping
              source-code files only (SOURCE_SUFFIXES) and dropping test files
              (is_test_path in agent/scoring/regions.py)
  gold_lines  each core span widened by 20 lines on both sides, clipped to the file, overlaps merged

--check fails unless both files match the sha256 in data/manifests/expected.json, i.e. the exact
data the published model was trained on. --window W (default 20) writes data/window{W}/ instead,
e.g. --window 0 for the raw-patch ablation.
"""

import argparse
import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from scoring.regions import is_test_path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

SOURCES = {
    "nebius/SWE-rebench-V2": (
        "10483de0f50fe5da545942705a76c6150171af7f",
        ["data/train-00000-of-00001.parquet"],
    ),
    "nebius/SWE-rebench": (
        "89cdfbab4ab1bd8f5a658bb212d1b63624f4f881",
        ["data/test-00000-of-00002.parquet", "data/test-00001-of-00002.parquet"],
    ),
}
DATA_SOURCE = "swegrep_v3"
LANGUAGE_NAMES = {
    "js": "javascript",
    "ts": "typescript",
}  # SWE-rebench-V2 codes; SWE-rebench rows are all Python
SOURCE_SUFFIXES = {
    ".c", ".cc", ".clj", ".cpp", ".cs", ".ex", ".exs", ".go", ".h", ".hpp", ".java", ".jl", ".js", ".jsx",
    ".kt", ".kts", ".lua", ".php", ".py", ".pyi", ".pyx", ".R", ".rs", ".scala", ".sh", ".sql", ".swift",
    ".ts", ".tsx", ".vue",
}  # fmt: skip
HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+\d+(?:,(\d+))? @@")


def load_sources(ids):
    rows = {}
    for repo_id, (revision, files) in SOURCES.items():
        for name in files:
            path = hf_hub_download(repo_id, name, repo_type="dataset", revision=revision)
            columns = ["instance_id", "repo", "base_commit", "problem_statement", "patch"]
            columns += ["language"] if repo_id.endswith("V2") else []
            for row in pq.read_table(path, columns=columns).to_pylist():
                if row["instance_id"] in ids and row["instance_id"] not in rows:
                    rows[row["instance_id"]] = row | {"source": repo_id}
    return rows


def core_lines(patch, excluded):
    """Old-side hunk spans per source file, as {path: [[start, end], ...]}."""
    spans, path = defaultdict(list), None
    old_left = new_left = 0  # lines still to come in the current hunk, per side
    for line in patch.splitlines():
        if old_left > 0 or new_left > 0:
            # Hunk content, even when it looks like a header: a removed "-- comment" reads "--- comment".
            old_left -= line[:1] in (" ", "", "-")
            new_left -= line[:1] in (" ", "", "+")
        elif line.startswith("--- "):
            old = line[4:].split("\t")[0]
            path = None if old == "/dev/null" else old.removeprefix("a/")
        elif line.startswith("@@"):
            match = HUNK.match(line)
            start, count = (
                int(match.group(1)),
                int(match.group(2) if match.group(2) is not None else 1),
            )
            old_left, new_left = count, int(match.group(3) if match.group(3) is not None else 1)
            if (
                path
                and count
                and Path(path).suffix in SOURCE_SUFFIXES
                and not is_test_path(path)
                and path not in excluded
            ):
                spans[path].append([start, start + count - 1])
    return dict(spans)


def merge(spans):
    out = []
    for start, end in sorted(spans):
        if out and start <= out[-1][1] + 1:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return out


def line_count(repo, commit, path, cache):
    """Lines in the file at that commit, counted as the snapshots were: newlines + 1."""
    key = f"{repo}@{commit}:{path}"
    if key not in cache:
        url = f"https://raw.githubusercontent.com/{repo}/{commit}/{path}"
        for attempt in range(5):
            try:
                with urllib.request.urlopen(url, timeout=60) as response:
                    cache[key] = response.read().count(b"\n") + 1
                break
            except urllib.error.URLError:
                if attempt == 4:
                    raise
                time.sleep(2**attempt)
    return cache[key]


def build_row(task, source, excluded, cache, window):
    core = core_lines(source["patch"], excluded)
    gold = {
        path: merge(
            [
                [max(1, s - window), min(line_count(source["repo"], source["base_commit"], path, cache), e + window)]
                for s, e in spans
            ]
        )
        for path, spans in core.items()
    }  # fmt: skip
    return {
        "instance_id": task["instance_id"],
        "repo": source["repo"],
        "base_commit": source["base_commit"],
        "language": LANGUAGE_NAMES.get(source.get("language"), source.get("language")) or "python",
        "source": source["source"],
        "data_source": DATA_SOURCE,
        "problem_statement": source["problem_statement"],
        "core_lines": core,
        "gold_lines": gold,
        "gold_files": sorted(gold),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail unless outputs match data/manifests/expected.json",
    )
    parser.add_argument("--window", type=int, default=20, help="lines added on each side of a patch hunk")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if args.check and args.window != 20:
        parser.error("--check verifies the published (window 20) data only")

    tasks = [json.loads(line) for line in (DATA / "manifests/training_tasks.jsonl").read_text().splitlines()]
    excluded = defaultdict(set)
    for item in json.loads((DATA / "manifests/excluded_files.json").read_text())["excluded"]:
        excluded[item["instance_id"]].add(item["path"])
    sources = load_sources({t["instance_id"] for t in tasks})
    missing = [t["instance_id"] for t in tasks if t["instance_id"] not in sources]
    if missing:
        raise SystemExit(f"{len(missing)} tasks not found in the pinned sources, e.g. {missing[:3]}")

    cache_path = DATA / ".cache/line_counts.json"
    cache_path.parent.mkdir(exist_ok=True)
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    with ThreadPoolExecutor(args.workers) as pool:
        rows = list(
            pool.map(
                lambda t: build_row(t, sources[t["instance_id"]], excluded[t["instance_id"]], cache, args.window),
                tasks,
            )
        )
    cache_path.write_text(json.dumps(cache))

    out = DATA if args.window == 20 else DATA / f"window{args.window}"
    out.mkdir(exist_ok=True)
    expected = json.loads((DATA / "manifests/expected.json").read_text())
    ok = True
    for split in ("train", "val"):
        blob = "".join(
            json.dumps(r, ensure_ascii=False) + "\n" for r, t in zip(rows, tasks, strict=True) if t["split"] == split
        ).encode()
        (out / f"{split}.jsonl").write_bytes(blob)
        digest = hashlib.sha256(blob).hexdigest()
        match = args.window == 20 and digest == expected[f"{split}.jsonl"]["sha256"]
        ok &= match
        verdict = f", {'matches' if match else 'DIFFERS from'} expected" if args.window == 20 else ""
        num_rows = blob.count(b"\n")
        print(f"{out / split}.jsonl: {num_rows} rows, sha256 {digest}{verdict}")
    if args.check and not ok:
        raise SystemExit("Rebuilt data differs from the published training data")


if __name__ == "__main__":
    main()
