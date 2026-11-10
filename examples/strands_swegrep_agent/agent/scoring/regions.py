"""Code regions cited in the agent's answer: parsing, validation against the checkout, test paths.

A region is ``path:start-end``: a repo-relative path and a 1-indexed, inclusive line range.

* ``parse_regions`` reads the lines of an ``<answer>`` block into ranked ``Region`` objects and
  lists malformed entries separately; it never guesses a range.
* ``validate_regions`` drops regions whose file is missing or outside the checkout and clips
  ranges that run past the end of the file.
* ``is_test_path`` recognises test files. It is used to filter the training labels
  (``data/build_training_data.py``), not to parse answers.

``retrieval.extract_answer`` calls the first two for the training reward and for the benchmark
scorers, so training and evaluation parse answers the same way.
"""

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Region:
    path: str
    start_line: int
    end_line: int


# Test files and directories. This filters the training labels (data/build_training_data.py), so a
# change here changes the data and its pinned sha256 in data/manifests/expected.json.
_TEST_PATH = re.compile(
    r"""
    # Any language: a test directory anywhere in the path (case-insensitive)
      (?i:(^|/)(tests?|__tests__|testing|testdata|testfixtures?|test_?utils|specs?)/)
    # Python and Rust: tests.py, test.rs, test_foo.py, conftest.py
    | (^|/)tests?\.(py|rs)$
    | (^|/)test_helpers?\.\w+$
    | (^|/)test_[^/]+\.py$
    # Go, Rust, C/C++, Ruby, Elixir, Python: foo_test.go, foo_test.cc, ...
    | _test\.(py|go|rs|cc|cpp|c|rb|exs?)$
    | (^|/)conftest\.py$
    # JVM, .NET, Swift, PHP: FooTest.java, FooTests.cs, TestFoo.kt
    | [a-z0-9_]Tests?\.(java|kt|scala|cs|swift|php)$
    | (^|/)Tests?[A-Z][^/]*\.(java|kt|cs)$
    # Ruby: foo_spec.rb
    | _spec\.rb$
    # JavaScript / TypeScript: foo.test.ts, foo.spec.jsx, foo.test.mjs
    | \.(test|spec)\.[cm]?[jt]sx?$
    """,
    re.VERBOSE,
)


def is_test_path(path: str) -> bool:
    # lstrip removes every leading "." and "/" (not just one "./"), so "../tests/x" and ".tests/x"
    # also match. Kept as is: the published labels were built with it.
    return bool(_TEST_PATH.search(path.replace("\\", "/").lstrip("./")))


_SINGLE_LINE = re.compile(r"^([^\s:]+):(\d+)$")
_CITATION = re.compile(r"^([^\s:]+):(\d+)-(\d+)$")


def parse_regions(block: str, *, allow_single_line: bool = False) -> tuple[list[Region], list[str]]:
    """Keep ranked entries, including multiple spans in one file; never invent spans."""
    regions, invalid = [], []
    for raw in block.splitlines():
        line = raw.strip()
        if not line:
            continue
        match = _CITATION.fullmatch(line)
        if not match and allow_single_line:
            single = _SINGLE_LINE.fullmatch(line)
            if single:
                match = _CITATION.fullmatch(f"{single[1]}:{single[2]}-{single[2]}")
        if not match:
            invalid.append(raw)
            continue
        path, start, end = match.groups()
        start, end = int(start), int(end)
        if start < 1 or end < start:
            invalid.append(raw)
            continue
        regions.append(Region(path, start, end))
    return regions, invalid


def validate_regions(regions: list[Region], repo_root: str | Path) -> tuple[list[Region], list[str]]:
    """Drop missing/outside paths and empty spans; clip ends to checkout line counts.

    The root must be the benchmark's actual base-commit snapshot. Validation
    does not establish relevance, and never fills in missing line ranges.
    """
    root = Path(repo_root).resolve()
    valid, invalid = [], []
    line_counts: dict[Path, int] = {}
    for region in regions:
        label = f"{region.path}:{region.start_line}-{region.end_line}"
        try:
            path = (root / region.path).resolve()
            relative = path.relative_to(root).as_posix()
            if not path.is_file():
                raise ValueError("not a file")
            if path not in line_counts:
                with path.open("rb") as stream:
                    line_counts[path] = sum(1 for _ in stream)
            end = min(region.end_line, line_counts[path])
            if region.start_line < 1 or end < region.start_line:
                raise ValueError("empty or invalid interval")
            valid.append(Region(relative, region.start_line, end))
        except (OSError, ValueError, RuntimeError):
            invalid.append(label)
    return valid, invalid
