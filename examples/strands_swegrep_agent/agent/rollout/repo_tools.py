"""The grep / glob / read tools the SWE-grep agent explores a checkout with.

``build_repo_tools(repo_root)`` returns them closed over one rollout's checkout (paths are
repo-root-relative and nothing escapes it). Their docstrings are the tool schemas the model
sees. The prompt is in ``agent_prompts.py`` and the search budget in ``turn_cap.py``.
"""

import os
import re
import subprocess
import tempfile
import threading
from collections import defaultdict
from pathlib import Path

from strands import tool

# Tool payload limits. The defaults are the published model's: they set what the policy sees.
# Lines per grep, across all files.
GREP_MAX_MATCHES = int(os.environ.get("SWEGREP_GREP_MAX_MATCHES", "120"))
# Per-file match cap (grep -m), so one chatty file cannot eat the whole grep budget.
GREP_MAX_PER_FILE = int(os.environ.get("SWEGREP_GREP_PER_FILE", "10"))
GLOB_MAX_RESULTS = int(os.environ.get("SWEGREP_GLOB_MAX_RESULTS", "200"))  # paths per glob
READ_MAX_LINES = int(os.environ.get("SWEGREP_READ_MAX_LINES", "800"))  # lines per read
_MAX_CHARS = int(os.environ.get("SWEGREP_TOOL_MAX_CHARS", "8000"))  # hard cap per tool payload
# A grep alternation wider than this (or a pattern longer than _MAX_PATTERN_CHARS) is rejected:
# long invented-identifier chains never hit and burn the turn budget.
GREP_MAX_ALTERNATIVES = int(os.environ.get("SWEGREP_GREP_MAX_ALTERNATIVES", "8"))
_MAX_PATTERN_CHARS = 120

# Never searched: VCS internals, dependency/build trees and caches. Gold files never live there
# and build copies (e.g. cdk.out) shadow the real source with identical content.
_ALWAYS_SKIP_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        ".tox",
        ".nox",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".eggs",
        "site-packages",
        "dist",
        "build",
        "cdk.out",
        ".idea",
        ".vscode",
    }
)


def _gitignored_dirs(root: Path) -> set[str]:
    """Directory names the root .gitignore excludes outright ('foo/', '/foo', 'foo' with no
    glob characters). Only bare names: the exclusion is applied by name at any depth."""
    try:
        lines = (root / ".gitignore").read_text(errors="replace").splitlines()
    except OSError:
        return set()
    names: set[str] = set()
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith(("#", "!")):
            continue
        name = line.strip("/")
        if name and "/" not in name and not any(ch in name for ch in "*?[]"):
            names.add(name)
    return names


def _too_broad(pattern: str) -> str | None:
    """Reason a grep pattern is rejected, or None."""
    if len(pattern) > _MAX_PATTERN_CHARS:
        return f"pattern is {len(pattern)} chars (max {_MAX_PATTERN_CHARS})"
    alternatives = len(re.findall(r"(?<!\\)\|", pattern)) + 1
    if alternatives > GREP_MAX_ALTERNATIVES:
        return f"{alternatives} alternatives (max {GREP_MAX_ALTERNATIVES})"
    return None


def top_level_listing(repo_root: str | Path, limit: int = 60) -> str:
    """Top-level listing (dirs get a trailing '/') seeded into the system prompt
    so the explorer doesn't burn its first turn discovering the repo layout."""
    root = Path(repo_root)
    try:
        entries = sorted((e.name + "/" if e.is_dir() else e.name) for e in root.iterdir())
    except OSError:
        return ""
    if len(entries) > limit:
        entries = entries[:limit] + [f"... (+{len(entries) - limit} more)"]
    return "\n".join(entries)


def _resolve(repo_root: Path, rel: str) -> Path | None:
    """Resolve ``rel`` under ``repo_root``, refusing paths that escape it."""
    root = repo_root.resolve()
    target = (root / str(rel or ".").lstrip("/")).resolve()
    # Bare startswith would let /root-sibling match /root; require a separator.
    if target != root and not str(target).startswith(str(root) + "/"):
        return None
    return target


def _search_page(lines: list[str], offset: int, max_lines: int) -> str:
    """Bound output without making the remaining results inaccessible."""
    if offset < 0:
        return "error: offset must be non-negative."
    if offset >= len(lines):
        return f"No more results at offset {offset} (total {len(lines)})."
    selected = []
    used = 0
    # Reserve space for a continuation marker. Keep complete paths/records.
    budget = max(128, _MAX_CHARS - 160)
    for line in lines[offset : offset + max_lines]:
        if selected and used + len(line) + 1 > budget:
            break
        if len(line) > budget:
            # Retain the complete grep path:line prefix; only shorten source text.
            prefix = ":".join(line.split(":", 2)[:2]) + ":" if ":" in line else line
            line = line[: max(len(prefix), budget - 45)] + " ... [match text shortened; use read]"
        selected.append(line)
        used += len(line) + 1
    next_offset = offset + len(selected)
    if next_offset < len(lines):
        selected.append(
            f"... (more results: next_offset={next_offset}; "
            f"repeat the same query with offset={next_offset}, or narrow the query)"
        )
    return "\n".join(selected)


def _run_grep(args: list[str], cwd: Path, timeout: float = 60) -> tuple[int, str, str] | None:
    """grep's (return code, stdout, first stderr line), or None on timeout. Each output line is cut
    to _MAX_CHARS while it is read, so matches in minified files cannot exhaust memory;
    _search_page shortens any line over its budget (< _MAX_CHARS) anyway, so the result is the same.
    """
    with tempfile.TemporaryFile("w+", errors="replace") as stderr:
        proc = subprocess.Popen(args, cwd=str(cwd), stdout=subprocess.PIPE, stderr=stderr, text=True, errors="replace")
        timed_out = threading.Event()
        timer = threading.Timer(timeout, lambda: (timed_out.set(), proc.kill()))
        timer.start()
        lines = []
        try:
            # readline(n) returns at most n characters; drain the rest of an over-long line.
            while line := proc.stdout.readline(_MAX_CHARS):
                if not line.endswith("\n"):
                    while (rest := proc.stdout.readline(_MAX_CHARS)) and not rest.endswith("\n"):
                        pass
                lines.append(line.rstrip("\n"))
            returncode = proc.wait()
        finally:
            timer.cancel()
            proc.stdout.close()
        if timed_out.is_set():
            return None
        stderr.seek(0)
        return returncode, "\n".join(lines), stderr.readline().strip()


def _diverse_grep_lines(text: str) -> list[str]:
    """Round-robin matching files so one file cannot consume the first page."""
    grouped = defaultdict(list)
    for line in text.splitlines():
        grouped[line.split(":", 1)[0]].append(line)
    return [
        grouped[path][i]
        for i in range(max(map(len, grouped.values()), default=0))
        for path in sorted(grouped)
        if i < len(grouped[path])
    ]


def build_repo_tools(repo_root: str | Path) -> list:
    """Build the explorer tools bound to one rollout's checkout root."""
    root = Path(repo_root)
    skip_dirs = _ALWAYS_SKIP_DIRS | _gitignored_dirs(root)
    exclude_args = [f"--exclude-dir={d}" for d in sorted(skip_dirs)]

    @tool
    def grep(pattern: str = "", path: str = ".", offset: int = 0) -> str:
        """Search the repository text with a regular expression (POSIX ERE).
        Returns matching `path:line: text` lines, repo-root-relative (at most a
        few matches per file — read the file for full context).

        Args:
            pattern: Regex to search for (e.g. "def resolve_url" or "TimeoutError").
            path: Optional repo-root-relative subtree to search (default: whole repo).
            offset: Result offset; use next_offset from a previous page with the same query.
        """
        if not pattern or not pattern.strip():
            return "error: provide a non-empty pattern."
        if reason := _too_broad(pattern):
            return f"error: pattern too broad ({reason}). Grep ONE concrete symbol or message per call."
        target = _resolve(root, path)
        if target is None or not target.exists():
            return f"error: path not found in repo: {path}"
        rel = "." if target == root.resolve() else str(target.relative_to(root.resolve()))
        args = ["grep", "-rnHIE", "-m", str(GREP_MAX_PER_FILE), *exclude_args, "--", pattern, rel]
        if (result := _run_grep(args, root)) is None:
            return f"error: grep timed out (60s) for /{pattern}/. Narrow the pattern or path."
        returncode, stdout, err = result
        # grep exits 1 on no matches, >1 on real errors (e.g. a bad regex).
        if returncode > 1:
            return f"error: grep failed: {err or 'unknown error'}"
        stdout = stdout.replace("\r", "").strip()
        if not stdout:
            return f"No matches for /{pattern}/. Try a different pattern."
        # Strip grep's leading "./" so paths read as repo-root-relative.
        cleaned = "\n".join(line.removeprefix("./") for line in stdout.splitlines())
        return _search_page(_diverse_grep_lines(cleaned), int(offset), GREP_MAX_MATCHES)

    @tool
    def glob(pattern: str, offset: int = 0) -> str:
        """Discover files whose path matches a glob pattern (e.g. "**/*router*.py",
        "src/**/*.py"). Returns repo-root-relative paths.

        Args:
            pattern: Glob pattern, repo-root-relative.
            offset: Result offset; use next_offset from a previous page with the same query.
        """
        pat = str(pattern or "").strip()
        if not pat or ".." in Path(pat).parts:
            return "error: provide a valid repo-root-relative glob (no '..')."
        paths = sorted(
            {
                rel
                for p in root.glob(pat.lstrip("/"))
                if p.is_file()
                and _resolve(root, rel := str(p.relative_to(root))) is not None
                and skip_dirs.isdisjoint(Path(rel).parts[:-1])
            }
        )
        if not paths:
            return f"No files match {pat!r}."
        return _search_page(paths, int(offset), GLOB_MAX_RESULTS)

    @tool
    def read(path: str, start_line: int = 1, end_line: int | None = None) -> str:
        """Read a file's contents with line numbers.

        Args:
            path: Repo-root-relative file path (e.g. "src/router.py").
            start_line: 1-based first line to return (default 1).
            end_line: 1-based last line. Reads have line and character limits;
                a wide window is truncated
                (with a "... (truncated)" marker); page with start_line to see more.
        """
        target = _resolve(root, path)
        if target is None or not target.exists():
            return f"error: file not found: {path}"
        if not target.is_file():
            return f"error: not a file: {path}"
        data = target.read_bytes()
        if b"\0" in data[:8192]:  # grep -I's binary heuristic
            return f"error: binary file: {path}"
        # Universal newlines, trailing whitespace dropped: the same text the published model was
        # trained on (it read files through the Strands community file_read tool). Split on "\n" only,
        # so line numbers match grep -n and the gold patches (str.splitlines also breaks on \f, \v,
        # \x85, \u2028, ...).
        text = data.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
        all_lines = text.rstrip().split("\n") if text.strip() else []
        start = max(1, int(start_line))
        stop = int(end_line) if end_line else start + READ_MAX_LINES - 1
        if stop < start:
            return f"error: end_line {stop} is before start_line {start}."
        stop = min(stop, start + READ_MAX_LINES - 1, len(all_lines))
        if start > len(all_lines):
            return f"error: start_line {start} beyond end of file ({len(all_lines)} lines)."
        numbered = [f"{i}\t{all_lines[i - 1]}" for i in range(start, stop + 1)]
        header = f"# {path} (lines {start}-{stop} of {len(all_lines)})"
        out = header + "\n" + "\n".join(numbered)
        return out[:_MAX_CHARS] + "\n... (truncated)" if len(out) > _MAX_CHARS else out

    return [grep, glob, read]
