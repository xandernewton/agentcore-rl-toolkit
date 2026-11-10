"""Metrics on how the agent searched, read from a Strands transcript.

Diagnostics only, never reward terms: which paths each tool result showed the model,
how well-aimed the searches were, and which turns found nothing new.
"""

import re
from typing import Any

from scoring.retrieval import normalize_list, normalize_path

# `read` echoes the file it opened: "# path/to/f.py (lines 10-40 of 900)".
READ_HEADER_RE = re.compile(r"^#\s+(\S+)\s+\(lines\s+\d+-\d+\s+of\s+\d+\)", re.MULTILINE)
# grep prefixes hits with "path:line:" only when searching a DIRECTORY; a
# single-file grep emits a bare "line:text", so the file comes from the call.
GREP_HIT_RE = re.compile(r"^([^\s:]+):(\d+):", re.MULTILINE)
# Tool results that gave the model nothing to work with (repo_tools.grep / glob wording).
_ZERO_HIT_PREFIXES = ("No matches for ", "No files match ", "error: pattern too broad")


def _blocks(messages: list[dict[str, Any]]):
    """Every dict content block of a transcript, in order."""
    for message in messages:
        for block in message.get("content", []) or []:
            if isinstance(block, dict):
                yield block


def _result_texts(block: dict) -> list[str]:
    """The text pieces of a toolResult block."""
    content = (block["toolResult"] or {}).get("content") or []
    return [b["text"] for b in content if isinstance(b, dict) and isinstance(b.get("text"), str)]


def _call(block: dict) -> tuple[str, tuple[str, dict]]:
    """A toolUse block as (toolUseId, (tool name, args))."""
    tool_use = block["toolUse"] or {}
    return tool_use.get("toolUseId"), (tool_use.get("name", ""), tool_use.get("input") or {})


def surfaced_recall(tool_output_blob: str, gold: set[str]) -> float:
    """Fraction of gold paths appearing anywhere in tool results.

    Deliberately loose — a broad glob listing counts as "surfaced" even though
    the agent never opened the file, so read this as an upper bound on what the
    search found, not as evidence the model located anything.
    """
    if not gold:
        return 0.0
    return sum(1 for g in gold if g and g in tool_output_blob) / len(gold)


def _looks_like_path(token: str) -> bool:
    """Reject the bare line numbers a single-file grep emits."""
    return bool(token) and not token.isdigit() and ("/" in token or "." in token)


def credited_paths(name: str, args: dict, text: str) -> tuple[list[str], list[str]]:
    """Paths one tool result credits the agent with: ``(read, listed)``.

    A read counts only if the result carries the echo header — an errored read
    showed the model nothing.
    """
    if name == "read":
        return normalize_list(READ_HEADER_RE.findall(text)), []
    if name == "grep":
        listed = normalize_list(hit[0] for hit in GREP_HIT_RE.findall(text) if _looks_like_path(hit[0]))
        if not listed:
            # Single-file grep: the hit lines carry no path, so credit the arg.
            arg_path = normalize_path(str(args.get("path") or ""))
            if arg_path and text.strip() and _looks_like_path(arg_path.rsplit("/", 1)[-1]):
                listed = [arg_path]
        return [], listed
    if name == "glob":
        tokens = (line.strip() for line in text.splitlines())
        return [], normalize_list(t for t in tokens if _looks_like_path(t) and " " not in t)
    return [], []


def search_quality(messages: list[dict[str, Any]], gold: set[str]) -> dict[str, float]:
    """How well-aimed the search was, measured structurally. Diagnostic only —
    never a reward term."""
    listed: set[str] = set()
    read: set[str] = set()
    for turn in calls_by_turn(messages):
        for name, args, text in turn:
            read_hits, listed_hits = credited_paths(name, args, text)
            read.update(read_hits)
            listed.update(listed_hits)

    pool = listed | read
    return {
        "search_pool": float(len(pool)),
        # Gold density of what the searches surfaced: how well-aimed the queries were.
        "search_precision": (len(pool & gold) / len(pool)) if pool else 0.0,
        "num_read": float(len(read)),
        # Gold density of what the agent chose to open out of that pool.
        "read_precision": (len(read & gold) / len(read)) if read else 0.0,
    }


def zero_hit_calls(messages: list[dict[str, Any]]) -> int:
    """Number of grep/glob calls that returned no hits or were rejected as too broad."""
    return sum(
        any(text.startswith(_ZERO_HIT_PREFIXES) for text in _result_texts(block))
        for block in _blocks(messages)
        if "toolResult" in block
    )


def _as_int(value, default: int) -> int:
    """Tolerate the malformed numbers a small model emits in tool arguments."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def idle_turns(turns: list[list[tuple[str, dict, str]]]) -> int:
    """Turns that yielded no new information. Input: per turn, (tool name, args, result text).

    A search is productive if it lists a path not seen before; a read is productive if its
    (path, window) was not read before. Zero-hit, errored and repeat calls are not. A turn
    is idle only when every call in it is unproductive — so a confirming read or a wide
    sweep where some patterns miss is never idle, a hallucinated-identifier turn is.
    """
    seen: set[str] = set()
    windows: set[tuple[str, int, int]] = set()
    idle = 0
    for turn in turns:
        productive = False
        for name, args, text in turn:
            if text.startswith((*_ZERO_HIT_PREFIXES, "error:")):
                continue
            read, listed = credited_paths(name, args, text)
            if name == "read":
                path = normalize_path(str(args.get("path") or ""))
                window = (
                    path,
                    _as_int(args.get("start_line"), 1),
                    _as_int(args.get("end_line"), 0),
                )
                if read and window not in windows:
                    windows.add(window)
                    productive = True
                seen.update(read)
            elif any(p not in seen for p in listed):
                seen.update(listed)
                productive = True
        if turn and not productive:
            idle += 1
    return idle


def calls_by_turn(messages: list[dict[str, Any]]) -> list[list[tuple[str, dict, str]]]:
    """Strands messages -> per assistant turn, (tool name, args, result text) for ``idle_turns``."""
    calls: dict[str, tuple[str, dict]] = {}
    results: dict[str, str] = {}
    for block in _blocks(messages):
        if "toolUse" in block:
            call_id, call = _call(block)
            calls[call_id] = call
        elif "toolResult" in block:
            results[(block["toolResult"] or {}).get("toolUseId")] = "\n".join(_result_texts(block))
    turns = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        ids = [_call(block)[0] for block in _blocks([message]) if "toolUse" in block]
        if ids:
            turns.append([(*calls[i], results.get(i, "")) for i in ids if i in calls])
    return turns


def tool_result_blob(messages: list[dict[str, Any]]) -> str:
    """All tool RESULT text concatenated — what the agent actually saw."""
    return "\n".join(text for block in _blocks(messages) if "toolResult" in block for text in _result_texts(block))
