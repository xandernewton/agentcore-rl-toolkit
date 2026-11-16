"""Session message matcher for history replayed by Strands' stock ``OpenAIModel``.

Strands does not replay an assistant turn as the session returned it: it drops
``reasoning_content``, sends text as a list of parts, strips whitespace-only text from
tool-call turns, replaces other blank text with ``[blank text]``, renames invalid tool
calls to ``INVALID_TOOL_NAME`` and re-serializes tool arguments. This matcher accepts
exactly those rewrites and delegates everything else to Miles' ``loose_tool_call``.
On a match the session keeps its stored message and token checkpoint, so training
sees what the model generated.
"""

from typing import Any

from miles.utils.chat_template_utils.message_matcher_hub import loose_tool_call_message_matches

BLANK_TEXT = "[blank text]"  # strands.event_loop.streaming._normalize_messages
INVALID_TOOL_NAME = "INVALID_TOOL_NAME"


def strands_message_matches(stored: dict[str, Any], replayed: dict[str, Any]) -> bool:
    stored, replayed = _flatten(stored), _flatten(replayed)
    if stored.get("role") == replayed.get("role") == "assistant":
        replayed = _undo_strands_rewrites(stored, replayed)
    return loose_tool_call_message_matches(stored, replayed)


def _flatten(message: dict[str, Any]) -> dict[str, Any]:
    content = message.get("content")
    if isinstance(content, list) and all(isinstance(part, dict) and part.get("type") == "text" for part in content):
        return {**message, "content": "".join(str(part.get("text", "")) for part in content)}
    return message


def _is_blank(text: Any) -> bool:
    return text is None or (isinstance(text, str) and not text.strip())


def _undo_strands_rewrites(stored: dict[str, Any], replayed: dict[str, Any]) -> dict[str, Any]:
    """Return a comparison copy of ``replayed`` with Strands' rewrites mapped back to ``stored``."""
    replayed = dict(replayed)
    if not replayed.get("reasoning_content") and stored.get("reasoning_content"):
        replayed["reasoning_content"] = stored["reasoning_content"]

    content = replayed.get("content")
    if _is_blank(stored.get("content")) and (
        (stored.get("tool_calls") and _is_blank(content)) or (not stored.get("tool_calls") and content == BLANK_TEXT)
    ):
        replayed["content"] = stored.get("content")

    stored_calls, replayed_calls = stored.get("tool_calls"), replayed.get("tool_calls")
    if isinstance(stored_calls, list) and isinstance(replayed_calls, list) and len(stored_calls) == len(replayed_calls):
        replayed["tool_calls"] = [_restore_name(s, r) for s, r in zip(stored_calls, replayed_calls, strict=True)]
    return replayed


def _restore_name(stored: Any, replayed: Any) -> Any:
    if not (isinstance(stored, dict) and isinstance(replayed, dict)) or stored.get("id") != replayed.get("id"):
        return replayed
    stored_fn, replayed_fn = stored.get("function"), replayed.get("function")
    if isinstance(stored_fn, dict) and isinstance(replayed_fn, dict) and replayed_fn.get("name") == INVALID_TOOL_NAME:
        return {**replayed, "function": {**replayed_fn, "name": stored_fn.get("name")}}
    return replayed
