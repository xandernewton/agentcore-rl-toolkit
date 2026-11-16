import copy

import pytest

pytest.importorskip("strands.models.openai")

from email_miles.message_matcher import strands_message_matches  # noqa: E402
from miles.utils.chat_template_utils.message_matcher_hub import (  # noqa: E402
    loose_tool_call_message_matches,
    resolve_session_message_matcher,
)
from strands.event_loop.streaming import _normalize_messages  # noqa: E402
from strands.models.openai import OpenAIModel  # noqa: E402

SERVED_CALL = {
    "role": "assistant",
    "content": "\n\n",
    "reasoning_content": "Look for the parser.",
    "tool_calls": [
        {
            "id": "call_1",
            "index": 0,
            "type": "function",
            "function": {"name": "grep", "arguments": '{"pattern":"parse", "path":"src"}'},
        }
    ],
}
SERVED_ANSWER = {"role": "assistant", "content": "\n\n", "reasoning_content": "Done."}


def strands_replay(served, name=None):
    """Return what stock Strands sends back for ``served`` after its own history handling."""
    blocks = [{"reasoningContent": {"reasoningText": {"text": served["reasoning_content"]}}}]
    if served.get("content"):
        blocks.append({"text": served["content"]})
    for call in served.get("tool_calls") or []:
        tool_use = {
            "toolUseId": call["id"],
            "name": name or call["function"]["name"],
            "input": {"path": "src", "pattern": "parse"},
        }
        blocks.append({"toolUse": tool_use})
    history = [{"role": "user", "content": [{"text": "Find the parser."}]}, {"role": "assistant", "content": blocks}]
    (replayed,) = [
        m for m in OpenAIModel.format_request_messages(_normalize_messages(history)) if m["role"] == "assistant"
    ]
    return replayed


@pytest.mark.parametrize(
    "served,name",
    [
        (SERVED_CALL, None),
        (SERVED_CALL, "repo.grep"),
        ({**SERVED_CALL, "content": "Searching.\n\n"}, None),
        (SERVED_ANSWER, None),
    ],
)
def test_accepts_strands_replay_that_loose_tool_call_rejects(served, name):
    if name:
        served = copy.deepcopy(served)
        served["tool_calls"][0]["function"]["name"] = name
    replayed = strands_replay(served, name)
    assert not loose_tool_call_message_matches(served, replayed)
    assert strands_message_matches(served, replayed)


def test_accepts_identical_messages_of_any_role():
    for message in (
        SERVED_CALL,
        {"role": "user", "content": [{"type": "text", "text": "q"}]},
        {"role": "tool", "content": "x"},
    ):
        assert strands_message_matches(message, copy.deepcopy(message))


@pytest.mark.parametrize(
    "change",
    [
        lambda m: m.update(reasoning_content="Summarized."),
        lambda m: m.update(content=[{"type": "text", "text": "Other text."}]),
        lambda m: m["tool_calls"][0].update(id="call_2"),
        lambda m: m["tool_calls"][0]["function"].update(name="read_file"),
        lambda m: m["tool_calls"][0]["function"].update(arguments='{"path": "lib", "pattern": "parse"}'),
        lambda m: m["tool_calls"].append(copy.deepcopy(m["tool_calls"][0])),
    ],
)
def test_rejects_real_edits(change):
    replayed = strands_replay(SERVED_CALL)
    change(replayed)
    assert not strands_message_matches(SERVED_CALL, replayed)


def test_blank_text_rules_are_specific_to_the_turn_kind():
    # A tool-call turn never becomes "[blank text]", and an answer turn never loses its text.
    assert not strands_message_matches(SERVED_CALL, {**strands_replay(SERVED_CALL), "content": "[blank text]"})
    assert not strands_message_matches(SERVED_ANSWER, {"role": "assistant"})


def test_resolves_as_a_session_message_matcher():
    matcher = resolve_session_message_matcher("email_miles.message_matcher.strands_message_matches")
    assert matcher(SERVED_CALL, strands_replay(SERVED_CALL)) is True
