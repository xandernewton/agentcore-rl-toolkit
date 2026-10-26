"""Tool names in a reply stay within the OpenAI function-name charset."""

import pytest

from agentcore_rl_toolkit.rollout_gateway.adapters.openai import _build_reply_parts, _valid_tool_name
from agentcore_rl_toolkit.rollout_gateway.render import ParsedOutput


@pytest.mark.parametrize(
    "name,expected",
    [
        ("read_file", "read_file"),
        ("repo.read-file", "repo_read-file"),
        ("functions:grep", "functions_grep"),
        ("a" * 70, "a" * 64),
        ("", "invalid_tool"),
    ],
)
def test_valid_tool_name(name, expected):
    assert _valid_tool_name(name) == expected


def test_reply_uses_the_coerced_name_on_both_sides():
    parsed = ParsedOutput(reasoning="", text="", tool_uses=[{"name": "repo.grep", "input": {"q": "x"}}])
    wire, manager, _ = _build_reply_parts(parsed, "stop")
    assert wire["tool_calls"][0]["function"]["name"] == "repo_grep"
    assert manager["tool_calls"][0]["function"]["name"] == "repo_grep"
