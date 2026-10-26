"""Reasoning that a client omits from the echoed history is restored from the served reply.

Strands' ``OpenAIModel`` never sends ``reasoning_content`` back. Without restoring it,
the echoed assistant turn differs from the stored reply, the history match fails, and
the turn re-renders without the reasoning the model generated.
"""

import pytest

from agentcore_rl_toolkit.rollout_gateway import RolloutGateway
from agentcore_rl_toolkit.rollout_gateway.adapters.common import _reply_key, _restore_omitted_reasoning
from agentcore_rl_toolkit.rollout_gateway.render import ParsedOutput

from .test_gateway_integration import FakeBackend, FakeRenderer, bearer, serve

TOOLS = [{"type": "function", "function": {"name": "calculator", "parameters": {"type": "object"}}}]


class ThinkingRenderer(FakeRenderer):
    """FakeRenderer whose replies may open with ``THINK ... /THINK`` reasoning."""

    async def render(self, messages, *, tools=None, add_generation_prompt=True):
        ids: list[int] = []
        for m in messages:
            ids += self._encode(f"{m['role']}:")
            if m.get("reasoning_content"):
                ids += self._encode(f"THINK {m['reasoning_content']} /THINK")
            ids += self._encode(m.get("content") or "")
            for call in m.get("tool_calls") or []:
                fn = call["function"]
                args = " ".join(f"{k}={v}" for k, v in sorted(fn["arguments"].items()))
                ids += self._encode(f"CALL {fn['name']} {args}")
        if add_generation_prompt:
            ids += self._encode("assistant:")
        return ids

    def parse(self, output_ids, *, tools_schema=None):
        text = self.decode(output_ids)
        reasoning = ""
        if text.startswith("THINK ") and " /THINK" in text:
            reasoning, text = text[len("THINK ") :].split(" /THINK", 1)
        out = super().parse(self._encode(text.strip()), tools_schema=tools_schema)
        return ParsedOutput(reasoning=reasoning, text=out.text, tool_uses=out.tool_uses, ill_formed=out.ill_formed)


async def _two_turns(gateway, sid, *, echo_reasoning):
    async with serve(gateway) as client:
        gateway.create_session(sid)
        body1 = {"model": "m", "messages": [{"role": "user", "content": "two plus two"}], "tools": TOOLS}
        msg1 = (await (await client.post("/v1/chat/completions", json=body1, headers=bearer(sid))).json())["choices"][
            0
        ]["message"]
        assert msg1["reasoning_content"] == "add them"
        echoed = {"role": "assistant", "content": None, "tool_calls": msg1["tool_calls"]}
        if echo_reasoning is not None:
            echoed["reasoning_content"] = echo_reasoning
        body2 = {
            "model": "m",
            "messages": [
                {"role": "user", "content": "two plus two"},
                echoed,
                {"role": "tool", "tool_call_id": msg1["tool_calls"][0]["id"], "content": "4"},
            ],
            "tools": TOOLS,
        }
        await client.post("/v1/chat/completions", json=body2, headers=bearer(sid))
        return await gateway.finish_session(sid, reward=1.0)


def _gateway(history_mode):
    renderer = ThinkingRenderer()
    backend = FakeBackend(renderer, ["THINK add them /THINK CALL calculator expr=2+2", "the answer is 4"])
    return RolloutGateway(backend=backend, renderer=renderer, tokenizer=renderer, history_mode=history_mode), backend


@pytest.mark.asyncio
@pytest.mark.parametrize("history_mode", ["tree", "linear"])
async def test_omitted_reasoning_keeps_one_trajectory(history_mode):
    gateway, backend = _gateway(history_mode)
    records = await _two_turns(gateway, f"sid-{history_mode}", echo_reasoning=None)
    assert len(records) == 1
    turn2_prompt = backend.renderer.decode(backend.calls[1]["prompt_ids"])
    assert "THINK add them /THINK" in turn2_prompt
    if history_mode == "linear":
        assert records[0].metadata["linear_healer"].get("nonlinear", 0) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("history_mode", ["tree", "linear"])
async def test_echoed_reasoning_is_unchanged(history_mode):
    gateway, _ = _gateway(history_mode)
    records = await _two_turns(gateway, f"sid-echo-{history_mode}", echo_reasoning="add them")
    assert len(records) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("history_mode", ["tree", "linear"])
async def test_edited_reasoning_still_diverges(history_mode):
    gateway, backend = _gateway(history_mode)
    await _two_turns(gateway, f"sid-edit-{history_mode}", echo_reasoning="summarized")
    turn2_prompt = backend.renderer.decode(backend.calls[1]["prompt_ids"])
    assert "THINK summarized /THINK" in turn2_prompt
    assert "add them" not in turn2_prompt


def test_restore_only_fills_matching_assistant_turns():
    call = [{"type": "function", "function": {"name": "calculator", "arguments": {"expr": "2+2"}}}]
    reply = {"role": "assistant", "content": "", "tool_calls": call, "reasoning_content": "add them"}
    served = {_reply_key(reply): "add them"}
    same = {"role": "assistant", "content": "", "tool_calls": call}
    other = {"role": "assistant", "content": "different", "tool_calls": call}
    user = {"role": "user", "content": ""}
    out = _restore_omitted_reasoning([user, same, other], served)
    assert out[1]["reasoning_content"] == "add them"
    assert "reasoning_content" not in out[2]
    assert out[0] == user
    assert "reasoning_content" not in same  # input is not mutated


def test_no_served_reasoning_is_a_no_op():
    messages = [{"role": "assistant", "content": "hi"}]
    assert _restore_omitted_reasoning(messages, {}) is messages
