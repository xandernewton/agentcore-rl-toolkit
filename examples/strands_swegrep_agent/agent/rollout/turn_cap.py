"""TurnCapHook ends the search at the turn or context budget and asks for the answer."""

import os

from strands.hooks import (
    AfterModelCallEvent,
    AfterToolCallEvent,
    BeforeModelCallEvent,
    BeforeToolCallEvent,
    HookProvider,
    HookRegistry,
)

from rollout.agent_prompts import SEARCH_TURNS, final_turn_nudge

# Context budget (tokens) at which TurnCapHook forces the answer turn; 0 = turn cap only.
# Must leave room for one worst-case turn + the reply under the 64k window.
CONTEXT_BUDGET_TOKENS = int(os.environ.get("SWEGREP_CONTEXT_BUDGET_TOKENS", "44000"))
# "always" adds the [FINAL ANSWER PHASE] user turn at the budget, as the agent that trained and evaluated the
# published model did. "auto" skips it when the last tool result already carries the nudge: a new user turn
# makes Qwen3.5's template drop earlier reasoning, which splits the captured trajectory.
FINAL_USER_TURN = os.environ.get("SWEGREP_FINAL_USER_TURN", "always")
if FINAL_USER_TURN not in {"auto", "always"}:
    raise ValueError("SWEGREP_FINAL_USER_TURN must be 'auto' or 'always'")

MAX_FINAL_ATTEMPTS = 2


def _count_turns(messages) -> int:
    """Assistant turns so far. Before the cap every turn is a search turn (a text-only turn
    ends the episode), including one whose malformed tool calls Strands dropped."""
    return sum(m.get("role") == "assistant" for m in messages)


def _approx_tokens(messages) -> int:
    """Cheap transcript size estimate (chars / 3.5) over every content block —
    text, toolUse inputs and toolResult text. Deliberately pessimistic."""
    chars = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            chars += len(content)
            continue
        for block in content or []:
            if not isinstance(block, dict):
                chars += len(str(block))
            elif "text" in block:
                chars += len(str(block["text"]))
            elif "toolUse" in block:
                chars += len(str((block["toolUse"] or {}).get("input", "")))
            elif "toolResult" in block:
                for item in (block["toolResult"] or {}).get("content", []) or []:
                    chars += len(str(item.get("text", "") if isinstance(item, dict) else item))
    return int(chars / 3.5)


def _context_tokens(agent) -> int:
    """Estimate the NEXT prompt's size in tokens: server-reported usage for what
    vLLM has already seen plus a chars/3.5 estimate of the tool results since."""
    messages = agent.messages
    try:
        exact = agent.event_loop_metrics.projected_context_size
    except Exception:  # noqa: BLE001 — metrics are best-effort
        exact = None
    if exact is None:
        return _approx_tokens(messages)
    last_assistant = max((i for i, m in enumerate(messages) if m.get("role") == "assistant"), default=-1)
    return int(exact) + _approx_tokens(messages[last_assistant + 1 :])


class TurnCapHook(HookProvider):
    """Bound search while allowing a plain-text final message.

    Tool schemas stay fixed. At the budget limit search calls are cancelled;
    at most two final model calls are allowed if the model keeps calling tools.
    """

    def __init__(self) -> None:
        self.final_nudge = final_turn_nudge()
        self.always_final_user_turn = FINAL_USER_TURN == "always"
        self.final_phase = False  # no more searching; the next model calls must answer
        self.search_exhausted = False  # latched by _after_tool, read by the next _before_model
        self.final_attempts = 0
        self.final_user_turn_added = False
        self.nudge_in_tool_result = False

    def register_hooks(self, registry: HookRegistry, **kwargs) -> None:
        registry.add_callback(BeforeModelCallEvent, self._before_model)
        registry.add_callback(AfterModelCallEvent, self._after_model)
        registry.add_callback(BeforeToolCallEvent, self._before_tool)
        registry.add_callback(AfterToolCallEvent, self._after_tool)

    def _budget_reached(self, agent) -> bool:
        if _count_turns(agent.messages) >= SEARCH_TURNS:
            return True
        return CONTEXT_BUDGET_TOKENS > 0 and _context_tokens(agent) >= CONTEXT_BUDGET_TOKENS

    def _before_model(self, event):
        self.final_phase = self.final_phase or self.search_exhausted or self._budget_reached(event.agent)
        if not self.final_phase:
            return
        skip_user_turn = self.nudge_in_tool_result and not self.always_final_user_turn
        if not self.final_user_turn_added and not skip_user_turn:
            text = (
                "[FINAL ANSWER PHASE] Searching is finished. "
                "Do not call any tools, write a patch, or continue searching."
            )
            event.agent.messages.append({"role": "user", "content": [{"text": text + self.final_nudge}]})
            self.final_user_turn_added = True
        if self.final_attempts >= MAX_FINAL_ATTEMPTS:
            event.cancel = "Final submission attempts exhausted."
        else:
            self.final_attempts += 1

    def _after_model(self, event: AfterModelCallEvent) -> None:
        # A failed call produced no turn, so a retry of it is not another final attempt.
        if event.exception is not None and self.final_phase:
            self.final_attempts -= 1

    def _before_tool(self, event):
        if self.final_phase:
            event.cancel_tool = "Search budget exhausted. " + self.final_nudge

    def _after_tool(self, event: AfterToolCallEvent) -> None:
        if self.final_phase or not self._budget_reached(event.agent):
            return
        # Latch this decision: Strands' projected usage can change after the
        # tool batch completes. Never stop here; reserve final submission.
        self.search_exhausted = True
        if self.nudge_in_tool_result:
            return
        event.result.setdefault("content", []).append({"text": self.final_nudge})
        self.nudge_in_tool_result = True
