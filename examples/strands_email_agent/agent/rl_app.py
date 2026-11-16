"""AgentCore Runtime entrypoint of the email-search agent.

The trainer invokes it with one task (question, reference answer, the emails that hold it) plus
`_config`: the URL of this rollout's Miles session, the model id, Miles' sampling parameters and
the session's token limit. The agent searches and reads the inbox with two tools, gives its answer
as its final message, and returns `{"reward", "metrics"}`.

When the rollout says nothing about the policy (the model could not be reached, or the judge gave
no verdict) it returns `{"reward": 0.0, "error"}` instead; the trainer then runs the task again.
"""

import asyncio
import json
import logging
import os
import re

os.environ.setdefault("STRANDS_NON_INTERACTIVE", "true")
os.environ.setdefault("BYPASS_TOOL_CONSENT", "true")

from email_db import read_email as db_read_email  # noqa: E402
from email_db import search_emails  # noqa: E402
from judge import judge  # noqa: E402
from reward import declined, score, search_facts  # noqa: E402
from strands import Agent, tool  # noqa: E402
from strands.agent.conversation_manager import NullConversationManager  # noqa: E402
from strands.hooks import AfterToolCallEvent, BeforeModelCallEvent, HookProvider, HookRegistry  # noqa: E402
from strands.models.openai import OpenAIModel  # noqa: E402
from strands.types.exceptions import ContextWindowOverflowException, MaxTokensReachedException  # noqa: E402

from agentcore_rl_toolkit import AgentCoreRuntimeApp  # noqa: E402

logger = logging.getLogger(__name__)
app = AgentCoreRuntimeApp()

MAX_TURNS = 10
MAX_SEARCH_RESULTS = 20
SYSTEM_PROMPT = """You are an email search agent. You are given a user query and tools to search the user's email. Use the tools to find the answer to the user's query. You may take up to {max_turns} turns, so if your first search doesn't find the answer, try different keywords.

Search results are ranked by relevance, but the top result is not always the right one. Before answering, open the 2-3 most promising results with read_email and verify which one actually contains the answer. Do not answer after reading only a single email unless it clearly and fully answers the question. If the emails you read do not directly answer the question, search again with different keywords before answering.

When you have the answer, reply with it as your final message, without calling a tool. State the answer completely: include every name, date, time, number and other specific detail the email gives for what the question asks. End the message with one line listing the message_ids of the emails you used:
Sources: <message_id>, <message_id>

User's email address is {inbox_address}
Today's date is {query_date}"""  # noqa: E501


def build_tools(inbox_address, query_date):
    """The agent's two tools, bound to one task's inbox and date."""

    @tool
    def search_inbox(keywords: list[str], from_address: str | None = None, to_address: str | None = None) -> str:
        """Search the inbox for emails matching ANY of the given keywords, ranked by relevance (best
        match first). Returns up to 20 results, each with message_id, subject, date, from_address, and
        a snippet of the matching text.

        Args:
            keywords: List of keywords to search for (e.g. ["budget", "Q3", "Joe"]).
            from_address: Optional sender filter. Set it only when the question makes the direction
                unambiguous (e.g. "who did X email").
            to_address: Optional recipient filter, same guidance as from_address.
        """
        if not keywords:
            return json.dumps({"error": "Provide at least one keyword."})
        try:
            results = search_emails(inbox_address, keywords, from_address, to_address, query_date, MAX_SEARCH_RESULTS)
        except Exception as exc:
            return json.dumps({"error": f"search failed: {exc}"})
        return json.dumps(results or {"error": f"No emails matched keywords {keywords}. Try different keywords."})

    @tool
    def read_email(message_id: str) -> str:
        """Read the full content of an email by its message_id (subject, body, from, to, cc, bcc, date).

        Args:
            message_id: A message_id returned by search_inbox.
        """
        return json.dumps(db_read_email(message_id) or {"error": f"Email not found: {message_id}"})

    return [search_inbox, read_email]


class TextOpenAIModel(OpenAIModel):
    """OpenAIModel that sends text as plain strings.

    Strands sends every message's text as a list of parts. Qwen3's chat template renders only
    string content, so the question and every tool result would reach the model empty.
    """

    @classmethod
    def format_request_messages(cls, *args, **kwargs):
        messages = super().format_request_messages(*args, **kwargs)
        for message in messages:
            parts = message.get("content")
            if isinstance(parts, list) and all(part.get("type") == "text" for part in parts):
                message["content"] = "".join(part["text"] for part in parts)
        return messages


class OutOfBudget(Exception):
    """The rollout used all its turns, or its next request would not fit in the session."""


class Budget(HookProvider):
    """Ends the rollout at the turn cap, or before a request that would not fit in the session.

    Miles truncates a session at its token limit but does not stop the agent, and an over-length
    request fails in the server. Stopping here makes it a scored outcome instead of a lost rollout.
    """

    def __init__(self, max_turns, max_tokens):
        self.max_turns = max_turns
        self.max_tokens = max_tokens

    def register_hooks(self, registry: HookRegistry, **kwargs) -> None:
        registry.add_callback(BeforeModelCallEvent, self.check)

    def check(self, event):
        messages = event.agent.messages
        turns = sum(message["role"] == "assistant" for message in messages)
        # Tokens the server reported for the context so far, plus about 3 characters per token for
        # what has been added since (tool results).
        last_reply = max((i for i, message in enumerate(messages) if message["role"] == "assistant"), default=-1)
        seen = event.agent.event_loop_metrics.projected_context_size or 0
        tokens = seen + len(json.dumps(messages[last_reply + 1 :], default=str)) // 3
        if turns >= self.max_turns or (self.max_tokens and tokens >= self.max_tokens):
            raise OutOfBudget(f"{turns} turns, about {tokens} tokens")


class ToolCallHelp(HookProvider):
    """Tells the model how to fix a tool call whose arguments were rejected.

    A base model often writes the parameters beside "name" instead of inside "arguments". The
    server then parses a call with no arguments, and the bare validation error does not say why,
    so the model repeats the same call until it runs out of turns.
    """

    def register_hooks(self, registry: HookRegistry, **kwargs) -> None:
        registry.add_callback(AfterToolCallEvent, self.explain)

    def explain(self, event):
        text = " ".join(str(block.get("text", "")) for block in event.result.get("content") or [])
        if event.result.get("status") == "error" and "Validation failed" in text:
            name = event.tool_use.get("name", "tool")
            help_text = (
                f"The {name} call was rejected: its parameters were missing or invalid. Put every parameter "
                f'inside "arguments", for example {{"name": "{name}", "arguments": {{...}}}}. Details: {text}'
            )
            event.result = {**event.result, "content": [{"text": help_text}]}


def build_agent(payload):
    config = payload["_config"]
    sampling = dict(config["sampling_params"])
    model = TextOpenAIModel(
        # No retries: a retried request would be sampled twice in one session.
        client_args={"base_url": config["base_url"], "api_key": "EMPTY", "timeout": 600, "max_retries": 0},
        model_id=config["model_id"],
        # The session server rejects training requests whose sampling differs from its own.
        params={"stream": False, "extra_body": sampling},
    )
    # Leave room in the session for one full response and the tool schemas.
    max_tokens = (config.get("max_seq_len") or 0) - sampling.get("max_tokens", 0) - 2048
    return Agent(
        model=model,
        tools=build_tools(payload["inbox_address"], payload["query_date"]),
        system_prompt=SYSTEM_PROMPT.format(
            max_turns=MAX_TURNS, inbox_address=payload["inbox_address"], query_date=payload["query_date"]
        ),
        hooks=[Budget(MAX_TURNS, max(max_tokens, 0)), ToolCallHelp()],
        conversation_manager=NullConversationManager(),  # never drop history: the task is in it
        retry_strategy=None,
        callback_handler=None,
    )


def caused_by(exc, *kinds):
    """Whether `exc` or anything it was raised from is one of `kinds` (Strands wraps hook errors)."""
    while exc is not None and not isinstance(exc, kinds):
        exc = exc.__cause__
    return exc is not None


def final_answer(messages):
    """(answer, cited message_ids) from the agent's final message; ("", []) if it did not give one.

    SGLang's qwen3 reasoning parser returns a reply that contains no tool call as reasoning when
    the model did not think first, so the text may arrive in either kind of block.
    """
    last = messages[-1] if messages else {}
    blocks = last.get("content") or []
    if last.get("role") != "assistant" or any("toolUse" in block for block in blocks):
        return "", []
    text = "".join(block.get("text", "") for block in blocks).strip()
    if not text:
        text = "".join(
            block["reasoningContent"].get("reasoningText", {}).get("text", "")
            for block in blocks
            if "reasoningContent" in block
        ).strip()
    sources = list(re.finditer(r"^\W*sources?\b(.*)$", text, re.IGNORECASE | re.MULTILINE))
    if not sources:
        return text, []
    return text[: sources[-1].start()].strip(), re.findall(r"<[^<>\s]+@[^<>\s]+>", text[sources[-1].start() :])


@app.entrypoint
def invoke_agent(payload):
    agent, out_of_budget = None, False
    try:
        agent = build_agent(payload)
        asyncio.run(agent.invoke_async(payload["question"]))
    except Exception as exc:
        out_of_budget = caused_by(exc, OutOfBudget, ContextWindowOverflowException, MaxTokensReachedException)
        if not out_of_budget:
            logger.exception("Rollout failed")
            return {"reward": 0.0, "error": repr(exc)}

    messages = list(agent.messages)
    answer, sources = ("", []) if out_of_budget else final_answer(messages)
    correct = None
    if answer and not declined(answer):
        correct = judge(payload["question"], payload["answer"], answer)
        if correct is None:
            # Scoring an unjudged answer as wrong would teach the policy the answer was wrong.
            return {"reward": 0.0, "error": "the judge gave no verdict"}
    facts = search_facts(messages, payload["message_ids"])
    reward, metrics = score(facts, answer, sources, payload["message_ids"], correct, out_of_budget)
    return {"reward": reward, "metrics": metrics}


if __name__ == "__main__":
    app.run()
