"""AgentCore Runtime entrypoint for the SWE-grep code-search agent.

The trainer invokes ``invoke_agent`` once per task with the dataset row plus a ``_config``
dict (gateway ``base_url``, ``model_id`` and the rollout's ``api_key``). The agent explores
the S3 checkout of ``repo@base_commit``, submits regions, and returns the scalar ``reward``
(``scoring/reward.py``) with its metrics and its assistant turns (for offline re-scoring).
"""

import logging
import os

import httpx
import openai
from openai import AsyncOpenAI
from rollout.agent_prompts import build_system_prompt
from rollout.custom_openai_model import CustomOpenAIModel
from rollout.repo_store import ensure_checkout
from rollout.repo_tools import build_repo_tools, top_level_listing
from rollout.turn_cap import TurnCapHook
from scoring.reward import compute_reward
from strands import Agent, ModelRetryStrategy
from strands.agent.conversation_manager import NullConversationManager

from agentcore_rl_toolkit import AgentCoreRuntimeApp

logger = logging.getLogger(__name__)

app = AgentCoreRuntimeApp()

# Sampling is the trainer's: the gateway applies its rollout/eval sampling params to requests that omit them,
# and for benchmarks endpoint/serve_eval.sh fixes them on the vLLM server.

LLM_RETRIES = 3  # retries after the first attempt
LLM_READ_TIMEOUT = float(os.environ.get("SWEGREP_LLM_READ_TIMEOUT", "300"))
LLM_CONNECT_TIMEOUT = float(os.environ.get("SWEGREP_LLM_CONNECT_TIMEOUT", "120"))


class RetryTransientErrors(ModelRetryStrategy):
    """Strands' model retry, extended from rate limits to transient vLLM and network errors.

    Strands drops a failed turn and calls the model again, so a stream that breaks halfway is
    simply requested again. Timeouts are not retried: each one has already used up
    ``LLM_READ_TIMEOUT``.
    """

    def is_retryable(self, exc: Exception) -> bool:
        if isinstance(exc, (openai.APITimeoutError, httpx.TimeoutException)):
            return False
        return (
            super().is_retryable(exc)  # rate limits
            or isinstance(exc, (openai.APIConnectionError, openai.InternalServerError, httpx.TransportError))
            or type(exc) is openai.APIError  # the server sent an error event inside a stream
            # vLLM's tokenizer race ("Already borrowed") comes back as a 400.
            or (isinstance(exc, openai.BadRequestError) and "already borrowed" in str(exc).lower())
        )


def _llm_client(base_url: str, api_key: str) -> AsyncOpenAI:
    """OpenAI client for the rollout gateway (training) or the vLLM server (benchmarks).

    ``api_key`` is the rollout's session key (``_config.api_key``): the gateway uses it to
    attribute each LLM turn to this rollout. A static key ("EMPTY") trains nothing.
    """
    return AsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
        # The SDK's 5 s connect timeout is too tight when a validation burst starts.
        timeout=httpx.Timeout(timeout=LLM_READ_TIMEOUT, connect=LLM_CONNECT_TIMEOUT),
        max_retries=0,  # RetryTransientErrors is the only retry layer
    )


@app.entrypoint
def invoke_agent(payload: dict, context=None) -> dict:
    config = payload["_config"]
    client = _llm_client(config["base_url"], config.get("api_key") or "EMPTY")
    result, _ = run_episode(payload, CustomOpenAIModel(client=client, model_id=config["model_id"]))
    return result


def run_episode(payload: dict, model) -> tuple[dict, Agent | None]:
    """One task with ``model`` as the policy: the result the runtime returns, and the agent.

    eval/claude_bench.py calls this directly with Claude as the policy.
    """
    repo = str(payload.get("repo", ""))
    base_commit = str(payload.get("base_commit", ""))
    problem_statement = str(payload.get("problem_statement") or payload.get("question", ""))
    gold_files = list(payload.get("gold_files", []) or [])
    gold_lines = payload.get("gold_lines") or "{}"  # JSON str {path: [[s,e],..]}

    rollout_exception = False
    agent = None
    repo_root = ""
    try:
        # Fetch + extract the repo snapshot from S3 (no internet in ACR VPC mode).
        repo_root = str(ensure_checkout(repo, base_commit))

        agent = Agent(
            model=model,
            tools=build_repo_tools(repo_root),
            system_prompt=build_system_prompt(repo, top_level=top_level_listing(repo_root)),
            hooks=[TurnCapHook()],
            retry_strategy=RetryTransientErrors(max_attempts=LLM_RETRIES + 1, initial_delay=0.5, max_delay=4),
            # Never trim history: the default sliding window drops the task on long episodes.
            conversation_manager=NullConversationManager(),
            callback_handler=None,
        )
        agent(problem_statement)
    except Exception as exc:  # noqa: BLE001
        rollout_exception = True
        logger.warning("rollout failed: %s", exc)

    messages = list(agent.messages) if agent is not None else []
    # An exception before any assistant turn is infrastructure: scored 0, not -1.
    infra_exception = rollout_exception and not any(m.get("role") == "assistant" for m in messages)
    reward, metrics = compute_reward(
        messages,
        gold_files,
        rollout_exception and not infra_exception,
        repo_root=repo_root,
        gold_lines=gold_lines,
        infra_exception=infra_exception,
    )
    result = {"reward": float(reward), "metrics": metrics, "messages": assistant_turns(messages)}
    return result, agent


def assistant_turns(messages: list[dict]) -> list[dict]:
    """The assistant messages reduced to text and toolUse blocks, for offline re-scoring.

    That is everything ``extract_answer`` reads; reasoning and tool results are dropped
    to keep the result small.
    """
    turns = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        blocks = content if isinstance(content, list) else [{"text": str(content or "")}]
        kept = []
        for block in blocks:
            if not isinstance(block, dict):
                continue
            if "text" in block:
                kept.append({"text": block["text"]})
            elif isinstance(block.get("toolUse"), dict):
                call = block["toolUse"]
                kept.append({"toolUse": {"name": call.get("name"), "input": call.get("input")}})
        turns.append({"role": "assistant", "content": kept})
    return turns


if __name__ == "__main__":
    app.run()
