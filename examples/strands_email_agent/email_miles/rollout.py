"""The three Miles hooks that connect a rollout to the agent on AgentCore.

  --custom-agent-function-path     run        send one task and its session URL to the agent
  --custom-generate-function-path  generate   Miles' agentic rollout, rerunning rollouts that were lost
  --custom-rm-path                 reward     read the reward the agent computed

Miles gives each rollout a session: an OpenAI-compatible URL that records the tokens and log-probs
of every model call and assembles the training sample. The agent only makes chat-completion calls.
"""

import asyncio
import gzip
import json
import logging
import os
import random
import time
import uuid
from functools import cache
from pathlib import Path

from miles.rollout.agentic.session import openai_session_url
from miles.rollout.generate_hub import agentic_tool_call
from miles.utils.types import Sample

from agentcore_rl_toolkit.concurrency.rate_limiter import LocalRateLimiter
from agentcore_rl_toolkit.rollout_session.agentcore_http_session import AgentCoreHttpSession

logger = logging.getLogger(__name__)

ROLLOUT_TIMEOUT = 1800  # seconds for one rollout on AgentCore
ROLLOUT_ATTEMPTS = 3  # a lost rollout is run again in a fresh session
RETRY_DELAY = 5.0  # seconds before the first rerun; doubles each attempt
# AgentCore starts at most 25 new sessions per second per account by default. Reruns add sessions
# on top of a batch, so stay well under it.
SESSIONS_PER_SECOND = 15

_running = set()  # rollout tasks in flight, for abort()


@cache
def session_rate_limiter():
    return LocalRateLimiter(SESSIONS_PER_SECOND)


async def run(base_url, prompt, request_kwargs, metadata, **kwargs):
    """Miles agent function: run one task on AgentCore and return what to add to the sample's metadata.

    `{"reward", "metrics"}` for a scored rollout; `{"lost": reason}` when the rollout says nothing
    about the policy (AgentCore or the model server failed, or the judge gave no verdict).
    """
    task = asyncio.current_task()
    _running.add(task)
    payload = metadata["payload"] | {
        "_config": {
            "base_url": openai_session_url(base_url.rstrip("/")),
            "model_id": "default",
            "sampling_params": dict(request_kwargs),
            # Miles truncates the session at this many tokens but does not stop the agent.
            "max_seq_len": metadata.get("max_seq_len"),
        }
    }
    try:
        session = AgentCoreHttpSession(str(uuid.uuid4()), None, agentcore_runtime_arn=os.environ["AGENT_RUNTIME_ARN"])
        async with session:
            await session_rate_limiter().wait_async()
            async with asyncio.timeout(ROLLOUT_TIMEOUT):
                dump = await session.run(payload)
        if dump.exception is not None:
            return {"lost": dump.failure_reason()}
        result = dump.task_output
        if result.get("error"):
            return {"lost": result["error"]}
        return {"reward": float(result["reward"]), "metrics": result["metrics"]}
    except Exception as exc:
        logger.exception("AgentCore rollout failed")
        return {"lost": repr(exc)}
    finally:
        _running.discard(task)


async def abort(args):
    """Miles abort hook: cancel the rollouts in flight; each one stops its AgentCore session on exit."""
    tasks = list(_running)
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


def is_lost(sample):
    """No scored result: the agent function reported a loss, or the session recorded no model calls."""
    return sample.status == Sample.Status.ABORTED or "reward" not in sample.metadata


async def generate(input):
    """Miles' agentic_tool_call.generate, with a policy for rollouts that were lost.

    A dropped connection between Miles' servers or a throttled AgentCore session turns up about
    once in a thousand rollouts. Each call below opens a new session, so a lost rollout is run
    again, after a pause. If it is lost every time, training stops rather than learn from a made-up
    reward; an evaluation rollout is scored 0 and its loss recorded.
    """
    for attempt in range(1, ROLLOUT_ATTEMPTS + 1):
        output = await agentic_tool_call.generate(input)
        sample = output.samples
        if not is_lost(sample):
            break
        logger.warning("Rollout lost (attempt %d of %d): %s", attempt, ROLLOUT_ATTEMPTS, sample.metadata.get("lost"))
        if attempt < ROLLOUT_ATTEMPTS:
            await asyncio.sleep(RETRY_DELAY * 2 ** (attempt - 1) * random.uniform(0.5, 1.5))
    if is_lost(sample):
        if not input.evaluation:
            raise RuntimeError(f"Rollout lost {ROLLOUT_ATTEMPTS} times: {sample.metadata.get('lost')}")
        sample.status = Sample.Status.COMPLETED  # Miles skips the reward hook for aborted samples
        sample.metadata.update(reward=0.0, metrics={}, lost=sample.metadata.get("lost") or "no model calls recorded")
    if log_dir := os.environ.get("ROLLOUT_LOG_DIR"):
        text = await asyncio.to_thread(input.state.tokenizer.decode, sample.tokens)
        log_rollout(Path(log_dir), sample, text, input.evaluation)
    return output


generate.add_arguments = agentic_tool_call.generate.add_arguments


async def reward(args, samples, **kwargs):
    """Miles reward hook: the agent already scored the rollout."""
    if isinstance(samples, list):
        return [float(sample.metadata["reward"]) for sample in samples]
    return float(samples.metadata["reward"])


def log_rollout(log_dir, sample, text, evaluation):
    """Append the rollout's transcript and metrics to {train,eval}.jsonl.gz."""
    record = {
        "task_id": sample.metadata["payload"]["id"],
        "reward": sample.metadata["reward"],
        "metrics": sample.metadata["metrics"],
        "lost": sample.metadata.get("lost"),
        "status": sample.status.value,
        "tokens": len(sample.tokens),
        "logged_at": time.time(),
        "text": text,
    }
    log_dir.mkdir(parents=True, exist_ok=True)
    # One gzip member per append; no await between open and close, so writes never interleave.
    with gzip.open(log_dir / f"{'eval' if evaluation else 'train'}.jsonl.gz", "at", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
