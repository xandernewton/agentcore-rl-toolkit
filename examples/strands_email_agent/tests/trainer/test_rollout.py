"""The Miles hooks, with AgentCore and Miles' own generate function replaced by stand-ins."""

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from email_miles import rollout
from miles.utils.types import Sample

from agentcore_rl_toolkit.rollout_session.wire import RolloutDumpResponse


class Session:
    """Stand-in for AgentCoreHttpSession."""

    last = None

    def __init__(self, session_id, state, *, agentcore_runtime_arn, output=None, exception=None, blocked=False):
        self.output, self.exception, self.blocked = output, exception, blocked
        self.payload, self.stopped = None, False
        Session.last = self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        self.stopped = True

    async def run(self, payload):
        self.payload = payload
        if self.blocked:
            await asyncio.Event().wait()
        return RolloutDumpResponse(task_output=self.output, reward=None, metrics={}, exception=self.exception)


def agentcore(**kwargs):
    return patch.object(rollout, "AgentCoreHttpSession", lambda *a, **kw: Session(*a, **kw, **kwargs))


def miles_generate(*samples):
    """Miles' agentic_tool_call.generate returning `samples` in turn, one per call."""
    remaining = list(samples)

    async def generate(input):
        return SimpleNamespace(samples=remaining.pop(0))

    return patch.object(rollout.agentic_tool_call, "generate", generate)


def sample(status=Sample.Status.COMPLETED, **metadata):
    return SimpleNamespace(status=status, metadata=metadata, tokens=[1, 2])


@pytest.fixture(autouse=True)
def environment(monkeypatch):
    monkeypatch.setenv("AGENT_RUNTIME_ARN", "arn:aws:bedrock-agentcore:us-east-1:1:runtime/x")
    monkeypatch.delenv("ROLLOUT_LOG_DIR", raising=False)
    monkeypatch.setattr(rollout, "RETRY_DELAY", 0.0)
    Session.last = None


RUN = ("http://host:9100/sessions/s1", [], {"max_tokens": 8192}, {"payload": {"id": 7}, "max_seq_len": 40960})


@pytest.mark.asyncio
async def test_run_sends_the_task_and_session_to_the_agent_and_returns_its_score():
    with agentcore(output={"reward": 1.4, "metrics": {"answer_correct": 1.0}}):
        result = await rollout.run(*RUN)
    assert result == {"reward": 1.4, "metrics": {"answer_correct": 1.0}}
    assert Session.last.payload["id"] == 7 and Session.last.stopped and not rollout._running
    assert Session.last.payload["_config"] == {
        "base_url": "http://host:9100/sessions/s1/v1",
        "model_id": "default",
        "sampling_params": {"max_tokens": 8192},
        "max_seq_len": 40960,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome",
    [{"output": {"reward": 0.0, "error": "no verdict"}}, {"exception": "Traceback ...\nRuntimeError: boom"}],
)
async def test_run_reports_a_rollout_that_was_not_scored_as_lost(outcome):
    with agentcore(**outcome):
        result = await rollout.run(*RUN)
    assert set(result) == {"lost"}


@pytest.mark.asyncio
async def test_abort_cancels_rollouts_and_stops_their_sessions():
    with agentcore(blocked=True):
        task = asyncio.create_task(rollout.run(*RUN))
        while Session.last is None or Session.last.payload is None:
            await asyncio.sleep(0)
        await rollout.abort(None)
    assert task.cancelled() and Session.last.stopped and not rollout._running


@pytest.mark.asyncio
async def test_lost_rollout_is_run_again():
    scored = sample(reward=1.0, metrics={})
    with miles_generate(sample(lost="dropped connection"), sample(Sample.Status.ABORTED), scored):
        output = await rollout.generate(SimpleNamespace(evaluation=False))
    assert output.samples is scored and await rollout.reward(None, output.samples) == 1.0


@pytest.mark.asyncio
async def test_training_stops_when_a_rollout_is_lost_every_time():
    with miles_generate(*[sample(lost="throttled")] * rollout.ROLLOUT_ATTEMPTS):
        with pytest.raises(RuntimeError, match="lost 3 times: throttled"):
            await rollout.generate(SimpleNamespace(evaluation=False))


@pytest.mark.asyncio
async def test_lost_evaluation_rollout_scores_zero_and_is_recorded():
    lost = [sample(Sample.Status.ABORTED) for _ in range(rollout.ROLLOUT_ATTEMPTS)]
    with miles_generate(*lost):
        output = await rollout.generate(SimpleNamespace(evaluation=True))
    assert output.samples.status == Sample.Status.COMPLETED
    assert await rollout.reward(None, [output.samples]) == [0.0] and output.samples.metadata["lost"]


def test_rollouts_are_logged(tmp_path):
    import gzip
    import json

    logged = Sample(tokens=[1, 2, 3], status=Sample.Status.TRUNCATED)
    logged.metadata = {"payload": {"id": 7}, "reward": 0.5, "metrics": {"answer_correct": 0.0}}
    rollout.log_rollout(tmp_path, logged, "transcript", evaluation=False)
    (record,) = [json.loads(line) for line in gzip.open(tmp_path / "train.jsonl.gz", "rt")]
    assert record["task_id"] == 7 and record["status"] == "truncated" and record["text"] == "transcript"
