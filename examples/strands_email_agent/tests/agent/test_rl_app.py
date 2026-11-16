"""The agent against a scripted OpenAI-compatible server: no GPU, database or Bedrock.

uv run --project agent --with pytest pytest tests/agent
"""

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "agent"))

import judge  # noqa: E402
import rl_app  # noqa: E402

RIGHT = "<right@enron>"
TASK = {
    "id": 1,
    "question": "When is the meeting?",
    "answer": "Tuesday",
    "message_ids": [RIGHT],
    "inbox_address": "a@enron.com",
    "query_date": "2001-05-27",
}
SEARCH = {"name": "search_inbox", "arguments": {"keywords": ["meeting"]}}


def tool_call(name, arguments):
    call = {"id": f"call_{name}", "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}
    return {"role": "assistant", "content": None, "tool_calls": [call]}


def final(text, as_reasoning=False):
    """A final message. SGLang's qwen3 parser returns it as reasoning when the model did not think."""
    if as_reasoning:
        return {"role": "assistant", "content": None, "reasoning_content": text}
    return {"role": "assistant", "content": text}


@pytest.fixture
def server(monkeypatch):
    """Serves `script` (assistant messages, or HTTP status codes) in order and records the requests."""
    script, requests = [], []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            step = script.pop(0)
            if isinstance(step, int):
                status, body = step, {"error": {"message": "scripted failure"}}
            else:
                choice = {
                    "index": 0,
                    "message": step,
                    "finish_reason": "tool_calls" if step.get("tool_calls") else "stop",
                }
                usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
                status = 200
                body = {"id": "c", "object": "chat.completion", "model": "m", "choices": [choice], "usage": usage}
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    # The judge passes an answer that contains the reference; searches find one other email.
    monkeypatch.setattr(rl_app, "judge", lambda question, reference, answer: reference in answer)
    monkeypatch.setattr(rl_app, "search_emails", lambda *args: [{"message_id": "<other@enron>", "snippet": "..."}])
    monkeypatch.setattr(rl_app, "db_read_email", lambda message_id: {"message_id": message_id, "body": "..."})
    config = {
        "base_url": f"http://127.0.0.1:{httpd.server_port}/sessions/s/v1",
        "model_id": "m",
        "sampling_params": {"temperature": 1.0, "max_tokens": 64, "top_k": -1},
    }

    def run(*steps, **config_overrides):
        script.extend(steps)
        return rl_app.invoke_agent(TASK | {"_config": config | config_overrides})

    yield SimpleNamespace(run=run, requests=requests)
    httpd.shutdown()


def test_correct_answer_with_the_right_source(server):
    result = server.run(tool_call(**SEARCH), final(f"The meeting is on Tuesday.\nSources: {RIGHT}"))
    assert result["metrics"]["answer_correct"] == result["metrics"]["citation_overlap"] == 1.0
    assert result["reward"] == pytest.approx(1.0 + 0.3 + 0.1)  # correct, cited exactly the right email
    assert len(server.requests) == 2  # a message without a tool call ends the rollout


def test_requests_carry_miles_sampling_and_plain_string_content(server):
    server.run(tool_call(**SEARCH), final("Tuesday"))
    first, second = server.requests
    assert first["temperature"] == 1.0 and first["max_tokens"] == 64 and first["top_k"] == -1
    assert [tool["function"]["name"] for tool in first["tools"]] == ["search_inbox", "read_email"]
    # Qwen3's chat template renders nothing for a list of content parts.
    system, user, _, tool = second["messages"]
    assert system["content"].startswith("You are an email search agent") and user["content"] == TASK["question"]
    assert isinstance(tool["content"], str) and "<other@enron>" in tool["content"]


def test_final_message_returned_as_reasoning_is_still_the_answer(server):
    text = f"The meeting is on Tuesday.\n\n**Sources:** {RIGHT}, <other@enron>"
    metrics = server.run(tool_call(**SEARCH), final(text, as_reasoning=True))["metrics"]
    assert metrics["answer_correct"] == 1.0 and metrics["num_sources"] == 2.0 and metrics["citation_overlap"] == 0.5


def test_reward_orders_the_outcomes(server, monkeypatch):
    correct = server.run(tool_call(**SEARCH), final("The meeting is on Tuesday."))
    declined = server.run(tool_call(**SEARCH), final("The emails do not specify when the meeting is."))
    wrong = server.run(tool_call(**SEARCH), final("The meeting is on Friday."))
    monkeypatch.setattr(rl_app, "MAX_TURNS", 2)
    out_of_turns = server.run(tool_call(**SEARCH), tool_call(**SEARCH))
    assert declined["metrics"]["declined"] == 1.0 and declined["metrics"]["answered"] == 0.0
    assert out_of_turns["metrics"]["out_of_budget"] == 1.0 and len(server.requests) == 8
    # Never answering must not beat a wrong answer, or the policy learns to loop until the cap.
    assert correct["reward"] > declined["reward"] > wrong["reward"] > out_of_turns["reward"]


def test_opening_a_right_email_earns_partial_credit(server):
    read = server.run(tool_call("read_email", {"message_id": RIGHT}), final("The meeting is on Friday."))
    other = server.run(tool_call("read_email", {"message_id": "<other@enron>"}), final("The meeting is on Friday."))
    assert read["metrics"]["read_right_email"] == 1.0 and read["reward"] - other["reward"] == pytest.approx(0.15)


def test_rejected_tool_call_gets_a_helpful_error_and_still_counts_as_a_turn(server, monkeypatch):
    monkeypatch.setattr(rl_app, "MAX_TURNS", 3)
    malformed = tool_call("search_inbox", {})  # no keywords: rejected before the tool runs
    result = server.run(malformed, malformed, malformed)
    assert result["metrics"]["out_of_budget"] == 1.0 and len(server.requests) == 3
    assert '"arguments"' in server.requests[1]["messages"][-1]["content"]


def test_request_that_would_not_fit_the_session_ends_the_rollout(server):
    # Room for the question (about 25 tokens) but not for the question plus a search result.
    result = server.run(tool_call(**SEARCH), max_seq_len=64 + 2048 + 60)
    assert result["metrics"]["out_of_budget"] == 1.0 and len(server.requests) == 1


def test_model_failure_is_reported_not_scored(server):
    assert "error" in server.run(500) and len(server.requests) == 1  # and not retried
    assert "error" in server.run(tool_call(**SEARCH), 500)


def test_answer_without_a_verdict_is_reported_not_scored(server, monkeypatch):
    monkeypatch.setattr(rl_app, "judge", lambda *args: None)
    assert server.run(final("The meeting is on Tuesday.")) == {"reward": 0.0, "error": "the judge gave no verdict"}


@pytest.mark.parametrize(
    "replies,verdict,calls",
    [
        (["True", "True"], True, 2),
        (["False", "True", "False."], False, 3),
        (["", "True", "true"], True, 3),  # an empty reply is not a vote
        (["", "", "True"], None, 3),  # one vote is not a majority
    ],
)
def test_judge_takes_a_majority_and_stops_when_two_agree(monkeypatch, replies, verdict, calls):
    asked = []

    def converse(**request):
        asked.append(request)
        return {"output": {"message": {"content": [{"text": replies[len(asked) - 1]}]}}}

    monkeypatch.setattr(judge, "_client", SimpleNamespace(converse=converse))
    assert judge.judge("q", "reference", "answer") == verdict and len(asked) == calls
    assert asked[0]["inferenceConfig"] == {"maxTokens": 2048}  # room for the model to reason first
