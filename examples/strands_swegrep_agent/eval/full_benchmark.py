"""Run one served model on a benchmark task file through the deployed AgentCore agent.

  python eval/full_benchmark.py --data data/benchmarks/swe-explore.jsonl --out OUT --model NAME

The agent calls the model directly on vLLM's OpenAI endpoint (``--base-url``; endpoint/serve_eval.sh, which
fixes the sampling settings server-side). It returns its assistant turns with its reward; answers are parsed
offline with the agent's own parser and saved per task, and eval/score_*.py compute the line metrics.
Every task is retained and gets one attempt. A task that runs past --timeout is scored as no answer
(timed_out in results.jsonl), like any other unanswered task. Infrastructure failures are recorded as
errors, never silently excluded, and a rerun retries only those.
"""

import argparse
import asyncio
import gzip
import hashlib
import json
import logging
import os
import socket
import statistics
import time
import urllib.request
import uuid
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import boto3
from botocore.config import Config
from dotenv import load_dotenv
from rollout import repo_store
from scoring.retrieval import extract_answer, fbeta, normalize_list, normalize_set
from scoring.reward import line_count, merge_spans

from agentcore_rl_toolkit.aws_tools.boto3_tools import get_async_session
from agentcore_rl_toolkit.concurrency.rate_limiter import LocalRateLimiter
from agentcore_rl_toolkit.rollout_session.agentcore_http_session import AgentCoreHttpSession
from agentcore_rl_toolkit.runtime import AgentCoreHttpClient

ROOT = Path(__file__).resolve().parents[1]

logger = logging.getLogger(__name__)

# Enforced by the vLLM server (endpoint/serve_eval.sh), not sent by the agent; recorded in the manifest.
SAMPLING = {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "max_new_tokens": 16384}
MAX_CONTEXT_TOKENS = 65536
ENGINE_PORT = 8399

METRIC_KEYS = [
    "precision", "recall", "f0.5", "f1", "f1.5", "f2", "num_predicted",
    "recall@1", "recall@3", "recall@5", "recall@8", "ranked_capped_recall",
]  # fmt: skip


# --- Scoring (also imported by eval/score_*.py) ----------------------------------------


def parse_messages(messages, repo, *, max_regions=0, allow_single_line=None):
    """Convert OpenAI-shaped assistant messages to the same Strands answer-parser input."""
    converted = []
    for msg in messages:
        blocks = [{"text": msg.get("content") or ""}]
        for call in msg.get("tool_calls") or []:
            fn = call["function"]
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = {}
            blocks.append({"toolUse": {"name": fn["name"], "input": args}})
        converted.append({"role": "assistant", "content": blocks})
    ans = extract_answer(
        converted,
        max_regions=max_regions,
        repo_root=repo,
        allow_single_line=allow_single_line,
    )
    return ans, normalize_list(ans.files, repo)


def score(pred, gold):
    gold = normalize_set(gold)
    pred = normalize_list(pred)
    hits = len(set(pred) & gold)
    p = hits / len(pred) if pred else 0.0
    r = hits / len(gold) if gold else 0.0
    out = {
        "precision": p,
        "recall": r,
        "num_predicted": len(pred),
        "num_gold": len(gold),
        "has_gold": bool(gold),
    }
    out.update({f"f{b:g}": fbeta(p, r, b) for b in [0.5, 1, 1.5, 2]})
    for k in [1, 3, 5, 8]:
        top_hits = len(set(pred[:k]) & gold)
        out[f"recall@{k}"] = top_hits / len(gold) if gold else 0.0
        out[f"capped_recall@{k}"] = top_hits / min(k, len(gold)) if gold else 0.0
    out["ranked_capped_recall"] = 0.8 * out["capped_recall@8"] + 0.2 * out["capped_recall@3"]
    return out


def summarize(records, expected):
    groups = defaultdict(list)
    for r in records:
        groups["all"].append(r)
        groups[r["data_source"]].append(r)
        groups["seen_validation" if r["in_validation"] else "outside_validation"].append(r)

    out = {
        "expected": expected,
        "completed": len(records),
        "errors": sum(bool(r["error"]) for r in records),
        "timeouts": sum(bool(r.get("timed_out")) for r in records),
        "groups": {},
    }
    for name, rs in sorted(groups.items()):
        labeled = [r for r in rs if r["scores"]["has_gold"]]
        g = {
            "n": len(rs),
            "labeled_n": len(labeled),
            "unlabeled_n": len(rs) - len(labeled),
            "errors": sum(bool(r["error"]) for r in rs),
            "timeouts": sum(bool(r.get("timed_out")) for r in rs),
        }
        g.update({k: statistics.mean(r["scores"][k] for r in labeled) if labeled else None for k in METRIC_KEYS})
        g["submitted"] = statistics.mean(r["submitted"] for r in rs)
        g["rollout_exceptions"] = sum(bool(r.get("agent_metrics", {}).get("rollout_exception")) for r in rs)
        out["groups"][name] = g
    return out


def unique_lines(regions):
    """Number of distinct lines the regions cover, merging overlaps within each file."""
    return line_count(merge_spans((r.path, r.start_line, r.end_line) for r in regions))


def parity_error(scores, submitted, metrics, answer):
    """Return an error if the agent's own metrics disagree with the offline parse, so we
    never silently grade a different answer than the deployed parser saw."""
    checks = [
        ("precision", scores["precision"]),
        ("recall", scores["recall"]),
        ("num_predicted", scores["num_predicted"]),
        ("submitted", float(submitted)),
    ]
    for key, want in checks:
        if key not in metrics or abs(float(metrics[key]) - want) > 1e-6:
            return f"Parser parity mismatch {key}: agent={metrics.get(key)}, offline={want}"
    # File-level numbers can agree while regions differ, and the benchmarks score lines.
    for key, want in [
        ("num_predicted_lines", unique_lines(answer.regions)),
        ("invalid_regions", len(answer.invalid_regions)),
    ]:
        if key in metrics and abs(float(metrics[key]) - want) > 1e-6:
            return f"Parser parity mismatch {key}: agent={metrics[key]}, offline={want}"
    return None


def accepts_single_line(agent_env):
    """Whether the deployed agent's parser takes bare ``path:line`` answers, from its runtime
    settings. Pass this to ``parse_messages`` so scoring parses as the agent did."""
    return agent_env.get("SWEGREP_ACCEPT_SINGLE_LINE", "1") == "1"


def to_openai_message(turn):
    """One agent-returned Strands assistant turn as the OpenAI-shaped message that
    ``parse_messages`` and eval/score_*.py read from the traces."""
    blocks = turn.get("content") or []
    calls = [b["toolUse"] for b in blocks if "toolUse" in b]
    return {
        "role": "assistant",
        "content": "\n".join(b["text"] for b in blocks if "text" in b),
        "tool_calls": [{"function": {"name": c["name"], "arguments": c.get("input") or {}}} for c in calls],
    }


# --- Setup -----------------------------------------------------------------------------


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--model",
        required=True,
        help="Served model name; for a LoRA, the --lora-modules adapter name",
    )
    ap.add_argument("--tokenizer", help="Served model directory; recorded in the manifest only")
    ap.add_argument(
        "--base-url",
        help=f"vLLM /v1 URL reachable from the AgentCore runtime "
        f"(default: $SWEGREP_GATEWAY_HOST or this host's IP, port {ENGINE_PORT})",
    )
    ap.add_argument("--concurrency", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--qualifier", default="DEFAULT", help="AgentCore runtime endpoint")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--max-answer-regions", type=int, default=0)
    args = ap.parse_args()
    if not args.base_url:
        host = os.environ.get("SWEGREP_GATEWAY_HOST") or socket.gethostbyname(socket.gethostname())
        assert not host.startswith("127."), f"{host} is loopback, unreachable from the runtime; pass --base-url"
        args.base_url = f"http://{host}:{ENGINE_PORT}/v1"
    return args


def load_rows(args):
    rows = [json.loads(line) for line in args.data.read_text().splitlines() if line.strip()]
    if args.limit:
        rows = rows[: args.limit]
    assert len({r["instance_id"] for r in rows}) == len(rows), "Duplicate instance_id in data"
    assert not any("_config" in r for r in rows), "Dataset rows must not carry _config"
    return rows


def build_manifest(args, rows, agent_env):
    return {
        "data": str(args.data.resolve()),
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "model": args.model,
        "tokenizer": args.tokenizer,
        "engine": "vllm-openai",
        "reps": args.reps,
        "sampling": SAMPLING,
        "max_context_tokens": MAX_CONTEXT_TOKENS,
        "task_ids": [r["instance_id"] for r in rows],
        "scoring": "macro file metrics over labeled tasks, failed tasks count as zero; "
        "unlabeled tasks tracked separately",
        "max_answer_regions": args.max_answer_regions,
        "region_scoring": "Validated region citations projected to files; "
        "file metrics only, not official region or trajectory-context scores",
        # The agent's tool, budget and answer settings define the benchmark; a resumed run must not change them.
        "agent_env": agent_env,
    }


def write_or_check_manifest(out_dir, manifest):
    path = out_dir / "manifest.json"
    if path.exists():
        assert json.loads(path.read_text()) == manifest, "Refusing to resume different evaluation configuration"
    else:
        path.write_text(json.dumps(manifest, indent=2) + "\n")


def check_engine(base_url, model):
    """Fail fast if vLLM is unreachable, isn't serving ``model``, or has another context length.
    (Sampling can't be read back; it is fixed by serve_eval.sh.)"""
    with urllib.request.urlopen(f"{base_url}/models", timeout=10) as r:
        served = {m["id"]: m for m in json.load(r)["data"]}
    assert model in served, f"{model!r} not served at {base_url}; serving {sorted(served)}"
    # A LoRA adapter card has no max_model_len; its base model's card does.
    lengths = {m.get("max_model_len") for m in served.values()} - {None}
    assert lengths == {MAX_CONTEXT_TOKENS}, f"Expected max_model_len {MAX_CONTEXT_TOKENS}, server has {lengths}"


def runtime_agent_env(arn, qualifier):
    """Check the runtime endpoint is READY and return its SWEGREP_* settings (buckets excluded)."""
    control = boto3.client("bedrock-agentcore-control", region_name=arn.split(":")[3])
    runtime_id = arn.split("/")[-1]
    endpoint = control.get_agent_runtime_endpoint(agentRuntimeId=runtime_id, endpointName=qualifier)
    assert endpoint["status"] == "READY", endpoint
    env = control.get_agent_runtime(agentRuntimeId=runtime_id, agentRuntimeVersion=endpoint["liveVersion"])[
        "environmentVariables"
    ]
    if env.get("SWEGREP_FINAL_USER_TURN", "always") != "always":
        logger.warning(
            'SWEGREP_FINAL_USER_TURN=%s: the published numbers used "always"',
            env["SWEGREP_FINAL_USER_TURN"],
        )
    return {k: v for k, v in sorted(env.items()) if k.startswith("SWEGREP_") and "BUCKET" not in k}


class QualifiedAgentCoreClient:
    """``aws_client`` provider for ``AgentCoreHttpClient``: one shared ``bedrock-agentcore``
    client whose invoke and stop calls target the runtime endpoint ``qualifier``
    (``AgentCoreHttpClient`` itself always uses the default endpoint)."""

    def __init__(self, region, qualifier):
        self.region = region
        self.qualifier = qualifier
        self._client = None
        self._context = None
        self._lock = asyncio.Lock()

    def _pin_qualifier(self, params, **kwargs):
        params["qualifier"] = self.qualifier

    async def __call__(self):
        async with self._lock:
            if self._client is None:
                config = Config(
                    max_pool_connections=200,
                    read_timeout=900,
                    user_agent_extra="agentcore-rl-toolkit",
                )
                self._context = (await get_async_session()).create_client(
                    "bedrock-agentcore", region_name=self.region, config=config
                )
                self._client = await self._context.__aenter__()
                for op in ["InvokeAgentRuntime", "StopRuntimeSession"]:
                    self._client.meta.events.register(
                        f"before-parameter-build.bedrock-agentcore.{op}", self._pin_qualifier
                    )
            return self._client

    async def aclose(self):
        if self._context is not None:
            await self._context.__aexit__(None, None, None)
            self._client = self._context = None


# --- Run -------------------------------------------------------------------------------


class Evaluation:
    def __init__(self, args, rows, client):
        self.args = args
        self.rows = rows
        self.client = client
        self.sem = asyncio.Semaphore(args.concurrency)
        self.session_starts = LocalRateLimiter(25)
        self.results_path = args.out / "results.jsonl"
        self.records = self._load_records()

    def _load_records(self):
        records = {}
        if self.results_path.exists():
            for line in self.results_path.read_text().splitlines():
                r = json.loads(line)
                records[(r["rep"], r["instance_id"])] = r
        return records

    def save_summary(self):
        summary = summarize(list(self.records.values()), len(self.rows) * self.args.reps)
        (self.args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary

    async def _invoke(self, row, sid):
        """Run one agent rollout; returns (result, error, repo_root, timed_out). Running past
        --timeout is the model's outcome, not an error: the task is scored as no answer."""
        args = self.args
        result = {}
        error = None
        repo_root = ""
        deadline = None
        # The session id doubles as the API key so vLLM's request logs can be joined to a task.
        task = {
            **row,
            "_config": {"base_url": args.base_url, "model_id": args.model, "api_key": sid},
        }
        try:
            # Match the deployed parser's file existence/EOF validation.
            repo_root = str(await asyncio.to_thread(repo_store.ensure_checkout, row["repo"], row["base_commit"]))
            # The session stops its Runtime session on exit: success, failure or timeout.
            async with AgentCoreHttpSession.with_client(sid, self.client) as session:
                await self.session_starts.wait_async()
                async with asyncio.timeout(args.timeout) as deadline:
                    dump = await session.run(task)
            if dump.exception:
                raise RuntimeError(f"Agent invocation failed: {dump.exception}")
            result = dump.task_output or {}
            if result.get("metrics", {}).get("infra_exception"):
                raise RuntimeError("Agent infrastructure exception")
            if "messages" not in result:
                raise RuntimeError("Agent result has no messages: the runtime predates returning assistant turns")
        except Exception as e:  # noqa: BLE001 - any failure is recorded as this task's error
            if deadline is not None and deadline.expired():
                return {}, None, repo_root, True
            error = repr(e)
        return result, error, repo_root, False

    async def run_task(self, row, rep):
        args = self.args
        async with self.sem:
            sid = str(uuid.uuid4())
            start = time.monotonic()
            result, error, repo_root, timed_out = await self._invoke(row, sid)

            turns = [{"message": to_openai_message(t)} for t in result.pop("messages", None) or []]
            ans, pred = parse_messages(
                [t["message"] for t in turns],
                repo_root,
                max_regions=args.max_answer_regions,
                allow_single_line=args.allow_single_line,
            )
            metrics = result.get("metrics", {})
            scores = score(pred, row["gold_files"])
            if not error and not timed_out:
                error = parity_error(scores, ans.submitted, metrics, ans)
            if error:
                scores = score([], row["gold_files"])

            record = {
                "rep": rep,
                "instance_id": row["instance_id"],
                "data_source": row["data_source"],
                "in_validation": row["in_validation"],
                "error": error,
                "timed_out": timed_out,
                "submitted": float(ans.submitted and not error),
                "predicted": pred,
                "predicted_regions": [asdict(r) for r in ans.regions],
                "invalid_regions": ans.invalid_regions,
                "scores": scores,
                "agent_metrics": metrics,
                "seconds": time.monotonic() - start,
                "session_id": sid,
            }

            trace = {"payload": row, "result": result, "turns": turns, "summary": record}
            with gzip.open(args.out / f"{sid}.json.gz", "wt") as f:
                json.dump(trace, f)
            with self.results_path.open("a") as f:
                f.write(json.dumps(record) + "\n")
            self.records[(rep, row["instance_id"])] = record

            if len(self.records) % 20 == 0 or error:
                self.save_summary()
                print(
                    f"{args.out.name}: {len(self.records)}/{len(self.rows) * args.reps} error={error}",
                    flush=True,
                )
            return record

    async def run(self):
        for rep in range(self.args.reps):
            # Resume: skip finished tasks, retry failed ones.
            pending = [
                r for r in self.rows
                if (rep, r["instance_id"]) not in self.records or self.records[(rep, r["instance_id"])]["error"]
            ]  # fmt: skip
            await asyncio.gather(*(self.run_task(r, rep) for r in pending))

        summary = self.save_summary()
        print(json.dumps(summary), flush=True)
        if summary["errors"]:
            raise RuntimeError(f"{summary['errors']} errors: rerun to retry only failed tasks")
        assert summary["completed"] == summary["expected"]
        (self.args.out / "COMPLETED").write_text("finished\n")


async def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args)

    load_dotenv(ROOT / ".env")
    arn = os.environ["AGENTCORE_AGENT_ARN"]
    check_engine(args.base_url, args.model)
    agent_env = runtime_agent_env(arn, args.qualifier)
    args.allow_single_line = accepts_single_line(agent_env)
    write_or_check_manifest(args.out, build_manifest(args, rows, agent_env))

    aws = QualifiedAgentCoreClient(arn.split(":")[3], args.qualifier)
    client = AgentCoreHttpClient(arn, request_rate_limiter=LocalRateLimiter(100), aws_client=aws)
    try:
        await Evaluation(args, rows, client).run()
    finally:
        await aws.aclose()


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(main())
