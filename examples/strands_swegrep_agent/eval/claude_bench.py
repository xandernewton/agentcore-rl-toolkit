"""Run the agent with Claude on Amazon Bedrock as the policy on a benchmark task file, on this host.

  python eval/claude_bench.py --data data/benchmarks/swe-explore.jsonl --out OUT \\
      --model global.anthropic.claude-sonnet-5-5 [--concurrency 8] [--region us-east-1]

The Claude rows of the README table. It is the agent's own episode (agent/rl_app.py run_episode: system
prompt, repo tools, TurnCapHook, no history trimming, reward and metrics) with the deployed runtime's
SWEGREP_* settings; only the model differs. Claude runs with adaptive thinking at effort high, prompt
caching, 16k output tokens per turn (the Qwen cap) and no refusal fallback (a refusal is no answer).

Output matches eval/full_benchmark.py's, so eval/score_*.py score it the same
way, and re-running resumes in the same way. Needs `uv sync --extra claude` and Bedrock access to the
model; the inference profiles are global, so --region is any region where they are enabled.
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import threading
from pathlib import Path

import boto3
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
MAX_TOKENS = 16384
MODEL_PARAMS = {
    "thinking": {"type": "adaptive"},
    "output_config": {"effort": "high"},
    "cache_control": {"type": "ephemeral"},
}
# List prices per million tokens (input, output, cache read, cache write), for the running cost only.
PRICES = {
    "global.anthropic.claude-sonnet-5-5": (2.0, 10.0, 0.20, 2.50),
    "global.anthropic.claude-sonnet-5": (2.0, 10.0, 0.20, 2.50),
}
USAGE_KEYS = ("inputTokens", "outputTokens", "cacheReadInputTokens", "cacheWriteInputTokens")


def use_runtime_settings(arn):
    """Give this process the deployed agent's SWEGREP_* settings (tool limits, turn and context
    budgets, answer parsing, reward). The agent modules read them at import, so this runs first.
    The checkout cache and buckets stay local (.env)."""
    control = boto3.client("bedrock-agentcore-control", region_name=arn.split(":")[3])
    env = control.get_agent_runtime(agentRuntimeId=arn.split("/")[-1])["environmentVariables"]
    for key, value in env.items():
        if key.startswith("SWEGREP_") and key != "SWEGREP_CACHE_DIR" and "BUCKET" not in key:
            os.environ[key] = value


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", required=True, help="Bedrock model or inference profile id")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--region", help="Bedrock region (default: the runtime's)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    load_dotenv(ROOT / ".env")
    arn = os.environ["AGENTCORE_AGENT_ARN"]
    region = args.region or arn.split(":")[3]
    use_runtime_settings(arn)

    import full_benchmark as harness
    import rl_app
    from anthropic import AsyncAnthropicBedrock
    from strands.models.anthropic import AnthropicModel

    rows = harness.load_rows(args)
    agent_env = harness.runtime_agent_env(arn, "DEFAULT")
    args.allow_single_line = harness.accepts_single_line(agent_env)
    args.max_answer_regions = 0
    args.reps = 1
    harness.write_or_check_manifest(
        args.out,
        {
            "data": str(args.data.resolve()),
            "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
            "model": args.model,
            "engine": "bedrock-anthropic",
            "max_tokens": MAX_TOKENS,
            "model_params": MODEL_PARAMS,
            "task_ids": [r["instance_id"] for r in rows],
            "agent_env": agent_env,
        },
    )

    price = PRICES.get(args.model)
    usage_path = args.out / "usage.json"
    totals = json.loads(usage_path.read_text()) if usage_path.exists() else dict.fromkeys(USAGE_KEYS, 0)
    lock = threading.Lock()

    def episode(row):
        model = AnthropicModel(model_id=args.model, max_tokens=MAX_TOKENS, params=MODEL_PARAMS)
        model.client = AsyncAnthropicBedrock(aws_region=region)
        result, agent = rl_app.run_episode(row, model)
        if agent is not None:
            usage = agent.event_loop_metrics.accumulated_usage
            with lock:
                for key in USAGE_KEYS:
                    totals[key] += usage.get(key, 0)
                usage_path.write_text(json.dumps(totals, indent=2) + "\n")
                if price:
                    cost = sum(totals[k] * p for k, p in zip(USAGE_KEYS, price, strict=True)) / 1e6
                    print(f"{args.out.name}: cost so far ${cost:.2f}", flush=True)
        return result

    class ClaudeEvaluation(harness.Evaluation):
        async def _invoke(self, row, sid):
            """The episode on this host instead of in the runtime; same (result, error, repo_root)."""
            result, error, root = {}, None, ""
            try:
                root = str(await asyncio.to_thread(harness.repo_store.ensure_checkout, row["repo"], row["base_commit"]))
                result = await asyncio.to_thread(episode, row)
                if result["metrics"].get("infra_exception"):
                    raise RuntimeError("Agent infrastructure exception")
            except Exception as e:  # noqa: BLE001 - recorded as this task's error, retried on re-run
                result, error = {}, repr(e)
            return result, error, root

    asyncio.run(ClaudeEvaluation(args, rows, client=None).run())


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
