"""SWE-grep training: the AgentCore RL toolkit's Tinker loop, unchanged, plus recipes, agent metrics
and resume.

  EXP=<name> .venv/bin/python swegrep_train.py [recipe] [--test-run]
  EXP=<name> .venv/bin/python swegrep_train.py [recipe] --resume-from CHECKPOINT --resume-step N
"""

import argparse
import asyncio
import json
import logging
import os
import re
import subprocess
import time
from pathlib import Path

import boto3
from dotenv import load_dotenv

from agentcore_rl_toolkit.backends.tinker_api import train as base
from agentcore_rl_toolkit.rollout_session.agentcore_http_session import AgentCoreHttpSession

logger = logging.getLogger("swegrep_train")

ROOT = Path(__file__).resolve().parent
QWEN_SNAPSHOT = "hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"

# --test-run: one optimizer step end to end (2 prompts x 4 rollouts, a policy update, sampler weight
# sync and a checkpoint), no evaluation, no W&B.
TEST_RUN = {
    "steps": 1,
    "batch_size": 2,
    "group_size": 4,
    "checkpoint_interval": 1,
    "evaluation_dataset": None,
    "wandb_project": None,
    "rollout_timeout": 2400,
}


def load_recipe(path: Path, test_run: bool = False) -> tuple[base.Config, dict]:
    """The recipe and this host's settings as upstream's Config, plus the recipe's "swegrep" section
    (the agent settings the AgentCore runtime must have)."""
    raw = json.loads(path.read_text())
    raw.pop("_comment", None)
    swegrep = raw.pop("swegrep", {})
    if test_run:
        raw |= TEST_RUN
    return base.Config(**raw, **host_settings()), swegrep


def host_settings() -> dict:
    """Config fields that depend on this host, deployment and run, not on the experiment."""
    load_dotenv(".env")  # AGENTCORE_AGENT_ARN, written by cdk/deploy.sh
    hf_home = os.environ.get("HF_HOME", f"{Path.home()}/.cache/huggingface")
    snapshot = f"{hf_home}/{QWEN_SNAPSHOT}"
    model = os.environ.get("MODEL_PATH") or os.environ.get("MODEL_DIR") or snapshot
    exp = run_name()
    return {
        "endpoint": f"http://127.0.0.1:{os.environ.get('TINKER_PORT', '18080')}",
        "base_model": model,
        "tokenizer": model,
        "agent_runtime_arn": required_env("AGENTCORE_AGENT_ARN"),
        "gateway_host": os.environ.get("SWEGREP_GATEWAY_HOST") or private_ip(),
        "output_dir": f"runs/{exp}",
        "exp_id": exp,
    }


def run_name() -> str:
    exp = required_env("EXP")
    # SkyRL rejects other checkpoint names with HTTP 422, at the first checkpoint, hours in.
    if not re.fullmatch(r"[A-Za-z0-9_-]+", exp):
        raise SystemExit(f"EXP must match [A-Za-z0-9_-]+, not {exp!r}")
    return exp


def required_env(name: str) -> str:
    if not (value := os.environ.get(name)):
        raise SystemExit(f"{name} is not set (see README.md)")
    return value


def private_ip() -> str:
    """The runtime reaches the gateway on this host's private IP (its security group opens the port)."""
    return subprocess.check_output(["hostname", "-I"], text=True).split()[0]


def check_runtime(config: base.Config, swegrep: dict, recipe: Path) -> None:
    """The reward a run trains on is set in the runtime by cdk/deploy.sh: check it is the recipe's."""
    arn = config.agent_runtime_arn
    control = boto3.client("bedrock-agentcore-control", region_name=arn.split(":")[3])
    env = control.get_agent_runtime(agentRuntimeId=arn.split("/")[-1])["environmentVariables"]
    want = swegrep.get("runtime_env", {})
    if wrong := {key: (env.get(key), value) for key, value in want.items() if env.get(key) != value}:
        raise SystemExit(
            f"runtime settings (deployed, recipe) differ: {wrong}; "
            f"redeploy with bash cdk/deploy.sh -c recipe={recipe}"
        )


class Hooks:
    """SWE-grep's additions to upstream's loop, which calls its helpers by module name: install()
    replaces them with the wrappers below.

    Agent metrics: upstream's rollout keeps only the agent's reward, but the AgentCore session sees
    the agent's whole result. Keep it per session, write each rollout's metrics (agent/scoring/
    reward.py) to rollouts.jsonl, and add their means to metrics.jsonl (swegrep/...),
    evaluation-NNNN.json (eval/swegrep/...) and W&B.

    Resume: the new LoRA client loads the checkpoint's weights and Adam state as its first request
    (SkyRL allows it only then), training skips the batches already trained, and the step-0
    evaluation is skipped. Steps count from the resume point; scripts/watch.py adds the offset.
    """

    def __init__(self, config: base.Config, resume_from: str | None = None, resume_step: int = 0):
        self.config = config
        self.resume_from = resume_from
        self.resume_step = resume_step
        self.rollouts_path = Path(config.output_dir) / "rollouts.jsonl"
        self.rows: dict[str, dict] = {}  # rollout id -> its row in rollouts.jsonl
        self.agent_results: dict[str, dict] = {}  # session id -> agent result, until popped
        self.phase = "train"  # "eval" while evaluate() runs; upstream never runs the two at once

    def install(self) -> None:
        self.upstream = {
            "session_run": AgentCoreHttpSession.run,
            "rollout": base.rollout,
            "training_data": base.training_data,
            "evaluate": base.evaluate,
            "create_client": base.tinker.ServiceClient.create_lora_training_client_async,
            "read_payloads": base.read_payloads,
        }
        base.rollout = self.rollout
        base.training_data = self.training_data
        base.evaluate = self.evaluate

        # Methods of other classes: the wrapper receives their instance (session, service) first.
        async def session_run(session, task):
            return await self.session_run(session, task)

        async def create_client(service, *args, **kwargs):
            return await self.create_client(service, *args, **kwargs)

        AgentCoreHttpSession.run = session_run
        if self.resume_from:
            base.tinker.ServiceClient.create_lora_training_client_async = create_client
            base.read_payloads = self.read_payloads

    # --- agent metrics

    async def session_run(self, session, task):
        dump = await self.upstream["session_run"](session, task)
        self.agent_results[session.session_id] = dump.task_output or {}
        return dump

    async def rollout(self, config, gateway, client, payload, *args, **kwargs):
        start = time.monotonic()
        episode = await self.upstream["rollout"](config, gateway, client, payload, *args, **kwargs)
        metrics = self.agent_results.pop(episode.rollout_id, {}).get("metrics")
        row = {
            "phase": self.phase,
            "rollout_id": episode.rollout_id,
            "instance_id": payload.get("instance_id"),
            "reward": episode.reward,
            "error": episode.error,
            "metrics": metrics if isinstance(metrics, dict) else None,
            "trained_tokens": sum(sum(record.loss_mask) for record in episode.records),
            "elapsed": time.monotonic() - start,
        }
        self.rows[episode.rollout_id] = row
        with self.rollouts_path.open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        return episode

    def training_data(self, groups):
        datums, metrics = self.upstream["training_data"](groups)
        ids = [episode.rollout_id for group in groups for episode in group]
        metrics.update(self.summarize(ids, "swegrep/"))
        return datums, metrics

    async def evaluate(self, config, gateway, client, payloads, step, **kwargs):
        if step == 0 and self.resume_from:
            logger.info("Resumed run: skipping the step-0 evaluation")
            return {}
        before = set(self.rows)
        self.phase = "eval"
        try:
            metrics = await self.upstream["evaluate"](config, gateway, client, payloads, step, **kwargs)
        finally:
            self.phase = "train"
        metrics.update(self.summarize(set(self.rows) - before, "eval/swegrep/"))
        # Upstream writes this file too; rewrite it with the agent metrics added.
        path = Path(config.output_dir) / f"evaluation-{step:04d}.json"
        path.write_text(json.dumps({"step": step, **metrics}) + "\n")
        return metrics

    def summarize(self, rollout_ids, prefix: str) -> dict[str, float]:
        """Means of the agent's numeric metrics over scored rollouts, plus rollout-level numbers."""
        rows = [self.rows[i] for i in rollout_ids if i in self.rows]
        scored = [row["metrics"] for row in rows if row["reward"] is not None and row["metrics"]]
        out = {}
        for key in sorted({key for metrics in scored for key in metrics}):
            values = [m[key] for m in scored if isinstance(m.get(key), (int, float))]
            if values:
                out[prefix + key] = sum(values) / len(values)
        if rows:
            out[prefix + "trained_tokens_mean"] = sum(r["trained_tokens"] for r in rows) / len(rows)
            out[prefix + "elapsed_mean"] = sum(r["elapsed"] for r in rows) / len(rows)
            out[prefix + "timeouts"] = sum("Timeout" in (r["error"] or "") for r in rows)
        return out

    # --- resume

    async def create_client(self, service, *args, **kwargs):
        client = await self.upstream["create_client"](service, *args, **kwargs)
        await (await client.load_state_with_optimizer_async(self.resume_from)).result_async()
        logger.info("Loaded weights and optimizer state from %s", self.resume_from)
        return client

    def read_payloads(self, path):
        rows = self.upstream["read_payloads"](path)
        if path != self.config.dataset:
            return rows
        return rows[self.resume_step * self.config.batch_size :]


def prepare_run(args: argparse.Namespace) -> tuple[base.Config, Hooks]:
    """Load and check the run's config, and create its output directory."""
    recipe = args.recipe.resolve()
    os.chdir(ROOT)  # recipes and outputs are repo-relative
    config, swegrep = load_recipe(recipe, args.test_run)
    # Without a key, wandb.init prompts for one, which fails in the background; log locally instead.
    if config.wandb_project and not os.environ.get("WANDB_API_KEY"):
        logger.warning("W&B off: WANDB_API_KEY is not set")
        config.wandb_project = None
    output = Path(config.output_dir)
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"{output} is not empty; pick a new EXP")
    check_runtime(config, swegrep, args.recipe)
    resume_step = args.resume_step or 0
    if args.resume_from:
        if not 0 < resume_step < config.steps:
            raise SystemExit(f"--resume-step must be between 1 and {config.steps - 1}")
        config.steps -= resume_step
    logger.info("exp=%s recipe=%s test_run=%s", config.exp_id, args.recipe, args.test_run)

    output.mkdir(parents=True, exist_ok=True)
    run_info = {**swegrep, "resume_from": args.resume_from, "resume_step": resume_step}
    (output / "swegrep.json").write_text(json.dumps(run_info, indent=2) + "\n")
    return config, Hooks(config, args.resume_from, resume_step)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recipe", type=Path, nargs="?", default=Path("recipes/final.json"))
    parser.add_argument("--test-run", action="store_true", help="one-step end-to-end test run")
    parser.add_argument("--resume-from", metavar="CHECKPOINT", help="tinker:// checkpoint path")
    parser.add_argument("--resume-step", type=int, help="the step that checkpoint was saved at")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    args = parser.parse_args()
    if bool(args.resume_from) != (args.resume_step is not None):
        parser.error("--resume-from and --resume-step go together")
    return args


def main():
    args = parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for name in ("httpx", "agentcore_rl_toolkit.client", "agentcore_rl_toolkit.rollout_gateway"):
        logging.getLogger(name).setLevel(logging.DEBUG if args.log_level == "DEBUG" else logging.WARNING)
    config, hooks = prepare_run(args)
    hooks.install()
    asyncio.run(base.train(config))


if __name__ == "__main__":
    main()
