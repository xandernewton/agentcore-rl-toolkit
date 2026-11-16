"""Train the email-search agent on Miles.

  python -m email_miles.launch recipes/qwen3_8b.json --run NAME [--smoke] [--dry-run]

Run it in the training image (scripts/docker.sh train ...). Rerunning with the same --run continues
from the newest checkpoint in runs/NAME.
"""

import argparse
import asyncio
import json
import os
import shlex
import sys
from pathlib import Path
from unittest.mock import patch

import ray
from dotenv import dotenv_values
from miles.utils.arguments import parse_args
from miles.utils.async_utils import with_disposer
from miles.utils.chat_template_utils import resolve_reasoning_and_tool_call_parser
from miles.utils.external_utils.model_args_utils import load_model_args
from miles.utils.orchestration_utils import ArgvOrchestratorStartupInfo
from train import train

from email_miles.resume import latest_checkpoint

ROOT = Path(__file__).resolve().parents[1]
SESSION_PORT = 9100  # session servers listen on SESSION_PORT .. SESSION_PORT + session_servers - 1
# --smoke: 2 tasks x 4 rollouts, one update, eval and checkpoint after it.
SMOKE = {
    "dataset": "data/smoke_train.jsonl", "eval_dataset": "data/smoke_val.jsonl", "steps": 1, "batch_size": 2,
    "group_size": 4, "eval_interval": 1, "eval_samples": 1, "save_interval": 1,
}  # fmt: skip


def model_path(hf_model):
    """MODEL_PATH, or the model's newest snapshot in the Hugging Face cache."""
    if path := os.environ.get("MODEL_PATH"):
        return path
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
    snapshots = sorted((hub / f"models--{hf_model.replace('/', '--')}" / "snapshots").glob("*"))
    if not snapshots:
        raise FileNotFoundError(f"{hf_model} is not in the Hugging Face cache; run: hf download {hf_model}")
    return str(snapshots[-1])


def miles_args(recipe, model, run_dir):
    """The Miles arguments this example sets. Everything else is left to Miles' defaults."""
    # Miles binds SGLang's reasoning and tool-call parsers to the session-tokenizer family.
    reasoning_parser, tool_call_parser = resolve_reasoning_and_tool_call_parser(recipe["tito_model"])
    options = {
        # Model: Hugging Face weights, trained as a LoRA adapter.
        "hf-checkpoint": model,
        "megatron-to-hf-mode": "bridge",
        "lora-rank": recipe["lora_rank"],
        "lora-alpha": recipe["lora_alpha"],
        "target-modules": "attn,mlp",
        "sglang-max-lora-rank": recipe["lora_rank"],
        # Every GPU alternates between rollouts and training, so each update is on-policy.
        "actor-num-gpus-per-node": recipe["gpus"],
        "rollout-num-gpus": recipe["gpus"],
        "sglang-mem-fraction-static": 0.7,  # leave room on the shared GPUs for the trainer
        # Data: one task per row; the task itself rides in metadata (see prepare_data.py).
        "prompt-data": ROOT / recipe["dataset"],
        "input-key": "prompt",
        "num-rollout": recipe["steps"],
        "rollout-batch-size": recipe["batch_size"],
        "n-samples-per-prompt": recipe["group_size"],
        "num-steps-per-rollout": 1,
        "rollout-max-response-len": recipe["max_tokens_per_turn"],
        # Agentic rollout: the agent runs on AgentCore and talks to a Miles session (rollout.py).
        "custom-generate-function-path": "email_miles.rollout.generate",
        "custom-agent-function-path": "email_miles.rollout.run",
        "custom-agent-function-mode": "inline",  # run() only waits on AgentCore; no process per rollout
        "custom-rm-path": "email_miles.rollout.reward",
        "max-seq-len": recipe["max_seq_len"],  # at most the model's context length
        "tito-model": recipe["tito_model"],
        "sglang-reasoning-parser": reasoning_parser,
        "sglang-tool-call-parser": tool_call_parser,
        # Accepts the way Strands replays earlier turns; see message_matcher.py.
        "session-message-matcher": "email_miles.message_matcher.strands_message_matches",
        # AgentCore reaches the session servers on these ports (see "Network" in the README).
        "session-server-ip": "0.0.0.0",
        "session-server-port": SESSION_PORT,
        "session-server-workers": recipe["session_servers"],
        # Whole rollouts in flight per engine, tools and judging included. A rollout spends most of
        # its time outside the model, so the engines stay busy only with many more than Miles' usual.
        "sglang-server-concurrency": recipe["rollouts_per_gpu"],
        # Optimizer: Megatron would otherwise decay the learning rate and apply dropout of 0.1.
        "lr": recipe["learning_rate"],
        "lr-decay-style": "constant",
        "attention-dropout": 0,
        "hidden-dropout": 0,
        # Trainer memory: positions up to the session limit, activation recomputation, and a per-GPU
        # token budget for packing variable-length samples.
        "seq-length": recipe["max_seq_len"],
        "max-tokens-per-gpu": recipe["max_tokens_per_gpu"],
        "qkv-format": "thd",
        "recompute-granularity": "full",
        "recompute-method": "uniform",
        "recompute-num-layers": 1,
        # Evaluation and checkpoints.
        "eval-interval": recipe["eval_interval"],
        "n-samples-per-eval-prompt": recipe["eval_samples"],
        "save": run_dir / "checkpoints",
        "save-interval": recipe["save_interval"],
    }
    argv = [item for key, value in options.items() for item in (f"--{key}", str(value))]
    argv += ["--eval-prompt-data", "email", str(ROOT / recipe["eval_dataset"])]
    argv += [
        "--use-session-server",
        "--use-miles-router",  # forwards request bodies to the engines unchanged
        "--colocate",
        "--lora-base-cpu-backup",
        "--use-dynamic-batch-size",
        "--no-gradient-accumulation-fusion",
        # Take the policy ratio against the log-probs the rollout engine sampled with.
        "--use-rollout-logprobs",
        # Fast paths: FlashAttention-3 in SGLang, Transformer Engine's fused attention in the
        # trainer, and the Triton log-prob op. Do not add --recompute-loss-function: it turns off
        # that op's in-place backward, which doubles the logits memory (tokens x vocabulary).
        "--sglang-attention-backend", "fa3",
        "--attention-backend", "fused",
        "--log-probs-backend", "fused",
    ]  # fmt: skip
    if host := os.environ.get("EMAIL_GATEWAY_HOST"):
        argv += ["--session-server-external-host", host]
    if os.environ.get("WANDB_API_KEY"):
        argv += ["--use-wandb", "--wandb-project", "email_agent_miles", "--wandb-group", run_dir.name]
    return argv


def resume_args(run_dir):
    """Arguments that continue from the newest complete checkpoint in run_dir, if there is one."""
    checkpoint = latest_checkpoint(run_dir)
    if not checkpoint:
        return []
    update, adapter = checkpoint
    print(f"Resuming after update {update + 1}", flush=True)
    # --load restores the dataset position and update count; the adapter directory holds the LoRA
    # weights and optimizer state.
    load = ["--load", str(run_dir / "checkpoints"), "--lora-adapter-path", str(adapter)]
    return load + ["--use-checkpoint-opt-param-scheduler"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recipe", type=Path)
    parser.add_argument("--run", required=True, help="run name; outputs go to runs/<name>")
    parser.add_argument("--smoke", action="store_true", help="2 tasks x 4 rollouts, one update")
    parser.add_argument("--dry-run", action="store_true", help="check the arguments with Miles and print them")
    cli = parser.parse_args()

    os.environ.update({k: v for k, v in dotenv_values(ROOT / ".env").items() if v and k not in os.environ})
    recipe = json.loads(cli.recipe.read_text()) | (SMOKE if cli.smoke else {})
    run_dir = ROOT / "runs" / cli.run
    args = miles_args(recipe, model_path(recipe["hf_model"]), run_dir) + resume_args(run_dir)
    sys.argv = ["train.py", *shlex.split(load_model_args(recipe["miles_model"])), *args]

    if cli.dry_run:
        # Miles' parser asks for the GPU architecture; a dry run has no GPU, so assume H100.
        with patch("megatron.training.arguments.get_device_arch_version", return_value=9):
            parse_args()
        print(shlex.join(sys.argv))
        return
    if (run_dir / "COMPLETED").exists():
        sys.exit(f"{run_dir} already completed")
    if not os.environ.get("AGENT_RUNTIME_ARN") or not os.environ.get("EMAIL_GATEWAY_HOST"):
        sys.exit("Set AGENT_RUNTIME_ARN and EMAIL_GATEWAY_HOST in .env (see .env.example)")
    saved = run_dir / "recipe.json"
    if saved.exists() and json.loads(saved.read_text()) != recipe:
        sys.exit(f"{saved} differs from this recipe; use a new --run name")
    parsed = parse_args()
    run_dir.mkdir(parents=True, exist_ok=True)
    saved.write_text(json.dumps(recipe, indent=2) + "\n")
    os.environ["ROLLOUT_LOG_DIR"] = str(run_dir / "rollouts")
    os.environ["WANDB_DIR"] = str(run_dir)
    ray.init(address="local", namespace=cli.run, _temp_dir=f"/tmp/miles-ray-{os.getpid()}")
    try:
        asyncio.run(with_disposer(train, ArgvOrchestratorStartupInfo.create(parsed)))
        (run_dir / "COMPLETED").touch()
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
