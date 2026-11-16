# Email search with Miles

Reinforcement learning for email search, trained with [Miles](https://github.com/radixark/miles).
The agent answers questions about a user's inbox (the Enron corpus) by searching and reading
emails. This example trains Qwen3-8B on it with GRPO and a LoRA adapter.

The agent is a [Strands](https://strandsagents.com) agent running on AWS Bedrock AgentCore. Miles
runs the model and the trainer on one GPU host:

```
GPU host                                              AgentCore Runtime
┌────────────────────────────────────────┐            ┌──────────────────────────────┐
│ Miles trainer (Megatron, LoRA)         │   task     │ agent/rl_app.py              │
│ rollout hooks (email_miles/rollout.py) ├───────────►│   Strands agent              │
│                                        │◄───────────┤   search_inbox, read_email   │
│ session servers :9100-9103             │   reward   │   reward + Claude judge      │
│   └─► SGLang engines                   │◄───────────┤                              │
└────────────────────────────────────────┘ chat calls └──────────────────────────────┘
```

For each rollout Miles opens a *session*: an OpenAI-compatible URL that records the tokens and
log-probs of every model call and assembles the training sample. The toolkit's
`AgentCoreHttpSession` starts the agent on AgentCore with the task and that URL. The agent makes
ordinary chat-completion calls, scores itself and returns the reward, so it needs no tokenizer and
no knowledge of the trainer.

## Results

Accuracy on 128 held-out questions, two attempts each, graded by Claude Haiku 5.5:

| Qwen3-8B | Correct | Wrong | Declined | Out of turns | Opened a right email |
|---|---:|---:|---:|---:|---:|
| Base model | 43% | 25% | 20% | 7% | 32% |
| After 5 updates | 72% | 25% | 3% | 0% | 43% |
| After 10 updates | 83% | 15% | 2% | 0% | 75% |
| After 15 updates | 86% | 13% | 2% | 0% | 91% |
| After 20 updates | 86% | 13% | 2% | 0% | 95% |

Twenty updates take about an hour and a half on 8× H100, evaluations included. Accuracy stops
improving after about 15 updates; two evaluations of the same checkpoint differ by about 2 points,
and two runs by about 5 at the same update.

## Recipe

| | |
|---|---|
| Data | 1,024 training and 128 evaluation questions from [`corbt/enron_emails_sample_questions`](https://huggingface.co/datasets/corbt/enron_emails_sample_questions), over the [`corbt/enron-emails`](https://huggingface.co/datasets/corbt/enron-emails) corpus |
| Agent | Two tools (`search_inbox`, `read_email`), 10 turns, 20 results per search; the answer is the final message, ending with a `Sources:` line of message IDs |
| Reward | +1 for a correct answer, −1 for a wrong one, 0 for declining, −2 for running out of turns, plus small credit for finding, opening and citing the right emails ([`agent/reward.py`](agent/reward.py)) |
| Judge | Claude Haiku 5.5 on Amazon Bedrock, majority of up to three votes |
| Training | GRPO, LoRA rank 16, learning rate 5e-5, 24 questions × 16 rollouts per update, 20 updates ([`recipes/qwen3_8b.json`](recipes/qwen3_8b.json)) |
| Hardware | All 8 GPUs alternate between rollouts and training, so every update is on-policy |

## Requirements

- 8 NVIDIA GPUs with 80 GB each (tested on 8× H100)
- Docker with the NVIDIA container toolkit
- [uv](https://docs.astral.sh/uv/)
- An AWS account with Bedrock AgentCore and access to the judge model; see [Network](#network)

## Quickstart

Run every command from `examples/strands_email_agent`.

```bash
# 1. Build the training image (Miles v0.2.0 + this toolkit) and download the model
bash scripts/docker.sh build
uvx --from huggingface_hub hf download Qwen/Qwen3-8B

# 2. Write the training and evaluation data to data/
uv run --with datasets python -m email_miles.prepare_data

# 3. Deploy the agent into this host's VPC (the image build downloads the emails and builds
#    the database). Use your own subnets, security group and region.
uv tool install bedrock-agentcore-starter-toolkit
cd agent
agentcore configure --entrypoint rl_app.py --name strands_email_agent \
  --requirements-file pyproject.toml --deployment-type container --protocol HTTP \
  --vpc --subnets subnet-xxxxxxxxxxxxxxxxx --security-groups sg-xxxxxxxxxxxxxxxxx \
  --disable-memory --non-interactive
agentcore deploy --agent strands_email_agent --env AWS_REGION=<region>
agentcore status --agent strands_email_agent    # wait for the endpoint to be READY (a few minutes)
cd ..
cp .env.example .env    # set AGENT_RUNTIME_ARN (printed by the deploy) and EMAIL_GATEWAY_HOST

# 4. One-update test run (about 10 minutes)
bash scripts/docker.sh train recipes/qwen3_8b.json --run smoke --smoke

# 5. Train
bash scripts/docker.sh train recipes/qwen3_8b.json --run qwen3-8b
```

Outputs go to `runs/<name>`: checkpoints every 5 updates, and every rollout's transcript, reward
and metrics in `rollouts/{train,eval}.jsonl.gz`. If training stops, rerun the same command: it
continues from the newest checkpoint. Add `--dry-run` to check the arguments with Miles and print
the full command. Set `WANDB_API_KEY` in `.env` to log to W&B.

`agentcore deploy` builds the image in AWS CodeBuild, so your credentials need CodeBuild, ECR and
IAM role permissions; a GPU host's instance role often lacks them. Either deploy from a machine
that has them, or add `--local-build` to build here (on x86 that needs Docker's ARM64 emulation).

To redeploy after editing the agent, rerun the `agentcore deploy` command. `JUDGE_MODEL` (`--env`)
selects another Bedrock model or inference profile for the judge.

## Network

The agent calls the Miles session servers on the GPU host's private IP, so the runtime runs in the
host's VPC, with the subnets and security group given to `agentcore configure`:

- The host's security group must allow inbound TCP 9100-9103 (one port per session server in the
  recipe) from the runtime's security group.
- A runtime in a VPC has no public IP, so an internet gateway alone (a default VPC's public
  subnets) gives it no way out. Its subnets need either a NAT gateway, or VPC endpoints for
  `bedrock-runtime` (the judge), `ecr.api`, `ecr.dkr`, `logs` and S3, with security groups that
  allow HTTPS from the runtime's. Without a route to Bedrock the judge hangs and every rollout is
  lost.
- `AWS_REGION` is the region the judge is called in; set it to the runtime's region.
- `EMAIL_GATEWAY_HOST` in `.env` is the host's private IP (`hostname -I`).

## Using another model

Copy the recipe and change three keys, then check the tokenization test still passes
(`bash scripts/docker.sh test`, after adding the model to `tests/trainer/test_tito_template.py`):

| Key | Meaning |
|---|---|
| `hf_model` | The Hugging Face repository |
| `miles_model` | Miles' model arguments, a file name in its `scripts/models/` |
| `tito_model` | Miles' session-tokenizer family, which also fixes SGLang's tool-call parser |

Set `max_seq_len` to at most the model's context length: SGLang refuses to start otherwise.

## Things to know

- **Miles loads its own defaults.** The launcher sets only what this example needs
  ([`email_miles/launch.py`](email_miles/launch.py)); sampling, the optimizer and parallelism are
  Miles'.
- **Message text must be a plain string.** Strands sends text as a list of parts, which Qwen3's
  chat template renders as nothing. `TextOpenAIModel` in `agent/rl_app.py` flattens it. After
  changing the agent or the model, read a transcript in `runs/<name>/rollouts/`.
- **Miles does not stop the agent at the session's token limit**, so the agent stops itself
  (`Budget` in `agent/rl_app.py`), and that rollout is scored.
- **Running out of turns must score below a wrong answer.** Otherwise never answering is the
  safest policy, and training collapses into repeating a tool call until the turn cap.
- **A lost rollout is run again.** A dropped connection or a throttled AgentCore session affects
  about one rollout in a thousand. `email_miles/rollout.py` reruns it in a new session, up to three
  times, and stops training only if it is lost every time. AgentCore starts at most 25 new sessions
  per second per account by default; the hooks pace themselves at 15.
- **The judge needs room to reason.** Claude Haiku 5.5 thinks before answering harder comparisons;
  with a small output limit the reply is cut off empty. An answer the judge gives no verdict on is
  never scored as wrong: the rollout is rerun.
- **`rollouts_per_gpu` counts whole rollouts**, tools and judging included, not model requests. A
  rollout spends most of its time outside the model, so the engines need many in flight.

## Layout

| Path | Purpose |
|---|---|
| `agent/` | The AgentCore container: `rl_app.py` (agent and entrypoint), `email_db.py`, `reward.py`, `judge.py`; `build_db.py` builds the email database into the image |
| `email_miles/rollout.py` | The Miles hooks: send a task to the agent, rerun lost rollouts, read the reward |
| `email_miles/launch.py` | Recipe → Miles arguments, and resume |
| `email_miles/message_matcher.py` | Tells Miles which of Strands' rewrites of earlier turns are harmless |
| `recipes/` | Training recipes |
| `tests/` | `tests/agent` (`uv run --project agent --with pytest pytest tests/agent`) and `tests/trainer` (`bash scripts/docker.sh test`) |
