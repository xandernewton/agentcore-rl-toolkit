# SWE-Grep agent (AgentCore container)

The rollout body deployed to AWS Bedrock AgentCore Runtime. Each task becomes one AgentCore session:
the trainer (or the benchmark harness) sends the task row as the payload, the agent explores a
read-only checkout of `repo@base_commit` with three tools, cites the relevant code regions, and
returns the reward with its metrics.

## Files

- **`rl_app.py`**: the entrypoint (`AgentCoreRuntimeApp` + `@app.entrypoint`; the episode itself is `run_episode`, which `eval/claude_bench.py` also calls). Reads the payload
  (`repo`, `base_commit`, `problem_statement`, `gold_files`, `gold_lines`) and the
  injected `_config` (`base_url` of the rollout gateway in training or of the benchmark vLLM server,
  `model_id`, the rollout's `api_key`), runs a Strands `Agent` on the issue text and returns
  `{"reward": float, "metrics": {...}, "messages": [...]}`; `messages` are its assistant turns (text and
  tool calls), which the benchmark harness re-parses. Transient LLM errors are retried through Strands'
  `retry_strategy`. Sampling, output-token limits and thinking mode come from the trainer through the
  gateway, or from `../endpoint/serve_eval.sh` for benchmarks.
- **`local_test.py`**: a local test run with Claude on Amazon Bedrock as the policy (no gateway or S3):
  dummy repositories, the real tools, hook and reward.
- **`Dockerfile`**: the ARM64 image the CDK stack builds; dependencies pinned in `pyproject.toml`.

**`rollout/`**: what runs inside a rollout.

- **`agent_prompts.py`**: the system prompt and the final-turn nudges.
- **`repo_tools.py`**: the tools, built per rollout over the checkout: `grep(pattern, path)` (the
  system `grep`), `glob(pattern)` and `read(path, start_line, end_line)` (line-numbered).
- **`turn_cap.py`**: `TurnCapHook`, which allows `SWEGREP_MAX_TURNS - 1` search turns or the context
  budget, then asks for the final answer and blocks further tool calls (at most two final model calls).
- **`custom_openai_model.py`**: `CustomOpenAIModel`, the model provider. A Strands `OpenAIModel` that
  replays each assistant turn as generated: it sends the turn's reasoning back as `reasoning_content`
  (stock Strands drops it) and restores tool names Strands rewrote to `INVALID_TOOL_NAME`. The gateway
  then sees the history it served, and the model sees its earlier reasoning in training and in
  benchmarks.
- **`repo_store.py`**: downloads `s3://$SWEGREP_CORPUS_BUCKET/repos/{owner}__{name}__{commit}.tar.gz`
  (built by `../scripts/prepare_corpus.py`; the runtime has no internet) into a per-container cache.

**`scoring/`**: answer parsing and scoring, also imported by `../eval/` and `../data/` so
training and evaluation parse identically.

- **`regions.py`**: parses `path:start-end` entries, validates them against the checkout, and
  recognises test paths (used to filter the training labels).
- **`retrieval.py`**: finds the answer in the transcript (`extract_answer`), path normalisation and
  file-level scoring.
- **`reward.py`**: the reward, line-level region scoring (`score_regions`) and the metrics.
- **`search_metrics.py`**: how the agent searched: credited paths, search quality and idle turns.

## Answer format

The final assistant message lists ranked regions, one per line:

```text
<answer>
src/router.py:120-165
tests/test_router.py:40-68
</answer>
```

Each entry is a repository-relative path and a 1-indexed inclusive range. With a checkout root,
`extract_answer` rejects missing or outside files and empty spans and clips ends to EOF; malformed
entries are recorded as `invalid_regions`, never replaced by guesses. `SWEGREP_MAX_ANSWER_REGIONS=0`
(as trained) puts no limit on the number of regions.

## Reward

```
infra exception                  ->  0.0   (a -1 would shift every GRPO sibling's advantage)
rollout exception, no answer     -> -1.0
no or empty answer               -> -0.3
otherwise                        ->  F-beta(line precision, line recall, SWEGREP_BETA)   (beta 2)
                                     - invalid-region penalty (0.05 each, capped at 0.3)
```

Precision and recall are over unique cited lines against the task's `gold_lines`
(`{path: [[start, end], ...]}`); overlapping spans count once. The training labels are each reference
patch hunk widened by 20 lines (see `../data/build_training_data.py`). Metrics include file
`precision`/`recall`, `region_precision`/`region_recall`, `num_predicted_lines`, `submitted`,
`num_turns` and the exception flags.

## Settings that define the published agent

Tool limits and budgets are runtime environment variables whose defaults are the published values:
16 turns, a 44k-token context budget, 8,000-character tool results and 800-line reads.

`SWEGREP_FINAL_USER_TURN` controls how the agent asks for the answer once the search budget runs out
(`rollout/turn_cap.py`). When a tool call uses up the budget, the "submit your answer now" nudge is
appended to that tool's result. What happens before the next model call depends on the setting:

| Value | Before the answer turn | Effect |
|---|---|---|
| `always` (default) | Also appends a `[FINAL ANSWER PHASE]` user message, even though the nudge is already in the tool result. | Matches the agent that trained and evaluated the published model. |
| `auto` | Skips that user message when the nudge is already in the tool result. It is still added if the budget was hit some other way. | Keeps the earlier reasoning in the rendered history (see below). |

Qwen3.5's chat template removes the reasoning of every assistant turn that comes before the latest user
message. Under `always`, the extra user turn therefore drops the search turns' reasoning from the final
prompt, and the captured trajectory splits at that point. Keep `always` to reproduce the published
model; `auto` is an option for new runs.

The prompt and tool code live in the image. The reward and answer settings in a recipe's
`swegrep.runtime_env` are applied by `../cdk/deploy.sh` (`-c recipe=...`): changing them means a
redeploy, and `../swegrep_train.py` refuses to start if the runtime's settings differ from the recipe's.
