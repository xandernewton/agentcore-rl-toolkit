#!/usr/bin/env bash
# Benchmark one served model through the deployed agent: SWE-Explore (848 tasks) and ContextBench Lite
# (500), then score both.
#   bash eval/run_benchmarks.sh OUT MODEL_DIR SERVED_NAME
# vLLM must serve MODEL_DIR as SERVED_NAME on :8399 (endpoint/serve_eval.sh), reachable from the runtime
# at this host's private IP; the agent calls it directly. Progress: OUT/STATUS. Scores: OUT/RESULTS.txt.
# Re-running resumes: finished stages are skipped and only failed tasks are retried.
set -euo pipefail
OUT="${1:?usage: run_benchmarks.sh OUT MODEL_DIR SERVED_NAME}" MODEL_DIR="${2:?}" NAME="${3:?}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
DATA="$ROOT/data/benchmarks"
mkdir -p "$OUT"
[[ ! -f "$OUT/SCORING_COMPLETED" ]] || { cat "$OUT/RESULTS.txt"; exit 0; }

# One pipeline per OUT. Take the lock before the ERR trap, so a second invocation does not mark the
# running one as failed.
exec 9>"$OUT/pipeline.lock"
flock -n 9 || { echo "FATAL: another run_benchmarks.sh is using $OUT" >&2; exit 1; }
trap 'echo failed >"$OUT/STATUS"' ERR
stage() { echo "$1" >"$OUT/STATUS"; }
die() { echo "FATAL: $*" >&2; stage failed; exit 1; }

[[ -f "$ROOT/.env" ]] || die "$ROOT/.env missing (written by cdk/deploy.sh)"
set -a; source "$ROOT/.env"; set +a

stage waiting_for_evaluation_server
health="http://${HOST:-127.0.0.1}:8399/health"
for _ in $(seq 360); do curl --max-time 5 -sf "$health" >/dev/null && break; sleep 5; done
curl --max-time 5 -sf "$health" >/dev/null || die "no vLLM at $health after 30 min"

# full_benchmark.py resumes (skips finished tasks, retries failed ones) and writes COMPLETED only once
# every task has finished without error; give it three attempts.
run_eval() {
  local name="$1" data="$2" concurrency="$3"; shift 3
  [[ ! -f "$OUT/$name/COMPLETED" ]] || return 0
  for _ in 1 2 3; do
    "$PY" -u "$ROOT/eval/full_benchmark.py" --data "$data" --out "$OUT/$name" \
      --concurrency "$concurrency" --model "$NAME" --tokenizer "$MODEL_DIR" "$@" && return 0
  done
  return 1
}

# Guard against a truncated or wrong benchmark file: the run must cover the full task count.
expect_tasks() {
  jq -e --argjson n "$2" '.completed == $n and .expected == $n and .errors == 0' "$1/summary.json" >/dev/null \
    || die "$1 is not $2 error-free tasks: $(cat "$1/summary.json")"
}

stage test_run_evaluation
run_eval test-run "$DATA/contextbench-lite.jsonl" 4 --limit 4

stage swe_explore_evaluation
run_eval swe-explore "$DATA/swe-explore.jsonl" 64
expect_tasks "$OUT/swe-explore" 848
stage scoring_swe_explore
"$PY" "$ROOT/eval/score_swe_explore.py" --evaluation "$OUT/swe-explore" --out "$OUT/line-scores" \
  | tee "$OUT/RESULTS.txt"

stage contextbench_evaluation
run_eval contextbench "$DATA/contextbench-lite.jsonl" 64
expect_tasks "$OUT/contextbench" 500
stage scoring_contextbench
"$PY" "$ROOT/eval/score_contextbench.py" "$OUT" | tee -a "$OUT/RESULTS.txt"  # writes SCORING_COMPLETED

stage completed
