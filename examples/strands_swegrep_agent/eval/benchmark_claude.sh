#!/usr/bin/env bash
# Benchmark Claude on Amazon Bedrock as the agent's policy (eval/claude_bench.py), then score it:
#   bash eval/benchmark_claude.sh [MODEL]    (default global.anthropic.claude-sonnet-5-5)
# Scores: evals/<model>/eval/RESULTS.txt, e.g. evals/claude-sonnet-5-5/eval. Re-running resumes and
# retries failed tasks.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
MODEL="${1:-global.anthropic.claude-sonnet-5-5}"
OUT="evals/${MODEL##*.}/eval"
set -a; source .env; set +a  # the corpus bucket and checkout cache, for the scorers
run() { uv run --frozen --extra claude "$@"; }  # adds the anthropic SDK without removing other extras

bench() {  # claude_bench.py writes COMPLETED once every task has finished without error
  for _ in 1 2 3; do
    [[ -f "$OUT/$1/COMPLETED" ]] && return 0
    run eval/claude_bench.py --model "$MODEL" --data "data/benchmarks/$2.jsonl" --out "$OUT/$1" || true
  done
  [[ -f "$OUT/$1/COMPLETED" ]] || { echo "FATAL: $OUT/$1 has failed tasks; rerun to retry them" >&2; exit 1; }
}
bench swe-explore swe-explore
bench contextbench contextbench-lite
run eval/score_swe_explore.py --evaluation "$OUT/swe-explore" --out "$OUT/line-scores" | tee "$OUT/RESULTS.txt"
run eval/score_contextbench.py "$OUT" | tee -a "$OUT/RESULTS.txt"
