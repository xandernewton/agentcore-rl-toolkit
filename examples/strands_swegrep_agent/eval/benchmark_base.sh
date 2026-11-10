#!/usr/bin/env bash
# Benchmark the base model (Qwen3.5-9B at the pinned revision, downloaded by scripts/endpoint.sh setup):
#   bash eval/benchmark_base.sh
# Serves it on :8399, runs eval/run_benchmarks.sh and stops the server. Scores: evals/base/eval/RESULTS.txt.
# Re-running resumes; after a failure the eval server stays up so the rerun reuses it.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
MODEL_DIR="$(source endpoint/environment.sh && echo "$MODEL_DIR")"
OUT=evals/base/eval CONTAINER=swegrep-eval-base
mkdir -p "$OUT" logs

if [[ -z "$(docker ps -q --filter "name=^$CONTAINER\$")" ]]; then
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
  ((used < 1000)) || { echo "FATAL: GPUs busy; stop the training endpoint first (bash scripts/endpoint.sh stop)" >&2; exit 1; }
  MODEL_DIR="$MODEL_DIR" SKYRL_CONTAINER="$CONTAINER" DETACH_LOG=logs/eval-base.log \
    bash endpoint/docker.sh bash serve_eval.sh base
fi
bash eval/run_benchmarks.sh "$OUT" "$MODEL_DIR" base
docker stop -t 60 "$CONTAINER" >/dev/null
