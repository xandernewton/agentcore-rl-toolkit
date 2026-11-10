#!/usr/bin/env bash
# Train a recipe, export + merge the last checkpoint, and benchmark it.
#   bash scripts/run_recipe.sh RECIPE EXP STEPS [launch args...]
# Results: evals/<EXP>/eval/RESULTS.txt. Re-running skips finished stages (training, export, merge, eval),
# and continues an interrupted training run from its last checkpoint (every 5 steps).
# The agent must be deployed with this recipe (bash cdk/deploy.sh -c recipe=RECIPE).
# It runs for hours in the foreground; to survive logout, start it with
#   setsid nohup bash scripts/run_recipe.sh ... >>logs/<EXP>-recipe.log 2>&1 </dev/null &
set -uo pipefail
RECIPE="$1" EXP="$2" STEPS="$3"; shift 3
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
BASE="${MODEL_DIR:-${HF_HOME:-$HOME/.cache/huggingface}/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a}"
STATE="${SKYRL_STATE_DIR:-$HOME/skyrl-state}"
P="$ROOT/evals/$EXP" ADAPTER="$ROOT/exports/$EXP-adapter" MERGED="$ROOT/exports/$EXP"
cd "$ROOT"
mkdir -p "$P"

# "<step> <tinker path>" of the latest checkpoint of runs/<EXP> or an interrupted attempt moved aside
# (runs/<EXP>-to-step-*), the step counted from the start of training (a resumed run numbers its
# steps from its resume point, recorded in swegrep.json).
latest_checkpoint() {
  local best="0 -" run offset line
  for run in "runs/$EXP" "runs/$EXP"-to-step-*; do
    [[ -f "$run/checkpoint.json" ]] || continue
    offset=$(jq -r '.resume_step // 0' "$run/swegrep.json" 2>/dev/null || echo 0)
    line=$(jq -r --argjson offset "$offset" '"\(.step + $offset) \(.path)"' "$run/checkpoint.json")
    (( ${line%% *} > ${best%% *} )) && best="$line"
  done
  echo "$best"
}

# Whether launch.sh's trainer for EXP is still running: a zombie has an empty cmdline, a reused pid another one.
training() { local pid; pid=$(cat "logs/$EXP.pid" 2>/dev/null) && [[ "$(tr '\0' ' ' 2>/dev/null <"/proc/$pid/cmdline")" == *swegrep_train.py* ]]; }
container_running() { [[ -n "$(docker ps -q --filter "name=^$1\$")" ]]; }

gpus_free() { for _ in $(seq 1 120); do [[ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ]] && return; sleep 5; done; echo "FATAL: GPUs still busy after 10 min" >&2; return 1; }

# Checked before the ERR trap, so they leave a running pipeline's STATUS alone.
[[ "$(jq .steps "$RECIPE")" == "$STEPS" ]] || { echo "FATAL: $RECIPE trains $(jq .steps "$RECIPE") steps, not $STEPS" >&2; exit 1; }
! training || { echo "FATAL: $EXP is still training (pid $(cat "logs/$EXP.pid"))" >&2; exit 1; }

set -eE
trap 'echo failed >"$P/STATUS"' ERR
read -r DONE CHECKPOINT <<<"$(latest_checkpoint)"
if (( DONE < STEPS )); then
  # Interrupted: keep its outputs, and continue from the last checkpoint if there is one.
  [[ ! -e "runs/$EXP" ]] || mv "runs/$EXP" "runs/$EXP-to-step-$DONE-$(date -u +%H%M)"
  resume=()
  (( DONE == 0 )) || resume=(--resume-from "$CHECKPOINT" --resume-step "$DONE")
  # After an interruption the endpoint is usually still up and holding the GPUs: reuse it.
  bash scripts/endpoint.sh running || gpus_free
  MODEL_DIR="$BASE" bash scripts/endpoint.sh start  # no-op if already up; fails if up on another port
  echo training >"$P/STATUS"
  EXP="$EXP" bash scripts/launch.sh "$RECIPE" "${resume[@]}" "$@"
  while training; do sleep 60; done
  read -r DONE CHECKPOINT <<<"$(latest_checkpoint)"
  (( DONE == STEPS )) || { echo "training stopped at step $DONE of $STEPS; rerun to resume" >"$P/STATUS"; exit 1; }
fi
if [[ ! -f "$ADAPTER/SOURCE.txt" ]]; then
  echo exporting >"$P/STATUS"
  MODEL_DIR="$BASE" bash scripts/endpoint.sh start  # no-op if already up; fails if up on another port
  "$PY" -u scripts/export_adapter.py "$CHECKPOINT" "$ADAPTER" --state-dir "$STATE"
fi
[[ "$(cat "$ADAPTER/SOURCE.txt")" == "$CHECKPOINT" ]]
bash scripts/endpoint.sh stop
if [[ ! -f "$MERGED/MERGED_FROM.json" ]]; then
  echo merging >"$P/STATUS"
  OMP_NUM_THREADS=8 "$PY" -u scripts/merge_adapter.py "$ADAPTER" "$MERGED" --base "$BASE"
fi
U="swegrep-eval-$EXP"
if [[ ! -f "$P/eval/SCORING_COMPLETED" ]]; then
  # A rerun after a failed eval reuses the eval server still holding the GPUs.
  if ! container_running "$U"; then
    gpus_free
    MODEL_DIR="$MERGED" SKYRL_CONTAINER="$U" SKYRL_STATE_DIR="$STATE" DETACH_LOG="$P/server.log" \
      bash endpoint/docker.sh bash serve_eval.sh "$EXP"
  fi
  echo eval >"$P/STATUS"
  bash eval/run_benchmarks.sh "$P/eval" "$MERGED" "$EXP"
fi
docker stop -t 60 "$U" >/dev/null 2>&1 || true
cat "$P/eval/RESULTS.txt"
echo completed >"$P/STATUS"
