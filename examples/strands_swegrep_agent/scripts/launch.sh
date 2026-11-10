#!/usr/bin/env bash
# Start a training run in the background, plus a watcher that writes runs/<EXP>/REPORT.md.
#   EXP=<name> bash scripts/launch.sh [recipe]
# Same arguments as swegrep_train.py, which also runs in the foreground on its own.
# Both keep running after you log out. Logs: logs/<EXP>.log   PID: logs/<EXP>.pid
# Stop: kill $(cat logs/<EXP>.pid)
set -euo pipefail
cd "$(dirname "$0")/.."
: "${EXP:?set EXP (the run name: runs/EXP and the W&B run)}"
export EXP
LOG="$PWD/logs/$EXP.log"
PIDFILE="$PWD/logs/$EXP.pid"
mkdir -p logs
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "FATAL: $EXP is already training (pid $(cat "$PIDFILE"))" >&2
  exit 1
fi

# setsid + nohup: own session, so closing the terminal or logging out does not stop it.
setsid nohup .venv/bin/python -u swegrep_train.py "$@" >>"$LOG" 2>&1 </dev/null &
pid=$!
echo "$pid" >"$PIDFILE"
setsid nohup .venv/bin/python -u scripts/watch.py "runs/$EXP" --pid "$pid" \
  >>"logs/$EXP-watch.log" 2>&1 </dev/null &
echo "started $EXP (pid $pid). Log: $LOG  Report: runs/$EXP/REPORT.md  Stop: kill $pid"
