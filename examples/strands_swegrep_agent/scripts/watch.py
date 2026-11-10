"""Background watcher for a SWEgrep Tinker run: rewrites runs/<EXP>/REPORT.md every minute.

  .venv/bin/python scripts/watch.py runs/<EXP> --pid <trainer pid>

Reads metrics.jsonl, evaluation-NNNN.json and rollouts.jsonl (written by the toolkit's loop and
swegrep_train.py), tabulates training and held-out metrics per step, and exits after a final report
once the trainer process has exited.
"""

import argparse
import json
import time
from pathlib import Path

TRAIN_COLS = [
    ("step", "step", "{:.0f}"),
    ("reward", "reward/mean", "{:.3f}"),
    ("region F1", "swegrep/region_f1", "{:.3f}"),
    ("region P", "swegrep/region_precision", "{:.3f}"),
    ("region R", "swegrep/region_recall", "{:.3f}"),
    ("F-beta", "swegrep/f_beta", "{:.3f}"),
    ("precision", "swegrep/precision", "{:.3f}"),
    ("recall", "swegrep/recall", "{:.3f}"),
    ("submitted", "swegrep/submitted", "{:.2f}"),
    ("n_pred", "swegrep/num_predicted", "{:.1f}"),
    ("turns", "swegrep/num_turns", "{:.1f}"),
    ("active grp", "train/active_groups", "{:.0f}"),
    ("datums", "train/datums", "{:.0f}"),
    ("failed", "rollout/failed_episodes", "{:.0f}"),
    ("trunc", "rollout/truncated_episodes", "{:.0f}"),
    ("updates", "train/optimizer_steps", "{:.0f}"),
    ("min", "time/batch_seconds", "{:.1f}"),
]
# Held-out evaluation (evaluation-NNNN.json), temperature 1.0.
EVAL_COLS = [
    ("step", "step", "{:.0f}"),
    ("reward", "eval/reward_mean", "{:.3f}"),
    ("line F1", "eval/swegrep/region_f1", "{:.3f}"),
    ("line P", "eval/swegrep/region_precision", "{:.3f}"),
    ("line R", "eval/swegrep/region_recall", "{:.3f}"),
    ("cited lines", "eval/swegrep/num_predicted_lines", "{:.0f}"),
    ("file recall", "eval/swegrep/recall", "{:.3f}"),
    ("submitted", "eval/swegrep/submitted", "{:.2f}"),
    ("failed", "eval/failed_episodes", "{:.0f}"),
]


def table(title: str, cols, rows) -> list[str]:
    lines = [
        "",
        f"## {title}",
        "",
        "| " + " | ".join(c[0] for c in cols) + " |",
        "|" + "---|" * len(cols),
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(key), spec) for _, key, spec in cols) + " |")
    return lines


def fmt(value, spec):
    if value is None:
        return "-"
    try:
        return spec.format(value)
    except (TypeError, ValueError):
        return str(value)


def trainer_running(pid: int) -> bool:
    """Whether pid is still swegrep_train.py: a zombie has an empty cmdline, a reused pid another one."""
    try:
        return b"swegrep_train.py" in Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return False


def report(run: Path, pid: int, state: str) -> str:
    metadata = run / "swegrep.json"
    # A resumed run counts steps from its resume point (swegrep_train.py --resume-step).
    offset = json.loads(metadata.read_text()).get("resume_step", 0) if metadata.exists() else 0
    lines = [
        f"# {run.name}",
        "",
        f"Trainer pid {pid}: **{state}** (updated {time.strftime('%Y-%m-%d %H:%M:%S')})",
    ]
    metrics = (
        [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
        if (run / "metrics.jsonl").exists()
        else []
    )
    for row in metrics:
        row["step"] += offset  # local step -> global step of a resumed run
        if "time/batch_seconds" in row:
            row["time/batch_seconds"] /= 60
    lines += table("Training", TRAIN_COLS, metrics)
    evals = [json.loads(path.read_text()) for path in sorted(run.glob("evaluation-[0-9]*.json"))]
    for row in evals:
        row["step"] += offset
    if evals:
        lines += table("Evaluation", EVAL_COLS, evals)
    rollouts = run / "rollouts.jsonl"
    if rollouts.exists():
        n = errors = 0
        last = None
        with rollouts.open() as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:  # the trainer is still writing this line
                    break
                n += 1
                errors += row.get("error") is not None
                last = row
        lines += [
            "",
            f"Rollouts logged: {n} ({errors} with errors); last phase: {last['phase']}",
        ]
    checkpoint = run / "checkpoint.json"
    if checkpoint.exists():
        lines += [f"Latest checkpoint: `{json.loads(checkpoint.read_text())['path']}`"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--pid", type=int, required=True, help="the swegrep_train.py process")
    parser.add_argument("--interval", type=int, default=60)
    args = parser.parse_args()
    while True:
        state = "running" if trainer_running(args.pid) else "exited"
        if args.run.exists():
            (args.run / "REPORT.md").write_text(report(args.run, args.pid, state))
        if state == "exited":
            print(f"trainer pid {args.pid} exited; final report written", flush=True)
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
