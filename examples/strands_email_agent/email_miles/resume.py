"""Find the newest checkpoint Miles can resume from.

Miles (v0.2.0) saves LoRA runs as <save>/iter_N/adapter: a symlink published only after the adapter
directory is complete, holding the adapter, optimizer and scheduler state (loadable under any GPU
layout). The rollout side (dataset cursor) is saved separately under <save>/rollout/N.
"""

import json
from pathlib import Path


def latest_checkpoint(run_dir: Path):
    """(iteration, adapter_dir) of the newest complete checkpoint; None for a fresh run."""
    root = run_dir / "checkpoints"
    candidates = sorted(root.glob("iter_[0-9]*"), reverse=True)
    for candidate in candidates:
        iteration = int(candidate.name.removeprefix("iter_"))
        adapter = candidate / "adapter"
        if problem := _problem(root, adapter, iteration):
            print(f"Skipping incomplete checkpoint {candidate}: {problem}", flush=True)
        else:
            return iteration, adapter
    if candidates:
        raise RuntimeError("Checkpoints exist but none is resumable; refusing to reset training")
    return None


def _problem(root, adapter, iteration):
    """Why this checkpoint cannot be resumed, or None."""
    required = [
        adapter / "adapter_model.safetensors",
        adapter / "adapter_state" / ".metadata",
        root / "rollout" / str(iteration) / "data_source" / "state.pt",
    ]
    for path in required:
        if not path.is_file():
            return f"missing {path.relative_to(root)}"
    try:
        state = json.loads((adapter / "training_state.json").read_text())
    except (OSError, ValueError) as exc:
        return f"unreadable training_state.json: {exc}"
    if state.get("iteration") != iteration:
        return f"training_state.json is for iteration {state.get('iteration')}"
    return None
