import json

import pytest
from email_miles.resume import latest_checkpoint


def write_checkpoint(root, iteration):
    """The layout Miles v0.2.0 writes: iter_N/adapter -> .adapter.<commit>, plus the rollout cursor."""
    step = root / "checkpoints" / f"iter_{iteration:07d}"
    hidden = step / ".adapter.abc123"
    (hidden / "adapter_state").mkdir(parents=True)
    (hidden / "adapter_state" / ".metadata").write_bytes(b"x")
    (hidden / "adapter_model.safetensors").write_bytes(b"x")
    (hidden / "training_state.json").write_text(json.dumps({"iteration": iteration}))
    (step / "adapter").symlink_to(hidden.name)
    cursor = root / "checkpoints/rollout" / str(iteration) / "data_source"
    cursor.mkdir(parents=True)
    (cursor / "state.pt").write_bytes(b"x")
    return step / "adapter"


def test_fresh_run(tmp_path):
    assert latest_checkpoint(tmp_path) is None


def test_newest_complete_checkpoint_wins(tmp_path):
    write_checkpoint(tmp_path, 3)
    newest = write_checkpoint(tmp_path, 4)
    assert latest_checkpoint(tmp_path) == (4, newest)


def test_unpublished_adapter_is_skipped(tmp_path):
    """Miles publishes the adapter symlink last; a save interrupted before that is not resumable."""
    previous = write_checkpoint(tmp_path, 3)
    (write_checkpoint(tmp_path, 4)).unlink()
    assert latest_checkpoint(tmp_path) == (3, previous)


def test_missing_dataset_cursor_falls_back(tmp_path):
    previous = write_checkpoint(tmp_path, 0)
    write_checkpoint(tmp_path, 1)
    (tmp_path / "checkpoints/rollout/1/data_source/state.pt").unlink()
    assert latest_checkpoint(tmp_path) == (0, previous)


def test_never_silently_restarts(tmp_path):
    write_checkpoint(tmp_path, 0).unlink()
    with pytest.raises(RuntimeError, match="refusing to reset"):
        latest_checkpoint(tmp_path)
