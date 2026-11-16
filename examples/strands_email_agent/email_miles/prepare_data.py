"""Write data/*.jsonl (Miles prompt data) from the ART-E scenarios on Hugging Face.

  uv run --with datasets python -m email_miles.prepare_data

Each row is {"prompt": [user question], "metadata": {"payload": <task>}, "label": ""}; the payload is
what agent/rl_app.py receives. Rows are a seeded sample, so every machine gets the same tasks.
"""

import hashlib
import json
import random
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = "corbt/enron_emails_sample_questions"
SIZES = {"train": 1024, "test": 128}  # tasks sampled from each split
FIELDS = ("id", "question", "answer", "message_ids", "inbox_address", "query_date")


def sample_split(split, count):
    rows = list(load_dataset(SCENARIOS, split=split))
    random.Random(42).shuffle(rows)
    payloads = []
    for row in rows[:count]:
        payload = {key: row[key] for key in FIELDS}
        payload["message_ids"] = [str(m) for m in payload["message_ids"]]
        assert all(payload[key] not in (None, "", []) for key in FIELDS), row["id"]
        payloads.append(payload)
    assert len(payloads) == count == len({p["id"] for p in payloads})
    return payloads


def write_split(name, payloads):
    rows = [
        {
            "prompt": [{"role": "user", "content": p["question"]}],
            "metadata": {"payload": p},
            "label": "",
        }
        for p in payloads
    ]
    path = ROOT / "data" / f"{name}.jsonl"
    path.parent.mkdir(exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    return {"rows": len(rows), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    train, val = (sample_split(split, count) for split, count in SIZES.items())
    splits = {"train": train, "val": val, "smoke_train": train[:2], "smoke_val": val[:2]}
    manifest = {name: write_split(name, payloads) for name, payloads in splits.items()}
    (ROOT / "data/manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
