#!/usr/bin/env python3
"""Build enron_emails.db (SQLite + FTS5, ~650 MB) from the Hugging Face dataset `corbt/enron-emails`.

The agent's Dockerfile runs this, so the image carries the database.
"""

import sqlite3
from datetime import datetime

from datasets import Features, Sequence, Value, load_dataset
from email_db import DB_PATH

SCHEMA = """
CREATE TABLE emails (
    id INTEGER PRIMARY KEY AUTOINCREMENT, message_id TEXT UNIQUE, subject TEXT, from_address TEXT,
    date TEXT, body TEXT, file_name TEXT
);
CREATE TABLE recipients (email_id TEXT, recipient_address TEXT, recipient_type TEXT);
"""
INDEXES = """
CREATE INDEX idx_emails_from ON emails(from_address);
CREATE INDEX idx_emails_date ON emails(date);
CREATE INDEX idx_recipients_address_email ON recipients(recipient_address, email_id);
CREATE VIRTUAL TABLE emails_fts USING fts5(subject, body, content='emails', content_rowid='id');
INSERT INTO emails_fts(emails_fts) VALUES('rebuild');
"""
FEATURES = Features(
    {
        "message_id": Value("string"),
        "subject": Value("string"),
        "from": Value("string"),
        "to": Sequence(Value("string")),
        "cc": Sequence(Value("string")),
        "bcc": Sequence(Value("string")),
        "date": Value("timestamp[us]"),
        "body": Value("string"),
        "file_name": Value("string"),
    }
)


def main():
    if DB_PATH.exists():
        print(f"{DB_PATH} exists; delete it to rebuild")
        return
    connection = sqlite3.connect(DB_PATH)
    connection.executescript(SCHEMA)
    seen = set()
    for row in load_dataset("corbt/enron-emails", features=FEATURES, split="train"):
        recipients = {kind: [a for a in row[kind] if a] for kind in ("to", "cc", "bcc")}
        # Skip very long emails and mass mailings, and keep one copy of each duplicate.
        if len(row["body"] or "") > 5000 or sum(map(len, recipients.values())) > 30:
            continue
        key = (row["subject"], row["body"], row["from"])
        if key in seen:
            continue
        seen.add(key)
        date = row["date"].strftime("%Y-%m-%d %H:%M:%S") if isinstance(row["date"], datetime) else str(row["date"])
        connection.execute(
            "INSERT INTO emails (message_id, subject, from_address, date, body, file_name) VALUES (?, ?, ?, ?, ?, ?)",
            (row["message_id"], row["subject"], row["from"], date, row["body"], row["file_name"]),
        )
        connection.executemany(
            "INSERT INTO recipients VALUES (?, ?, ?)",
            [(row["message_id"], address, kind) for kind, addresses in recipients.items() for address in addresses],
        )
    connection.executescript(INDEXES)
    connection.commit()
    connection.close()
    print(f"Built {DB_PATH}: {len(seen)} emails")


if __name__ == "__main__":
    main()
