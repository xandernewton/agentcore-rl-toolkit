"""Read-only search and lookup over the Enron email database (SQLite + FTS5, built by build_db.py)."""

import os
import re
import sqlite3
import threading
from pathlib import Path

DB_PATH = Path(os.environ.get("EMAIL_DB_PATH", Path(__file__).parent / "enron_emails.db"))
SNIPPET_TOKENS = 64  # the most FTS5's snippet() returns

_local = threading.local()


def _cursor():
    """One connection per thread: SQLite cursors are not safe to share."""
    if not hasattr(_local, "connection"):
        _local.connection = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    return _local.connection.cursor()


def _fts_query(keywords, exact_phrases):
    """An OR of the keywords' words, plus each keyword as an exact phrase when asked."""
    terms = []
    for keyword in keywords:
        phrase = str(keyword).replace('"', " ").strip()
        if phrase and exact_phrases:
            terms.append(f'"{phrase}"')
        for word in re.findall(r"[A-Za-z0-9]+(?:[./:_-][A-Za-z0-9]+)*", phrase.lower()):
            if len(word) > 1:
                terms.append(f'"{word}"')
    if not terms:
        raise ValueError("No usable keywords.")
    return " OR ".join(dict.fromkeys(terms))


def search_emails(inbox, keywords, from_address=None, to_address=None, sent_before=None, max_results=20):
    """Emails in `inbox` matching any keyword, best match first.

    Falls back to dropping the sender/recipient filters, then the exact-phrase terms, when nothing matches.
    """

    def run(fts_query, sender, recipient):
        where = [
            "fts.emails_fts MATCH ?",
            "(e.from_address = ? OR EXISTS (SELECT 1 FROM recipients r"
            " WHERE r.recipient_address = ? AND r.email_id = e.message_id))",
        ]
        params = [fts_query, inbox, inbox]
        if sender:
            where.append("e.from_address = ?")
            params.append(sender)
        if recipient:
            where.append(
                "EXISTS (SELECT 1 FROM recipients r WHERE r.recipient_address = ? AND r.email_id = e.message_id)"
            )
            params.append(recipient)
        if sent_before:
            where.append("e.date < ?")
            params.append(f"{sent_before} 00:00:00")
        rows = _cursor().execute(
            f"""
            SELECT e.message_id, snippet(emails_fts, -1, '<b>', '</b>', ' ... ', {SNIPPET_TOKENS}),
                   e.subject, e.from_address, e.date
            FROM emails e JOIN emails_fts fts ON e.id = fts.rowid
            WHERE {" AND ".join(where)}
            ORDER BY bm25(emails_fts), e.date DESC
            LIMIT ?
            """,
            [*params, max_results],
        )
        columns = ("message_id", "snippet", "subject", "from_address", "date")
        return [dict(zip(columns, row, strict=True)) for row in rows]

    query = _fts_query(keywords, exact_phrases=True)
    results = run(query, from_address, to_address)
    if not results and (from_address or to_address):
        results = run(query, None, None)
    return results or run(_fts_query(keywords, exact_phrases=False), None, None)


def read_email(message_id):
    """The email with this message_id and its recipients, or None."""
    cursor = _cursor()
    row = cursor.execute(
        "SELECT message_id, date, subject, from_address, body, file_name FROM emails WHERE message_id = ?",
        (message_id,),
    ).fetchone()
    if not row:
        return None
    recipients = {"to": [], "cc": [], "bcc": []}
    for address, kind in cursor.execute(
        "SELECT recipient_address, recipient_type FROM recipients WHERE email_id = ?", (message_id,)
    ):
        recipients[kind.lower()].append(address)
    message_id, date, subject, from_address, body, file_name = row
    return {
        "message_id": message_id,
        "date": date,
        "subject": subject,
        "from_address": from_address,
        "to_addresses": recipients["to"],
        "cc_addresses": recipients["cc"],
        "bcc_addresses": recipients["bcc"],
        "body": body,
        "file_name": file_name,
    }
