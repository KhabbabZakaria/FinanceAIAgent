"""A durable record of what was asked and what came back.

The app is expected to run continuously, so this accumulates for as long as it
is up. A question five months from now can therefore see what you said today.

Retrieval is keyed on tickers, not similarity. Asking about BTC-USD pulls the
turns that mentioned BTC-USD, however old they are, plus the last few turns for
immediate context. That is a SELECT, so it is exact, cheap and testable --
embeddings would make an exact lookup fuzzy for no gain.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
import json
import logging
import sqlite3

logger = logging.getLogger("financeai.conversations")

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    TEXT NOT NULL,
    asked_at   DATETIME NOT NULL,
    question   TEXT NOT NULL,
    intent     TEXT,
    tickers    TEXT,
    answer     TEXT,
    findings   TEXT,
    provider   TEXT,
    latency_ms INTEGER
)
"""
INDEX = "CREATE INDEX IF NOT EXISTS idx_conv_user_time ON conversations (user_id, asked_at DESC)"


@dataclass
class Turn:
    asked_at: datetime
    question: str
    intent: Optional[str]
    tickers: List[str]
    answer: Optional[str]
    findings: List[str]

    def age_days(self, now: datetime = None) -> int:
        return ((now or datetime.now()) - self.asked_at).days

    def summarise(self, now: datetime = None) -> str:
        """One compact line. Raw transcript is noise; a small model garbles it."""
        when = self.age_days(now)
        ago = "today" if when == 0 else f"{when}d ago"
        bits = [f"[{ago}] asked: {self.question.strip()[:120]}"]
        if self.intent:
            bits.append(f"(intent {self.intent})")
        if self.findings:
            bits.append(f"flagged: {', '.join(self.findings)}")
        return " ".join(bits)


def ensure_schema(db_path: str) -> None:
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute(SCHEMA)
    conn.execute(INDEX)
    conn.commit()
    conn.close()


def record(db_path: str, user_id: str, question: str, intent: str = None,
           tickers: List[str] = None, answer: str = None,
           findings: List[str] = None, provider: str = None,
           latency_ms: int = None) -> None:
    """Append one turn. Never raises: losing a log line must not lose the answer."""
    try:
        ensure_schema(db_path)
        conn = sqlite3.connect(db_path, timeout=30)
        conn.execute(
            "INSERT INTO conversations (user_id, asked_at, question, intent, tickers,"
            " answer, findings, provider, latency_ms) VALUES (?,?,?,?,?,?,?,?,?)",
            (user_id, datetime.now().isoformat(timespec="seconds"), question, intent,
             json.dumps([t.upper() for t in tickers or []]),
             (answer or "")[:4000], json.dumps(findings or []), provider, latency_ms))
        conn.commit()
        conn.close()
    except Exception as e:                       # noqa: BLE001 - deliberate
        logger.warning("Could not record turn: %s", e)


def _rows_to_turns(rows) -> List[Turn]:
    out = []
    for asked_at, question, intent, tickers, answer, findings in rows:
        out.append(Turn(datetime.fromisoformat(asked_at), question, intent,
                        json.loads(tickers or "[]"), answer,
                        json.loads(findings or "[]")))
    return out


_SELECT = ("SELECT asked_at, question, intent, tickers, answer, findings "
           "FROM conversations WHERE user_id=?")


def recent(db_path: str, user_id: str, limit: int = 3) -> List[Turn]:
    ensure_schema(db_path)
    conn = sqlite3.connect(db_path, timeout=30)
    rows = conn.execute(_SELECT + " ORDER BY asked_at DESC, id DESC LIMIT ?",
                        (user_id, limit)).fetchall()
    conn.close()
    return _rows_to_turns(rows)


def mentioning(db_path: str, user_id: str, tickers: List[str],
               limit: int = 5) -> List[Turn]:
    """Turns that named any of these tickers, however long ago."""
    wanted = {t.upper() for t in tickers or []}
    if not wanted:
        return []
    ensure_schema(db_path)
    conn = sqlite3.connect(db_path, timeout=30)
    clause = " OR ".join(["tickers LIKE ?"] * len(wanted))
    rows = conn.execute(f"{_SELECT} AND ({clause}) ORDER BY asked_at DESC, id DESC LIMIT ?",
                        (user_id, *[f'%"{t}"%' for t in sorted(wanted)], limit)).fetchall()
    conn.close()
    return _rows_to_turns(rows)


def history_for(db_path: str, user_id: str, tickers: List[str],
                recent_n: int = 2, matched_n: int = 4) -> List[Turn]:
    """Context for the critic: what was said about these tickers, plus the last
    couple of turns. Oldest first, deduplicated, bounded."""
    seen, combined = set(), []
    for turn in mentioning(db_path, user_id, tickers, matched_n) + \
            recent(db_path, user_id, recent_n):
        key = (turn.asked_at, turn.question)
        if key in seen:
            continue
        seen.add(key)
        combined.append(turn)
    return sorted(combined, key=lambda t: t.asked_at)
