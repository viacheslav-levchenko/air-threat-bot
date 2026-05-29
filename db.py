"""SQLite + Postgres storage for air-threat-bot.

Mirrors the pattern used in photo-quest-bot: pick backend by DATABASE_URL.
Synchronous API — handler concurrency is low and the poller runs in a
background asyncio task that delegates blocking DB work to `asyncio.to_thread`.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

log = logging.getLogger("db")


# ---------- Schema (works in both SQLite and Postgres) ----------

SCHEMA_STATEMENTS = [
    """CREATE TABLE IF NOT EXISTS messages (
        channel     TEXT NOT NULL,
        msg_id      BIGINT NOT NULL,
        ts          TIMESTAMP NOT NULL,
        text        TEXT,
        tags        TEXT NOT NULL DEFAULT '[]',
        fetched_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (channel, msg_id)
    )""",
    "CREATE INDEX IF NOT EXISTS idx_messages_ts ON messages(ts)",
    "CREATE INDEX IF NOT EXISTS idx_messages_channel_ts ON messages(channel, ts DESC)",
    """CREATE TABLE IF NOT EXISTS subscribers (
        user_id        BIGINT PRIMARY KEY,
        username       TEXT,
        subscribed_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        muted_until    TIMESTAMP,
        min_level      INT NOT NULL DEFAULT 3,
        is_admin       BOOLEAN NOT NULL DEFAULT FALSE
    )""",
    """CREATE TABLE IF NOT EXISTS alerts_sent (
        user_id    BIGINT NOT NULL,
        sent_at    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        level      INT NOT NULL,
        summary    TEXT,
        trigger_tags TEXT
    )""",
    "CREATE INDEX IF NOT EXISTS idx_alerts_user_time ON alerts_sent(user_id, sent_at DESC)",
    """CREATE TABLE IF NOT EXISTS state (
        key   TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )""",
]


@dataclass
class Subscriber:
    user_id: int
    username: str | None
    subscribed_at: datetime
    muted_until: datetime | None
    min_level: int
    is_admin: bool


@dataclass
class MessageRow:
    channel: str
    msg_id: int
    ts: datetime
    text: str
    tags: list[str]


class DB:
    """Backend-agnostic DB wrapper. Postgres when DATABASE_URL is set, SQLite otherwise."""

    def __init__(self, db_path: Path, database_url: str | None = None) -> None:
        self.db_path = db_path
        self.database_url = database_url
        self.is_pg = bool(database_url and database_url.startswith(("postgres://", "postgresql://")))
        if self.is_pg:
            import psycopg  # noqa: F401  (lazy import; ensures requirements.txt is installed)
            self._pg_conninfo = (
                database_url.replace("postgres://", "postgresql://", 1)
                if database_url and database_url.startswith("postgres://")
                else database_url
            )
            self._pg_conn = None
        else:
            self._sqlite = sqlite3.connect(str(self.db_path), isolation_level=None, check_same_thread=False)
            self._sqlite.execute("PRAGMA journal_mode=WAL")
            self._sqlite.execute("PRAGMA foreign_keys=ON")
            self._sqlite.row_factory = sqlite3.Row
        self._init_schema()

    # ---------- connection helpers ----------

    def _pg_connection(self):
        import psycopg
        if self._pg_conn is None or self._pg_conn.closed:
            self._pg_conn = psycopg.connect(self._pg_conninfo, autocommit=True)
        return self._pg_conn

    @contextmanager
    def cursor(self) -> Iterator[Any]:
        if self.is_pg:
            conn = self._pg_connection()
            with conn.cursor() as cur:
                yield cur
        else:
            cur = self._sqlite.cursor()
            try:
                yield cur
            finally:
                cur.close()

    def _ph(self) -> str:
        """Placeholder for current backend (SQLite uses ?, Postgres uses %s)."""
        return "%s" if self.is_pg else "?"

    def _adapt_sql(self, sql: str) -> str:
        """Adapt the schema statement for Postgres-specific syntax differences."""
        if self.is_pg:
            sql = sql.replace("BOOLEAN NOT NULL DEFAULT FALSE", "BOOLEAN NOT NULL DEFAULT FALSE")
        return sql

    def _init_schema(self) -> None:
        with self.cursor() as cur:
            for stmt in SCHEMA_STATEMENTS:
                cur.execute(self._adapt_sql(stmt))

    def close(self) -> None:
        try:
            if self.is_pg and self._pg_conn:
                self._pg_conn.close()
            elif not self.is_pg:
                self._sqlite.close()
        except Exception as e:  # noqa: BLE001
            log.warning("DB close error: %s", e)

    # ---------- messages ----------

    def upsert_message(
        self,
        channel: str,
        msg_id: int,
        ts: datetime,
        text: str,
        tags: list[str],
    ) -> bool:
        """Insert if new, otherwise no-op. Returns True if newly inserted."""
        ph = self._ph()
        if self.is_pg:
            sql = (
                f"INSERT INTO messages (channel, msg_id, ts, text, tags) "
                f"VALUES ({ph}, {ph}, {ph}, {ph}, {ph}) "
                f"ON CONFLICT (channel, msg_id) DO NOTHING"
            )
        else:
            sql = (
                f"INSERT OR IGNORE INTO messages (channel, msg_id, ts, text, tags) "
                f"VALUES ({ph}, {ph}, {ph}, {ph}, {ph})"
            )
        with self.cursor() as cur:
            cur.execute(
                sql,
                (channel, msg_id, ts.astimezone(timezone.utc).replace(tzinfo=None) if self.is_pg else ts.isoformat(),
                 text, json.dumps(tags, ensure_ascii=False)),
            )
            return cur.rowcount > 0

    def max_msg_id(self, channel: str) -> int | None:
        ph = self._ph()
        with self.cursor() as cur:
            cur.execute(f"SELECT MAX(msg_id) FROM messages WHERE channel = {ph}", (channel,))
            row = cur.fetchone()
            return row[0] if row and row[0] is not None else None

    def recent_messages(self, since_minutes: int) -> list[MessageRow]:
        """All messages from all channels within the past N minutes."""
        ph = self._ph()
        with self.cursor() as cur:
            if self.is_pg:
                cur.execute(
                    f"SELECT channel, msg_id, ts, text, tags FROM messages "
                    f"WHERE ts >= (NOW() AT TIME ZONE 'UTC') - INTERVAL '1 minute' * {ph} "
                    f"ORDER BY ts ASC",
                    (since_minutes,),
                )
            else:
                cur.execute(
                    f"SELECT channel, msg_id, ts, text, tags FROM messages "
                    f"WHERE datetime(ts) >= datetime('now', {ph}) "
                    f"ORDER BY ts ASC",
                    (f"-{since_minutes} minutes",),
                )
            rows = cur.fetchall()
        return [_row_to_message(r) for r in rows]

    def recent_messages_by_channel(self, channel: str, limit: int) -> list[MessageRow]:
        ph = self._ph()
        with self.cursor() as cur:
            cur.execute(
                f"SELECT channel, msg_id, ts, text, tags FROM messages "
                f"WHERE channel = {ph} ORDER BY ts DESC LIMIT {ph}",
                (channel, limit),
            )
            rows = cur.fetchall()
        return list(reversed([_row_to_message(r) for r in rows]))

    # ---------- subscribers ----------

    def upsert_subscriber(self, user_id: int, username: str | None, is_admin: bool = False) -> None:
        ph = self._ph()
        if self.is_pg:
            sql = (
                f"INSERT INTO subscribers (user_id, username, is_admin) VALUES ({ph}, {ph}, {ph}) "
                f"ON CONFLICT (user_id) DO UPDATE SET username = EXCLUDED.username, "
                f"is_admin = subscribers.is_admin OR EXCLUDED.is_admin"
            )
        else:
            sql = (
                f"INSERT INTO subscribers (user_id, username, is_admin) VALUES ({ph}, {ph}, {ph}) "
                f"ON CONFLICT(user_id) DO UPDATE SET username = excluded.username, "
                f"is_admin = subscribers.is_admin OR excluded.is_admin"
            )
        with self.cursor() as cur:
            cur.execute(sql, (user_id, username, is_admin))

    def remove_subscriber(self, user_id: int) -> None:
        ph = self._ph()
        with self.cursor() as cur:
            cur.execute(f"DELETE FROM subscribers WHERE user_id = {ph}", (user_id,))

    def mute_subscriber(self, user_id: int, until: datetime) -> None:
        ph = self._ph()
        val = until.astimezone(timezone.utc).replace(tzinfo=None) if self.is_pg else until.isoformat()
        with self.cursor() as cur:
            cur.execute(
                f"UPDATE subscribers SET muted_until = {ph} WHERE user_id = {ph}",
                (val, user_id),
            )

    def set_min_level(self, user_id: int, min_level: int) -> None:
        ph = self._ph()
        with self.cursor() as cur:
            cur.execute(
                f"UPDATE subscribers SET min_level = {ph} WHERE user_id = {ph}",
                (min_level, user_id),
            )

    def list_subscribers(self) -> list[Subscriber]:
        with self.cursor() as cur:
            cur.execute(
                "SELECT user_id, username, subscribed_at, muted_until, min_level, is_admin FROM subscribers"
            )
            rows = cur.fetchall()
        out: list[Subscriber] = []
        for r in rows:
            out.append(
                Subscriber(
                    user_id=r[0],
                    username=r[1],
                    subscribed_at=_to_dt(r[2]),
                    muted_until=_to_dt(r[3]) if r[3] else None,
                    min_level=r[4] or 3,
                    is_admin=bool(r[5]),
                )
            )
        return out

    def get_subscriber(self, user_id: int) -> Subscriber | None:
        ph = self._ph()
        with self.cursor() as cur:
            cur.execute(
                f"SELECT user_id, username, subscribed_at, muted_until, min_level, is_admin "
                f"FROM subscribers WHERE user_id = {ph}",
                (user_id,),
            )
            r = cur.fetchone()
        if not r:
            return None
        return Subscriber(
            user_id=r[0],
            username=r[1],
            subscribed_at=_to_dt(r[2]),
            muted_until=_to_dt(r[3]) if r[3] else None,
            min_level=r[4] or 3,
            is_admin=bool(r[5]),
        )

    # ---------- alerts log ----------

    def last_alert_at(self, user_id: int) -> datetime | None:
        ph = self._ph()
        with self.cursor() as cur:
            cur.execute(
                f"SELECT MAX(sent_at) FROM alerts_sent WHERE user_id = {ph}",
                (user_id,),
            )
            r = cur.fetchone()
        return _to_dt(r[0]) if r and r[0] else None

    def record_alert(
        self,
        user_id: int,
        level: int,
        summary: str,
        trigger_tags: list[str],
    ) -> None:
        ph = self._ph()
        with self.cursor() as cur:
            cur.execute(
                f"INSERT INTO alerts_sent (user_id, level, summary, trigger_tags) "
                f"VALUES ({ph}, {ph}, {ph}, {ph})",
                (user_id, level, summary, json.dumps(trigger_tags, ensure_ascii=False)),
            )

    # ---------- generic key/value state ----------

    def get_state(self, key: str) -> str | None:
        ph = self._ph()
        with self.cursor() as cur:
            cur.execute(f"SELECT value FROM state WHERE key = {ph}", (key,))
            r = cur.fetchone()
        return r[0] if r else None

    def set_state(self, key: str, value: str) -> None:
        ph = self._ph()
        if self.is_pg:
            sql = (
                f"INSERT INTO state (key, value) VALUES ({ph}, {ph}) "
                f"ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value"
            )
        else:
            sql = (
                f"INSERT INTO state (key, value) VALUES ({ph}, {ph}) "
                f"ON CONFLICT(key) DO UPDATE SET value = excluded.value"
            )
        with self.cursor() as cur:
            cur.execute(sql, (key, value))


# ---------- helpers ----------


def _to_dt(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def _row_to_message(row: Any) -> MessageRow:
    tags_raw = row[4] or "[]"
    try:
        tags = json.loads(tags_raw) if isinstance(tags_raw, str) else list(tags_raw)
    except (TypeError, json.JSONDecodeError):
        tags = []
    return MessageRow(
        channel=row[0],
        msg_id=row[1],
        ts=_to_dt(row[2]) or datetime.now(tz=timezone.utc),
        text=row[3] or "",
        tags=tags,
    )
