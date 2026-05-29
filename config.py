"""Env-based configuration for air-threat-bot."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


def _require_env(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise RuntimeError(
            f"Missing env var {key}. Copy .env.example to .env and fill it in."
        )
    return value


def _csv(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


@dataclass(frozen=True)
class Settings:
    bot_token: str
    admin_ids: frozenset[int]
    channels: tuple[str, ...]
    channel_labels: dict[str, str]
    poll_interval_idle: int
    poll_interval_active: int
    alert_min_level: int
    alert_cooldown_sec: int
    tz: str
    db_path: Path
    database_url: str | None
    webhook_base_url: str | None
    webhook_secret_path: str
    port: int


def load_settings() -> Settings:
    channels = tuple(_csv(os.getenv("CHANNELS", "kpszsu,kyiv_alarm,war_monitor")))
    labels_raw = _csv(os.getenv("CHANNEL_LABELS", ""))
    labels = {ch: (labels_raw[i] if i < len(labels_raw) else ch) for i, ch in enumerate(channels)}

    return Settings(
        bot_token=_require_env("BOT_TOKEN"),
        admin_ids=frozenset(int(x) for x in _csv(_require_env("ADMIN_IDS"))),
        channels=channels,
        channel_labels=labels,
        poll_interval_idle=int(os.getenv("POLL_INTERVAL_IDLE", "60")),
        poll_interval_active=int(os.getenv("POLL_INTERVAL_ACTIVE", "20")),
        alert_min_level=int(os.getenv("ALERT_MIN_LEVEL", "3")),
        alert_cooldown_sec=int(os.getenv("ALERT_COOLDOWN_SEC", "600")),
        tz=os.getenv("TZ", "Europe/Kyiv"),
        db_path=Path(os.getenv("DB_PATH", str(ROOT / "threat.db"))),
        database_url=os.getenv("DATABASE_URL") or None,
        webhook_base_url=os.getenv("WEBHOOK_BASE_URL") or None,
        webhook_secret_path=os.getenv("WEBHOOK_SECRET_PATH", "/tg/webhook"),
        port=int(os.getenv("PORT", "10000")),
    )
