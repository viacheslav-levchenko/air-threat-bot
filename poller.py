"""Background polling loop.

For each configured channel, periodically:
  1. Fetch HTML mirror at https://t.me/s/<channel>
  2. Parse + tag new messages
  3. Persist new messages to DB
  4. Recompute threat state from last 6h of all messages
  5. If state crossed an alert threshold and any subscriber qualifies — push DM

Adaptive interval: polls every POLL_INTERVAL_IDLE seconds when threat level is
0-1, switches to POLL_INTERVAL_ACTIVE during elevated/active alerts. This both
respects Telegram's web-mirror rate limits during quiet periods and gives us
denser samples when it matters most.
"""

from __future__ import annotations

import asyncio
import logging
import urllib.error
from datetime import datetime, timedelta, timezone
from typing import Callable

from aiogram import Bot
from aiogram.exceptions import TelegramAPIError, TelegramForbiddenError, TelegramRetryAfter

from classifier import ThreatState, classify
from config import Settings
from db import DB
from parser import (
    ParsedMessage,
    PreviewDisabledError,
    fetch_html,
    parse_html,
    tag_text,
)

log = logging.getLogger("poller")


CLASSIFY_WINDOW_MINUTES = 6 * 60  # 6h
STATE_CACHE_KEY = "last_threat_state"


def _now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)


def _fetch_channel_blocking(channel: str) -> list[ParsedMessage]:
    """Blocking HTTP fetch + parse (called from a thread)."""
    try:
        html_text = fetch_html(channel)
    except PreviewDisabledError:
        raise
    except urllib.error.HTTPError as e:
        log.warning("HTTP %s for @%s", e.code, channel)
        return []
    except (TimeoutError, urllib.error.URLError) as e:
        log.warning("Network error for @%s: %s", channel, e)
        return []
    msgs = parse_html(channel, html_text)
    for m in msgs:
        m.tags = tag_text(m.text)
    return msgs


async def fetch_channel(channel: str) -> list[ParsedMessage]:
    return await asyncio.to_thread(_fetch_channel_blocking, channel)


def _ingest(db: DB, msgs: list[ParsedMessage]) -> int:
    """Persist new messages, return count actually inserted."""
    inserted = 0
    for m in msgs:
        try:
            if db.upsert_message(m.channel, m.msg_id, m.ts, m.text, sorted(m.tags)):
                inserted += 1
        except Exception as e:  # noqa: BLE001
            log.warning("DB upsert failed %s: %s", m.post_path, e)
    return inserted


def _level_crossed_threshold(prev_level: int, new_level: int, threshold: int) -> bool:
    """True only when we move UP across `threshold`. No re-fire on equal levels."""
    return prev_level < threshold <= new_level


async def _push_alert(
    bot: Bot,
    db: DB,
    settings: Settings,
    state: ThreatState,
    reason: str,
) -> int:
    """Push current state to all subscribers who qualify; return count delivered."""
    now = _now_utc()
    delivered = 0
    subs = await asyncio.to_thread(db.list_subscribers)
    for sub in subs:
        # admins always receive, regardless of muted/min_level settings
        if not sub.is_admin and sub.user_id not in settings.admin_ids:
            if sub.muted_until and sub.muted_until > now:
                continue
            if state.level < sub.min_level:
                continue
            last = await asyncio.to_thread(db.last_alert_at, sub.user_id)
            if last and (now - last).total_seconds() < settings.alert_cooldown_sec:
                continue
        text = (
            f"🚨 <b>Загроза для Києва — {state.level_name}</b>\n"
            f"<i>{reason}</i>\n\n"
            f"{state.summary}\n\n"
            f"Снапшот: /status   Підписка: /subscribe   Пауза 1 год: /mute"
        )
        try:
            await bot.send_message(sub.user_id, text, parse_mode="HTML", disable_web_page_preview=True)
            await asyncio.to_thread(
                db.record_alert, sub.user_id, state.level, state.summary, state.trigger_tags
            )
            delivered += 1
        except TelegramRetryAfter as e:
            log.warning("Rate-limited by Telegram, sleeping %s sec", e.retry_after)
            await asyncio.sleep(e.retry_after + 1)
        except TelegramForbiddenError:
            log.info("User %s blocked the bot — removing", sub.user_id)
            await asyncio.to_thread(db.remove_subscriber, sub.user_id)
        except TelegramAPIError as e:
            log.warning("send_message failed for %s: %s", sub.user_id, e)
    return delivered


async def _ensure_admin_subscribers(db: DB, settings: Settings) -> None:
    """ADMIN_IDS always exist as subscribers."""
    for admin_id in settings.admin_ids:
        try:
            sub = await asyncio.to_thread(db.get_subscriber, admin_id)
            if not sub:
                await asyncio.to_thread(db.upsert_subscriber, admin_id, None, True)
                log.info("Auto-added admin %s as subscriber", admin_id)
        except Exception as e:  # noqa: BLE001
            log.warning("Could not seed admin subscriber %s: %s", admin_id, e)


async def poll_once(
    bot: Bot,
    db: DB,
    settings: Settings,
) -> tuple[ThreatState, int]:
    """One iteration of the polling loop. Returns (current_state, new_msg_count)."""
    new_messages_total = 0
    for channel in settings.channels:
        try:
            msgs = await fetch_channel(channel)
        except PreviewDisabledError:
            log.error(
                "Channel @%s has web preview disabled — cannot read via HTML mirror. "
                "Remove from CHANNELS env or switch to a different source.",
                channel,
            )
            continue
        except Exception as e:  # noqa: BLE001
            log.warning("Fetch failed for @%s: %s", channel, e)
            continue
        if not msgs:
            continue
        max_known = await asyncio.to_thread(db.max_msg_id, channel)
        if max_known is not None:
            msgs = [m for m in msgs if m.msg_id > max_known]
        if msgs:
            new = await asyncio.to_thread(_ingest, db, msgs)
            new_messages_total += new
            if new:
                log.info("@%s: ingested %d new msg(s) (last=%d)", channel, new, msgs[-1].msg_id)

    window = await asyncio.to_thread(db.recent_messages, CLASSIFY_WINDOW_MINUTES)
    pm_window = [
        ParsedMessage(channel=r.channel, msg_id=r.msg_id, ts=r.ts, text=r.text)
        for r in window
    ]
    for pm, r in zip(pm_window, window):
        pm.tags = set(r.tags)
    state = classify(pm_window)

    prev_level_raw = await asyncio.to_thread(db.get_state, "last_level")
    prev_level = int(prev_level_raw) if prev_level_raw and prev_level_raw.isdigit() else 0
    prev_combined_raw = await asyncio.to_thread(db.get_state, "last_combined")
    prev_combined = prev_combined_raw == "1"

    new_alert_needed = False
    reason = ""

    if _level_crossed_threshold(prev_level, state.level, settings.alert_min_level):
        new_alert_needed = True
        reason = f"Рівень загрози піднявся: {prev_level} → {state.level}"
    elif state.is_combined_attack and not prev_combined:
        new_alert_needed = True
        reason = "Зафіксовано ознаки комбінованої атаки"
    elif state.level >= 5 and prev_level < 5:
        new_alert_needed = True
        reason = "УДАР — підтверджені вибухи / спуск балістики на Київ"

    if new_alert_needed:
        log.warning("ALERT push (%s): %s", state.short_summary(), reason)
        delivered = await _push_alert(bot, db, settings, state, reason)
        log.warning("Alert delivered to %d subscribers", delivered)

    await asyncio.to_thread(db.set_state, "last_level", str(state.level))
    await asyncio.to_thread(db.set_state, "last_combined", "1" if state.is_combined_attack else "0")
    await asyncio.to_thread(
        db.set_state,
        "last_state_at",
        state.computed_at.isoformat(),
    )
    return state, new_messages_total


async def run_poller_loop(bot: Bot, db: DB, settings: Settings) -> None:
    """Long-running task: poll forever with adaptive interval. Cancel-safe."""
    await _ensure_admin_subscribers(db, settings)
    log.info(
        "Poller started: channels=%s idle=%ds active=%ds min_level=%s cooldown=%ds",
        settings.channels,
        settings.poll_interval_idle,
        settings.poll_interval_active,
        settings.alert_min_level,
        settings.alert_cooldown_sec,
    )
    backoff = 1
    while True:
        try:
            state, new_count = await poll_once(bot, db, settings)
            backoff = 1
            interval = (
                settings.poll_interval_active
                if state.level >= 2
                else settings.poll_interval_idle
            )
            log.debug(
                "Poll done: level=%s new_msgs=%s next in %ss",
                state.level, new_count, interval,
            )
            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            log.info("Poller cancelled, exiting")
            raise
        except Exception as e:  # noqa: BLE001
            log.exception("Poller iteration crashed: %s", e)
            await asyncio.sleep(min(60, settings.poll_interval_idle * backoff))
            backoff = min(backoff * 2, 8)
