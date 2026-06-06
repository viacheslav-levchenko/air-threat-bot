"""Daily 9:00 Kyiv-time digest scheduler.

A long-lived asyncio task that:
  1. Sleeps until the next 09:00 Europe/Kyiv.
  2. Generates the digest payload (last 24h activity, forecast 24h, top factors).
  3. Pushes to all subscribers (admins + opted-in users).
  4. Records that today's digest was sent so we don't re-send on restart.

The task runs inside the same Python process as the bot+poller; no external
cron needed. On Render that means the digest comes from wherever the bot
lives (Frankfurt) at 09:00 Europe/Kyiv (06:00-07:00 UTC depending on DST).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from aiogram import Bot
from aiogram.exceptions import TelegramAPIError, TelegramForbiddenError, TelegramRetryAfter

from analytics import compute_attack_stats, detect_incidents
from classifier import classify
from config import Settings
from db import DB
from forecast import forecast_24h, render_forecast_text
from parser import ParsedMessage

if TYPE_CHECKING:  # avoid import cycle at runtime
    pass

log = logging.getLogger("scheduler")


DIGEST_HOUR_LOCAL = 9
DIGEST_MINUTE_LOCAL = 0


def _now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)


def _next_digest_time(now: datetime, tz_name: str) -> datetime:
    """Compute the next 09:00 Kyiv-time in UTC."""
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(tz_name)
    except Exception:  # noqa: BLE001
        tz = timezone.utc

    now_local = now.astimezone(tz)
    target_local = now_local.replace(
        hour=DIGEST_HOUR_LOCAL, minute=DIGEST_MINUTE_LOCAL,
        second=0, microsecond=0,
    )
    if target_local <= now_local:
        target_local = target_local + timedelta(days=1)
    return target_local.astimezone(timezone.utc)


def build_digest(db: DB, settings: Settings, now: datetime | None = None) -> str:
    """Build the 9:00 daily digest message (HTML-safe Telegram markup)."""
    now = now or _now_utc()
    yesterday = now - timedelta(hours=24)

    # 1. Pull last 7 days of messages for stats; last 24h for "what happened"
    last_24h_rows = db.recent_messages(24 * 60)
    last_7d_rows = db.recent_messages(7 * 24 * 60)

    pms_7d = [ParsedMessage(channel=r.channel, msg_id=r.msg_id, ts=r.ts, text=r.text) for r in last_7d_rows]
    for pm, r in zip(pms_7d, last_7d_rows):
        pm.tags = set(r.tags)

    pms_24h = [pm for pm in pms_7d if pm.ts >= yesterday]

    state_now = classify(pms_7d, now=now)

    incidents_7d = detect_incidents(pms_7d)
    incidents_24h = [i for i in incidents_7d if i.started_at >= yesterday]
    stats = compute_attack_stats(incidents_7d, now=now)
    fcast = forecast_24h(state_now, stats, recent_incidents=incidents_7d, now=now)

    # --- Yesterday's activity summary ---
    by_type_yest: dict[str, list] = {}
    for inc in incidents_24h:
        by_type_yest.setdefault(inc.type, []).append(inc)

    def fmt_local(dt: datetime) -> str:
        try:
            from zoneinfo import ZoneInfo
            return dt.astimezone(ZoneInfo(settings.tz)).strftime("%H:%M")
        except Exception:  # noqa: BLE001
            return dt.strftime("%H:%M UTC")

    kyiv_alarm_count = sum(1 for m in pms_24h if "kyiv_alarm_active" in m.tags)

    lines: list[str] = []
    lines.append("🌅 <b>Доброго ранку.</b>")
    lines.append("")
    lines.append("<b>За добу:</b>")
    if kyiv_alarm_count:
        lines.append(f"  • Тривог у Києві: {kyiv_alarm_count}")
    else:
        lines.append("  • Тривог у Києві: 0")
    if by_type_yest:
        type_labels_ua = {
            "shahed_mass": "Масовані Шахеди",
            "shahed_kyiv": "Шахеди на Київ",
            "cruise_kyiv": "Калібри/Х-101",
            "ballistic_kyiv": "Балістика",
            "kinzhal": "Кинджал",
            "kab_kyiv": "КАБи",
            "explosions_kyiv": "Вибухи в Києві",
            "ballistic_descent_kyiv": "Спуск балістики на Київ",
        }
        for t, incs in by_type_yest.items():
            label = type_labels_ua.get(t, t)
            first = fmt_local(incs[0].started_at)
            lines.append(f"  • {label} × {len(incs)} (перший о {first})")
    else:
        lines.append("  • Атак на Київ: не зафіксовано ✓")
    lines.append("")

    # --- Forecast ---
    lines.append(render_forecast_text(fcast, stats))
    lines.append("")

    # --- Practical recommendation ---
    if fcast.tier == "VERY_HIGH":
        lines.append("💡 <b>Рекомендація:</b> Тримай телефон зарядженим і поряд. "
                     "Ризик найвищий вночі (02:00-05:00). Перевір укриття поблизу.")
    elif fcast.tier == "HIGH":
        lines.append("💡 <b>Рекомендація:</b> Будь готовий до нічних тривог. "
                     "Зарядка телефону, ліхтарик, вода — поряд.")
    elif fcast.tier == "MEDIUM":
        lines.append("💡 <b>Рекомендація:</b> Звичайна обережність. "
                     "Якщо чергуєш — варто бути уважним до сповіщень.")
    else:
        lines.append("💡 <b>Рекомендація:</b> Звичайний день. Слідкуй за оновленнями /status.")

    lines.append("")
    lines.append("<i>Сповіщення в реальному часі — /subscribe. Пауза — /mute.</i>")
    return "\n".join(lines)


async def _send_digest_to_all(bot: Bot, db: DB, settings: Settings) -> int:
    """Send the daily digest to every subscriber. Returns count delivered."""
    text = await asyncio.to_thread(build_digest, db, settings)
    subs = await asyncio.to_thread(db.list_subscribers)
    delivered = 0
    for sub in subs:
        # Daily digest goes to EVERYONE — it is not an emergency alert, so
        # mute / min_level filters don't apply to the morning briefing.
        try:
            await bot.send_message(
                sub.user_id, text,
                parse_mode="HTML",
                disable_web_page_preview=True,
            )
            delivered += 1
        except TelegramRetryAfter as e:
            log.warning("Digest rate-limited, sleeping %s sec", e.retry_after)
            await asyncio.sleep(e.retry_after + 1)
        except TelegramForbiddenError:
            log.info("User %s blocked the bot — removing", sub.user_id)
            await asyncio.to_thread(db.remove_subscriber, sub.user_id)
        except TelegramAPIError as e:
            log.warning("Digest send_message failed for %s: %s", sub.user_id, e)
    return delivered


async def run_digest_scheduler(bot: Bot, db: DB, settings: Settings) -> None:
    """Long-running task: sleep until 9:00 Kyiv, send digest, repeat."""
    log.info("Digest scheduler started")
    while True:
        try:
            now = _now_utc()
            next_run = _next_digest_time(now, settings.tz)
            sleep_sec = max(1, (next_run - now).total_seconds())
            log.info("Next digest at %s (sleeping %.0fs)", next_run.isoformat(), sleep_sec)
            await asyncio.sleep(sleep_sec)

            today_key = next_run.date().isoformat()
            last_sent = await asyncio.to_thread(db.get_state, "last_digest_date")
            if last_sent == today_key:
                log.info("Digest already sent today (%s), skipping", today_key)
                # Small jitter so we don't busy-loop if scheduling drifts
                await asyncio.sleep(60)
                continue

            log.warning("=== Sending 09:00 digest ===")
            delivered = await _send_digest_to_all(bot, db, settings)
            log.warning("Digest sent to %d subscribers", delivered)
            await asyncio.to_thread(db.set_state, "last_digest_date", today_key)
        except asyncio.CancelledError:
            log.info("Digest scheduler cancelled")
            raise
        except Exception as e:  # noqa: BLE001
            log.exception("Digest loop crashed: %s — retrying in 5min", e)
            await asyncio.sleep(300)


__all__ = [
    "build_digest",
    "run_digest_scheduler",
]
