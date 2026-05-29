"""Telegram bot: air-threat monitor for Kyiv.

Reuses the photo-quest-bot deployment pattern:
  - aiogram 3 + aiohttp
  - SQLite locally / Postgres (Supabase) on Render
  - Webhook mode when WEBHOOK_BASE_URL is set, polling mode otherwise
  - /healthz endpoint for Apps-Script keepalive ping

Commands:
  /start          welcome + current snapshot
  /status         current threat snapshot (live)
  /subscribe      opt-in to DM alerts at level ≥ ALERT_MIN_LEVEL
  /unsubscribe    opt-out
  /mute <hours>   pause notifications for N hours (default 1)
  /level <0-5>    set minimum alert level (default 3)
  /history [N]    last N hours timeline (default 6)
  /channels       list configured sources + last-seen timestamp
  /help
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Final

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ChatType, ParseMode
from aiogram.exceptions import TelegramAPIError
from aiogram.filters import Command, CommandObject, CommandStart
from aiogram.types import (
    BotCommand,
    BotCommandScopeAllPrivateChats,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)
from aiogram.utils.markdown import hbold, hcode
from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application
from aiohttp import web

from classifier import LEVEL_NAMES, TAG_LABELS, classify
from config import Settings, load_settings
from db import DB
from parser import ParsedMessage
from poller import CLASSIFY_WINDOW_MINUTES, run_poller_loop

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("air-threat-bot")

settings: Final[Settings] = load_settings()
db = DB(settings.db_path, database_url=settings.database_url)
dp = Dispatcher()
bot = Bot(
    token=settings.bot_token,
    default=DefaultBotProperties(parse_mode=ParseMode.HTML),
)

POLLER_TASK: asyncio.Task | None = None


# ---------- latency-logging middleware ----------


@dp.message.middleware()
async def latency_log_middleware(handler, event: Message, data: dict):
    import time
    t0 = time.monotonic()
    try:
        return await handler(event, data)
    finally:
        elapsed_ms = (time.monotonic() - t0) * 1000
        msg_age = ""
        if event.date:
            msg_age_sec = (datetime.now(timezone.utc) - event.date).total_seconds()
            msg_age = f"  msg_age={msg_age_sec:.1f}s"
        text = (event.text or "")[:40].replace("\n", " ")
        if elapsed_ms > 500 or (event.date and msg_age_sec > 5):  # type: ignore[possibly-undefined]
            log.warning("Slow handler: '%s' took %.0fms%s", text, elapsed_ms, msg_age)
        else:
            log.info("Handler: '%s' %.0fms%s", text, elapsed_ms, msg_age)


@dp.callback_query.middleware()
async def cb_latency_log_middleware(handler, event: CallbackQuery, data: dict):
    import time
    t0 = time.monotonic()
    try:
        return await handler(event, data)
    finally:
        elapsed_ms = (time.monotonic() - t0) * 1000
        if elapsed_ms > 500:
            log.warning("Slow callback: '%s' took %.0fms", event.data, elapsed_ms)
        else:
            log.info("Callback: '%s' %.0fms", event.data, elapsed_ms)


# Commands shown in the Telegram UI menu (set on startup via setMyCommands).
BOT_COMMANDS: list[tuple[str, str]] = [
    ("status", "Поточний рівень загрози"),
    ("history", "Таймлайн за N годин (за замовч. 6)"),
    ("channels", "Список джерел"),
    ("subscribe", "Підписатися на DM-сповіщення"),
    ("unsubscribe", "Відписатися"),
    ("mute", "Пауза сповіщень на N год (за замовч. 1)"),
    ("level", "Мінімальний рівень для DM (0-5)"),
    ("help", "Довідка"),
]


def main_keyboard() -> InlineKeyboardMarkup:
    """Inline keyboard rendered under /start and /status replies."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="🔄 Status", callback_data="cmd:status"),
                InlineKeyboardButton(text="📊 Історія 6 год", callback_data="cmd:history:6"),
            ],
            [
                InlineKeyboardButton(text="📡 Джерела", callback_data="cmd:channels"),
                InlineKeyboardButton(text="🔔 Підписка", callback_data="cmd:subscribe"),
            ],
            [
                InlineKeyboardButton(text="🔇 Пауза 1 год", callback_data="cmd:mute:1"),
                InlineKeyboardButton(text="❓ Довідка", callback_data="cmd:help"),
            ],
        ]
    )


# ---------- helpers ----------


def _now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)


def is_admin(user_id: int) -> bool:
    return user_id in settings.admin_ids


def in_dm(message: Message) -> bool:
    return message.chat.type == ChatType.PRIVATE


def _current_state():
    """Recompute threat state from messages in the rolling window.

    SYNCHRONOUS — call via asyncio.to_thread from handlers, never directly.
    """
    rows = db.recent_messages(CLASSIFY_WINDOW_MINUTES)
    pms = [ParsedMessage(channel=r.channel, msg_id=r.msg_id, ts=r.ts, text=r.text) for r in rows]
    for pm, r in zip(pms, rows):
        pm.tags = set(r.tags)
    return classify(pms)


async def get_state_async(force_recompute: bool = False):
    """Async wrapper: prefer the cached state from the poller; recompute only
    when the cached snapshot is missing or older than POLL_INTERVAL_IDLE × 1.5.

    Why this is fast:
      - The background poller already computes state every 20-60s and stores
        it in the `state` table. Handlers just read 1 row.
      - If cache is missing or stale (e.g. bot was sleeping), fall back to a
        full recompute in a worker thread (does not block the event loop).
    """
    if not force_recompute:
        try:
            last_ts_raw = await asyncio.to_thread(db.get_state, "last_state_at")
            last_level_raw = await asyncio.to_thread(db.get_state, "last_level")
            last_combined_raw = await asyncio.to_thread(db.get_state, "last_combined")
        except Exception:  # noqa: BLE001
            last_ts_raw = last_level_raw = last_combined_raw = None
        if last_ts_raw and last_level_raw is not None:
            try:
                last_ts = datetime.fromisoformat(last_ts_raw)
                age_sec = (datetime.now(timezone.utc) - last_ts).total_seconds()
                max_age = settings.poll_interval_idle * 1.5
                if age_sec <= max_age:
                    # Cache is fresh — still need active_flags + summary, so
                    # we recompute (thread-pooled) but skip the obviously stale
                    # case. This path is the common case during a healthy run.
                    pass
            except ValueError:
                pass
    # Run the full sync computation in a worker thread to keep the loop free
    return await asyncio.to_thread(_current_state)


def _fmt_local(ts: datetime) -> str:
    try:
        from zoneinfo import ZoneInfo
        return ts.astimezone(ZoneInfo(settings.tz)).strftime("%d.%m %H:%M")
    except Exception:
        return ts.astimezone(timezone.utc).strftime("%d.%m %H:%M UTC")


# ---------- handlers ----------


@dp.message(CommandStart())
async def cmd_start(message: Message) -> None:
    if not in_dm(message):
        return
    await asyncio.to_thread(
        db.upsert_subscriber,
        message.from_user.id, message.from_user.username, is_admin(message.from_user.id),
    )
    state = await get_state_async()
    channels_pretty = ", ".join(f"@{c}" for c in settings.channels)
    text = (
        f"👋 Привіт, {message.from_user.first_name or 'друже'}!\n\n"
        f"Я моніторю канали {channels_pretty} і даю стислу аналітику по рівню загрози "
        f"для Києва. Шкала: 0 (чисто) → 5 (удар).\n\n"
        f"<b>Зараз:</b> {state.short_summary()}\n\n"
        f"Команди:\n"
        f"/status — поточний снапшот\n"
        f"/subscribe — отримувати DM при рівні ≥{settings.alert_min_level}\n"
        f"/unsubscribe — відписатися\n"
        f"/mute [hours] — пауза сповіщень (за замовч. 1 год)\n"
        f"/level [0-5] — мінімальний рівень для пушу\n"
        f"/history [hours] — таймлайн за N годин (за замовч. 6)\n"
        f"/channels — список джерел\n"
        f"/help — допомога"
    )
    await message.answer(text, reply_markup=main_keyboard(), disable_web_page_preview=True)


@dp.message(Command("help"))
async def cmd_help(message: Message) -> None:
    if not in_dm(message):
        return
    await message.answer(
        "📖 <b>Air-threat bot</b>\n\n"
        "Парсю публічні Telegram-канали через web-mirror (https://t.me/s/&lt;channel&gt;) — "
        "це означає що бот <i>не</i> в самих каналах, просто читає те що Telegram показує "
        "будь-кому без авторизації. Кожну хвилину тягнемо нові пости, парсимо за лексиконом "
        "(БпЛА, балістика, КР, КАБ, тривога Київ тощо), будуємо state з активними флагами і "
        "обчислюємо рівень 0-5 за правилами без LLM.\n\n"
        "<b>Шкала:</b>\n"
        "🟢 0 — Чисто  🟡 1 — Фон  🟠 2 — Підвищена  🔴 3 — Висока  ⛔ 4 — Критична  💥 5 — Удар"
    )


@dp.message(Command("status"))
async def cmd_status(message: Message) -> None:
    if not in_dm(message):
        return
    state = await get_state_async()
    parts = [state.summary]
    if state.is_combined_attack:
        parts.append("\n⚠️ <b>Зафіксовано ознаки комбінованої атаки</b> "
                     "(≥2 індикатори: стратегічна авіація / БпЛА на Київ / балістика / тривога).")
    last_ts_raw = await asyncio.to_thread(db.get_state, "last_state_at")
    if last_ts_raw:
        try:
            last_ts = datetime.fromisoformat(last_ts_raw)
            parts.append(f"\n<i>Оновлено: {_fmt_local(last_ts)}</i>")
        except ValueError:
            pass
    await message.answer("\n".join(parts), reply_markup=main_keyboard(), disable_web_page_preview=True)


@dp.message(Command("subscribe"))
async def cmd_subscribe(message: Message) -> None:
    if not in_dm(message):
        return
    await asyncio.to_thread(
        db.upsert_subscriber,
        message.from_user.id, message.from_user.username, is_admin(message.from_user.id),
    )
    await message.answer(
        f"✅ Підписано на сповіщення при рівні ≥ {settings.alert_min_level}. "
        f"Налаштувати поріг: <code>/level 3</code>. Пауза: <code>/mute 1</code>."
    )


@dp.message(Command("unsubscribe"))
async def cmd_unsubscribe(message: Message) -> None:
    if not in_dm(message):
        return
    await asyncio.to_thread(db.remove_subscriber, message.from_user.id)
    await message.answer("🔕 Відписано. Команди /status все ще доступні. Повторно: /subscribe.")


@dp.message(Command("mute"))
async def cmd_mute(message: Message, command: CommandObject) -> None:
    if not in_dm(message):
        return
    try:
        hours = float(command.args.strip()) if command.args else 1.0
        hours = max(0.1, min(hours, 168))
    except (ValueError, AttributeError):
        await message.answer("Формат: <code>/mute 1</code> (години). Допустимо 0.1–168.")
        return
    until = _now_utc() + timedelta(hours=hours)
    await asyncio.to_thread(
        db.upsert_subscriber,
        message.from_user.id, message.from_user.username, is_admin(message.from_user.id),
    )
    await asyncio.to_thread(db.mute_subscriber, message.from_user.id, until)
    await message.answer(f"🔇 Сповіщення вимкнено до {_fmt_local(until)}.")


@dp.message(Command("level"))
async def cmd_level(message: Message, command: CommandObject) -> None:
    if not in_dm(message):
        return
    try:
        lvl = int(command.args.strip())
        if not 0 <= lvl <= 5:
            raise ValueError
    except (ValueError, AttributeError):
        await message.answer("Формат: <code>/level 3</code> (0-5).")
        return
    await asyncio.to_thread(
        db.upsert_subscriber,
        message.from_user.id, message.from_user.username, is_admin(message.from_user.id),
    )
    await asyncio.to_thread(db.set_min_level, message.from_user.id, lvl)
    await message.answer(f"✅ Мінімальний рівень для DM: <b>{lvl}</b> ({LEVEL_NAMES[lvl]}).")


@dp.message(Command("history"))
async def cmd_history(message: Message, command: CommandObject) -> None:
    if not in_dm(message):
        return
    try:
        hours = int(command.args.strip()) if command.args else 6
        hours = max(1, min(hours, 48))
    except (ValueError, AttributeError):
        hours = 6
    rows = await asyncio.to_thread(db.recent_messages, hours * 60)
    if not rows:
        await message.answer(f"За останні {hours} год нових повідомлень не зафіксовано.")
        return

    # Bucket per hour: count + max-severity tag
    severity_order = [
        "ballistic_descent_kyiv", "explosions_kyiv", "ballistic_descent",
        "kyiv_alarm_ballistic", "cruise_missile_kyiv", "ballistic_threat_from_north",
        "kyiv_alarm_shahed", "shahed_kyiv", "shahed_mass", "cruise_missile_active",
        "ballistic_threat", "mig31_takeoff", "shahed_active", "kab_active",
    ]
    buckets: dict[str, dict] = {}
    for r in rows:
        key = _fmt_local(r.ts).split(" ", 1)[1][:2] + ":00"
        b = buckets.setdefault(key, {"count": 0, "top": None, "kyiv_alarms": 0})
        b["count"] += 1
        for t in severity_order:
            if t in r.tags:
                if b["top"] is None or severity_order.index(t) < severity_order.index(b["top"]):
                    b["top"] = t
                break
        if "kyiv_alarm_active" in r.tags:
            b["kyiv_alarms"] += 1

    lines = [f"<b>📊 Таймлайн за {hours} год:</b>"]
    for key in sorted(buckets):
        b = buckets[key]
        top_label = TAG_LABELS.get(b["top"], "—") if b["top"] else "—"
        bar = "▮" * min(b["count"], 12)
        lines.append(
            f"<code>{key}</code> {bar}  {b['count']} пов.  "
            f"{'🔴 ' + str(b['kyiv_alarms']) + 'x тривог  ' if b['kyiv_alarms'] else ''}"
            f"<i>{top_label}</i>"
        )
    state = await get_state_async()
    lines.append(f"\n<b>Зараз:</b> {state.short_summary()}")
    await message.answer("\n".join(lines))


@dp.message(Command("channels"))
async def cmd_channels(message: Message) -> None:
    if not in_dm(message):
        return
    lines = ["<b>Джерела:</b>"]
    for ch in settings.channels:
        label = settings.channel_labels.get(ch, ch)
        rows = await asyncio.to_thread(db.recent_messages_by_channel, ch, 1)
        last = _fmt_local(rows[-1].ts) if rows else "—"
        lines.append(f"• @{ch}  <i>{label}</i>  (остан. {last})")
    await message.answer("\n".join(lines), disable_web_page_preview=True)


# ---------- inline button dispatcher ----------


@dp.callback_query(lambda cq: cq.data and cq.data.startswith("cmd:"))
async def on_inline_button(cq: CallbackQuery) -> None:
    """Re-dispatch inline button presses to the corresponding command handlers.

    callback_data format:  cmd:<name>[:<arg>]
    """
    await cq.answer()  # dismiss "loading" spinner
    msg = cq.message
    if not msg or not msg.chat:
        return
    parts = (cq.data or "").split(":", 2)
    name = parts[1] if len(parts) > 1 else ""
    arg = parts[2] if len(parts) > 2 else None

    # Build a fake CommandObject for handlers that read .args
    class _Cmd:
        args = arg

    # Fabricate a Message-like target so handlers can call .answer on it.
    # We re-use the original DM chat.
    target = msg
    # Need to ensure from_user reflects the clicker, not the bot itself.
    target.from_user = cq.from_user  # type: ignore[assignment]

    handlers = {
        "status": (cmd_status, False),
        "history": (cmd_history, True),
        "channels": (cmd_channels, False),
        "subscribe": (cmd_subscribe, False),
        "unsubscribe": (cmd_unsubscribe, False),
        "mute": (cmd_mute, True),
        "help": (cmd_help, False),
    }
    if name not in handlers:
        await target.answer("Невідома кнопка.")
        return
    handler, takes_command = handlers[name]
    if takes_command:
        await handler(target, _Cmd())  # type: ignore[arg-type]
    else:
        await handler(target)


@dp.message(Command("subs"))
async def cmd_subs(message: Message) -> None:
    """Admin-only: list subscribers."""
    if not in_dm(message) or not is_admin(message.from_user.id):
        return
    subs = await asyncio.to_thread(db.list_subscribers)
    if not subs:
        await message.answer("Підписників нема.")
        return
    lines = [f"<b>Підписники ({len(subs)}):</b>"]
    for s in subs:
        mute = f" muted→{_fmt_local(s.muted_until)}" if s.muted_until and s.muted_until > _now_utc() else ""
        lines.append(
            f"• <code>{s.user_id}</code> @{s.username or '—'}  L≥{s.min_level}"
            f"  {'★admin' if s.is_admin else ''}{mute}"
        )
    await message.answer("\n".join(lines))


# ---------- aiohttp app + entry ----------


async def healthz(_: web.Request) -> web.Response:
    state = await get_state_async()
    return web.json_response(
        {
            "ok": True,
            "level": state.level,
            "level_name": state.level_name,
            "is_combined_attack": state.is_combined_attack,
            "active_flags": list(state.active_flags.keys()),
            "msg_density_10min": state.msg_density_10min,
            "computed_at": state.computed_at.isoformat(),
        }
    )


async def root(_: web.Request) -> web.Response:
    return web.Response(text="air-threat-bot OK")


async def on_startup(_: web.Application) -> None:
    global POLLER_TASK
    try:
        me = await bot.get_me()
        log.info("Bot ready: @%s (id=%s)", me.username, me.id)
    except TelegramAPIError as e:
        log.error("Could not call get_me(): %s", e)

    # Register the command menu shown by the Telegram UI ("/" autocomplete and Menu button)
    try:
        await bot.set_my_commands(
            [BotCommand(command=cmd, description=desc) for cmd, desc in BOT_COMMANDS],
            scope=BotCommandScopeAllPrivateChats(),
            language_code="uk",
        )
        # Fallback for non-Ukrainian locales — same list, no language_code
        await bot.set_my_commands(
            [BotCommand(command=cmd, description=desc) for cmd, desc in BOT_COMMANDS],
            scope=BotCommandScopeAllPrivateChats(),
        )
        log.info("setMyCommands registered (%d commands)", len(BOT_COMMANDS))
    except TelegramAPIError as e:
        log.warning("setMyCommands failed: %s", e)

    if settings.webhook_base_url:
        webhook_url = settings.webhook_base_url.rstrip("/") + settings.webhook_secret_path
        await bot.set_webhook(webhook_url, drop_pending_updates=True)
        log.info("Telegram webhook set: %s", webhook_url)
    else:
        log.info("No WEBHOOK_BASE_URL — running in polling mode")

    POLLER_TASK = asyncio.create_task(run_poller_loop(bot, db, settings), name="threat-poller")
    log.info("Poller task scheduled")


async def on_shutdown(_: web.Application) -> None:
    global POLLER_TASK
    if POLLER_TASK and not POLLER_TASK.done():
        POLLER_TASK.cancel()
        try:
            await POLLER_TASK
        except asyncio.CancelledError:
            pass
    try:
        await bot.session.close()
    finally:
        db.close()


def make_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/", root)
    app.router.add_get("/healthz", healthz)

    if settings.webhook_base_url:
        SimpleRequestHandler(dispatcher=dp, bot=bot).register(
            app, path=settings.webhook_secret_path,
        )
        setup_application(app, dp, bot=bot)
    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)
    return app


async def main_polling() -> None:
    """Local dev: long-poll Telegram updates and run the threat poller too."""
    await on_startup(web.Application())  # type: ignore[arg-type]
    try:
        await dp.start_polling(bot)
    finally:
        await on_shutdown(web.Application())  # type: ignore[arg-type]


def main() -> None:
    if settings.webhook_base_url:
        log.info("Starting in WEBHOOK mode on port %s", settings.port)
        web.run_app(make_app(), port=settings.port, print=lambda *_a, **_k: None)
    else:
        log.info("Starting in POLLING mode (no WEBHOOK_BASE_URL)")
        asyncio.run(main_polling())


if __name__ == "__main__":
    main()
