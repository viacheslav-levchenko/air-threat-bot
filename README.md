# air-threat-bot

Telegram bot для моніторингу повітряної загрози в Києві.

Парсить публічні Telegram-канали через web-mirror (`https://t.me/s/<channel>`),
складає лексиконний rules engine (без LLM) і дає:

- `/status` — поточний рівень загрози 0-5 + активні флаги
- DM-сповіщення при суттєвій загрозі (≥ рівня 3 за замовч.)
- Окремий DM-сигнал при ознаках комбінованої атаки (Зліт МіГ-31К + БпЛА на Київ + балістика тощо)
- `/history` — таймлайн за N годин

## Архітектура

| Шар | Файл | Що робить |
|---|---|---|
| HTTP fetch + HTML parse | `parser.py` | Тягне `https://t.me/s/<channel>`, парсить пости, накидає тегі з regex-лексикону |
| Threat rules engine | `classifier.py` | TTL-флаги + правила обчислення рівня 0-5, детектор "комбінованої атаки" |
| Storage | `db.py` | SQLite локально / Postgres (Supabase) у продакшені — повідомлення, підписники, alert log |
| Background poller | `poller.py` | Цикл fetch → ingest → classify → push alert. Adaptive interval (60с idle / 20с active) |
| Bot | `bot.py` | aiogram 3 + aiohttp. Webhook у продакшені, polling локально |

## Шкала загрози

| Рівень | Назва | Тригери |
|---|---|---|
| 0 | 🟢 Чисто | Активних флагів нема, тиша 30+ хв |
| 1 | 🟡 Фон | Залишковий шум (поодинокі БпЛА не в нашому напрямку, обстановочні брифи) |
| 2 | 🟠 Підвищена | Зліт МіГ-31К / Ту-95, активна стратегічна авіація, БпЛА у повітрі, ракетна небезпека по країні |
| 3 | 🔴 Висока | БпЛА на Київ; тривога Київ; крилаті ракети активні; ≥5 повідомлень за 10 хв |
| 4 | ⛔ Критична | Кр.ракети курсом на Київ + балістика з Брянська/Воронежа АБО тривога Київ Балістика; спуск балістики десь в Україні; ≥10 повід./10хв |
| 5 | 💥 Удар | Спуск балістики на Київ або вибухи в Києві |

**"Комбінована атака"** = ≥2 з: Зліт МіГ-31К, Ту-95, БпЛА на Київ, масовані БпЛА, КР активні, балістична загроза, тривога Київ (Балістика чи Шахеди). Тригерить окремий DM незалежно від рівня.

## Локальний запуск

```bash
cd air-threat-bot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # заповни BOT_TOKEN, ADMIN_IDS
python bot.py
```

У polling-режимі бот логінться у Telegram, відкриває long-poll сесію і паралельно крутить poller-цикл по каналах.

## Deploy на Render

1. Запуш цей repo на GitHub (свій акаунт або boltable org)
2. Render → New Web Service → connect repo → Render знайде `render.yaml`
3. У Render dashboard заповни env vars:
   - `BOT_TOKEN` (з @BotFather)
   - `ADMIN_IDS` (твій user id — отримати через @userinfobot)
   - `CHANNELS=kpszsu,kyiv_alarm,war_monitor`
   - `DATABASE_URL` (Supabase Postgres — можна використати ту саму DB що photo-quest-bot, таблиці не перетинаються)
   - `WEBHOOK_BASE_URL` (`https://air-threat-bot.onrender.com` — стане доступним після першого deploy)
4. Deploy → Render побудує + запустить
5. Set Telegram webhook (одноразово після першого deploy):
   ```bash
   curl "https://api.telegram.org/bot<TOKEN>/setWebhook?url=https://air-threat-bot.onrender.com/tg/webhook"
   ```
6. Підключи до того ж Apps Script keepalive що photo-quest-bot — поміняй у `populate_doc.gs` константу `BOT_URL` на `https://air-threat-bot.onrender.com/healthz` АБО додай другий тригер.

## Як працює виявлення комбінованої атаки

Класифікатор веде stateful карту "активних флагів" з TTL. Наприклад:

- `mig31_takeoff` живе 3 год — бо вікно для пуску Кинджалів орієнтовно стільки ж
- `ballistic_threat` живе 60 хв — якщо за цей час прийшло `Відбій загрози`, флаг знімається

Кожні 60с (або 20с під час активної загрози) ми:
1. Тягнемо нові пости з `@kpszsu`, `@kyiv_alarm`, `@war_monitor`
2. Прогоняємо текст через regex-лексикон → отримуємо набір тегів
3. Складаємо тегі в активні флаги з TTL
4. Обчислюємо рівень 0-5
5. Якщо рівень піднявся над `ALERT_MIN_LEVEL` АБО з'явилась комбінована атака — пушимо DM
6. Cooldown між пушами на одного юзера — 10 хв за замовч.

Адміни (`ADMIN_IDS`) автоматично підписані і отримують пуш без cooldown / мут / порогу.

## Обмеження

- **Web preview може бути вимкнено**. Тоді канал недосяжний (бот напише попередження у логи і
  продовжить з рештою каналів). Перевір через `curl -sI https://t.me/s/<ch>` — якщо `302` →
  preview disabled.
- **HTML-парсер крихкий**: якщо Telegram змінить розмітку — повідомлення перестануть тагуватись.
  Запусти `python _test_corpus_replay.py` щоб перевірити що парсер живий.
- **Затримка ~ polling interval** (20-60с). Для real-time критичних попереджень (балістика
  з Брянська має ETA ~5хв) це може бути запізно. Якщо хочеш ще раніше — переходимо на
  MTProto (Telethon) з push-режимом.

## Тестування на історичних даних

```bash
# Витягни історичний корпус (~600 повідомлень за 4 міс)
python -c "
import json, pathlib
# ... див. /tmp/tg_corpus/*.json які генерує scripts/scrape_corpus.py
"
# Прогони парсер + класифікатор по корпусу:
python _test_corpus_replay.py
```

Дивись частоту тегів, hourly heatmap рівнів, моменти L4/L5 — все має збігатись з реальною
історією комбінованих атак (17, 23, 24 травня 2026).

## Файли

```
parser.py               regex-лексикон + HTML scrape
classifier.py           rules engine 0-5
db.py                   SQLite/Postgres storage
poller.py               background polling loop
bot.py                  aiogram bot + aiohttp app
config.py               env loader
_test_corpus_replay.py  manual replay sanity-check
requirements.txt        aiogram 3, aiohttp, psycopg
render.yaml             Render free tier config
.env.example            всі env vars з коментарями
```
