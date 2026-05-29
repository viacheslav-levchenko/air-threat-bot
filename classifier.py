"""Rule-based threat classifier (no LLM).

Consumes a rolling window of tagged ParsedMessage objects and produces:
  - threat_level (0-5)
  - active_flags (dict[str, datetime] — flag -> expires_at_utc)
  - is_combined_attack (bool)
  - summary (Ukrainian human-readable string)
  - trigger_tags (list[str]) for explainability

The TTL values were chosen from observed timings during the 17/23/24 May 2026
combined attacks:
  - MiG-31K takeoff → Kinzhal launch window ≈ 1-3 hours.
  - Strategic bomber takeoff (Tu-95) → cruise-missile launches ≈ 3-6 hours.
  - Cruise missile from Caspian/Black Sea to Kyiv ≈ 40-90 min.
  - Ballistic from Bryansk/Voronezh to Kyiv ≈ 4-10 min (descent window ≈ 1-3 min).
  - Shahed swarm from northern oblasts ≈ 1-3 hours travel.

These are not hard physics — they are *signal-decay* horizons after which the
flag stops contributing to the level. Real-time deescalation is handled by the
explicit `all_clear` tag.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Iterable, Sequence

from parser import ParsedMessage

log = logging.getLogger("classifier")


# TTL (seconds) — how long a tag's signal stays "active" after we saw it
TAG_TTL_SEC: dict[str, int] = {
    "mig31_takeoff": 180 * 60,            # 3h
    "tu95_takeoff": 360 * 60,             # 6h
    "ru_strategic_aviation_active": 240 * 60,
    "ballistic_threat": 60 * 60,
    "ballistic_threat_from_north": 60 * 60,
    "ballistic_descent": 5 * 60,
    "ballistic_descent_kyiv": 10 * 60,
    "cruise_missile_active": 90 * 60,
    "cruise_missile_kyiv": 90 * 60,
    "hypersonic_kinzhal": 30 * 60,
    "hypersonic_tsirkon": 30 * 60,
    "shahed_active": 120 * 60,
    "shahed_kyiv": 120 * 60,
    "shahed_mass": 120 * 60,
    "kab_active": 30 * 60,
    "explosions_kyiv": 30 * 60,
    "explosions_anywhere": 20 * 60,
    "kyiv_alarm_active": 90 * 60,
    "kyiv_alarm_ballistic": 30 * 60,
    "kyiv_alarm_shahed": 90 * 60,
    "country_wide_missile_alert": 60 * 60,
    "situation_brief": 60 * 60,
}

# Friendly Ukrainian labels for active flags (for /status and DM summaries)
TAG_LABELS: dict[str, str] = {
    "mig31_takeoff": "Зліт МіГ-31К",
    "tu95_takeoff": "Зліт Ту-95/160",
    "ru_strategic_aviation_active": "Стратегічна авіація РФ активна",
    "ballistic_threat": "Загроза балістики",
    "ballistic_threat_from_north": "Балістика з півночі (Брянськ/Воронеж)",
    "ballistic_descent": "Спуск балістики",
    "ballistic_descent_kyiv": "Спуск балістики на Київ",
    "cruise_missile_active": "Крилаті ракети в повітрі",
    "cruise_missile_kyiv": "Крилаті ракети курсом на Київ",
    "hypersonic_kinzhal": "Кинджал",
    "hypersonic_tsirkon": "Циркон",
    "shahed_active": "БпЛА у повітрі",
    "shahed_kyiv": "БпЛА курсом на Київ",
    "shahed_mass": "Масована атака БпЛА",
    "kab_active": "КАБи",
    "explosions_anywhere": "Вибухи",
    "explosions_kyiv": "Вибухи в Києві",
    "kyiv_alarm_active": "Тривога в Києві",
    "kyiv_alarm_ballistic": "Тривога Київ: Балістика",
    "kyiv_alarm_shahed": "Тривога Київ: Шахеди",
    "country_wide_missile_alert": "Ракетна небезпека по всій країні",
    "situation_brief": "Обстановочний брифінг",
}

LEVEL_NAMES = {
    0: "🟢 Чисто",
    1: "🟡 Фон",
    2: "🟠 Підвищена",
    3: "🔴 Висока",
    4: "⛔ Критична",
    5: "💥 Удар",
}


@dataclass
class ThreatState:
    level: int
    level_name: str
    active_flags: dict[str, datetime]  # flag -> expiry (UTC)
    trigger_tags: list[str]
    is_combined_attack: bool
    msg_density_10min: int
    summary: str
    computed_at: datetime
    last_kyiv_alarm_text: str | None = None

    def short_summary(self) -> str:
        return f"{self.level_name} (рівень {self.level}/5)"


def _now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)


def compute_active_flags(
    messages: Sequence[ParsedMessage],
    now: datetime | None = None,
) -> dict[str, datetime]:
    """Walk messages in chronological order, build expiring flags, honor all-clear.

    A flag is set when the corresponding tag appears in a message; its expiry
    is message_ts + TTL. An `all_clear` tag clears a fixed set of threat flags.

    Messages older than the largest TTL are skipped for efficiency.
    """
    now = now or _now_utc()
    horizon = now - timedelta(seconds=max(TAG_TTL_SEC.values()) + 60)

    flags: dict[str, datetime] = {}
    msgs = [m for m in messages if m.ts >= horizon]
    msgs.sort(key=lambda m: m.ts)

    CLEARABLE_BY_ALL_CLEAR = {
        "ballistic_threat",
        "ballistic_threat_from_north",
        "ballistic_descent",
        "ballistic_descent_kyiv",
        "cruise_missile_active",
        "cruise_missile_kyiv",
        "kyiv_alarm_active",
        "kyiv_alarm_ballistic",
        "kyiv_alarm_shahed",
        "country_wide_missile_alert",
    }
    # Strategic-aviation flags are cleared by an explicit landing/inactive signal
    # ("Відбій загрози МіГ-31К" / "Стратегічна авіація: Не активна").
    CLEARABLE_BY_STRATEGIC_INACTIVE = {
        "mig31_takeoff",
        "tu95_takeoff",
        "ru_strategic_aviation_active",
        "hypersonic_kinzhal",
    }

    for m in msgs:
        if "all_clear" in m.tags:
            for f in list(flags):
                if f in CLEARABLE_BY_ALL_CLEAR:
                    flags.pop(f, None)
        if "ru_strategic_aviation_inactive" in m.tags:
            for f in list(flags):
                if f in CLEARABLE_BY_STRATEGIC_INACTIVE:
                    flags.pop(f, None)
        for tag in m.tags:
            ttl = TAG_TTL_SEC.get(tag)
            if not ttl:
                continue
            expiry = m.ts + timedelta(seconds=ttl)
            prev = flags.get(tag)
            if not prev or expiry > prev:
                flags[tag] = expiry

    # Drop expired
    return {tag: exp for tag, exp in flags.items() if exp > now}


# Tags that are Kyiv-specific OR otherwise material to Kyiv threat.
# These are the ONLY tags the classifier uses to raise the level.
# Generic threats happening elsewhere in Ukraine (e.g. "БпЛА на Дніпро",
# "балістика з Воронежа на Полтаву") deliberately do NOT raise our level,
# because this bot is scoped to Kyiv. Exception: `shahed_mass` — a mass UAV
# wave is treated as Kyiv-relevant even before vector data, because Kyiv is
# the most likely target on any heavy night.
KYIV_RELEVANT_TAGS = {
    "explosions_kyiv",
    "ballistic_descent_kyiv",
    "kyiv_alarm_ballistic",
    "kyiv_alarm_shahed",
    "kyiv_alarm_active",
    "cruise_missile_kyiv",
    "ballistic_threat_from_north",  # Bryansk/Voronezh trajectory → Kyiv corridor
    "shahed_kyiv",
    "shahed_mass",
    # Strategic-aviation flags are preparatory and contribute up to L2.
    # They are not Kyiv-targeted by themselves, but they precede combined attacks.
    "mig31_takeoff",
    "tu95_takeoff",
    "ru_strategic_aviation_active",
    "country_wide_missile_alert",
}


def is_combined_attack(active: dict[str, datetime]) -> tuple[bool, list[str]]:
    """≥2 indicators of a Kyiv-targeted combined attack currently active.

    All indicators must be Kyiv-specific (or a mass-wave that statistically
    targets Kyiv). Strategic-aviation takeoff counts as ONE indicator only.
    """
    strategic_aviation_up = any(
        t in active for t in ("mig31_takeoff", "tu95_takeoff", "ru_strategic_aviation_active")
    )

    indicators: list[tuple[bool, str]] = [
        (strategic_aviation_up, "Стратегічна авіація РФ у повітрі"),
        ("shahed_kyiv" in active, "БпЛА курсом на Київ"),
        ("kyiv_alarm_shahed" in active, "Тривога Київ: Шахеди"),
        ("shahed_mass" in active, "Масована атака БпЛА"),
        ("cruise_missile_kyiv" in active, "Крилаті ракети на Київ"),
        ("ballistic_threat_from_north" in active, "Балістика з півночі (Брянськ/Воронеж) на Київ"),
        ("kyiv_alarm_ballistic" in active, "Тривога Київ: Балістика"),
    ]
    hits = [label for present, label in indicators if present]
    return len(hits) >= 2, hits


def compute_level(
    active: dict[str, datetime],
    msg_density_10min: int,
) -> tuple[int, list[str]]:
    """Return (level, trigger_tags).

    Rules are intentionally Kyiv-centric:
      - Threats over other oblasts do not raise our level by themselves.
      - Strategic-aviation takeoff alone is at most L2 (preparation, not impact).
      - L3+ requires a Kyiv-direction signal (alarm in Kyiv, BpLA/missiles on
        Kyiv, ballistic from northern launch zones whose trajectory IS Kyiv).
      - L4+ requires a confirmed-imminent strike on Kyiv (Kyiv ballistic alarm,
        OR ≥2 critical Kyiv-targeted signals, OR a message-density spike).
      - L5 = impact (explosions / descent over Kyiv).
    """
    triggers: list[str] = []

    # Level 5: physical impact on Kyiv RIGHT NOW
    if "explosions_kyiv" in active or "ballistic_descent_kyiv" in active:
        triggers.extend(t for t in ("explosions_kyiv", "ballistic_descent_kyiv") if t in active)
        return 5, triggers

    # Critical Kyiv-targeted signals
    crit_signals = [
        "kyiv_alarm_ballistic",
        "cruise_missile_kyiv",
        "ballistic_threat_from_north",
    ]
    crit_hits = [t for t in crit_signals if t in active]

    # Level 4: explicit Kyiv ballistic alarm, OR ≥2 critical Kyiv signals, OR density spike
    if "kyiv_alarm_ballistic" in active or len(crit_hits) >= 2 or msg_density_10min >= 10:
        triggers.extend(crit_hits)
        if msg_density_10min >= 10:
            triggers.append(f"msg_density_10min={msg_density_10min}")
        return 4, triggers

    # Level 3: any single Kyiv-direction signal, OR ≥2 high signals, OR density ≥5
    high_signals = [
        "shahed_kyiv",
        "kyiv_alarm_shahed",
        "kyiv_alarm_active",
        "ballistic_threat_from_north",
        "cruise_missile_kyiv",
    ]
    high_hits = [t for t in high_signals if t in active]
    if high_hits or "shahed_mass" in active or msg_density_10min >= 5:
        triggers.extend(high_hits)
        if "shahed_mass" in active:
            triggers.append("shahed_mass")
        if msg_density_10min >= 5:
            triggers.append(f"msg_density_10min={msg_density_10min}")
        return 3, triggers

    # Level 2: strategic posturing / countrywide preparation (NOT Kyiv-specific yet)
    mod_signals = [
        "mig31_takeoff",
        "tu95_takeoff",
        "ru_strategic_aviation_active",
        "country_wide_missile_alert",
    ]
    mod_hits = [t for t in mod_signals if t in active]
    if mod_hits or msg_density_10min >= 3:
        triggers.extend(mod_hits)
        if msg_density_10min >= 3:
            triggers.append(f"msg_density_10min={msg_density_10min}")
        return 2, triggers

    # Level 1: minor background activity that isn't Kyiv-relevant
    if any(t in active for t in KYIV_RELEVANT_TAGS) or msg_density_10min >= 1:
        triggers.extend(t for t in active if t in KYIV_RELEVANT_TAGS)
        return 1, triggers

    return 0, []


def make_summary(
    level: int,
    active: dict[str, datetime],
    triggers: list[str],
    msg_density_10min: int,
    last_kyiv_alarm_text: str | None,
) -> str:
    parts: list[str] = []
    parts.append(f"<b>{LEVEL_NAMES[level]}</b>")
    if level == 0:
        parts.append("Активних загроз для Києва не зафіксовано.")
        return " ".join(parts)

    if last_kyiv_alarm_text:
        parts.append(f"Київ: {last_kyiv_alarm_text}.")

    flag_lines: list[str] = []
    priority = [
        "ballistic_descent_kyiv",
        "explosions_kyiv",
        "ballistic_descent",
        "explosions_anywhere",
        "kyiv_alarm_ballistic",
        "kyiv_alarm_shahed",
        "cruise_missile_kyiv",
        "cruise_missile_active",
        "ballistic_threat_from_north",
        "ballistic_threat",
        "shahed_mass",
        "shahed_kyiv",
        "shahed_active",
        "hypersonic_kinzhal",
        "hypersonic_tsirkon",
        "mig31_takeoff",
        "tu95_takeoff",
        "ru_strategic_aviation_active",
        "kab_active",
        "country_wide_missile_alert",
    ]
    seen: set[str] = set()
    for tag in priority:
        if tag in active and tag not in seen:
            label = TAG_LABELS.get(tag, tag)
            mins_left = int((active[tag] - _now_utc()).total_seconds() // 60)
            flag_lines.append(f"• {label} (TTL ~{mins_left}хв)")
            seen.add(tag)
    if flag_lines:
        parts.append("\n<b>Активно:</b>\n" + "\n".join(flag_lines[:8]))

    if msg_density_10min >= 3:
        parts.append(f"\n<i>Інтенсивність потоку: {msg_density_10min} пов./10хв.</i>")

    return "\n".join(parts)


def classify(
    messages: Sequence[ParsedMessage],
    now: datetime | None = None,
) -> ThreatState:
    """Top-level: compute current ThreatState from a window of messages."""
    now = now or _now_utc()
    active = compute_active_flags(messages, now=now)

    window_start = now - timedelta(minutes=10)
    msg_density = sum(1 for m in messages if m.ts >= window_start)

    level, triggers = compute_level(active, msg_density)
    combined, _hits = is_combined_attack(active)

    last_alarm: str | None = None
    for m in sorted(messages, key=lambda m: m.ts, reverse=True):
        if "kyiv_alarm_active" in m.tags and m.text:
            last_alarm = m.text[:160]
            break

    summary = make_summary(level, active, triggers, msg_density, last_alarm)
    return ThreatState(
        level=level,
        level_name=LEVEL_NAMES[level],
        active_flags=active,
        trigger_tags=triggers,
        is_combined_attack=combined,
        msg_density_10min=msg_density,
        summary=summary,
        computed_at=now,
        last_kyiv_alarm_text=last_alarm,
    )


__all__ = [
    "LEVEL_NAMES",
    "TAG_LABELS",
    "TAG_TTL_SEC",
    "ThreatState",
    "classify",
    "compute_active_flags",
    "compute_level",
    "is_combined_attack",
    "make_summary",
]
