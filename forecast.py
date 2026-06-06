"""24-hour attack probability forecast for Kyiv.

Combines:
  • Current active flags (from classifier.ThreatState)
  • Historical attack stats (from analytics.compute_attack_stats)
  • Time-of-day and weekday priors
  • Active preparation signals (MiG-31K up, Tu-95 up, official warnings, etc.)

into a single probability estimate for the next 24 hours, plus an
*explainable* list of the top factors driving the score.

This is NOT a black-box ML model. It's a transparent weighted-sum with
expert-informed coefficients. We deliberately keep it simple so the
output is understandable and tunable.

Output tiers:
  🟢 LOW       < 0.25
  🟡 MEDIUM    0.25 - 0.50
  🟠 HIGH      0.50 - 0.75
  🔴 VERY_HIGH > 0.75
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Iterable

from analytics import (
    AttackStats,
    Incident,
    STOCKPILE_RECOVERY_DAYS,
)
from classifier import ThreatState

log = logging.getLogger("forecast")


# ---------- Tunable weights (revisit during validation) ----------

# Boost contributions when these preparation flags are CURRENTLY active.
# Sum can exceed 1.0; we clamp the final probability.
ACTIVE_FLAG_BOOSTS: dict[str, float] = {
    "mig31_takeoff":           0.35,
    "tu95_takeoff":            0.35,
    "tu95_carrier_movement":   0.25,
    "naval_carrier_movement":  0.30,
    "ru_strategic_aviation_active": 0.30,
    "shahed_launch_detected":  0.20,
    "official_warning_attack": 0.40,
    "presidential_statement_attack": 0.25,
    "country_wide_missile_alert":   0.20,
}

# Boosts derived from "recently happened" events (within last N hours).
RECENT_INCIDENT_BOOSTS: dict[str, float] = {
    # If we just saw mass Shahed wave, another in next 24h is less likely
    "shahed_mass_recent_within_24h": -0.20,
    "kinzhal_recent_within_24h": -0.15,
    "cruise_kyiv_recent_within_72h": -0.20,
}

# Time-of-day base rate adjustment (Russia historically prefers 22:00-05:00 Kyiv)
NIGHT_HOURS = (22, 23, 0, 1, 2, 3, 4, 5)
DAY_HOURS = tuple(h for h in range(24) if h not in NIGHT_HOURS)

# Day-of-week prior. Empirically Russian strikes cluster on Fri-Sat-Sun (per
# public analysis), but the effect is mild. 0=Mon, 6=Sun.
WEEKDAY_PRIORS: dict[int, float] = {
    0: 0.0,   # Mon
    1: 0.0,   # Tue
    2: 0.02,  # Wed
    3: 0.03,  # Thu
    4: 0.05,  # Fri
    5: 0.05,  # Sat
    6: 0.03,  # Sun
}

# Known politically-charged anniversaries that historically saw escalation.
# (MM, DD) -> note. We add a small boost on these days.
ANNIVERSARY_DATES: dict[tuple[int, int], str] = {
    (2, 24): "Річниця повномасштабного вторгнення",
    (5, 9): "9 травня — російське свято",
    (6, 22): "Початок Другої світової",
    (8, 24): "День Незалежності України",
    (10, 14): "День захисника",
}


# ---------- Public output type ----------

TIER_LABELS = {
    "LOW": "🟢 НИЗЬКА",
    "MEDIUM": "🟡 СЕРЕДНЯ",
    "HIGH": "🟠 ВИСОКА",
    "VERY_HIGH": "🔴 ДУЖЕ ВИСОКА",
}


@dataclass
class Factor:
    """One named contribution to the final probability."""
    name: str
    delta: float  # signed contribution to probability
    explanation: str


@dataclass
class Forecast:
    probability: float  # 0..1, clamped
    tier: str           # LOW | MEDIUM | HIGH | VERY_HIGH
    tier_label: str
    factors: list[Factor] = field(default_factory=list)
    computed_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))

    @property
    def percent(self) -> int:
        return int(round(self.probability * 100))

    def top_factors(self, n: int = 5) -> list[Factor]:
        return sorted(self.factors, key=lambda f: abs(f.delta), reverse=True)[:n]


# ---------- Forecast model ----------


def _tier_for(p: float) -> str:
    if p >= 0.75:
        return "VERY_HIGH"
    if p >= 0.50:
        return "HIGH"
    if p >= 0.25:
        return "MEDIUM"
    return "LOW"


def forecast_24h(
    state: ThreatState,
    stats_by_type: dict[str, AttackStats],
    recent_incidents: Iterable[Incident] = (),
    now: datetime | None = None,
) -> Forecast:
    """Produce a single 24h forecast for Kyiv.

    `state` = current ThreatState from classifier
    `stats_by_type` = dict from analytics.compute_attack_stats
    `recent_incidents` = all incidents from last 7 days (for "recently happened" boosts)
    """
    now = now or datetime.now(tz=timezone.utc)
    factors: list[Factor] = []

    # --- Base rate from per-type stockpile readiness ---
    # We take the MAXIMUM base across types, since the question is
    # "any major attack in next 24h", not "of a specific type".
    base_components: list[Factor] = []
    for t, s in stats_by_type.items():
        if s.days_since_last is None:
            continue
        recovery = STOCKPILE_RECOVERY_DAYS.get(t, 14.0)
        # Logistic-like: probability that today is a "stock ready" day.
        # If days_since_last > recovery → ready (boost), else penalize.
        readiness = s.stockpile_readiness
        # Daily base rate ≈ 1/typical_interval, scaled by readiness
        daily_base = readiness / max(recovery, 1.0)
        if daily_base > 0.01:
            base_components.append(
                Factor(
                    name=f"base_{t}",
                    delta=daily_base,
                    explanation=(
                        f"{t}: {int(s.days_since_last)}д від останнього "
                        f"(норма {int(recovery)}д, готовність {int(readiness*100)}%)"
                    ),
                )
            )

    # Use the largest single base contribution + half of the second largest.
    # This avoids stacking multiple "old type" readiness values incorrectly.
    base_components.sort(key=lambda f: f.delta, reverse=True)
    base = 0.0
    if base_components:
        base += base_components[0].delta
        if len(base_components) > 1:
            base += base_components[1].delta * 0.5
        # Re-scale: a "fully ready everything" day shouldn't exceed ~0.4 base
        base = min(base * 6.0, 0.45)
        factors.extend(base_components[:3])

    p = base

    # --- Boost from currently active preparation flags ---
    active = set(state.active_flags.keys())
    for flag, boost in ACTIVE_FLAG_BOOSTS.items():
        if flag in active:
            p += boost
            factors.append(Factor(
                name=f"active_{flag}",
                delta=boost,
                explanation=_flag_explanation(flag),
            ))

    # --- Already-elevated current state ---
    if state.is_combined_attack:
        p += 0.20
        factors.append(Factor(
            name="combined_attack_signature",
            delta=0.20,
            explanation="Активні ≥2 індикаторів комбінованої атаки",
        ))
    if state.level >= 3:
        p += 0.20
        factors.append(Factor(
            name=f"current_level_{state.level}",
            delta=0.20,
            explanation=f"Поточний рівень {state.level}/5 — атака триває або щойно була",
        ))

    # --- Recently happened: reduce probability of immediate repeat ---
    recent = list(recent_incidents)
    for inc in recent:
        age_h = (now - inc.started_at).total_seconds() / 3600
        if age_h < 0:
            continue
        if inc.type == "shahed_mass" and age_h < 24:
            d = -0.20
            p += d
            factors.append(Factor(
                name="recent_shahed_mass",
                delta=d,
                explanation=f"Масовані Шахеди {int(age_h)}г тому — наступна хвиля частіше через 2-3 доби",
            ))
            break
    for inc in recent:
        age_h = (now - inc.started_at).total_seconds() / 3600
        if inc.type == "cruise_kyiv" and age_h < 72:
            d = -0.20
            p += d
            factors.append(Factor(
                name="recent_cruise",
                delta=d,
                explanation=f"Калібри по Києву {int(age_h)}г тому — найближчий повтор зазвичай через 2-3 тижні",
            ))
            break
    for inc in recent:
        age_h = (now - inc.started_at).total_seconds() / 3600
        if inc.type == "kinzhal" and age_h < 24:
            d = -0.15
            p += d
            factors.append(Factor(
                name="recent_kinzhal",
                delta=d,
                explanation=f"Кинджал {int(age_h)}г тому — повтор зазвичай ≥2 тижні",
            ))
            break

    # --- Weekday prior ---
    try:
        from zoneinfo import ZoneInfo
        now_kyiv = now.astimezone(ZoneInfo("Europe/Kyiv"))
    except Exception:  # noqa: BLE001
        now_kyiv = now
    weekday = now_kyiv.weekday()
    if weekday in WEEKDAY_PRIORS:
        d = WEEKDAY_PRIORS[weekday]
        if d != 0.0:
            p += d
            factors.append(Factor(
                name=f"weekday_{weekday}",
                delta=d,
                explanation=f"{['Пн','Вт','Ср','Чт','Пт','Сб','Нд'][weekday]} — статистично "
                            f"{'активний' if d > 0 else 'тихий'} день",
            ))

    # --- Anniversary boost ---
    key = (now_kyiv.month, now_kyiv.day)
    if key in ANNIVERSARY_DATES:
        d = 0.15
        p += d
        factors.append(Factor(
            name=f"anniversary_{key[0]}_{key[1]}",
            delta=d,
            explanation=ANNIVERSARY_DATES[key],
        ))

    # --- Clamp ---
    p = max(0.0, min(0.95, p))
    tier = _tier_for(p)
    return Forecast(
        probability=p,
        tier=tier,
        tier_label=TIER_LABELS[tier],
        factors=factors,
        computed_at=now,
    )


def _flag_explanation(flag: str) -> str:
    return {
        "mig31_takeoff": "Зліт МіГ-31К — ризик пуску Кинджалу 30-90 хв",
        "tu95_takeoff": "Зліт Ту-95/Ту-160 — крилаті ракети у вікні 3-6 год",
        "tu95_carrier_movement": "Підготовка стратбомбера на аеродромі — старт за години",
        "naval_carrier_movement": "Ракетоносії виведені у море — Калібри готові",
        "ru_strategic_aviation_active": "Стратегічна авіація РФ активна",
        "shahed_launch_detected": "Зафіксовані пуски Шахедів — підльот за 1-3 год",
        "official_warning_attack": "Офіційне попередження про можливу атаку",
        "presidential_statement_attack": "Заява керівництва — публічна готовність",
        "country_wide_missile_alert": "Ракетна небезпека по країні",
    }.get(flag, flag)


# ---------- Pretty-print forecast for Telegram ----------


def render_forecast_text(forecast: Forecast, stats_by_type: dict[str, AttackStats]) -> str:
    """Format the forecast as an HTML-friendly Telegram message."""
    lines = [
        f"<b>🎯 Прогноз 24h для Києва: {forecast.tier_label} ({forecast.percent}%)</b>\n",
    ]

    # Top driving factors
    top = forecast.top_factors(5)
    if top:
        lines.append("<b>Ключові фактори:</b>")
        for f in top:
            lines.append(f"  • {f.explanation} ({f.delta:+.2f})")
        lines.append("")

    # Per-type readiness summary
    relevant_types = ["shahed_mass", "cruise_kyiv", "kinzhal", "ballistic_kyiv"]
    type_labels = {
        "shahed_mass": "Шахеди (масовані)",
        "cruise_kyiv": "Калібри/Х-101 на Київ",
        "kinzhal": "Кинджал",
        "ballistic_kyiv": "Балістика на Київ",
    }
    has_any = False
    for t in relevant_types:
        s = stats_by_type.get(t)
        if not s or s.days_since_last is None:
            continue
        if not has_any:
            lines.append("<b>Тренди по типах:</b>")
            has_any = True
        recovery = STOCKPILE_RECOVERY_DAYS.get(t, 14.0)
        readiness_bar = "█" * int(s.stockpile_readiness * 10)
        lines.append(
            f"  • {type_labels[t]}: <code>{int(s.days_since_last)}д</code> від останнього "
            f"(норма {int(recovery)}д)  {readiness_bar}"
        )

    return "\n".join(lines)


__all__ = [
    "ACTIVE_FLAG_BOOSTS",
    "ANNIVERSARY_DATES",
    "Factor",
    "Forecast",
    "TIER_LABELS",
    "WEEKDAY_PRIORS",
    "forecast_24h",
    "render_forecast_text",
]
