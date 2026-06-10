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
#
# These values were tuned against 185 days of historical replay on a corpus
# of 19,451 messages from 4 channels (Nov 2025 - Jun 2026). Tuning targets:
#   • Brier Skill Score > 0  (model must beat coin-flip)
#   • VERY_HIGH tier  >80% actual-attack rate
#   • LOW tier        <30% actual-serious-attack rate
#
# Methodology: open _test_forecast_validation.py and re-run after edits.

# Base rate floor — in a wartime regime with daily Shahed activity, the
# empirical probability of a SERIOUS attack on Kyiv tomorrow is ~55-65%
# (measured on 6 months of history: 67.6% of days had a medium+ incident).
# Anchoring at 0.55 means a "normal day with no signals" reports just
# above 50/50, which honestly reflects the reality of 2026 wartime.
BASE_RATE_DAILY: float = 0.55

# Boost contributions when these preparation flags are CURRENTLY active.
# Validation showed that individual flag boosts of 0.20-0.30 caused
# overconfidence at the top of the scale (predicting 85-90% on days when
# actual rate was only 76-77%). Coefficients here are about 30% smaller
# than initial estimates, calibrated against 185 days of replay.
ACTIVE_FLAG_BOOSTS: dict[str, float] = {
    "mig31_takeoff":           0.20,
    "tu95_takeoff":            0.20,
    "tu95_carrier_movement":   0.12,
    "naval_carrier_movement":  0.15,
    "ru_strategic_aviation_active": 0.12,
    "shahed_launch_detected":  0.10,
    "official_warning_attack": 0.20,
    # Zelenskyy speaks about strikes most days — barely predictive.
    "presidential_statement_attack": 0.02,
    "country_wide_missile_alert":   0.10,
}
# Maximum contribution from the *sum* of preparation flags.
ACTIVE_FLAG_BOOSTS_CAP: float = 0.25

# Recent-strike penalties REMOVED in v3.
#
# Theory was: after a major Kalibr/Kinzhal raid, stockpile depletion means
# next 24-72h is safer. In reality, validation showed this was a major
# source of false negatives — days AFTER a Kalibr raid often see further
# Shahed/ballistic attacks, just not necessarily of the same type. The
# penalty was effectively killing predictions for the wrong reason.
#
# Stockpile cycles are real over WEEKS but not at 24-72h granularity.
# We instead reflect them in stockpile_readiness (above) which gives a
# positive boost when 2+ weeks have passed since last cruise/kinzhal.
RECENT_INCIDENT_BOOSTS: dict[str, float] = {}

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
    """Tier thresholds calibrated for 2026 wartime base rate (~67% daily).

    Original "peacetime" tiers (LOW <25%, MEDIUM 25-50%, HIGH 50-75%,
    VERY_HIGH >75%) collapsed everything into HIGH/VERY_HIGH because no
    day has < 25% prob during active war. We rebalance:
      LOW         < 0.45  — truly quiet day (rare; <10% of days)
      MEDIUM      0.45-0.60  — normal wartime "background expectation"
      HIGH        0.60-0.78  — elevated, preparation signals or recent
                              activity raising the bar
      VERY_HIGH   ≥ 0.78  — strong combination of signals; treat as warning
    """
    if p >= 0.78:
        return "VERY_HIGH"
    if p >= 0.60:
        return "HIGH"
    if p >= 0.45:
        return "MEDIUM"
    return "LOW"


def forecast_24h(
    state: ThreatState,
    stats_by_type: dict[str, AttackStats],
    recent_incidents: Iterable[Incident] = (),
    now: datetime | None = None,
) -> Forecast:
    """Produce a single 24h forecast for Kyiv.

    Model (in order of application):
      1. Anchor at BASE_RATE_DAILY (~30%) — empirical prior for "serious
         attack on Kyiv tomorrow", measured on 6 months of history.
      2. Add stockpile-readiness boost ONLY for rare types (cruise / kinzhal /
         ballistic). Shahed readiness is meaningless (always ready), so it
         contributes nothing — saves us from inflating every day to high.
      3. Add boosts from currently active preparation flags (capped sum 0.40).
      4. Add boost for current threat level >= 3.
      5. Subtract recent-strike penalty for rare-cycle types.
      6. Tiny weekday / anniversary adjustments.
      7. Clamp to [0.05, 0.95].
    """
    now = now or datetime.now(tz=timezone.utc)
    factors: list[Factor] = []

    # --- 1. Anchor at empirical base rate ---
    p = BASE_RATE_DAILY
    factors.append(Factor(
        name="base_rate_daily",
        delta=BASE_RATE_DAILY,
        explanation=f"Базовий рівень {int(BASE_RATE_DAILY*100)}% (історія 6 міс)",
    ))

    # --- 2. Stockpile readiness for RARE types only ---
    # Shahed mass is constant background noise → no contribution
    # Cruise / Kinzhal / Ballistic have real production cycles
    for t in ("cruise_kyiv", "kinzhal", "ballistic_kyiv", "kab_kyiv"):
        s = stats_by_type.get(t)
        if s is None or s.days_since_last is None:
            continue
        recovery = STOCKPILE_RECOVERY_DAYS.get(t, 14.0)
        # Bonus proportional to readiness, but capped per type
        if s.stockpile_readiness >= 0.7:
            delta = 0.06 * s.stockpile_readiness
            p += delta
            factors.append(Factor(
                name=f"stockpile_{t}",
                delta=delta,
                explanation=(
                    f"{_type_label(t)}: {int(s.days_since_last)}д від останнього "
                    f"(норма {int(recovery)}д) — готові"
                ),
            ))

    # --- 3. Active preparation flags ---
    active_flags_delta = 0.0
    active = set(state.active_flags.keys())
    for flag, boost in ACTIVE_FLAG_BOOSTS.items():
        if flag in active:
            active_flags_delta += boost
    # Cap the *sum* of preparation flag boosts.
    if active_flags_delta > ACTIVE_FLAG_BOOSTS_CAP:
        active_flags_delta = ACTIVE_FLAG_BOOSTS_CAP
    if active_flags_delta > 0:
        # Show each contributing flag individually for transparency
        for flag, boost in ACTIVE_FLAG_BOOSTS.items():
            if flag in active:
                factors.append(Factor(
                    name=f"active_{flag}",
                    delta=boost,
                    explanation=_flag_explanation(flag),
                ))
        p += active_flags_delta

    # --- 4. Current threat level is a *weak* predictor of next 24h ---
    # Reduced from earlier versions: current level tells us about NOW, not
    # about tomorrow. A live L3 alarm at 09:00 (e.g. residual Shahed activity
    # from the night) often doesn't continue into the next 24h window.
    if state.level >= 4:
        d = 0.15
        p += d
        factors.append(Factor(
            name=f"current_level_{state.level}",
            delta=d,
            explanation=f"Поточний рівень {state.level}/5 — критична загроза прямо зараз",
        ))
    elif state.level >= 3:
        d = 0.07
        p += d
        factors.append(Factor(
            name=f"current_level_{state.level}",
            delta=d,
            explanation=f"Поточний рівень {state.level}/5 — атака триває або щойно була",
        ))

    if state.is_combined_attack:
        d = 0.08
        p += d
        factors.append(Factor(
            name="combined_attack_signature",
            delta=d,
            explanation="Активні ≥2 індикаторів комбінованої атаки",
        ))

    # --- 5. (REMOVED in v3) Recent-strike penalties caused more false
    # negatives than they prevented false positives. Stockpile-cycle effects
    # are now captured only by positive stockpile_readiness bonuses above. ---

    # --- 6. Weekday / anniversary ---
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
                explanation=f"{['Пн','Вт','Ср','Чт','Пт','Сб','Нд'][weekday]} — "
                            f"{'активний' if d > 0 else 'тихий'} день",
            ))
    key = (now_kyiv.month, now_kyiv.day)
    if key in ANNIVERSARY_DATES:
        d = 0.12
        p += d
        factors.append(Factor(
            name=f"anniversary_{key[0]}_{key[1]}",
            delta=d,
            explanation=ANNIVERSARY_DATES[key],
        ))

    # --- 7. Clamp ---
    # Tight clamp: in active war on Kyiv, no day is genuinely <30% prob
    # (Shahed background is constant), and the top of the model is
    # overconfident relative to actuals — cap at 0.85 to keep tier
    # discrimination honest.
    p = max(0.30, min(0.85, p))
    tier = _tier_for(p)
    return Forecast(
        probability=p,
        tier=tier,
        tier_label=TIER_LABELS[tier],
        factors=factors,
        computed_at=now,
    )


def _type_label(t: str) -> str:
    return {
        "shahed_mass": "Шахеди (масовані)",
        "shahed_kyiv": "Шахеди на Київ",
        "cruise_kyiv": "Калібри/Х-101",
        "ballistic_kyiv": "Балістика на Київ",
        "kinzhal": "Кинджал",
        "kab_kyiv": "КАБи",
    }.get(t, t)


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
