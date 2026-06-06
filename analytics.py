"""Historical-pattern analytics for Kyiv air-threat events.

This module is the analytic backbone of the bot:
  1. detect_incidents() — cluster raw tagged messages into incident-level
     attack records (type / intensity / time window).
  2. attack_stats()    — for each weapon type, compute days_since_last,
     mean_interval, and stockpile_readiness.
  3. context_factors() — derive day-of-week and time-of-day priors.

Outputs feed `forecast.py` which produces the 24h probability estimate.

The expert parameters (STOCKPILE_RECOVERY_DAYS, TYPICAL_INTERVAL_DAYS) are
**hardcoded constants** sourced from publicly available open-source
analysis of Russian missile/UAV employment patterns (ISW, Defense Express,
Samus/Romanenko/Musiyenko public commentary). They are not real
intelligence — they are an open-source informed best estimate.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Iterable, Sequence

from parser import ParsedMessage

log = logging.getLogger("analytics")


# ---------- Expert stockpile-recovery parameters ----------
#
# Source notes (public, open-source assessments — NOT classified data):
# - Kh-101/Kh-555 (Tu-95/Tu-160 cruise missiles): Russian production estimated
#   at ~30-40 missiles/month (Defense Express, 2025). Major raids 2-4 weeks
#   apart, accumulating ~50-100 missiles per major strike.
# - Iskander/Kalibr: regional ballistic + cruise. More frequent, ~daily strikes
#   somewhere in Ukraine, but Kyiv-targeted Kalibr strikes typically come every
#   2-3 weeks.
# - Kinzhal (Kh-47M2): low-rate production, ~5-15 missiles per major strike,
#   2-4 weeks between massed Kinzhal raids.
# - Shahed/Geran-2 industrial production has scaled to ~3000+ per month
#   (autumn 2025 onward). Mass swarms (>100 drones) every 2-3 days.
# - KAB (guided bombs) on Kyiv region: extremely rare due to range; main
#   targets are frontline oblasts.

STOCKPILE_RECOVERY_DAYS: dict[str, float] = {
    "shahed_mass": 2.5,       # mass UAV waves resume after ~2.5d cooldown
    "cruise_kyiv": 21.0,      # Kalibr/Kh-101 on Kyiv every ~21 days median
    "kinzhal": 14.0,
    "ballistic_kyiv": 18.0,   # Iskander on Kyiv specifically (ballistic from N)
    "kab_kyiv": 60.0,         # very rare for Kyiv
    "combined_attack": 14.0,
}

# Plausible typical interval for the SAME type to recur in a 30-day window.
# Used as the denominator in base_rate computation.
TYPICAL_INTERVAL_DAYS: dict[str, float] = STOCKPILE_RECOVERY_DAYS.copy()


# ---------- Incident detection ----------

# Mapping from tag set → incident type. Order matters (first match wins).
INCIDENT_TYPE_RULES: list[tuple[str, frozenset[str]]] = [
    ("ballistic_descent_kyiv", frozenset({"ballistic_descent_kyiv"})),
    ("explosions_kyiv",       frozenset({"explosions_kyiv"})),
    ("cruise_kyiv",           frozenset({"cruise_missile_kyiv"})),
    ("ballistic_kyiv",        frozenset({"ballistic_threat_from_north"})),
    ("ballistic_kyiv",        frozenset({"kyiv_alarm_ballistic"})),
    ("kinzhal",               frozenset({"hypersonic_kinzhal"})),
    ("kab_kyiv",              frozenset({"kab_kyiv"})),
    ("shahed_mass",           frozenset({"shahed_mass"})),
    ("shahed_kyiv",           frozenset({"shahed_kyiv"})),
    ("shahed_kyiv",           frozenset({"kyiv_alarm_shahed"})),
]

# Tags strictly required for an incident to be considered "Kyiv-direction"
KYIV_INCIDENT_TAGS: frozenset[str] = frozenset({
    "ballistic_descent_kyiv", "explosions_kyiv", "cruise_missile_kyiv",
    "ballistic_threat_from_north", "kyiv_alarm_ballistic", "kyiv_alarm_shahed",
    "shahed_kyiv", "kab_kyiv", "shahed_mass",
})


@dataclass
class Incident:
    type: str
    started_at: datetime
    ended_at: datetime
    intensity: str
    msg_count: int
    sources: list[str] = field(default_factory=list)
    first_msg_post: str | None = None
    notes: str | None = None


def _msg_to_incident_type(tags: set[str]) -> str | None:
    """Pick the most specific incident type for a message's tags."""
    for incident_type, required in INCIDENT_TYPE_RULES:
        if required.issubset(tags):
            return incident_type
    return None


def detect_incidents(
    messages: Sequence[ParsedMessage],
    cluster_window_min: int = 30,
) -> list[Incident]:
    """Cluster messages into incidents.

    Algorithm:
      1. Sort messages chronologically.
      2. For each message that matches a Kyiv-direction tag, determine its
         "incident_type" (cruise_kyiv / ballistic_kyiv / shahed_mass / ...).
      3. Within each type, merge messages that are within `cluster_window_min`
         of each other into one incident.
      4. Intensity is derived from message count + presence of mass markers.

    The cluster window is deliberately wide (default 30 min) so that bursts of
    related posts (e.g. 6 "Спуск балістики" messages in 5 min) collapse into
    one incident, not six.
    """
    relevant = [
        m for m in messages
        if m.tags & KYIV_INCIDENT_TAGS and m.text
    ]
    relevant.sort(key=lambda m: m.ts)

    by_type: dict[str, list[ParsedMessage]] = defaultdict(list)
    for m in relevant:
        t = _msg_to_incident_type(m.tags)
        if t:
            by_type[t].append(m)

    incidents: list[Incident] = []
    for incident_type, msgs in by_type.items():
        # Greedy clustering
        cluster: list[ParsedMessage] = []
        for m in msgs:
            if not cluster:
                cluster.append(m)
                continue
            if (m.ts - cluster[-1].ts) <= timedelta(minutes=cluster_window_min):
                cluster.append(m)
            else:
                incidents.append(_cluster_to_incident(incident_type, cluster))
                cluster = [m]
        if cluster:
            incidents.append(_cluster_to_incident(incident_type, cluster))

    incidents.sort(key=lambda i: i.started_at)
    return incidents


def _cluster_to_incident(incident_type: str, cluster: list[ParsedMessage]) -> Incident:
    intensity = _classify_intensity(cluster)
    sources = sorted({m.channel for m in cluster})
    return Incident(
        type=incident_type,
        started_at=cluster[0].ts,
        ended_at=cluster[-1].ts,
        intensity=intensity,
        msg_count=len(cluster),
        sources=sources,
        first_msg_post=cluster[0].post_path,
        notes=cluster[0].text[:120] if cluster[0].text else None,
    )


def _classify_intensity(cluster: list[ParsedMessage]) -> str:
    """Heuristic — LOW / MEDIUM / HIGH / MASSIVE based on volume and key tags."""
    n = len(cluster)
    duration_min = max(1, (cluster[-1].ts - cluster[0].ts).total_seconds() // 60)
    has_mass = any("shahed_mass" in m.tags for m in cluster)
    has_descent = any("ballistic_descent_kyiv" in m.tags or "ballistic_descent" in m.tags for m in cluster)
    has_explosions = any("explosions_kyiv" in m.tags for m in cluster)

    if has_explosions or (has_descent and n >= 3):
        return "massive"
    if has_mass or n >= 10:
        return "high"
    if n >= 4 or duration_min >= 20:
        return "medium"
    return "low"


# ---------- Aggregate statistics ----------

@dataclass
class AttackStats:
    type: str
    last_at: datetime | None
    days_since_last: float | None
    count_7d: int
    count_30d: int
    count_90d: int
    mean_interval_30d: float | None  # days
    stockpile_readiness: float  # 0..1


def compute_attack_stats(incidents: Sequence[Incident], now: datetime | None = None) -> dict[str, AttackStats]:
    """For each known type, compute days_since_last, rolling counts, readiness."""
    now = now or datetime.now(tz=timezone.utc)
    by_type: dict[str, list[Incident]] = defaultdict(list)
    for inc in incidents:
        by_type[inc.type].append(inc)

    out: dict[str, AttackStats] = {}
    for t in set(list(STOCKPILE_RECOVERY_DAYS.keys()) + list(by_type.keys())):
        events = sorted(by_type.get(t, []), key=lambda i: i.started_at)
        if not events:
            out[t] = AttackStats(
                type=t,
                last_at=None,
                days_since_last=None,
                count_7d=0,
                count_30d=0,
                count_90d=0,
                mean_interval_30d=None,
                stockpile_readiness=1.0,  # never seen → assume ready
            )
            continue

        last_at = events[-1].started_at
        days_since = (now - last_at).total_seconds() / 86400.0
        count_7d = sum(1 for e in events if e.started_at >= now - timedelta(days=7))
        count_30d = sum(1 for e in events if e.started_at >= now - timedelta(days=30))
        count_90d = sum(1 for e in events if e.started_at >= now - timedelta(days=90))

        # Mean interval over last 30 days
        recent = [e for e in events if e.started_at >= now - timedelta(days=30)]
        mean_interval: float | None = None
        if len(recent) >= 2:
            gaps = [
                (recent[i].started_at - recent[i - 1].started_at).total_seconds() / 86400.0
                for i in range(1, len(recent))
            ]
            mean_interval = sum(gaps) / len(gaps)

        recovery = STOCKPILE_RECOVERY_DAYS.get(t, 14.0)
        readiness = min(1.0, max(0.0, days_since / recovery))

        out[t] = AttackStats(
            type=t,
            last_at=last_at,
            days_since_last=days_since,
            count_7d=count_7d,
            count_30d=count_30d,
            count_90d=count_90d,
            mean_interval_30d=mean_interval,
            stockpile_readiness=readiness,
        )
    return out


# ---------- Time-of-day / day-of-week context ----------

def hourly_distribution(incidents: Sequence[Incident]) -> dict[int, int]:
    """Count of incidents per hour-of-day (Kyiv local time, 0..23)."""
    from zoneinfo import ZoneInfo
    tz = ZoneInfo("Europe/Kyiv")
    c: Counter[int] = Counter()
    for inc in incidents:
        c[inc.started_at.astimezone(tz).hour] += 1
    return dict(c)


def weekday_distribution(incidents: Sequence[Incident]) -> dict[int, int]:
    """Count of incidents per weekday (0=Mon, 6=Sun) in Kyiv local time."""
    from zoneinfo import ZoneInfo
    tz = ZoneInfo("Europe/Kyiv")
    c: Counter[int] = Counter()
    for inc in incidents:
        c[inc.started_at.astimezone(tz).weekday()] += 1
    return dict(c)


__all__ = [
    "AttackStats",
    "Incident",
    "STOCKPILE_RECOVERY_DAYS",
    "TYPICAL_INTERVAL_DAYS",
    "compute_attack_stats",
    "detect_incidents",
    "hourly_distribution",
    "weekday_distribution",
]
