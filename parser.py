"""HTML scraping + threat-lexicon tagging for Telegram public channels.

Two responsibilities:
1. fetch_messages(channel) -> list[ParsedMessage]
   reads https://t.me/s/<channel> and parses the public web preview HTML.
2. tag_text(text) -> set[str]
   matches Ukrainian threat-lexicon regex patterns and returns a set of tags.

Tags are stable string keys (e.g. "mig31_takeoff") that the classifier consumes.
The lexicon below was derived empirically from ~600 historical messages of
@kpszsu, @war_monitor, @kyiv_alarm in May 2026, covering quiet days and the
combined attacks of 17, 23, 24 May.
"""

from __future__ import annotations

import html as ihtml
import logging
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone

log = logging.getLogger("parser")

USER_AGENT = "Mozilla/5.0 (compatible; air-threat-bot/1.0)"
DEFAULT_TIMEOUT = 15


# ---------- Threat lexicon (Ukrainian) ----------
# Tag => list of regex patterns. Patterns are case-insensitive and matched
# against cleaned text (no HTML, collapsed whitespace).

LEXICON: dict[str, list[str]] = {
    # --- Strategic aviation (leading indicators of cruise/Kinzhal strike) ---
    "mig31_takeoff": [
        r"зліт\s+мі[гг]-?31\s*к",
        r"мі[гг]-?31\s*к.{0,20}злет",
        r"зліт.{0,15}мі[гг]\s*31",
        r"\d+\s*борт\w*\s+мі[гг]-?31",
        r"мі[гг]-?31\s*к.{0,20}у\s+повітр",
        r"\bмі[гг]-?31\s*к\s*$",  # bare "МіГ-31К" mention as a one-line update
    ],
    "tu95_takeoff": [
        r"зліт\s+ту-?95",
        r"зліт\s+ту-?160",
        r"зліт\s+ту-?22",
        r"ту-?95.{0,15}злет",
        r"\d+\s*борт\w*\s+ту-?(95|160|22)",
        r"ту-?(95|160|22).{0,20}у\s+повітр",
    ],
    "ru_strategic_aviation_active": [
        r"стратегічн.{1,10}авіац.{1,15}актив",
        r"ракетоносі[їі].{1,20}(виведен|у\s+мор)",
    ],
    "ru_strategic_aviation_inactive": [
        r"стратегічн.{1,10}авіац.{1,15}не\s+актив",
        r"ракетоносі[їі].{1,20}заведен[іиі].{1,20}порт",
        # Explicit "all-clear for MiG-31K / Tu-95" from @kpszsu and @war_monitor
        r"відбій\s+загроз\w*\s+(?:по\s+)?мі[гг]-?31",
        r"відбій\s+загроз\w*\s+(?:по\s+)?ту-?(?:95|160|22)",
        r"посадк\w*\s+мі[гг]-?31",
        r"посадк\w*\s+ту-?(?:95|160|22)",
        r"мі[гг]-?31.{0,15}поверн\w*",
        r"ту-?(?:95|160|22).{0,15}поверн\w*",
    ],

    # --- Cruise missiles ---
    "cruise_missile_active": [
        r"\bкр\s+калібр",
        r"\bкалібр\w*\b",
        r"\bх-?10[15]\b",
        r"\bх-?555\b",
        r"крилат\w*\s+ракет",
        r"група\s+кр\b",
    ],
    "cruise_missile_kyiv": [
        r"кр\s+калібр.{0,40}(київ|переяслав|білоцерків|бориспіль|вишгород)",
        r"калібр.{0,40}на\s+київ",
        r"група\s+кр.{0,30}на\s+київ",
        r"крилат\w*.{0,30}(київщин|на\s+київ)",
    ],

    # --- Hypersonic ---
    "hypersonic_kinzhal": [
        r"\bкинджал\w*\b",
        r"\bх-?47\b",
    ],
    "hypersonic_tsirkon": [
        r"\bциркон\w*\b",
    ],

    # --- Ballistic ---
    "ballistic_threat": [
        r"загроза\s+балісти(к|чн)",
        r"\bіскандер\w*\b",
        r"\bбр\b.{0,30}(брянськ|воронеж|курськ|крим)",
        r"\bбр\s+на\s+\w+",
        r"виход\w*\s+бр",
        r"балістична?\s+загроз",
    ],
    "ballistic_threat_from_north": [
        r"балісти.{0,30}з\s+(брянськ|воронеж|курськ)",
        r"(брянськ|воронеж|курськ).{0,30}балісти",
    ],
    "ballistic_descent": [
        r"спуск\s+балісти(к|чн)",
        r"\bспуск\s+бр\b",
    ],
    "ballistic_descent_kyiv": [
        r"київ.{0,20}спуск\s+балісти",
        r"спуск\s+балісти.{0,30}київ",
    ],

    # --- UAVs (Shahed / Geran-2) ---
    "shahed_active": [
        r"\bбпла\w*\b",
        r"\bшахед\w*\b",
        r"\bгеран\w*\b",
        r"ударн\w*\s+бпла",
    ],
    "shahed_kyiv": [
        r"бпла.{0,30}(на\s+київ|[вуy]\s+напрямку\s+києва|київщин)",
        r"\d+\s*х?\s*бпла.{0,30}(на\s+київ|київщин)",
        r"шахед.{0,30}(на\s+київ|київщин|[вуy]\s+напрямку\s+києва)",
        r"бпла.{0,30}через\s+славутич",
        r"бпла.{0,30}на\s+(вишгород|бориспіль|бровар|білоцеркі|переяслав|васильків)",
    ],
    "shahed_mass": [
        # >=10 БпЛА mentioned in a swarm — proxy for high intensity
        r"\b(1[0-9]|[2-9]\d|\d{3,})\s*х?\s*бпла\b",
        r"\b(1[0-9]|[2-9]\d|\d{3,})\s*х?\s*шахед",
        r"масован\w+\s+атак\w+\s+бпла",
    ],

    # --- KAB (guided bombs) ---
    "kab_active": [
        r"\bкаб\w*\b",
        r"авіабомб",
    ],

    # --- Explosions / impacts ---
    "explosions_anywhere": [
        r"💥",
        r"вибух\w+\s+(лунал|у\s+міст|в\s+\w+)",
        r"\bвибух\w+\b",
    ],
    "explosions_kyiv": [
        r"вибух\w+.{0,40}(у\s+києві|в\s+києві|по\s+києву|київ)",
        r"(у\s+києві|в\s+києві).{0,30}вибух",
    ],

    # --- Air-raid alarm for Kyiv (from @kyiv_alarm exact format) ---
    "kyiv_alarm_active": [
        r"тривога[!\s].{0,20}м\.\s*київ",
        r"повітряна\s+тривога.{0,20}київ",
    ],
    "kyiv_alarm_ballistic": [
        r"тривога[!\s].{0,30}київ.{0,20}балісти",
        r"тривога[!\s].{0,30}київ.{0,30}ракетн",
    ],
    "kyiv_alarm_shahed": [
        r"тривога[!\s].{0,30}київ.{0,30}шахед",
        r"тривога[!\s].{0,30}київ.{0,30}бпла",
        r"тривога[!\s].{0,30}київ.{0,30}\bкаб",
    ],

    # --- All clear / cancellation ---
    "all_clear": [
        r"⚪️\s*відбій",
        r"\bвідбій\s+загроз",
        r"\bвідбій\s+тривог",
        r"відбій\s+(балісти|повітря)",
    ],

    # --- Country-wide alerts (relevant context but not Kyiv-specific) ---
    "country_wide_missile_alert": [
        r"вся\s+україна.{0,30}ракетн\w*\s+небезпек",
        r"увага.{0,20}ракетн\w*\s+небезпек",
    ],

    # --- Situation brief (📡 from @war_monitor regular updates) ---
    "situation_brief": [
        r"📡\s*обстановк",
        r"обстановка\s+станом\s+на",
    ],
}

# Compile patterns once
COMPILED: dict[str, list[re.Pattern[str]]] = {
    tag: [re.compile(p, re.IGNORECASE | re.UNICODE) for p in patterns]
    for tag, patterns in LEXICON.items()
}


@dataclass
class ParsedMessage:
    channel: str
    msg_id: int
    ts: datetime  # UTC
    text: str
    tags: set[str] = field(default_factory=set)

    @property
    def post_path(self) -> str:
        return f"{self.channel}/{self.msg_id}"

    @property
    def url(self) -> str:
        return f"https://t.me/{self.channel}/{self.msg_id}"


# ---------- HTML fetching ----------


def fetch_html(channel: str, before: int | None = None, timeout: int = DEFAULT_TIMEOUT) -> str:
    """Fetch https://t.me/s/<channel> public web preview HTML."""
    url = f"https://t.me/s/{channel}"
    if before:
        url += f"?before={before}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        # Detect redirect to /<channel> (preview disabled / private channel)
        final = resp.geturl()
        if "/s/" not in final:
            raise PreviewDisabledError(channel)
        return resp.read().decode("utf-8", errors="replace")


class PreviewDisabledError(Exception):
    def __init__(self, channel: str) -> None:
        super().__init__(
            f"@{channel}: web preview disabled or channel is private (cannot read via HTML mirror)"
        )
        self.channel = channel


# ---------- HTML parsing ----------

_MSG_SPLIT = re.compile(r'<div class="tgme_widget_message_wrap')
_POST_RE = re.compile(r'data-post="([^"]+)"')
_TIME_RE = re.compile(r'datetime="([^"]+)"')
_BODY_BOUNDED_RE = re.compile(
    r'<div class="tgme_widget_message_text[^"]*"[^>]*>(.*?)</div>\s*'
    r'<div class="tgme_widget_message_(?:footer|reply|info)',
    re.S,
)
_BODY_OPEN_RE = re.compile(
    r'<div class="tgme_widget_message_text[^"]*"[^>]*>(.*)',
    re.S,
)


def _strip_html(raw: str) -> str:
    s = ihtml.unescape(re.sub(r"<br\s*/?>", "\n", raw))
    s = re.sub(r"<[^>]+>", "", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    # Trim trailing Telegram reaction tail: " 1🙏1.5K🤬944❤23 ..."
    s = re.sub(
        r"(\s+(?:[\d.]+K?))?(\s*(?:🙏|🤬|😭|❤|🔥|😢|😨|👍|👎|💔|⚡|🫡|🤯|👏|💯|🥺|😡)\s*[\d.]+K?){2,}\s*$",
        "",
        s,
    )
    # Trim Telegram preview footer: "Please open Telegram to view this post VIEW IN TELEGRAM"
    s = re.sub(
        r"\s*Please open Telegram to view this post\s*VIEW IN TELEGRAM\s*$",
        "",
        s,
        flags=re.IGNORECASE,
    )
    return s.strip()


def parse_html(channel: str, html_text: str) -> list[ParsedMessage]:
    """Parse the HTML page into ordered ParsedMessage list (oldest first)."""
    parts = _MSG_SPLIT.split(html_text)[1:]
    out: list[ParsedMessage] = []
    for blk in parts:
        post = _POST_RE.search(blk)
        dt = _TIME_RE.search(blk)
        if not post or not dt:
            continue
        try:
            ch, mid_str = post.group(1).split("/", 1)
            msg_id = int(mid_str)
        except (ValueError, AttributeError):
            continue
        # Force-match to expected channel (defensive)
        if ch.lower() != channel.lower():
            continue
        try:
            ts = datetime.fromisoformat(dt.group(1).replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            else:
                ts = ts.astimezone(timezone.utc)
        except ValueError:
            continue

        body_m = _BODY_BOUNDED_RE.search(blk) or _BODY_OPEN_RE.search(blk)
        raw_body = body_m.group(1) if body_m else ""
        text = _strip_html(raw_body[:8000])
        out.append(ParsedMessage(channel=channel, msg_id=msg_id, ts=ts, text=text))
    # Web preview lists oldest first; keep that order for deterministic ingestion
    out.sort(key=lambda m: m.msg_id)
    return out


# ---------- Tag extraction ----------


def tag_text(text: str) -> set[str]:
    """Apply lexicon regex to a cleaned message; return matching tag keys."""
    if not text:
        return set()
    tags: set[str] = set()
    for tag, patterns in COMPILED.items():
        for pat in patterns:
            if pat.search(text):
                tags.add(tag)
                break
    # Implied parent tags (so the classifier can rely on a simpler vocabulary)
    if "ballistic_descent_kyiv" in tags:
        tags.add("ballistic_descent")
    if "explosions_kyiv" in tags:
        tags.add("explosions_anywhere")
    if "shahed_kyiv" in tags or "shahed_mass" in tags:
        tags.add("shahed_active")
    if "cruise_missile_kyiv" in tags:
        tags.add("cruise_missile_active")
    if "ballistic_threat_from_north" in tags:
        tags.add("ballistic_threat")
    if "kyiv_alarm_ballistic" in tags or "kyiv_alarm_shahed" in tags:
        tags.add("kyiv_alarm_active")
    return tags


def fetch_messages(channel: str) -> list[ParsedMessage]:
    """High-level: fetch latest page and tag each message."""
    html_text = fetch_html(channel)
    msgs = parse_html(channel, html_text)
    for m in msgs:
        m.tags = tag_text(m.text)
    return msgs


__all__ = [
    "LEXICON",
    "ParsedMessage",
    "PreviewDisabledError",
    "fetch_html",
    "fetch_messages",
    "parse_html",
    "tag_text",
]
