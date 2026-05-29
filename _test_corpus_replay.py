"""Replay the /tmp/tg_corpus/*.json snapshots through parser+classifier.

This is a manual sanity-check, not a unit test. It walks the corpus message-by-
message in chronological order, classifies the state at each step, and prints
peaks + an hour-by-hour heatmap so we can visually verify that the rules light
up during the known combined-attack windows (17, 23, 24 May 2026).
"""

from __future__ import annotations

import json
import pathlib
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

from classifier import LEVEL_NAMES, classify
from parser import ParsedMessage, tag_text


def load_corpus(channel: str) -> list[ParsedMessage]:
    # Prefer the bundled test_data/ snapshot; fall back to /tmp scratchpad
    here = pathlib.Path(__file__).resolve().parent
    for candidate in (here / "test_data" / f"{channel}.json", pathlib.Path(f"/tmp/tg_corpus/{channel}.json")):
        if candidate.exists():
            path = candidate
            break
    else:
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: list[ParsedMessage] = []
    for r in raw:
        try:
            ch, mid = r["post"].split("/", 1)
            ts = datetime.fromisoformat(r["ts"].replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            else:
                ts = ts.astimezone(timezone.utc)
            text = r["text"]
            pm = ParsedMessage(channel=ch, msg_id=int(mid), ts=ts, text=text)
            pm.tags = tag_text(text)
            out.append(pm)
        except Exception as e:  # noqa: BLE001
            print(f"  skip {r.get('post')}: {e}")
    return out


def main() -> None:
    all_msgs: list[ParsedMessage] = []
    for ch in ("kpszsu", "kyiv_alarm", "war_monitor"):
        msgs = load_corpus(ch)
        print(f"loaded @{ch}: {len(msgs)} msgs")
        all_msgs.extend(msgs)
    all_msgs.sort(key=lambda m: m.ts)
    print(f"\ntotal corpus: {len(all_msgs)} msgs  "
          f"range {all_msgs[0].ts.isoformat()} → {all_msgs[-1].ts.isoformat()}\n")

    # ---------- Tag frequency (sanity: do regex hit anything?) ----------
    tag_counts: Counter[str] = Counter()
    untagged = 0
    for m in all_msgs:
        if not m.tags and m.text:
            untagged += 1
        for t in m.tags:
            tag_counts[t] += 1
    print("=== Tag frequency ===")
    for tag, cnt in tag_counts.most_common(25):
        print(f"  {tag:35s}  {cnt}")
    print(f"\nMessages with NO tag (and non-empty text): {untagged} / {len(all_msgs)}")

    # ---------- Hour heatmap of computed level ----------
    print("\n=== Hourly max threat level (replay) ===")
    by_hour: dict[str, int] = defaultdict(int)
    level_5_moments: list[tuple[datetime, str]] = []
    level_4_moments: list[tuple[datetime, str]] = []
    combined_moments: list[tuple[datetime, list[str]]] = []
    last_combined_at: datetime | None = None

    for i, m in enumerate(all_msgs):
        window = [x for x in all_msgs[: i + 1] if x.ts >= m.ts - timedelta(hours=6)]
        state = classify(window, now=m.ts)
        key = m.ts.strftime("%Y-%m-%d %H")
        if state.level > by_hour[key]:
            by_hour[key] = state.level
        if state.level == 5:
            level_5_moments.append((m.ts, m.text[:120]))
        elif state.level == 4:
            level_4_moments.append((m.ts, m.text[:120]))
        if state.is_combined_attack and (
            last_combined_at is None or (m.ts - last_combined_at).total_seconds() > 1800
        ):
            combined_moments.append((m.ts, list(state.active_flags.keys())[:6]))
            last_combined_at = m.ts

    for hour in sorted(by_hour):
        lvl = by_hour[hour]
        bar = ["·", "▁", "▃", "▅", "▇", "█"][lvl]
        print(f"  {hour}:00  L{lvl}  {bar * (lvl + 1)}  {LEVEL_NAMES[lvl]}")

    print(f"\n=== Level 5 moments ({len(level_5_moments)}) ===")
    for ts, txt in level_5_moments[:10]:
        print(f"  {ts.isoformat()}  {txt}")
    print(f"\n=== Level 4 moments ({len(level_4_moments)}) ===")
    for ts, txt in level_4_moments[:10]:
        print(f"  {ts.isoformat()}  {txt}")
    print(f"\n=== Combined-attack triggers ({len(combined_moments)}, dedup 30min) ===")
    for ts, flags in combined_moments[:15]:
        print(f"  {ts.isoformat()}  flags={flags}")


if __name__ == "__main__":
    main()
