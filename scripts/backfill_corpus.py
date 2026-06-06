"""One-shot backfill of historical messages from each configured channel.

Fetches the public web mirror at https://t.me/s/<channel>?before=<msg_id>
and walks backwards page-by-page until either:
  - we hit MAX_PAGES (safety),
  - the oldest message we got is older than CUTOFF_DAYS, OR
  - the channel runs out of messages.

Results are written to test_data/<channel>_full.json (deduped, sorted asc).
Resumable: if the file already exists, we continue from the oldest msg_id
we already have rather than starting from the latest page.
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys
import time
from datetime import datetime, timedelta, timezone

# Make the project root importable when running as a script
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from parser import fetch_html, parse_html, tag_text  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("backfill")

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "test_data"
OUT_DIR.mkdir(exist_ok=True)


def backfill_channel(
    channel: str,
    cutoff_days: int,
    max_pages: int,
    delay_sec: float,
) -> int:
    """Backfill one channel. Returns number of new unique messages written."""
    out_path = OUT_DIR / f"{channel}_full.json"
    existing: list[dict] = []
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log.warning("Corrupt JSON for @%s, starting fresh", channel)
            existing = []
    seen_ids = {m["post"] for m in existing}
    oldest_id = (
        min(int(m["post"].split("/")[-1]) for m in existing)
        if existing
        else None
    )
    log.info(
        "@%s: starting (existing=%d, oldest_id_to_continue_from=%s)",
        channel, len(existing), oldest_id,
    )

    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=cutoff_days)
    new_messages: list[dict] = []
    before = oldest_id

    for page in range(max_pages):
        try:
            html = fetch_html(channel, before=before)
        except Exception as e:  # noqa: BLE001
            log.warning("@%s page %d fetch failed: %s — sleeping 5s", channel, page, e)
            time.sleep(5)
            continue
        msgs = parse_html(channel, html)
        if not msgs:
            log.info("@%s page %d: empty — stopping", channel, page)
            break
        # parse_html returns ascending; for backfill we care about oldest first
        new_on_page = 0
        for m in msgs:
            if m.post_path in seen_ids:
                continue
            new_messages.append({
                "post": m.post_path,
                "ts": m.ts.isoformat(),
                "text": m.text,
            })
            seen_ids.add(m.post_path)
            new_on_page += 1
        oldest_msg = msgs[0]  # ascending → [0] is oldest
        log.info(
            "@%s page %d: +%d new (msgs %d-%d, oldest_ts=%s)",
            channel, page + 1, new_on_page,
            msgs[0].msg_id, msgs[-1].msg_id,
            oldest_msg.ts.isoformat()[:16],
        )
        if oldest_msg.ts < cutoff:
            log.info(
                "@%s: reached cutoff (oldest=%s < %s) — stopping",
                channel, oldest_msg.ts.isoformat()[:10], cutoff.isoformat()[:10],
            )
            break
        before = oldest_msg.msg_id
        if new_on_page == 0:
            log.info("@%s: no new messages on page — stopping", channel)
            break
        time.sleep(delay_sec)

    # Merge + sort
    all_msgs = existing + new_messages
    seen = set()
    deduped = []
    for m in all_msgs:
        if m["post"] in seen:
            continue
        seen.add(m["post"])
        deduped.append(m)
    deduped.sort(key=lambda m: int(m["post"].split("/")[-1]))

    out_path.write_text(
        json.dumps(deduped, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    log.info(
        "@%s: DONE (+%d new, total %d, %s → %s)",
        channel, len(new_messages), len(deduped),
        deduped[0]["ts"][:10] if deduped else "—",
        deduped[-1]["ts"][:10] if deduped else "—",
    )
    return len(new_messages)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", nargs="+", default=[
        "kpszsu", "kyiv_alarm", "war_monitor",
        "operativnoZSU", "V_Zelenskiy_official", "Pravda_Gerashchenko",
        "ssternenko", "serhii_flash", "milinua",
    ])
    ap.add_argument("--cutoff-days", type=int, default=180,
                    help="stop when we go further back than this many days")
    ap.add_argument("--max-pages", type=int, default=400,
                    help="hard cap on pages per channel (~20 msgs each)")
    ap.add_argument("--delay-sec", type=float, default=1.5,
                    help="sleep between page fetches (politeness)")
    args = ap.parse_args()

    t0 = time.monotonic()
    grand_total_new = 0
    for ch in args.channels:
        try:
            n = backfill_channel(ch, args.cutoff_days, args.max_pages, args.delay_sec)
            grand_total_new += n
        except KeyboardInterrupt:
            log.warning("Interrupted by user — stopping")
            break
        except Exception as e:  # noqa: BLE001
            log.exception("@%s: backfill crashed: %s", ch, e)
            continue
        time.sleep(2.0)  # extra breather between channels

    elapsed = time.monotonic() - t0
    log.info(
        "Backfill complete: %d new msgs across %d channels in %.0fs",
        grand_total_new, len(args.channels), elapsed,
    )


if __name__ == "__main__":
    main()
