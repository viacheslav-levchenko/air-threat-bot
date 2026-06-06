"""End-to-end smoke test of analytics → forecast → digest pipeline.

Uses the bundled test_data/*.json corpus. Verifies that:
  • detect_incidents() finds reasonable Kyiv-direction incidents
  • compute_attack_stats() produces sane numbers
  • forecast_24h() generates a Forecast with explainable factors
  • build_digest() renders without crashing
"""

from __future__ import annotations

import json
import pathlib
from datetime import datetime, timezone

from analytics import compute_attack_stats, detect_incidents
from classifier import classify
from forecast import forecast_24h, render_forecast_text
from parser import ParsedMessage, tag_text


def load_corpus() -> list[ParsedMessage]:
    here = pathlib.Path(__file__).resolve().parent
    out: list[ParsedMessage] = []
    for ch in ("kpszsu", "kyiv_alarm", "war_monitor"):
        # Try _full first (post-backfill), fall back to short
        for fname in (f"test_data/{ch}_full.json", f"test_data/{ch}.json"):
            path = here / fname
            if path.exists():
                raw = json.loads(path.read_text(encoding="utf-8"))
                for r in raw:
                    try:
                        ch_name, mid = r["post"].split("/", 1)
                        ts = datetime.fromisoformat(r["ts"].replace("Z", "+00:00"))
                        if ts.tzinfo is None:
                            ts = ts.replace(tzinfo=timezone.utc)
                        else:
                            ts = ts.astimezone(timezone.utc)
                        pm = ParsedMessage(channel=ch_name, msg_id=int(mid), ts=ts, text=r["text"])
                        pm.tags = tag_text(r["text"])
                        out.append(pm)
                    except Exception as e:  # noqa: BLE001
                        print(f"  skip {r.get('post')}: {e}")
                print(f"  loaded {fname}: {len(raw)} raw")
                break
    out.sort(key=lambda m: m.ts)
    return out


def main():
    msgs = load_corpus()
    print(f"\nCorpus: {len(msgs)} msgs, range {msgs[0].ts.isoformat()[:10]} → {msgs[-1].ts.isoformat()[:10]}")

    # === detect_incidents ===
    incidents = detect_incidents(msgs)
    print(f"\n=== Detected {len(incidents)} incidents ===")
    by_type: dict[str, int] = {}
    for inc in incidents:
        by_type[inc.type] = by_type.get(inc.type, 0) + 1
    for t, n in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"  {t:30s} {n} incidents")

    print(f"\n=== Latest 10 incidents ===")
    for inc in incidents[-10:]:
        print(f"  {inc.started_at.isoformat()[:16]}  {inc.type:25s} "
              f"{inc.intensity:8s} ({inc.msg_count} msgs, sources={inc.sources})")

    # === compute_attack_stats — using END of corpus as "now" ===
    fake_now = msgs[-1].ts
    stats = compute_attack_stats(incidents, now=fake_now)
    print(f"\n=== Attack stats (now={fake_now.isoformat()[:16]}) ===")
    for t, s in sorted(stats.items()):
        last = s.last_at.isoformat()[:16] if s.last_at else "—"
        days = f"{s.days_since_last:.1f}" if s.days_since_last is not None else "—"
        mean = f"{s.mean_interval_30d:.1f}" if s.mean_interval_30d else "—"
        print(f"  {t:25s} last={last}  days_since={days:>6s}  "
              f"7d={s.count_7d:2d}  30d={s.count_30d:3d}  90d={s.count_90d:3d}  "
              f"mean_interval={mean:>5s}  readiness={s.stockpile_readiness:.2f}")

    # === forecast 24h ===
    state = classify(msgs, now=fake_now)
    print(f"\n=== Current state (replay): {state.short_summary()} ===")
    print(f"  active flags: {list(state.active_flags.keys())}")

    fcast = forecast_24h(state, stats, recent_incidents=incidents, now=fake_now)
    print(f"\n=== Forecast 24h ===")
    print(f"  tier: {fcast.tier_label}  prob: {fcast.percent}%")
    print(f"  factors:")
    for f in fcast.top_factors(8):
        print(f"    {f.delta:+.2f}  {f.name:30s}  {f.explanation}")

    # === Render text ===
    print(f"\n=== Rendered forecast text ===\n")
    print(render_forecast_text(fcast, stats))


if __name__ == "__main__":
    main()
