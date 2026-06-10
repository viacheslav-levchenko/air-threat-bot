"""Validate forecast.py against 6-month historical corpus.

Methodology:
  1. Load all backfilled test_data/*_full.json (plus the original short
     test_data/*.json as fallback).
  2. Tag every message with the current production lexicon.
  3. Detect all Kyiv-direction incidents over the full window.
  4. For each day d in [start, end-1]:
       - "Freeze" state as-of 09:00 Kyiv that day.
       - Run classifier + analytics + forecast_24h on the as-of slice.
       - Record (day, forecast.prob, forecast.tier, actual_attack_in_next_24h).
  5. Report:
       - Confusion-style table: predicted tier vs actual outcome
       - Brier score (mean squared error of probability)
       - Calibration table (predicted band → actual %)
       - Top 10 false-positive days (high prob, no attack)
       - Top 10 false-negative days (low prob, attack happened)

This is the data we use to tune ACTIVE_FLAG_BOOSTS / RECENT_INCIDENT_BOOSTS /
WEEKDAY_PRIORS coefficients in forecast.py.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
from datetime import datetime, timedelta, timezone

from zoneinfo import ZoneInfo

from analytics import compute_attack_stats, detect_incidents
from classifier import classify
from forecast import TIER_LABELS, forecast_24h
from parser import ParsedMessage, tag_text

KYIV = ZoneInfo("Europe/Kyiv")


def load_full_corpus() -> list[ParsedMessage]:
    """Load all available test_data files. Prefer *_full.json (backfilled)."""
    here = pathlib.Path(__file__).resolve().parent
    out: list[ParsedMessage] = []
    seen: set[str] = set()
    test_data = here / "test_data"
    # Load _full files first (richer)
    for path in sorted(test_data.glob("*_full.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        for r in raw:
            if r["post"] in seen:
                continue
            seen.add(r["post"])
            try:
                ch, mid = r["post"].split("/", 1)
                ts = datetime.fromisoformat(r["ts"].replace("Z", "+00:00"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                else:
                    ts = ts.astimezone(timezone.utc)
                pm = ParsedMessage(channel=ch, msg_id=int(mid), ts=ts, text=r["text"])
                pm.tags = tag_text(r["text"])
                out.append(pm)
            except Exception:  # noqa: BLE001
                continue
        print(f"  loaded {path.name}: {len(raw)} raw (cumulative unique: {len(out)})")
    # Also load short test_data/*.json (the original 612-msg seed) for channels
    # we DIDN'T backfill yet
    for path in sorted(test_data.glob("*.json")):
        if path.name.endswith("_full.json"):
            continue
        ch_name = path.stem
        if (test_data / f"{ch_name}_full.json").exists():
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        for r in raw:
            if r["post"] in seen:
                continue
            seen.add(r["post"])
            try:
                ch, mid = r["post"].split("/", 1)
                ts = datetime.fromisoformat(r["ts"].replace("Z", "+00:00"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                else:
                    ts = ts.astimezone(timezone.utc)
                pm = ParsedMessage(channel=ch, msg_id=int(mid), ts=ts, text=r["text"])
                pm.tags = tag_text(r["text"])
                out.append(pm)
            except Exception:  # noqa: BLE001
                continue
        print(f"  loaded {path.name}: {len(raw)} raw (cumulative unique: {len(out)})")

    out.sort(key=lambda m: m.ts)
    return out


def validate(msgs: list[ParsedMessage], skip_warmup_days: int = 14) -> list[dict]:
    """Walk day-by-day, computing forecast as-of 09:00 each day."""
    if not msgs:
        return []

    all_incidents = detect_incidents(msgs)
    print(f"\nDetected {len(all_incidents)} total incidents")

    # The first `skip_warmup_days` days don't have enough lookback for stable
    # mean_interval / stockpile_readiness calculations.
    first_ts = msgs[0].ts.astimezone(KYIV)
    last_ts = msgs[-1].ts.astimezone(KYIV)
    start_local = (first_ts + timedelta(days=skip_warmup_days)).replace(hour=9, minute=0, second=0, microsecond=0)
    end_local = last_ts.replace(hour=9, minute=0, second=0, microsecond=0) - timedelta(days=1)

    print(f"Simulating forecasts {start_local.date()} → {end_local.date()} (one per day, at 09:00 Kyiv)")

    results: list[dict] = []
    current = start_local
    while current <= end_local:
        asof_utc = current.astimezone(timezone.utc)
        msgs_so_far = [m for m in msgs if m.ts <= asof_utc]
        incidents_so_far = [i for i in all_incidents if i.started_at <= asof_utc]

        state = classify(msgs_so_far, now=asof_utc)
        stats = compute_attack_stats(incidents_so_far, now=asof_utc)
        fc = forecast_24h(state, stats, recent_incidents=incidents_so_far, now=asof_utc)

        window_end = asof_utc + timedelta(hours=24)
        actual_in_24h = [
            i for i in all_incidents
            if asof_utc < i.started_at <= window_end
        ]
        actual = bool(actual_in_24h)
        # Stronger ground truth: "any Kyiv-direction incident with intensity >= medium"
        actual_serious = any(
            i.intensity in ("medium", "high", "massive") for i in actual_in_24h
        )

        results.append({
            "date": current.date(),
            "weekday": current.weekday(),
            "prob": fc.probability,
            "tier": fc.tier,
            "actual_any": actual,
            "actual_serious": actual_serious,
            "active_flags": list(state.active_flags.keys()),
            "incident_count_24h": len(actual_in_24h),
            "top_factors": [(f.name, f.delta) for f in fc.top_factors(3)],
        })

        current = current + timedelta(days=1)

    return results


def report(results: list[dict]) -> None:
    n = len(results)
    print(f"\n{'=' * 70}\nVALIDATION REPORT — {n} day-forecasts\n{'=' * 70}\n")

    # --- Tier breakdown ---
    by_tier = collections.defaultdict(lambda: {"n": 0, "any_hit": 0, "serious_hit": 0})
    for r in results:
        b = by_tier[r["tier"]]
        b["n"] += 1
        if r["actual_any"]:
            b["any_hit"] += 1
        if r["actual_serious"]:
            b["serious_hit"] += 1

    print("Per-tier accuracy:")
    print(f"{'tier':12s} {'n':>5s} {'%any_attack':>14s} {'%serious_attack':>17s}")
    print("-" * 60)
    for tier in ("LOW", "MEDIUM", "HIGH", "VERY_HIGH"):
        b = by_tier[tier]
        n_tier = b["n"]
        any_rate = (b["any_hit"] / n_tier * 100) if n_tier else 0
        ser_rate = (b["serious_hit"] / n_tier * 100) if n_tier else 0
        print(f"{TIER_LABELS[tier]:18s}  {n_tier:>5d}  {any_rate:>10.1f}%   {ser_rate:>13.1f}%")

    # --- Probability calibration table ---
    bands = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
             (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.001)]
    print("\nCalibration (predicted probability → actual %):")
    print(f"{'predicted band':>16s} {'n':>5s} {'%actual_any':>14s} {'%actual_serious':>17s}  (ideal: %actual ≈ midpoint)")
    print("-" * 80)
    for lo, hi in bands:
        in_band = [r for r in results if lo <= r["prob"] < hi]
        if not in_band:
            continue
        n_band = len(in_band)
        any_rate = sum(1 for r in in_band if r["actual_any"]) / n_band * 100
        ser_rate = sum(1 for r in in_band if r["actual_serious"]) / n_band * 100
        midpoint = (lo + hi) / 2 * 100
        print(f"{lo:.2f}-{hi:.2f}    {n_band:>5d}  {any_rate:>10.1f}%   {ser_rate:>13.1f}%   (mid: {midpoint:.0f}%)")

    # --- Brier score ---
    brier_any = sum((r["prob"] - (1 if r["actual_any"] else 0)) ** 2 for r in results) / n
    brier_serious = sum((r["prob"] - (1 if r["actual_serious"] else 0)) ** 2 for r in results) / n
    base_rate_any = sum(1 for r in results if r["actual_any"]) / n
    base_rate_serious = sum(1 for r in results if r["actual_serious"]) / n
    trivial_brier_any = base_rate_any * (1 - base_rate_any)
    trivial_brier_serious = base_rate_serious * (1 - base_rate_serious)

    print(f"\nBrier scores  (lower = better, 0=perfect, ≥trivial=useless):")
    print(f"  any_attack:    {brier_any:.3f}   (trivial baseline {trivial_brier_any:.3f}, base rate {base_rate_any*100:.1f}%)")
    print(f"  serious_attack:{brier_serious:.3f}   (trivial baseline {trivial_brier_serious:.3f}, base rate {base_rate_serious*100:.1f}%)")
    skill_any = 1 - brier_any / trivial_brier_any if trivial_brier_any else 0
    skill_serious = 1 - brier_serious / trivial_brier_serious if trivial_brier_serious else 0
    print(f"  Brier skill score (any): {skill_any:+.3f}  (positive = model beats coin-flip)")
    print(f"  Brier skill score (serious): {skill_serious:+.3f}")

    # --- False positives & negatives ---
    sorted_fp = sorted(
        [r for r in results if not r["actual_any"]],
        key=lambda r: r["prob"], reverse=True,
    )
    sorted_fn = sorted(
        [r for r in results if r["actual_serious"]],
        key=lambda r: r["prob"],
    )

    print(f"\nTop 10 FALSE POSITIVES (high predicted prob, no attack):")
    for r in sorted_fp[:10]:
        factors = ", ".join(f"{n}={d:+.2f}" for n, d in r["top_factors"])
        print(f"  {r['date']} ({['Пн','Вт','Ср','Чт','Пт','Сб','Нд'][r['weekday']]}) "
              f"prob={r['prob']*100:>5.1f}%  tier={r['tier']:9s}  factors=[{factors}]")

    print(f"\nTop 10 FALSE NEGATIVES (low predicted prob, serious attack happened):")
    for r in sorted_fn[:10]:
        n_inc = r["incident_count_24h"]
        factors = ", ".join(f"{n}={d:+.2f}" for n, d in r["top_factors"])
        print(f"  {r['date']} ({['Пн','Вт','Ср','Чт','Пт','Сб','Нд'][r['weekday']]}) "
              f"prob={r['prob']*100:>5.1f}%  tier={r['tier']:9s}  incidents_in_24h={n_inc}  "
              f"factors=[{factors}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", help="optional path to save per-day results as JSON")
    args = ap.parse_args()

    print("Loading corpus...")
    msgs = load_full_corpus()
    print(f"\nCorpus loaded: {len(msgs)} unique messages")
    if msgs:
        print(f"Range: {msgs[0].ts.isoformat()[:10]} → {msgs[-1].ts.isoformat()[:10]}")
        # Breakdown by channel
        by_ch = collections.Counter(m.channel for m in msgs)
        for ch, n in by_ch.most_common():
            print(f"  @{ch:25s} {n}")

    results = validate(msgs)
    report(results)

    if args.save:
        out_path = pathlib.Path(args.save)
        out_path.write_text(json.dumps(
            [{**r, "date": r["date"].isoformat()} for r in results],
            ensure_ascii=False, indent=1,
        ))
        print(f"\nSaved per-day results to {out_path}")


if __name__ == "__main__":
    main()
