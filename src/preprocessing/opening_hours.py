"""
preprocessing/opening_hours.py
-------------------------------
Parses OSM opening_hours strings into structured per-day open/close times.

Output columns per day (mo, tu, we, th, fr, sa, su):
  - {day}_open:  e.g. "09:00"
  - {day}_close: e.g. "21:00" or "24:00" for midnight

Additional:
  - is_24_7: bool - True if open ~10080 min/week (24h x 7 days)

Requires: pip install opening-hours-py
"""

import pandas as pd
from datetime import datetime, timedelta

try:
    from opening_hours import OpeningHours
    OPENING_HOURS_AVAILABLE = True
except ImportError:
    OPENING_HOURS_AVAILABLE = False
    print("[opening_hours] Warning - 'opening-hours-py' not installed.")
    print("                Run: pip install opening-hours-py")


# Reference Monday used for interval extraction (the exact date doesn't matter)
_MONDAY = datetime(2025, 5, 5, 0, 0)
DAYS = ["mo", "tu", "we", "th", "fr", "sa", "su"]


def parse_opening_hours(value) -> tuple[dict, bool]:
    """
    Parse a single OSM opening_hours string.

    Returns:
        (dict of {day_open/day_close: "HH:MM" or None}, is_24_7: bool)
    """
    empty = {f"{d}_{t}": None for d in DAYS for t in ["open", "close"]}

    if not value or str(value).strip().lower() in ("nan", "none", ""):
        return empty, False

    if not OPENING_HOURS_AVAILABLE:
        return empty, False

    text = str(value).strip()
    text = text.split("||")[0].strip()  # take only first rule if multiple

    result = {f"{d}_{t}": None for d in DAYS for t in ["open", "close"]}
    is_24_7 = False

    try:
        oh = OpeningHours(text)
        week_end = _MONDAY + timedelta(days=7)

        # Extract all OPEN intervals for the full week
        all_open = [
            iv
            for iv in oh.intervals(start=_MONDAY, end=week_end)
            if str(iv[2]) == "open"
        ]

        # is_24_7: open for ~10080 minutes in a week
        total_open_min = sum(
            (iv[1] - iv[0]).total_seconds() / 60 for iv in all_open
        )
        is_24_7 = total_open_min >= 7 * 24 * 60 - 1

        # Extract open/close per day
        for day_idx, day_key in enumerate(DAYS):
            day_start = _MONDAY + timedelta(days=day_idx)
            day_end = day_start + timedelta(days=1)

            day_intervals = []
            for iv in all_open:
                iv_start = max(iv[0], day_start)
                iv_end = min(iv[1], day_end)
                if iv_start < iv_end:
                    day_intervals.append((iv_start, iv_end))

            if day_intervals:
                result[f"{day_key}_open"] = day_intervals[0][0].strftime("%H:%M")
                last_close = day_intervals[-1][1]
                # If close falls into next day -> 24:00
                result[f"{day_key}_close"] = (
                    "24:00" if last_close >= day_end else last_close.strftime("%H:%M")
                )

    except Exception:
        pass  # Invalid OSM format -> leave all None

    return result, is_24_7


def add_opening_hours(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse opening_hours column and expand into per-day open/close columns.
    Also adds is_24_7 boolean column.
    """
    df = df.copy()

    if "opening_hours" not in df.columns:
        print("[opening_hours] Warning - 'opening_hours' column not found, skipping")
        for d in DAYS:
            df[f"{d}_open"] = None
            df[f"{d}_close"] = None
        df["is_24_7"] = False
        return df

    print(f"[opening_hours] Parsing {df['opening_hours'].notna().sum()} non-null values...")

    parsed = df["opening_hours"].apply(parse_opening_hours)

    hours_df = pd.DataFrame(
        [row for row, _ in parsed],
        index=df.index,
    )
    df = pd.concat([df, hours_df], axis=1)
    df["is_24_7"] = [is247 for _, is247 in parsed]

    df["opening_hours_norm"] = df["opening_hours"].apply(
        lambda x: str(x).strip() if pd.notna(x) else None
    )

    open_24_7_count = df["is_24_7"].sum()
    has_hours = hours_df.notna().any(axis=1).sum()
    print(f"[opening_hours] Rows with parsed hours: {has_hours}")
    print(f"[opening_hours] 24/7 places:            {open_24_7_count}")
    return df


def run(df: pd.DataFrame) -> pd.DataFrame:
    return add_opening_hours(df)
