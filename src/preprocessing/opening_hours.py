"""
preprocessing/opening_hours.py
-------------------------------
Validates OSM opening_hours strings and flags 24/7 POIs.

Per mentor guidance: no need to expand into 14+ per-day columns.
The opening_hours_py parser can parse the raw string directly at
query time (given a datetime), so we just keep opening_hours as-is
and add a lightweight is_24_7 boolean shortcut.

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



# Manual fixes for raw OSM opening_hours values that don't parse with
# opening_hours_py (non-standard syntax: AM/PM, lowercase days, 
# by_appointment, contradictory/seasonal rules, etc.)
OPENING_HOURS_FIXES = {
    "Mo-Su 09:00-17:00 Sat 10:00-17:00": "Mo-Fr,Su 09:00-17:00; Sa 10:00-17:00",
    "Wed, Thu, Sun: 4:00PM - 12:00AM; Fri & Sat: 4:00PM - 2:00AM": "We,Th,Su 16:00-24:00; Fr-Sa 16:00-02:00",
    "Sa-Th: 07:00-20:00; F: 07:00-18:00": "Sa-Th 07:00-20:00; Fr 07:00-18:00",
    "Mo-Tu,Th-Fr 10:00-18:00; We by_appointment; Sa 10:00-14:00": "Mo-Tu,Th-Fr 10:00-18:00; Sa 10:00-14:00",
    "By Appointment Only": None,
    "Mo-Su 11:00-18:00; 2026 Feb Mo-Su 11:00-17:00": "Mo-Su 11:00-18:00",
    "by_appointment": None,
    "tu-su 12:00-19:00": "Tu-Su 12:00-19:00",
    "Monday to Friday: 7:30\u202fAM–5:30\u202fPM": "Mo-Fr 07:30-17:30",
    "8am - 8pm": "Mo-Su 08:00-20:00",
    "closed Tuesday, Wednesday": "Mo,Th-Su open; Tu-We off",
    "Mo-Th 16:00-12:00am, Fr 16:00-01:00am, Sat 13:00-01:00, Sun 13:00-12:00": "Mo-Th 16:00-24:00; Fr 16:00-25:00; Sa 13:00-25:00; Su 13:00-24:00",
    '"Winter Hours": Mo-Fr 07:30-18:00, Sa 08:00-16:30, Su 09:00-16:30; "Summer Hours": Mo-Fr 07:30-18:00, Sa 08:00-17:30, Su 09:00-16:30': "Mo-Fr 07:30-18:00; Sa 08:00-17:30; Su 09:00-16:30",
    "Th-Sun 17:00-22:00": "Th-Su 17:00-22:00",
    "Wed 16:00-22:00, Th-Sat 16:00-23:00, Sun 16:00-21:00": "We 16:00-22:00; Th-Sa 16:00-23:00; Su 16:00-21:00",
}

_MONDAY = datetime(2025, 5, 5, 0, 0)


def is_valid_opening_hours(value) -> bool:
    """Check if a raw opening_hours string parses successfully."""
    if not value or str(value).strip().lower() in ("nan", "none", ""):
        return False
    if not OPENING_HOURS_AVAILABLE:
        return False
    try:
        OpeningHours(str(value).strip().split("||")[0].strip())
        return True
    except Exception:
        return False


def compute_is_24_7(value) -> bool:
    """Check if a raw opening_hours string represents a 24/7 schedule."""
    if not value or str(value).strip().lower() in ("nan", "none", ""):
        return False
    if not OPENING_HOURS_AVAILABLE:
        return False

    text = str(value).strip().split("||")[0].strip()

    try:
        oh = OpeningHours(text)
        week_end = _MONDAY + timedelta(days=7)
        all_open = [
            iv for iv in oh.intervals(start=_MONDAY, end=week_end)
            if str(iv[2]) == "open"
        ]
        total_open_min = sum(
            (iv[1] - iv[0]).total_seconds() / 60 for iv in all_open
        )
        return total_open_min >= 7 * 24 * 60 - 1
    except Exception:
        return False

def add_opening_hours(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep opening_hours as-is (no per-day expansion). Add:
      - opening_hours_norm: stripped/normalized raw string
      - is_24_7: boolean shortcut for 24/7 places

    Applies manual fixes for known non-standard raw OSM values
    (AM/PM format, lowercase days, by_appointment, etc.) before
    validation/parsing.
    """
    df = df.copy()

    if "opening_hours" not in df.columns:
        print("[opening_hours] Warning - 'opening_hours' column not found, skipping")
        df["opening_hours_norm"] = None
        df["is_24_7"] = False
        return df

    # Apply manual fixes for known non-standard raw values
    fixed_count = df["opening_hours"].isin(OPENING_HOURS_FIXES.keys()).sum()
    df["opening_hours"] = df["opening_hours"].apply(
        lambda x: OPENING_HOURS_FIXES.get(x, x) if pd.notna(x) else x
    )
    if fixed_count > 0:
        print(f"[opening_hours] Applied manual fixes to {fixed_count} rows")

    total = len(df)
    provided = df["opening_hours"].notna().sum()
    print(f"[opening_hours] Total POIs: {total}")
    print(f"[opening_hours] opening_hours provided: {provided} ({provided/total*100:.1f}%)")

    df["opening_hours_norm"] = df["opening_hours"].apply(
        lambda x: str(x).strip() if pd.notna(x) else None
    )

    if OPENING_HOURS_AVAILABLE:
        valid_mask = df["opening_hours"].apply(is_valid_opening_hours)
        valid_count = valid_mask.sum()
        invalid_count = provided - valid_count
        print(f"[opening_hours] Valid:   {valid_count} ({valid_count/provided*100:.1f}%)")
        print(f"[opening_hours] Invalid: {invalid_count} ({invalid_count/provided*100:.1f}%)")

        if invalid_count > 0:
            invalid_values = df.loc[df["opening_hours"].notna() & ~valid_mask, "opening_hours"]
            print(f"[opening_hours] Invalid values (fix manually if needed):")
            for v in invalid_values.unique():
                print(f"  - {v!r}")

    df["is_24_7"] = df["opening_hours"].apply(compute_is_24_7)
    print(f"[opening_hours] 24/7 places: {df['is_24_7'].sum()}")

    return df


def run(df: pd.DataFrame) -> pd.DataFrame:
    return add_opening_hours(df)