"""
opening_hours.py
----------------
Parses the 'opening_hours' OSM field into structured boolean flags.

Flags produced (all already exist in your dataset, this module standardizes them):
  - is_24_7           : True if open 24/7
  - works_every_day   : True if Mo-Su or 24/7
  - has_opening_hours : True if any valid hours string exists
  - by_appointment    : True if appointment-only
  - has_weekend_hours : True if Sa or Su mentioned
  - weekday_only      : True if Mo-Fr only, no weekend

Also produces:
  - opening_hours_norm: cleaned string representation (original preserved as-is)
"""

import re
import pandas as pd


# ── helpers ──────────────────────────────────────────────────────────────────

_EMPTY_VALUES = {"unknown", "", "nan", "none"}

_DAY_RANGES = {
    "Mo-Fr": ["Mo", "Tu", "We", "Th", "Fr"],
    "Mo-Sa": ["Mo", "Tu", "We", "Th", "Fr", "Sa"],
    "Mo-Su": ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"],
    "Tu-Sa": ["Tu", "We", "Th", "Fr", "Sa"],
    "Sa-Su": ["Sa", "Su"],
    "Fr,Sa": ["Fr", "Sa"],
    "We-Su": ["We", "Th", "Fr", "Sa", "Su"],
}

_ALL_DAYS = {"Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"}


def _is_valid(value) -> bool:
    if pd.isna(value):
        return False
    text = str(value).strip().lower()
    return text not in _EMPTY_VALUES and text != ""


def flag_is_24_7(value) -> bool:
    if not _is_valid(value):
        return False
    return str(value).strip().lower() in {"24/7", "24/7;", "mo-su 00:00-24:00"}


def flag_has_opening_hours(value) -> bool:
    return _is_valid(value)


def flag_by_appointment(value) -> bool:
    if not _is_valid(value):
        return False
    return "appointment" in str(value).lower()


def flag_has_weekend_hours(value) -> bool:
    if not _is_valid(value):
        return False
    text = str(value)
    return bool(re.search(r"\bSa\b|\bSu\b", text))


def flag_weekday_only(value) -> bool:
    if not _is_valid(value):
        return False
    text = str(value)
    has_weekdays = "Mo-Fr" in text
    has_weekend = bool(re.search(r"\bSa\b|\bSu\b", text))
    return has_weekdays and not has_weekend


def flag_works_every_day(value) -> bool:
    """True if all 7 days are covered."""
    if not _is_valid(value):
        return False

    text = str(value)

    if flag_is_24_7(value):
        return True

    days = set()

    for pattern, covered in _DAY_RANGES.items():
        if pattern in text:
            days.update(covered)

    for day in _ALL_DAYS:
        if re.search(rf"\b{day}\b", text):
            days.add(day)

    return days >= _ALL_DAYS


def normalize_opening_hours(value) -> str | None:
    """
    Lightly normalizes the hours string:
    - strips whitespace
    - returns None for empty/unknown values
    - preserves original format otherwise (OSM hours are structured)
    """
    if not _is_valid(value):
        return None
    return str(value).strip()


# ── main apply function ───────────────────────────────────────────────────────

def apply_opening_hours(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies all opening_hours processing.

    Input:  df with 'opening_hours' column
    Output: df with columns:
              opening_hours_norm  (cleaned string or None)
              is_24_7             (bool)
              has_opening_hours   (bool)
              by_appointment      (bool)
              has_weekend_hours   (bool)
              weekday_only        (bool)
              works_every_day     (bool)

    Note: If these boolean columns already exist (from your notebook),
          they will be overwritten with the standardized versions.
    """
    df = df.copy()

    if "opening_hours" not in df.columns:
        print("[opening_hours.py] Warning: no 'opening_hours' column found.")
        return df

    oh = df["opening_hours"]

    df["opening_hours_norm"] = oh.apply(normalize_opening_hours)
    df["is_24_7"] = oh.apply(flag_is_24_7)
    df["has_opening_hours"] = oh.apply(flag_has_opening_hours)
    df["by_appointment"] = oh.apply(flag_by_appointment)
    df["has_weekend_hours"] = oh.apply(flag_has_weekend_hours)
    df["weekday_only"] = oh.apply(flag_weekday_only)
    df["works_every_day"] = oh.apply(flag_works_every_day)

    # Summary
    print(f"[opening_hours.py] Processed {len(df)} rows.")
    print(f"  has_opening_hours : {df['has_opening_hours'].sum()}")
    print(f"  is_24_7           : {df['is_24_7'].sum()}")
    print(f"  works_every_day   : {df['works_every_day'].sum()}")
    print(f"  has_weekend_hours : {df['has_weekend_hours'].sum()}")
    print(f"  weekday_only      : {df['weekday_only'].sum()}")
    print(f"  by_appointment    : {df['by_appointment'].sum()}")

    return df
