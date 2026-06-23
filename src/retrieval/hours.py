"""
retrieval/hours.py
-------------------
Opening hours parser and "is open now" checker for OSM opening_hours format.

Handles the most common patterns found in the Portland dataset:
  - "24/7"
  - "Mo-Fr 09:00-17:00" (single day-range + time-range)
  - "Mo-Fr 09:00-17:00; Sa 09:00-13:00" (multiple semicolon-separated rules)
  - "Mo-Fr 08:00-14:00,17:30-22:00" (multiple time ranges per day, comma-separated)
  - "Tu-Th,Su 17:00-22:00" (comma-separated day groups)
  - "Su closed" / "Mo off" (explicit closed days)

Does NOT handle (logged as unparseable, treated as "unknown"):
  - Public holiday exceptions ("PH off", "PH 10:00-14:00")
  - Overnight ranges that wrap past midnight in complex ways
  - Seasonal variations (e.g. "Apr-Oct: ...")

Usage:
    from src.retrieval.hours import is_open_now, parse_opening_hours
    is_open_now("Mo-Fr 09:00-17:00", check_time=datetime.now())
"""

import re
from datetime import datetime, time

DAY_MAP = {
    "Mo": 0, "Tu": 1, "We": 2, "Th": 3, "Fr": 4, "Sa": 5, "Su": 6,
}
DAY_ORDER = ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"]


def _expand_day_range(day_str: str) -> list[int]:
    """
    Expand a day range/group like 'Mo-Fr' or 'Tu,Th,Su' into a list of weekday ints (0=Mon).
    Returns [] if unparseable.
    """
    days = []
    for part in day_str.split(","):
        part = part.strip()
        if "-" in part and part not in DAY_MAP:
            start, end = part.split("-")
            if start not in DAY_MAP or end not in DAY_MAP:
                continue
            start_i, end_i = DAY_MAP[start], DAY_MAP[end]
            if start_i <= end_i:
                days.extend(range(start_i, end_i + 1))
            else:
                # wraps around (e.g. Fr-Mo)
                days.extend(list(range(start_i, 7)) + list(range(0, end_i + 1)))
        elif part in DAY_MAP:
            days.append(DAY_MAP[part])
    return days


def _parse_time_range(time_str: str) -> tuple[time, time] | None:
    """Parse 'HH:MM-HH:MM' into (start_time, end_time). Returns None if unparseable."""
    match = re.match(r"^(\d{1,2}):(\d{2})-(\d{1,2}):(\d{2})$", time_str.strip())
    if not match:
        return None
    h1, m1, h2, m2 = map(int, match.groups())
    if h1 > 23 or h2 > 24 or m1 > 59 or m2 > 59:
        return None
    end_h = 23 if h2 == 24 else h2
    end_m = 59 if h2 == 24 else m2
    try:
        return time(h1, m1), time(end_h, end_m)
    except ValueError:
        return None


def parse_opening_hours(hours_str: str) -> dict | None:
    """
    Parse an OSM opening_hours string into a structured format:
        {weekday_int: [(start_time, end_time), ...]}

    Returns None if the string can't be parsed (caller should treat as "unknown").
    """
    if not hours_str or not isinstance(hours_str, str):
        return None

    hours_str = hours_str.strip()

    if hours_str == "24/7":
        return {d: [(time(0, 0), time(23, 59))] for d in range(7)}

    # Reject formats we don't support (public holidays, complex seasonal rules,
    # open-ended "+" times, specific date exceptions)
    if "PH" in hours_str or re.search(r"[A-Za-z]{3}-[A-Za-z]{3}\s*:", hours_str):
        return None
    if "+" in hours_str:
        return None
    if re.search(r"\b(19|20)\d{2}\b", hours_str):
        # contains a 4-digit year -> date-range exception, skip
        return None
    if re.search(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d", hours_str):
        # specific date like "Dec 24" -> skip
        return None

    schedule: dict[int, list[tuple[time, time]]] = {}

    rules = [r.strip() for r in hours_str.split(";") if r.strip()]

    for rule in rules:
        # Handle "Su closed" / "Mo off"
        closed_match = re.match(r"^([A-Za-z,\-]+)\s+(closed|off)$", rule, re.IGNORECASE)
        if closed_match:
            days = _expand_day_range(closed_match.group(1))
            for d in days:
                schedule[d] = []
            continue

        # Time-only rule with no day prefix (e.g. "05:00-19:00") -> applies to all days
        time_only_match = re.match(r"^\d{1,2}:\d{2}-\d{1,2}:\d{2}(\s*,\s*\d{1,2}:\d{2}-\d{1,2}:\d{2})*$", rule)
        if time_only_match:
            time_ranges = []
            for t in re.split(r"\s*,\s*", rule):
                parsed = _parse_time_range(t)
                if parsed is None:
                    return None
                time_ranges.append(parsed)
            for d in range(7):
                schedule.setdefault(d, [])
                schedule[d].extend(time_ranges)
            continue

        # General rule: "<days> <time1>,<time2>,..." (allow spaces around commas)
        general_match = re.match(r"^([A-Za-z,\-]+)\s+(.+)$", rule)
        if not general_match:
            return None  # unparseable rule -> whole string treated as unknown

        day_part, time_part = general_match.groups()
        days = _expand_day_range(day_part)
        if not days:
            return None

        time_ranges = []
        for t in re.split(r"\s*,\s*", time_part):
            parsed = _parse_time_range(t)
            if parsed is None:
                return None  # unparseable time -> unknown
            time_ranges.append(parsed)

        for d in days:
            schedule.setdefault(d, [])
            schedule[d].extend(time_ranges)

    return schedule if schedule else None


def is_open_now(hours_str: str, check_time: datetime = None) -> bool | None:
    """
    Check if a POI is open at check_time (default: now).

    Returns:
        True  -> open
        False -> closed
        None  -> unknown (unparseable or missing data)
    """
    if check_time is None:
        check_time = datetime.now()

    schedule = parse_opening_hours(hours_str)
    if schedule is None:
        return None

    weekday = check_time.weekday()
    current_time = check_time.time()

    ranges = schedule.get(weekday, [])
    for start, end in ranges:
        if start <= current_time <= end:
            return True

    return False


def filter_open_now(df, hours_col: str = "opening_hours", check_time: datetime = None):
    """
    Add an 'is_open_now' column to a dataframe: True/False/None (unknown).
    Does NOT filter out rows — caller decides how to handle unknowns.
    """
    df = df.copy()
    df["is_open_now"] = df[hours_col].apply(lambda h: is_open_now(h, check_time))
    return df