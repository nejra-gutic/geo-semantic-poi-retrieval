"""
retrieval/hours.py
-------------------
Opening hours status checker using the opening_hours_py library to parse
raw OSM opening_hours strings directly at query time (no custom regex
parser needed - opening_hours_py handles PH exceptions, seasonal rules,
overnight ranges, etc. that a hand-rolled parser can't).

Usage:
    from src.retrieval.hours import is_open_now, is_open_for_query
    is_open_now("Mo-Fr 09:00-17:00", check_time=datetime.now())
"""

from datetime import datetime, timedelta

try:
    from opening_hours import OpeningHours
    OPENING_HOURS_AVAILABLE = True
except ImportError:
    OPENING_HOURS_AVAILABLE = False
    print("[hours] Warning - 'opening-hours-py' not installed.")
    print("        Run: pip install opening-hours-py")


def is_open_now(hours_str: str, check_time: datetime = None) -> bool | None:
    """
    Check if a POI is open at check_time (default: now).

    Returns:
        True  -> open
        False -> closed
        None  -> unknown (missing data, unparseable string, or library
                 unavailable)
    """
    if not hours_str or not isinstance(hours_str, str) or not hours_str.strip():
        return None
    if not OPENING_HOURS_AVAILABLE:
        return None

    if check_time is None:
        check_time = datetime.now()

    text = hours_str.strip().split("||")[0].strip()  # take only first rule if multiple

    try:
        oh = OpeningHours(text)
        state = oh.state(check_time)
        state_name = str(state).split(".")[-1].upper()  # "State.OPEN" -> "OPEN"
        if state_name == "OPEN":
            return True
        if state_name == "CLOSED":
            return False
        return None  # unknown / unrecognized state
    except Exception:
        return None


def resolve_check_time(query: str, base_time: datetime = None) -> datetime:
    """
    Determine WHEN to check open status based on query phrasing, instead of
    always checking 'right now'.

    Handles:
      - "now" / "right now" / "currently" / "today" -> base_time as-is
      - "after midnight" -> estimate 00:30, shifted to the NEXT day (this is
        genuinely after 00:00, unlike "late"/"tonight" which are estimated
        at 22:00 the same day -- these were previously grouped together,
        which meant "after midnight" was checked at 22:00, i.e. BEFORE
        midnight, giving wrong answers for places that close before 00:00)
      - "late" / "tonight" / "this evening" / "all night" -> estimate 22:00
        same day
      - "early morning" / "before Xam" -> estimate 07:00 same day
      - "tomorrow" -> shift check date forward by one day (combines with
        the hour-of-day rules above, e.g. "open early tomorrow" -> 07:00
        next day)
      - specific weekday mentioned ("saturday", "sunday", ...) -> same time,
        but shifted to the next occurrence of that weekday
      - default -> base_time (treated as "now")

    This is a heuristic, not a full NLP time parser - it's meant to cover
    the common phrasings seen in the eval set.
    """
    if base_time is None:
        base_time = datetime.now()

    q = query.lower()

    weekday_names = {
        "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
        "friday": 4, "saturday": 5, "sunday": 6,
    }
    target_weekday = None
    for name, idx in weekday_names.items():
        if name in q:
            target_weekday = idx
            break

    check_time = base_time

    if "after midnight" in q:
        # Genuinely after 00:00 -> next day, early morning. NOT the same
        # bucket as "late"/"tonight" (22:00), which is still before midnight.
        check_time = (check_time + timedelta(days=1)).replace(
            hour=0, minute=30, second=0, microsecond=0
        )
    elif any(w in q for w in ["late", "tonight", "this evening", "all night"]):
        check_time = check_time.replace(hour=22, minute=0, second=0, microsecond=0)
    elif any(w in q for w in ["early morning", "before 7", "before 8"]):
        check_time = check_time.replace(hour=7, minute=0, second=0, microsecond=0)
    elif any(w in q for w in ["now", "right now", "currently", "rn", "yet", "still"]):
        pass  # keep as-is
    # else: "today", "open today" etc. -> keep as-is (default "now" check)

    if "tomorrow" in q and "after midnight" not in q:
        check_time = check_time + timedelta(days=1)

    if target_weekday is not None:
        days_ahead = (target_weekday - check_time.weekday()) % 7
        check_time = check_time + timedelta(days=days_ahead)

    return check_time


def is_open_for_query(hours_str: str, query: str, base_time: datetime = None) -> bool | None:
    """
    Convenience wrapper: resolve the right check_time from the query phrasing,
    then check open status at that time.
    """
    check_time = resolve_check_time(query, base_time)
    return is_open_now(hours_str, check_time)


def filter_open_now(df, hours_col: str = "opening_hours", check_time: datetime = None):
    """
    Add an 'is_open_now' column to a dataframe: True/False/None (unknown).
    Does NOT filter out rows — caller decides how to handle unknowns.
    """
    df = df.copy()
    df["is_open_now"] = df[hours_col].apply(lambda h: is_open_now(h, check_time))
    return df