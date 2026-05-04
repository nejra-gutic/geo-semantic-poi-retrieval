"""
preprocessing/flags.py
----------------------
Converts OSM boolean-like text attributes into clean binary (0/1) flags.

Handles:
  - wheelchair: yes / designated -> 1, else 0
  - takeaway:   yes / only       -> 1, else 0
"""

import pandas as pd


def parse_wheelchair(value) -> int:
    """Return 1 if wheelchair accessible, 0 otherwise."""
    if pd.isna(value):
        return 0
    v = str(value).strip().lower()
    return 1 if v in {"yes", "designated"} else 0


def parse_takeaway(value) -> int:
    """Return 1 if takeaway available, 0 otherwise."""
    if pd.isna(value):
        return 0
    v = str(value).strip().lower()
    return 1 if v in {"yes", "only"} else 0


def add_wheelchair_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Add wheelchair_clean binary column."""
    df = df.copy()
    if "wheelchair" not in df.columns:
        print("[flags] Warning - 'wheelchair' column not found, setting to 0")
        df["wheelchair_accessible"] = 0
        return df

    df["wheelchair_accessible"] = df["wheelchair"].apply(parse_wheelchair)
    accessible = df["wheelchair_accessible"].sum()
    print(f"[flags] Wheelchair accessible: {accessible} / {len(df)} ({round(accessible/len(df)*100,1)}%)")
    return df


def add_takeaway_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Add takeaway_clean binary column."""
    df = df.copy()
    if "takeaway" not in df.columns:
        print("[flags] Warning - 'takeaway' column not found, setting to 0")
        df["has_takeaway"] = 0
        return df

    df["has_takeaway"] = df["takeaway"].apply(parse_takeaway)
    available = df["has_takeaway"].sum()
    print(f"[flags] Takeaway available: {available} / {len(df)} ({round(available/len(df)*100,1)}%)")
    return df


def run(df: pd.DataFrame) -> pd.DataFrame:
    df = add_wheelchair_flag(df)
    df = add_takeaway_flag(df)
    return df
