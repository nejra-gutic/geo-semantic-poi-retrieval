"""
flags.py
--------
Converts structured OSM boolean-like fields into clean boolean values.

Fields handled:
  - wheelchair  → wheelchair_accessible (True / False / None)
  - takeaway    → has_takeaway          (True / False / None)

None = unknown/not specified (different from False = explicitly no)
"""

import pandas as pd


def parse_osm_bool(value) -> bool | None:
    """
    Parses OSM yes/no/limited values into Python booleans.
    
    Returns:
      True  → "yes", "designated"
      False → "no"
      None  → anything else (unknown, missing, "limited" treated as None)
    """
    if pd.isna(value):
        return None

    v = str(value).strip().lower()

    if v in {"yes", "designated"}:
        return True
    elif v == "no":
        return False
    elif v == "limited":
        return None  # keep as unknown, could be partially accessible
    else:
        return None


def apply_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies boolean flag parsing.

    Input:  df with 'wheelchair' and/or 'takeaway' columns
    Output: df with:
              wheelchair_accessible  (True / False / None)
              has_takeaway           (True / False / None)

    Preserves original columns.
    """
    df = df.copy()

    if "wheelchair" in df.columns:
        df["wheelchair_accessible"] = df["wheelchair"].apply(parse_osm_bool)
        counts = df["wheelchair_accessible"].value_counts(dropna=False)
        print(f"[flags.py] wheelchair_accessible:\n{counts.to_string()}\n")

    if "takeaway" in df.columns:
        df["has_takeaway"] = df["takeaway"].apply(parse_osm_bool)
        counts = df["has_takeaway"].value_counts(dropna=False)
        print(f"[flags.py] has_takeaway:\n{counts.to_string()}\n")

    return df
