"""
clean.py
--------
Handles null and non-informative values per column.
Step 1 in the preprocessing pipeline.
"""

import pandas as pd

# Values treated as missing across all text columns
_EMPTY_VALUES = {"unknown", "", "nan", "none", "yes", "no"}


def clean_text(value) -> str | None:
    """
    Returns None if value is null or non-informative, else stripped string.
    Used for: name, cuisine, addr, category
    """
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text.lower() in _EMPTY_VALUES:
        return None
    return text


def clean_name(value) -> str | None:
    """
    Name field: also treats 'unknown' as missing.
    If name is missing but brand exists, caller should fill name with brand first
    (done in pipeline.py).
    """
    return clean_text(value)


def clean_cuisine(value) -> str | None:
    """
    Cuisine field: strip, lowercase, return None if uninformative.
    """
    cleaned = clean_text(value)
    if cleaned is None:
        return None
    return cleaned.lower()


def clean_category(value) -> str | None:
    """
    Category field: treat noise labels as None.
    """
    noise = {"yes", "vacant", "trade", "bed", "art", "car", "no"}
    cleaned = clean_text(value)
    if cleaned is None:
        return None
    if cleaned.lower() in noise:
        return None
    return cleaned


def clean_addr(value) -> str | None:
    """
    Address field: strips whitespace, treats empty strings as None.
    Used for the already-merged 'addr' column OR raw 'addr:street'.
    """
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text == "":
        return None
    if text.lower() in _EMPTY_VALUES:
        return None
    return text


def apply_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies cleaning to all relevant columns.
    Returns df with new *_clean columns.

    Columns handled:
        name          → name_clean
        cuisine       → cuisine_clean
        category      → category_clean  (uses category_final if present)
        addr:street   → addr_clean      (preferred — city always Portland)
        addr          → addr_clean      (fallback if addr:street not present)

    NOTE: addr:city is intentionally ignored — always 'Portland' in this dataset,
    adds no discriminative value for POI matching.
    """
    df = df.copy()

    if "name" in df.columns:
        df["name_clean"] = df["name"].apply(clean_name)

    if "cuisine" in df.columns:
        df["cuisine_clean"] = df["cuisine"].apply(clean_cuisine)

    # Use category_final if available (already standardized), else raw category
    cat_col = "category_final" if "category_final" in df.columns else "category"
    if cat_col in df.columns:
        df["category_clean"] = df[cat_col].apply(clean_category)

    # addr:street preferred over merged addr (city dropped intentionally)
    if "addr:street" in df.columns:
        df["addr_clean"] = df["addr:street"].apply(clean_addr)
    elif "addr" in df.columns:
        df["addr_clean"] = df["addr"].apply(clean_addr)

    return df
