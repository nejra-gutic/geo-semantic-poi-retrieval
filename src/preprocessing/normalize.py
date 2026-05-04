"""
preprocessing/normalize.py
--------------------------
Applies lightweight text normalization to string columns:
  - unidecode: removes/replaces accented characters (café -> cafe)
  - lowercase
  - collapse multiple whitespace to single space

Produces _norm variant columns:
  name_norm, brand_norm, cuisine_norm, addr_norm
"""

import re
import pandas as pd
from unidecode import unidecode


def normalize_text(value) -> str | None:
    """
    Normalize a single text value.
    Returns None for missing/empty inputs.
    """
    if pd.isna(value) or str(value).strip() == "":
        return None
    text = str(value).strip()
    text = unidecode(text)           # café -> cafe, Ü -> U
    text = text.lower()              # lowercase
    text = re.sub(r"\s+", " ", text) # collapse whitespace
    return text.strip()


def add_normalized_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add _norm variants for: name, brand, cuisine, addr:street.
    Skips columns that don't exist in the dataframe.
    """
    df = df.copy()

    column_map = {
        "name":       "name_norm",
        "brand":      "brand_norm",
        "addr:street": "addr_norm",
    }

    for source_col, norm_col in column_map.items():
        if source_col in df.columns:
            df[norm_col] = df[source_col].apply(normalize_text)
            filled = df[norm_col].notna().sum()
            print(f"[normalize] {norm_col}: {filled} / {len(df)} filled")
        else:
            print(f"[normalize] Warning - '{source_col}' not found, skipping {norm_col}")
            df[norm_col] = None

    return df


def run(df: pd.DataFrame) -> pd.DataFrame:
    return add_normalized_columns(df)
