"""
preprocessing/address.py
------------------------
Cleans the address field from OSM data.
Replaces empty strings with None so missing values are consistent.
"""

import pandas as pd


def clean_address(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean addr:street column:
      - Replace empty strings with None for consistency
      - Result stored in 'addr' column
    """
    df = df.copy()

    if "addr:street" not in df.columns:
        print("[address] Warning - 'addr:street' column not found, skipping")
        df["addr"] = None
        return df

    df["addr"] = df["addr:street"].replace("", None)

    filled = df["addr"].notna().sum()
    pct = round(filled / len(df) * 100, 1)
    print(f"[address] Addresses filled: {filled} / {len(df)} ({pct}%)")
    return df


def run(df: pd.DataFrame) -> pd.DataFrame:
    return clean_address(df)
