"""
preprocessing/text_join.py
--------------------------
Creates a single 'poi_text' field by concatenating normalized columns.
Used for downstream NLP search and retrieval tasks.

Fields joined:
  name_norm + brand_norm + category_final + category_group + cuisine_clean + addr_norm
"""

import pandas as pd


POI_TEXT_COLS = [
    "name_norm",
    "name_norm",
    "name_norm",      # 3x
    "brand_norm",
    "category_final",
    "category_group",
    "cuisine_clean",
    "cuisine_clean",  # 2x
    "addr_norm",
]


def build_poi_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Concatenate normalized text columns into a single 'poi_text' field.
    Missing values are treated as empty strings.
    Multiple spaces are collapsed.
    """
    df = df.copy()

    available = [col for col in POI_TEXT_COLS if col in df.columns]
    missing = [col for col in POI_TEXT_COLS if col not in df.columns]
    if missing:
        print(f"[text_join] Warning - missing columns (will be skipped): {missing}")

    if not available:
        print("[text_join] Warning - no text columns available, poi_text set to empty")
        df["poi_text"] = ""
        return df


    df["poi_text"] = (
        df[available]
        .fillna("")
        .apply(lambda row: " ".join(row.values.astype(str)), axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    empty_count = (df["poi_text"] == "").sum()
    print(f"[text_join] poi_text created for {len(df) - empty_count} / {len(df)} rows")
    print(f"[text_join] Sample:\n{df['poi_text'].dropna().head(3).to_string()}")
    return df


def print_coverage(df: pd.DataFrame) -> None:
    """Print fill rate for key columns."""

    if len(df) == 0:
        print("[text_join] Empty dataframe")
        return
    
    cols = ["name", "brand", "category_final", "category_group", "cuisine_clean", "addr:street"]
    print("\n[text_join] Coverage report:")
    for col in cols:
        if col in df.columns:
            filled = df[col].notna().sum()
            pct = round(filled / len(df) * 100, 1)
            print(f"  {col}: {filled} / {len(df)} ({pct}%)")


def run(df: pd.DataFrame) -> pd.DataFrame:
    print_coverage(df)
    df = build_poi_text(df)
    return df
