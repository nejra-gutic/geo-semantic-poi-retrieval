"""
preprocessing/cuisine.py
------------------------
Normalizes OSM cuisine tags:
  - Maps variants/typos to canonical names (CUISINE_MAP)
  - Handles semicolon-separated multi-cuisine values
  - Deduplicates and joins into a clean string
"""

import pandas as pd


CUISINE_MAP = {
    # coffee variants
    "coffee_shop": "coffee",
    "coffe_shop": "coffee",
    "cafe": "coffee",
    # tex-mex -> mexican
    "tex-mex": "mexican",
    "tacos": "mexican",
    "taco": "mexican",
    "birrieria": "mexican",
    "mexican_restaurant": "mexican",
    # bubble_tea -> tea
    "bubble_tea": "tea",
    "teahouse": "tea",
    # donut -> bakery
    "donut": "bakery",
    "pastry": "bakery",
    "cupcakes": "bakery",
    "bread": "bakery",
    "cake": "bakery",
    "pretzel": "bakery",
    # steak_house -> steak
    "steak_house": "steak",
    # fish_and_chips -> seafood
    "fish_and_chips": "seafood",
    # asian_fusion -> asian
    "asian_fusion": "asian",
    # noodles -> noodle
    "noodles": "noodle",
    # brunch -> breakfast
    "brunch": "breakfast",
    "diner": "breakfast",
    "pancake": "breakfast",
    # bbq
    "barbecue": "bbq",
    "korean;barbecue": "bbq",
    # other
    "wings": "chicken",
    "fried_chicken": "chicken",
    "frozen_yogurt": "ice_cream",
    "smoothie": "juice",
    "fine_dining": "regional",
    "melts": "sandwich",
    "cheese_steak": "sandwich",
    "hot_dog": "sandwich",
    "fries": "american",
    "organic": "regional",
}


def parse_cuisine(value) -> str | None:
    """
    Normalize a raw OSM cuisine string.
    Handles semicolon-separated lists, maps aliases, deduplicates.
    Returns None for missing values.
    """
    if pd.isna(value) or str(value).strip() == "":
        return None
    tags = [t.strip().lower() for t in str(value).split(";")]
    tags = [CUISINE_MAP.get(t, t) for t in tags]
    tags = list(dict.fromkeys(tags))  # deduplicate while preserving order
    return " ".join(tags)


def add_cuisine_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Add cuisine_clean column to dataframe."""
    df = df.copy()
    if "cuisine" not in df.columns:
        print("[cuisine] Warning - 'cuisine' column not found, skipping")
        df["cuisine_clean"] = None
        return df

    df["cuisine_clean"] = df["cuisine"].apply(parse_cuisine)

    filled = df["cuisine_clean"].notna().sum()
    unique = df["cuisine_clean"].nunique()
    print(f"[cuisine] Cuisine filled: {filled} / {len(df)} ({round(filled/len(df)*100,1)}%)")
    print(f"[cuisine] Unique cuisine values: {unique}")
    return df


def run(df: pd.DataFrame) -> pd.DataFrame:
    return add_cuisine_clean(df)
