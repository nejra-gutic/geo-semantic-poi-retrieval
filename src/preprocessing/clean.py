"""
preprocessing/clean.py
----------------------
Handles:
  - Column selection from raw OSM data
  - Category column creation (amenity > shop > tourism priority)
  - Missing name resolution (name -> brand -> 'unknown')
  - Category standardization (grouping, rare -> other, typo fixes)
"""

import pandas as pd


SELECTED_COLS = [
    "name",
    "amenity",
    "shop",
    "tourism",
    "opening_hours",
    "wheelchair",
    "cuisine",
    "addr:street",
    "brand",
    "takeaway",
]

TRANSPORT_CATEGORIES = {
    "parking",
    "parking_space",
    "parking_entrance",
    "bicycle_parking",
    "bicycle_rental",
    "charging_station",
    "fuel",
    "car_repair",
    "car_parts",
    "car_wash",
    "car_rental",
    "motorcycle_parking",
    "bicycle_repair_station",
    "vehicle_inspection",
    "taxi",
}

UTILITY_CATEGORIES = {
    "bench",
    "waste_basket",
    "letter_box",
    "post_box",
    "toilets",
    "drinking_water",
    "public_bookcase",
    "recycling",
    "waste_disposal",
    "telephone",
    "compressed_air",
    "parcel_locker",
    "vending_machine",
}

COMMUNITY_CATEGORIES = {
    "community_centre",
    "social_facility",
    "social_centre",
}

TYPO_FIXES = {
    "student_accomodation": "student_accommodation",
}

NOISE_CATEGORIES = ["yes", "vacant", "trade", "bed", "art", "car"]

RARE_THRESHOLD = 10


def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only relevant OSM columns that exist in the dataframe."""
    existing = [col for col in SELECTED_COLS if col in df.columns]
    missing = [col for col in SELECTED_COLS if col not in df.columns]
    if missing:
        print(f"[clean] Warning - columns not found in data: {missing}")
    print(f"[clean] Selected {len(existing)} columns: {existing}")
    result = df[existing].copy()
    if "geometry" in df.columns:
        result["geometry"] = df["geometry"]
    return result


def create_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge amenity, shop, tourism into a single 'category' column.
    Priority: amenity > shop > tourism.
    """
    df = df.copy()
    df["category"] = df["amenity"].fillna(df["shop"]).fillna(df["tourism"])

    multiple = df[["amenity", "shop", "tourism"]].notna().sum(axis=1)
    print(f"[clean] Rows with 0 category tags: {(multiple == 0).sum()}")
    print(f"[clean] Rows with 1 category tag:  {(multiple == 1).sum()}")
    print(f"[clean] Rows with 2+ category tags: {(multiple >= 2).sum()}")
    return df


def fill_missing_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing name with brand where available.
    Remaining missing names -> 'unknown'.
    """
    df = df.copy()

    no_name_before = df["name"].isna().sum()
    df["name"] = df["name"].fillna(df["brand"])
    filled_from_brand = no_name_before - df["name"].isna().sum()

    df["name"] = df["name"].fillna("unknown")
    filled_as_unknown = (df["name"] == "unknown").sum()

    print(f"[clean] Names filled from brand:   {filled_from_brand}")
    print(f"[clean] Names set to 'unknown':    {filled_as_unknown}")
    print(f"[clean] Total rows:                {len(df)}")
    return df


def standardize_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map raw categories into standardized groups:
      - transport, utility, community buckets
      - rare categories (< threshold) -> 'other'
      - noise / invalid values -> 'other'
      - typo corrections
    """
    df = df.copy()
    category_counts = df["category"].value_counts()

    def map_category(cat):
        if pd.isna(cat):
            return None
        if cat in TRANSPORT_CATEGORIES:
            return "transport"
        if cat in UTILITY_CATEGORIES:
            return "utility"
        if cat in COMMUNITY_CATEGORIES:
            return "community"
        if category_counts.get(cat, 0) < RARE_THRESHOLD:
            return "other"
        return cat

    df["category_final"] = df["category"].apply(map_category)

    # Fix typos
    df["category_final"] = df["category_final"].replace(TYPO_FIXES)

    # Remove noisy values
    df["category_final"] = df["category_final"].replace(NOISE_CATEGORIES, "other")

    total = len(df["category_final"].value_counts())
    print(f"[clean] Final unique categories: {total}")
    print(df["category_final"].value_counts().head(15).to_string())
    return df


def run(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full cleaning pipeline on a raw OSM dataframe."""
    df = select_columns(df)
    df = create_category(df)
    df = fill_missing_names(df)
    df = standardize_category(df)
    return df
