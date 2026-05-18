"""
preprocessing/clean.py
----------------------
Handles:
  - Column selection from raw OSM data
  - Raw category creation (amenity > shop > tourism priority)
  - Missing name resolution (name -> brand -> 'unknown')
  - Category cleanup without losing specific POI type
  - High-level category grouping for filtering/reporting

Category columns:
  - category:       raw OSM category selected from amenity/shop/tourism
  - category_final: cleaned specific category, e.g. fuel, parking, cafe, pharmacy
  - category_group: broad group, e.g. transport, utility, food_drink, service, shop
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

FOOD_DRINK_CATEGORIES = {
    "restaurant",
    "fast_food",
    "cafe",
    "bar",
    "pub",
    "bakery",
    "ice_cream",
    "food_court",
}

SERVICE_CATEGORIES = {
    "pharmacy",
    "hospital",
    "clinic",
    "doctors",
    "dentist",
    "bank",
    "atm",
    "veterinary",
    "post_office",
    "police",
    "fire_station",
    "library",
}

SHOP_CATEGORIES = {
    "convenience",
    "supermarket",
    "clothes",
    "furniture",
    "gift",
    "books",
    "mobile_phone",
    "beauty",
    "hairdresser",
    "bicycle",
    "electronics",
    "shoes",
}

TYPO_FIXES = {
    "student_accomodation": "student_accommodation",
}

NOISE_CATEGORIES = {"yes", "vacant", "trade", "bed", "art", "car"}

RARE_THRESHOLD = 10


CATEGORY_GROUPS = {
    "transport": TRANSPORT_CATEGORIES,
    "utility": UTILITY_CATEGORIES,
    "community": COMMUNITY_CATEGORIES,
    "food_drink": FOOD_DRINK_CATEGORIES,
    "service": SERVICE_CATEGORIES,
    "shop": SHOP_CATEGORIES,
}


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
    Merge amenity, shop, tourism into a single raw 'category' column.
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


def clean_specific_category(cat):
    """
    Clean a raw OSM category while keeping the specific POI type.

    Important: transport-like categories are no longer collapsed into
    'transport'. For example, 'fuel' stays 'fuel' and 'parking' stays 'parking'.
    """
    if pd.isna(cat):
        return None

    cat = str(cat).strip().lower()
    cat = TYPO_FIXES.get(cat, cat)

    if cat in NOISE_CATEGORIES:
        return "other"

    return cat


def assign_category_group(category_final, category_counts):
    """
    Assign a broad group from a cleaned specific category.
    Rare categories are grouped as 'other' but remain preserved in category_final.
    """
    if pd.isna(category_final):
        return None

    if category_final == "other":
        return "other"

    for group_name, categories in CATEGORY_GROUPS.items():
        if category_final in categories:
            return group_name

    if category_counts.get(category_final, 0) < RARE_THRESHOLD:
        return "other"

    return "other"


def standardize_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create:
      - category_final: cleaned specific category
      - category_group: broad category used for high-level filtering/reporting

    This avoids losing specific categories like fuel, parking, taxi, etc.
    """
    df = df.copy()

    df["category_final"] = df["category"].apply(clean_specific_category)
    category_counts = df["category_final"].value_counts()
    df["category_group"] = df["category_final"].apply(
        lambda cat: assign_category_group(cat, category_counts)
    )

    print(f"[clean] Specific categories: {df['category_final'].nunique(dropna=True)}")
    print(df["category_final"].value_counts().head(15).to_string())
    print(f"\n[clean] Category groups: {df['category_group'].nunique(dropna=True)}")
    print(df["category_group"].value_counts().to_string())
    return df


def run(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full cleaning pipeline on a raw OSM dataframe."""
    df = select_columns(df)
    df = create_category(df)
    df = fill_missing_names(df)
    df = standardize_category(df)
    return df
