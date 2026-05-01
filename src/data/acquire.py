"""
acquire.py
----------
Fetches POI data from OpenStreetMap using osmnx and selects relevant columns.

Based on: 02_data_acquisition.ipynb + 05_osm_data_cleaning.ipynb

Steps:
  1. Fetch POIs from OSM for a given place
  2. Select relevant columns
  3. Merge amenity/shop/tourism → category
  4. Fill name from brand where missing
  5. Save raw selected data to CSV
"""

import osmnx as ox
import pandas as pd


# OSM tags to fetch
OSM_TAGS = {
    "amenity": True,
    "shop": True,
    "tourism": True,
}

# Columns to keep from the raw OSM fetch
# brand is used to fill name, not kept as a final column
SELECTED_COLS = [
    "geometry",
    "name",
    "brand",
    "amenity",
    "shop",
    "tourism",
    "cuisine",
    "opening_hours",
    "wheelchair",
    "takeaway",
    "addr:street",
]


def fetch_pois(place: str) -> pd.DataFrame:
    """
    Fetches POIs from OSM for a given place name.
    Returns a GeoDataFrame with all raw OSM columns.
    """
    print(f"[acquire.py] Fetching POIs for: {place}")
    pois = ox.features_from_place(place, OSM_TAGS)
    print(f"[acquire.py] Fetched {len(pois)} POIs with {len(pois.columns)} columns.")
    return pois


def select_columns(pois: pd.DataFrame) -> pd.DataFrame:
    """
    Selects relevant columns from the raw OSM GeoDataFrame.
    Only keeps columns that actually exist in the dataset.
    """
    existing = [col for col in SELECTED_COLS if col in pois.columns]
    missing = [col for col in SELECTED_COLS if col not in pois.columns]

    if missing:
        print(f"[acquire.py] Columns not found in dataset (skipped): {missing}")

    df = pois[existing].copy()
    print(f"[acquire.py] Selected {len(existing)} columns.")
    return df


def merge_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges amenity, shop, tourism into a single 'category' column.
    Priority: amenity > shop > tourism
    (amenity is most specific for search purposes)
    """
    cols = [c for c in ["amenity", "shop", "tourism"] if c in df.columns]
    if not cols:
        print("[acquire.py] Warning: no category columns found.")
        return df

    df["category"] = df[cols[0]]
    for col in cols[1:]:
        df["category"] = df["category"].fillna(df[col])

    # Drop the original separate columns
    df = df.drop(columns=cols)

    n_filled = df["category"].notna().sum()
    print(f"[acquire.py] category: {n_filled}/{len(df)} rows filled.")
    return df


def fill_name_from_brand(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fills missing name with brand where available (e.g. no name but brand = 'Starbucks').
    Drops brand column afterwards — not needed as a standalone field.
    """
    if "brand" not in df.columns:
        return df

    if "name" not in df.columns:
        df["name"] = None

    before = df["name"].isna().sum()
    df["name"] = df["name"].fillna(df["brand"])
    after = df["name"].isna().sum()

    print(f"[acquire.py] Filled {before - after} missing names from brand.")
    df = df.drop(columns=["brand"])
    return df


def acquire(place: str, save_path: str = None) -> pd.DataFrame:
    """
    Full acquisition pipeline:
      1. Fetch from OSM
      2. Select columns
      3. Merge category
      4. Fill name from brand

    Args:
        place     : place name for OSM query (e.g. "Portland, Oregon, USA")
        save_path : if provided, saves the result as CSV

    Returns:
        DataFrame with selected, lightly prepared columns.
    """
    pois = fetch_pois(place)
    df = select_columns(pois)
    df = merge_category(df)
    df = fill_name_from_brand(df)

    print(f"\n[acquire.py] Final shape: {df.shape}")
    print(f"[acquire.py] Columns: {df.columns.tolist()}")

    if save_path:
        df.to_csv(save_path, index=False)
        print(f"[acquire.py] Saved to: {save_path}")

    return df


if __name__ == "__main__":
    df = acquire(
        place="Portland, Oregon, USA",
        save_path="data/raw/pois_portland_raw.csv"
    )
