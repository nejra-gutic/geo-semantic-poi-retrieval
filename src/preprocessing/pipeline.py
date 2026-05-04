"""
preprocessing/pipeline.py
--------------------------
Orchestrates all preprocessing steps in order.
Each step is a separate module under src/preprocessing/.

Steps:
  1. clean        - column selection, category creation, name filling, category standardization
  2. address      - addr:street cleanup
  3. flags        - wheelchair_accessible, has_takeaway binary flags
  4. cuisine      - cuisine normalization and mapping
  5. opening_hours - parse OSM opening_hours into per-day columns + is_24_7
  6. normalize    - unidecode + lowercase for text columns
  7. text_join    - build poi_text field for NLP
  8. geometry     - validate/preserve geometry column
"""

import pandas as pd

from src.preprocessing import (
    clean,
    address,
    flags,
    cuisine,
    opening_hours,
    normalize,
    text_join,
    geometry,
)


FINAL_COLS = [
    "geometry",
    "name", "name_norm",
    "brand", "brand_norm",
    "amenity", "shop", "tourism",
    "category", "category_final",
    "cuisine", "cuisine_clean", 
    "opening_hours", "opening_hours_norm",
    "is_24_7",
    "mo_open", "mo_close",
    "tu_open", "tu_close",
    "we_open", "we_close",
    "th_open", "th_close",
    "fr_open", "fr_close",
    "sa_open", "sa_close",
    "su_open", "su_close",
    "wheelchair", "wheelchair_accessible",
    "takeaway", "has_takeaway",
    "addr:street", "addr_norm",
    "latitude", "longitude",
    "poi_text",
]


def run(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full preprocessing pipeline and return the cleaned dataframe."""
    print("=" * 50)
    print("Starting OSM POI preprocessing pipeline")
    print(f"Input shape: {df.shape}")
    print("=" * 50)

    print("\n--- Step 1: Clean ---")
    df = clean.run(df)

    print("\n--- Step 2: Address ---")
    df = address.run(df)

    print("\n--- Step 3: Flags ---")
    df = flags.run(df)

    print("\n--- Step 4: Cuisine ---")
    df = cuisine.run(df)

    print("\n--- Step 5: Opening Hours ---")
    df = opening_hours.run(df)

    print("\n--- Step 6: Normalize ---")
    df = normalize.run(df)

    print("\n--- Step 7: Text Join ---")
    df = text_join.run(df)

    print("\n--- Step 8: Geometry ---")
    df = geometry.run(df)

    # Keep only final columns that exist
    existing_final = [col for col in FINAL_COLS if col in df.columns]
    df_out = df[existing_final].copy()

    print("\n" + "=" * 50)
    print(f"Pipeline complete. Output shape: {df_out.shape}")
    print(f"Output columns: {df_out.columns.tolist()}")
    print("=" * 50)
    return df_out
