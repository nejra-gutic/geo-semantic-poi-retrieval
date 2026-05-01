"""
pipeline.py
-----------
Runs the full preprocessing pipeline in order.

Steps:
  1. clean.py          → *_clean columns
  2. normalize.py      → *_norm columns
  3. opening_hours.py  → boolean flags + opening_hours_norm
  4. flags.py          → wheelchair_accessible, has_takeaway
  5. cuisine.py        → cuisine_tags, cuisine_norm
  6. address.py        → addr_street, addr_street_num, addr_city
  7. geometry.py       → lat, lon
  8. text_join.py      → poi_text

Final output schema:
  # Original fields
  geometry, name, category, category_final, cuisine,
  opening_hours, wheelchair, takeaway, addr

  # Normalized text
  name_clean, name_norm
  category_clean, category_norm
  cuisine_clean, cuisine_norm, cuisine_tags
  addr_clean, addr_norm, addr_street, addr_street_num, addr_city

  # Opening hours flags
  opening_hours_norm, is_24_7, has_opening_hours,
  by_appointment, has_weekend_hours, weekday_only, works_every_day

  # Boolean flags
  wheelchair_accessible, has_takeaway

  # Geo
  lat, lon

  # Text for retrieval
  poi_text
"""

import pandas as pd
import time

from .clean import apply_cleaning
from .normalize import apply_normalization
from .opening_hours import apply_opening_hours
from .flags import apply_flags
from .cuisine import apply_cuisine
from .address import apply_address
from .geometry import extract_lat_lon
from .text_join import apply_text_join


def run_pipeline(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Runs the full preprocessing pipeline on a raw OSM POI DataFrame.

    Args:
        df      : raw DataFrame (from OSM / cleaned_osm_data_v2.csv)
        verbose : print step summaries

    Returns:
        Processed DataFrame with all normalized, flag, and text columns.
    """
    start = time.time()

    print("=" * 60)
    print("PREPROCESSING PIPELINE")
    print("=" * 60)
    print(f"Input shape: {df.shape}")
    print()

    steps = [
        ("1. Cleaning",       apply_cleaning),
        ("2. Normalization",  apply_normalization),
        ("3. Opening Hours",  apply_opening_hours),
        ("4. Flags",          apply_flags),
        ("5. Cuisine",        apply_cuisine),
        ("6. Address",        apply_address),
        ("7. Geometry",       extract_lat_lon),
        ("8. Text Join",      apply_text_join),
    ]

    for step_name, step_fn in steps:
        print(f"── {step_name}")
        df = step_fn(df)
        print()

    elapsed = time.time() - start
    print("=" * 60)
    print(f"Done. Output shape: {df.shape}")
    print(f"Time: {elapsed:.1f}s")
    print("=" * 60)

    return df


def get_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns only the final output columns in clean order.
    Drops intermediate _clean columns to keep output tidy.
    """
    desired_order = [
        # Geo
        "geometry", "lat", "lon",

        # Identity
        "name", "name_norm",
        "category", "category_final", "category_norm",

        # Cuisine
        "cuisine", "cuisine_norm", "cuisine_tags",

        # Address
        "addr", "addr_norm", "addr_street", "addr_street_num", "addr_city",

        # Opening hours
        "opening_hours", "opening_hours_norm",
        "is_24_7", "has_opening_hours", "works_every_day",
        "has_weekend_hours", "weekday_only", "by_appointment",

        # Flags
        "wheelchair", "wheelchair_accessible",
        "takeaway", "has_takeaway",

        # Text retrieval
        "poi_text",
    ]

    # Only keep columns that actually exist
    cols = [c for c in desired_order if c in df.columns]
    return df[cols]
