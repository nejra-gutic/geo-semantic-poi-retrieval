"""
utils/io.py
-----------
Load and save utilities for CSV and GeoJSON formats.
"""

import pandas as pd
import geopandas as gpd
from pathlib import Path


def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV file into a DataFrame."""
    df = pd.read_csv(path)
    print(f"[io] Loaded {len(df)} rows from {path}")
    return df


def save_csv(df: pd.DataFrame, path: str) -> None:
    """Save DataFrame to CSV."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[io] Saved {len(df)} rows to {path}")


def save_geojson(df: pd.DataFrame, path: str, crs: str = "EPSG:4326") -> None:
    """
    Save DataFrame with lat/lon columns to GeoJSON.
    Requires 'lat' and 'lon' columns (from geometry.py).
    """
    from shapely.geometry import Point

    Path(path).parent.mkdir(parents=True, exist_ok=True)

    geometry = [
        Point(lon, lat) if pd.notna(lat) and pd.notna(lon) else None
        for lat, lon in zip(df["lat"], df["lon"])
    ]

    gdf = gpd.GeoDataFrame(df.drop(columns=["geometry"], errors="ignore"),
                            geometry=geometry, crs=crs)

    gdf.to_file(path, driver="GeoJSON")
    print(f"[io] Saved GeoJSON ({len(gdf)} rows) to {path}")


def save_outputs(df: pd.DataFrame, base_path: str, name: str = "pois_processed") -> None:
    """
    Saves both CSV and GeoJSON to base_path/name.csv and base_path/name.geojson.
    """
    save_csv(df, f"{base_path}/{name}.csv")

    if "lat" in df.columns and "lon" in df.columns:
        save_geojson(df, f"{base_path}/{name}.geojson")
    else:
        print("[io] Skipping GeoJSON: no lat/lon columns found.")
