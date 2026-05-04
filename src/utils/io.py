"""
utils/io.py
-----------
Helper functions for reading and writing datasets.
Supports CSV and GeoJSON formats.
"""

import pandas as pd
import geopandas as gpd
from pathlib import Path


def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV file into a DataFrame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    print(f"[io] Loading CSV: {path}")
    df = pd.read_csv(path, index_col=0, low_memory=False)
    print(f"[io] Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def load_geojson(path: str) -> gpd.GeoDataFrame:
    """Load a GeoJSON file into a GeoDataFrame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    print(f"[io] Loading GeoJSON: {path}")
    gdf = gpd.read_file(path)
    print(f"[io] Loaded {len(gdf)} rows")
    return gdf


def save_csv(df: pd.DataFrame, path: str) -> None:
    """Save a DataFrame to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=True)
    print(f"[io] Saved CSV: {path}  shape={df.shape}")


def save_geojson(gdf: gpd.GeoDataFrame, path: str) -> None:
    """Save a GeoDataFrame to GeoJSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(path, driver="GeoJSON")
    print(f"[io] Saved GeoJSON: {path}")
