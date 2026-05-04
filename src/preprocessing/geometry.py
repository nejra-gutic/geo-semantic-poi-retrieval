"""
preprocessing/geometry.py
-------------------------
Handles geometry-related operations on OSM GeoDataFrames.
Preserves the geometry column through the pipeline when available.
"""

import pandas as pd


def preserve_geometry_col(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure geometry column is kept in the final column selection.
    If geometry is not present, logs a warning and continues.
    """
    if "geometry" in df.columns:
        print(f"[geometry] geometry column present, type: {df['geometry'].dtype}")
    else:
        print("[geometry] Warning - no geometry column found in dataframe")
    return df


def run(df: pd.DataFrame) -> pd.DataFrame:
    return preserve_geometry_col(df)
