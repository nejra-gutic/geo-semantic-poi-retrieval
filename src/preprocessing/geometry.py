"""
preprocessing/geometry.py
-------------------------
Extracts centroid coordinates from OSM geometry column.
Handles: Point, Polygon, MultiPolygon, LineString -> all reduced to centroid.
Adds latitude and longitude columns to the dataframe.
"""

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point


def extract_centroid_coords(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract latitude and longitude from geometry column.
    All geometry types (Polygon, MultiPolygon, LineString) are reduced to centroid.
    Point geometries are used directly.
    """
    df = df.copy()

    if "geometry" not in df.columns:
        print("[geometry] Warning - no geometry column found, skipping")
        df["latitude"] = None
        df["longitude"] = None
        return df

    gdf = gpd.GeoDataFrame(df, geometry="geometry")

    type_counts = gdf.geometry.geom_type.value_counts()
    print(f"[geometry] Geometry types:\n{type_counts.to_string()}")

    centroids = gdf.geometry.to_crs("EPSG:3857").centroid.to_crs("EPSG:4326")
    df["latitude"] = centroids.y
    df["longitude"] = centroids.x

    filled = df["latitude"].notna().sum()
    print(f"[geometry] Coordinates extracted: {filled} / {len(df)}")
    return df


def run(df: pd.DataFrame) -> pd.DataFrame:
    return extract_centroid_coords(df)