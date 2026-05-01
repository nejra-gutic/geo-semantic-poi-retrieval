"""
geometry.py
-----------
Extracts representative point (lat, lon) from OSM geometries.

Strategy:
  - Point      → use as-is
  - Polygon    → centroid
  - LineString → midpoint (interpolate at 0.5)
  - MultiX     → centroid of envelope

Both CSV (lat/lon columns) and GeoJSON (geometry column) outputs are preserved.
"""

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon, LineString, MultiPolygon, MultiLineString
from shapely import wkt


def _representative_point(geom):
    """
    Returns a shapely Point representing the 'center' of any geometry type.
    """
    if geom is None or (hasattr(geom, 'is_empty') and geom.is_empty):
        return None

    geom_type = geom.geom_type

    if geom_type == "Point":
        return geom

    elif geom_type == "Polygon":
        return geom.centroid

    elif geom_type == "LineString":
        # midpoint along the line
        return geom.interpolate(0.5, normalized=True)

    elif geom_type in ("MultiPolygon", "MultiLineString", "GeometryCollection"):
        return geom.envelope.centroid

    else:
        # fallback
        return geom.centroid


def extract_lat_lon(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds 'lat' and 'lon' columns from the 'geometry' column.

    Handles:
      - GeoDataFrame with shapely geometry objects
      - Regular DataFrame with geometry stored as WKT strings

    Returns df with 'lat' and 'lon' columns added.
    Original 'geometry' column is preserved for GeoJSON export.
    """
    df = df.copy()

    if "geometry" not in df.columns:
        print("[geometry.py] Warning: no 'geometry' column found, skipping.")
        df["lat"] = None
        df["lon"] = None
        return df

    # Parse WKT strings if geometry is not already shapely objects
    sample = df["geometry"].dropna().iloc[0] if df["geometry"].notna().any() else None
    if sample is not None and isinstance(sample, str):
        df["geometry"] = df["geometry"].apply(
            lambda x: wkt.loads(x) if pd.notna(x) and x != "" else None
        )

    def get_point(geom):
        pt = _representative_point(geom)
        if pt is None:
            return None, None
        return pt.y, pt.x  # lat = y, lon = x

    coords = df["geometry"].apply(get_point)
    df["lat"] = coords.apply(lambda x: x[0])
    df["lon"] = coords.apply(lambda x: x[1])

    n_missing = df["lat"].isna().sum()
    n_total = len(df)
    print(f"[geometry.py] Extracted lat/lon for {n_total - n_missing}/{n_total} rows.")

    return df


def to_geodataframe(df: pd.DataFrame, crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    """
    Converts df with lat/lon columns to a GeoDataFrame with Point geometry.
    Use this for GeoJSON export.
    """
    df = df.copy()

    if "lat" not in df.columns or "lon" not in df.columns:
        raise ValueError("df must have 'lat' and 'lon' columns. Run extract_lat_lon() first.")

    geometry = [
        Point(lon, lat) if pd.notna(lat) and pd.notna(lon) else None
        for lat, lon in zip(df["lat"], df["lon"])
    ]

    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=crs)
    return gdf
