"""
main.py
-------
Entry point for the OSM POI preprocessing pipeline.

Usage:
    python main.py --input data/raw/pois_portland_oregon_usa.geojson --output data/processed/cleaned_pois
"""

import argparse
import geopandas as gpd
from shapely.geometry import Point
from src.utils.io import load_data, save_csv, save_geojson
from src.preprocessing import pipeline


def main():
    parser = argparse.ArgumentParser(
        description="OSM POI data cleaning and preprocessing pipeline."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input file (.csv or .geojson)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path without extension (e.g. data/processed/cleaned_pois)",
    )
    args = parser.parse_args()

    df = load_data(args.input)
    df_clean = pipeline.run(df)

    # Save CSV
    save_csv(df_clean, args.output + ".csv")

    # Save GeoJSON — reconstruct Point geometry from lat/lon
    if "latitude" in df_clean.columns and "longitude" in df_clean.columns:
        gdf = gpd.GeoDataFrame(
            df_clean,
            geometry=df_clean.apply(
                lambda row: Point(row["longitude"], row["latitude"])
                if row["longitude"] is not None and row["latitude"] is not None
                else None,
                axis=1,
            ),
            crs="EPSG:4326",
        )
        time_cols = [f"{d}_{t}" for d in ["mo","tu","we","th","fr","sa","su"] for t in ["open","close"]]
        for col in time_cols:
            if col in gdf.columns:
                gdf[col] = gdf[col].where(gdf[col].notna(), other=None).astype(str)
                gdf[col] = gdf[col].replace("nan", None)
        save_geojson(gdf, args.output + ".geojson")
    else:
        print("[main] Warning - no lat/lon columns, skipping GeoJSON output")


if __name__ == "__main__":
    main()